#!/usr/bin/env python3
import argparse
import glob
import os
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import BalancedPureDataset, load_jsonl
from engine import (
    AxisDeltaInjector,
    compute_axis_slices,
    discover_ffn_layers,
    get_amp_dtype,
    get_device,
    get_model_dtype,
    resolve_model_path,
)

def resolve_domain_path(name_or_path):
    if os.path.sep not in name_or_path and not name_or_path.endswith(".jsonl"):
        resolved = os.path.join("data", f"{name_or_path}.jsonl")
        if not os.path.exists(resolved):
            sys.exit(f"No data file for domain '{name_or_path}' (expected {resolved}). Run python dataset.py --domains {name_or_path} first.")
        return resolved
    return name_or_path

def auto_ood_paths(domain_path):
    data_dir = "data"
    if not os.path.isdir(data_dir):
        return []
    domain_abs = os.path.abspath(domain_path)
    all_jsonl = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    return [p for p in all_jsonl if os.path.abspath(p) != domain_abs]

def auto_domain_index():
    grafts = sorted(glob.glob("*.graft"))
    if not grafts:
        return 0
    max_idx = -1
    for g in grafts:
        try:
            with safe_open(g, framework="pt", device="cpu") as f:
                meta = f.metadata()
                if meta and "domain_index" in meta:
                    max_idx = max(max_idx, int(meta["domain_index"]))
        except Exception:
            continue
    return max_idx + 1


def endless_batches(loader):
    while True:
        for batch in loader:
            yield batch

def main():
    parser = argparse.ArgumentParser(description="Axis ARW 0.3 - Train Graft")
    parser.add_argument("--model", required=True)
    parser.add_argument("--domain_data", required=True,
                        help="Domain name (e.g. medical) or path to a JSONL file.")
    parser.add_argument("--ood_data", nargs="*", default=None,
                        help="OOD JSONL files. Defaults to all other .jsonl files in data/.")
    parser.add_argument("--domain_index", type=int, default=None,
                        help="Axis slice index. Defaults to next available after existing grafts.")
    parser.add_argument("--max_domains", type=int, default=4)
    parser.add_argument("--layer_range", default=None)
    parser.add_argument("--lambda_silence", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--fa2", action="store_true", help="Use FlashAttention 2 when supported by the model and device.")
    parser.add_argument("--output", default=None, help="Output graft path. Defaults to <domain>.graft.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    domain_arg = args.domain_data
    args.domain_data = resolve_domain_path(args.domain_data)

    if args.output is None:
        stem = os.path.splitext(os.path.basename(domain_arg))[0]
        args.output = f"{stem}.graft"

    if args.ood_data is not None and len(args.ood_data) == 0:
        sys.exit("--ood_data requires at least one file if provided.")
    if args.ood_data is None:
        args.ood_data = auto_ood_paths(args.domain_data)
        if not args.ood_data:
            sys.exit("No OOD data files found in data/. Run python dataset.py or provide --ood_data explicitly.")

    if args.domain_index is None:
        args.domain_index = auto_domain_index()
        print(f"[train] Auto domain_index: {args.domain_index}")
        print(f"[train] Auto OOD: {args.ood_data}")

    device = get_device(args.device)
    amp_dtype = get_amp_dtype(device)
    model_dtype = get_model_dtype(device)
    amp_device_type = device.type if device.type in ["cuda", "mps"] else "cpu"
    use_amp = amp_device_type != "cpu" and amp_dtype != torch.float32
    pin_memory = (device.type == "cuda")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    model_path = resolve_model_path(args.model)
    print(f"[train] Axis ARW 0.3 | device={device} | amp={amp_dtype}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True, "dtype": model_dtype}
    if args.fa2 and device.type == "cuda":
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).to(device)
    model.config.use_cache = False
    model.train()

    layers = discover_ffn_layers(model, args.layer_range)
    if not layers:
        sys.exit("No FFN layers found. Check the model architecture or --layer_range.")

    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        sys.exit("Model config does not expose hidden_size; cannot compute residual slices.")

    slices = compute_axis_slices(layers, args.domain_index, args.max_domains, hidden_size)
    if not slices:
        sys.exit("No valid slices computed. Check --domain_index and --max_domains.")

    for p in model.parameters():
        p.requires_grad = False

    delta_injector = AxisDeltaInjector(layers, slices)
    delta_params = list(delta_injector.parameters())
    if not delta_params:
        sys.exit("No trainable delta parameters were created.")

    optimizer = torch.optim.AdamW(delta_params, lr=args.lr, weight_decay=args.weight_decay)

    dataset = BalancedPureDataset(load_jsonl(args.domain_data), [t for p in args.ood_data for t in load_jsonl(p)], tokenizer, max_len=args.max_len)
    loader = endless_batches(DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=pin_memory))

    print(f"[train] Starting {args.steps} steps (lambda_silence={args.lambda_silence})...")
    running_lm, running_sil = 0.0, 0.0

    def save_artifact(step_count, out_path):
        delta_injector.detach()
        try:
            tensors = {}
            with torch.no_grad():
                for name, info in layers.items():
                    if name not in slices:
                        continue
                    safe_name = name.replace(".", "_")
                    tensors[name] = delta_injector.deltas[safe_name].detach().to(torch.bfloat16).cpu()
            meta = {
                "version": "0.3-axis-arw",
                "model": args.model,
                "domain_index": str(args.domain_index),
                "max_domains": str(args.max_domains),
                "layer_range": args.layer_range or "",
                "steps": str(step_count)
            }
            graft_path = out_path
            save_file(tensors, graft_path, metadata=meta)
            delta_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
            print(f"[train] Saved {graft_path} ({delta_bytes / (1024**2):.2f} MB)")
        finally:
            delta_injector.attach()

    for step in range(1, args.steps + 1):
        input_ids, mask = next(loader)
        input_ids = input_ids.view(-1, input_ids.size(-1)).to(device, non_blocking=pin_memory)
        mask = mask.view(-1, mask.size(-1)).to(device, non_blocking=pin_memory)
        delta_injector.clear_saved_energy()

        with torch.amp.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=use_amp):
            out = model(input_ids=input_ids)
            shift_logits = out.logits[:, :-1].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = mask[:, 1:].contiguous()
            ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none").view(shift_labels.shape)
            in_mask = (shift_mask == 1.0).float()
            in_count = in_mask.sum()
            lm_loss = (ce * in_mask).sum() / in_count if in_count > 0 else torch.tensor(0.0, device=device)

            activation_out_mask = (mask == 0.0).float()
            silence_loss, n_layers = torch.tensor(0.0, device=device), 0
            for safe_name in delta_injector.deltas:
                tok_energy = delta_injector.delta_token_energy(safe_name)
                if tok_energy is None:
                    continue
                om = activation_out_mask[:, :tok_energy.shape[1]]
                om_count = om.sum()
                if om_count > 0:
                    silence_loss += (tok_energy * om).sum() / om_count
                    n_layers += 1
            if n_layers > 0:
                silence_loss /= n_layers

        total_loss = lm_loss + args.lambda_silence * silence_loss
        total_loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(delta_params, max_norm=args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        delta_injector.clear_saved_energy()

        running_lm += lm_loss.item()
        running_sil += silence_loss.item()
        if step % 50 == 0:
            print(f"  step {step:5d}/{args.steps} | lm={running_lm/50:.4f} | silence={running_sil/50:.6f}")
            running_lm, running_sil = 0.0, 0.0

    save_artifact(args.steps, args.output)
    print("[train] Training complete.")

if __name__ == "__main__":
    main()