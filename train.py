#!/usr/bin/env python3
import argparse
import sys

import torch
import torch.nn.functional as F
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

def endless_batches(loader):
    while True:
        for batch in loader:
            yield batch

def main():
    parser = argparse.ArgumentParser(description="Axis ARW 0.3 - Train Graft")
    parser.add_argument("--model", required=True)
    parser.add_argument("--domain_data", required=True)
    parser.add_argument("--ood_data", nargs="+", required=True)
    parser.add_argument("--domain_index", type=int, required=True)
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
    parser.add_argument("--output", default="graft.pt")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

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

    model_kwargs = {"trust_remote_code": True, "torch_dtype": model_dtype}
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
            graft = {}
            with torch.no_grad():
                for name, info in layers.items():
                    if name not in slices:
                        continue
                    safe_name = name.replace(".", "_")
                    s = slices[name]
                    graft[name] = {
                        "delta_slice": delta_injector.deltas[safe_name].detach().to(torch.bfloat16).cpu(),
                        "category": s["category"],
                        "inter_start": s["inter_start"], "inter_end": s["inter_end"],
                        "res_start": s["res_start"], "res_end": s["res_end"],
                        "weight_shape": list(info["module"].weight.shape),
                    }
            delta_bytes = sum(g["delta_slice"].numel() * g["delta_slice"].element_size() for g in graft.values())
            torch.save({"version": "0.3-axis-arw", "model": args.model, "layer_range": args.layer_range, "domain_index": args.domain_index, "max_domains": args.max_domains, "steps": step_count, "grafts": graft}, out_path)
            print(f"[train] Saved {out_path} ({delta_bytes / (1024**2):.2f} MB)")
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
