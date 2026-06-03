#!/usr/bin/env python3
"""Evaluate, stack-test, and install AxisARW graft artifacts."""

import argparse
import copy
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import EvalDataset, load_jsonl
from engine import discover_ffn_layers, get_device, get_model_dtype, resolve_model_path


DEFAULT_MODEL = "HuggingFaceTB/SmolLM3-3B"
GRAFT_VERSION = "0.3-axis-arw"
REQUIRED_GRAFT_KEYS = {
    "category",
    "delta_slice",
    "inter_start",
    "inter_end",
    "res_start",
    "res_end",
}


def compute_ppl(model, dataset, device):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    total_loss, total_tokens = 0.0, 0
    model.eval()

    with torch.inference_mode():
        for input_ids in loader:
            input_ids = input_ids.to(device)
            out = model(input_ids=input_ids)
            shift_logits = out.logits[:, :-1].contiguous().float()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = shift_labels != dataset.pad
            num_tokens = shift_mask.sum().item()
            if num_tokens == 0:
                continue

            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view(shift_labels.shape)
            total_loss += ce[shift_mask].sum().item()
            total_tokens += num_tokens

    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


def load_graft(path):
    try:
        art = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        art = torch.load(path, map_location="cpu")

    if not isinstance(art, dict) or art.get("version") != GRAFT_VERSION:
        sys.exit(f"{path} is not a valid {GRAFT_VERSION} graft.")
    if not isinstance(art.get("grafts"), dict) or not art["grafts"]:
        sys.exit(f"{path} has no graft tensors.")

    art["_path"] = path
    return art


def resolve_base_model(artifacts, model_arg):
    if model_arg:
        return model_arg

    artifact_models = {a.get("model") for a in artifacts if a.get("model")}
    if len(artifact_models) == 1:
        return next(iter(artifact_models))
    if len(artifact_models) > 1:
        sys.exit("Grafts reference multiple base models; pass --model explicitly.")
    return DEFAULT_MODEL


def load_model(model_path, device, fa2=False):
    kwargs = {"trust_remote_code": True, "torch_dtype": get_model_dtype(device)}
    if fa2 and device.type == "cuda":
        kwargs["attn_implementation"] = "flash_attention_2"
    return AutoModelForCausalLM.from_pretrained(model_path, **kwargs).to(device)


def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _summarize(items, limit=3):
    shown = ", ".join(items[:limit])
    return shown if len(items) <= limit else f"{shown}, ... (+{len(items) - limit} more)"


def validate_graft_against_layers(art, path, layers):
    missing_layers, invalid = [], []

    for name, graft in art["grafts"].items():
        if name not in layers:
            missing_layers.append(name)
            continue

        missing_keys = sorted(REQUIRED_GRAFT_KEYS - set(graft))
        if missing_keys:
            invalid.append(f"{name}: missing keys {missing_keys}")
            continue

        mod = layers[name]["module"]
        out_features, in_features = mod.weight.shape
        category = graft["category"]
        inter_s, inter_e = graft["inter_start"], graft["inter_end"]
        res_s, res_e = graft["res_start"], graft["res_end"]

        if category != layers[name]["category"]:
            invalid.append(f"{name}: category mismatch")
            continue
        if category not in {"ffn_expand", "ffn_contract"}:
            invalid.append(f"{name}: invalid category")
            continue
        if inter_e <= inter_s or res_e <= res_s:
            invalid.append(f"{name}: invalid slice")
            continue
        if not torch.is_tensor(graft["delta_slice"]):
            invalid.append(f"{name}: delta_slice is not a tensor")
            continue

        inter_axis = out_features if category == "ffn_expand" else in_features
        res_axis = in_features if category == "ffn_expand" else out_features
        if inter_e > inter_axis or res_e > res_axis:
            invalid.append(f"{name}: slice exceeds axis")
            continue

        expected_shape = (
            (inter_e - inter_s, res_e - res_s)
            if category == "ffn_expand"
            else (res_e - res_s, inter_e - inter_s)
        )
        if tuple(graft["delta_slice"].shape) != expected_shape:
            invalid.append(f"{name}: shape mismatch")
            continue

        if "weight_shape" in graft and list(mod.weight.shape) != list(graft["weight_shape"]):
            invalid.append(f"{name}: source weight shape mismatch")

    if missing_layers:
        sys.exit(f"{path} references missing layers: {_summarize(missing_layers)}")
    if invalid:
        sys.exit(f"{path} is incompatible: {_summarize(invalid)}")


def validate_non_overlapping_grafts(artifacts, paths):
    max_domains = {art.get("max_domains") for art in artifacts}
    if len(max_domains) != 1 or None in max_domains:
        sys.exit("Stacked grafts must all declare the same max_domains value.")

    seen_domains, occupied = {}, {}
    for art, path in zip(artifacts, paths):
        domain_index = art.get("domain_index")
        if not isinstance(domain_index, int):
            sys.exit(f"{path} does not declare an integer domain_index.")
        if domain_index in seen_domains:
            sys.exit(f"Duplicate domain_index {domain_index}: {seen_domains[domain_index]} and {path}")
        seen_domains[domain_index] = path

        for name, graft in art["grafts"].items():
            key = (name, graft["category"])
            i_s, i_e = graft["inter_start"], graft["inter_end"]
            r_s, r_e = graft["res_start"], graft["res_end"]

            for pi_s, pi_e, pr_s, pr_e, prev_path in occupied.get(key, []):
                inter_overlaps = max(i_s, pi_s) < min(i_e, pi_e)
                residual_overlaps = max(r_s, pr_s) < min(r_e, pr_e)
                if inter_overlaps and residual_overlaps:
                    sys.exit(f"Overlapping slices for {name}: {prev_path} and {path}")

            occupied.setdefault(key, []).append((i_s, i_e, r_s, r_e, path))


def delta_output_energy(x, delta_info):
    category = delta_info["category"]
    i_s, i_e = delta_info["inter_start"], delta_info["inter_end"]
    r_s, r_e = delta_info["res_start"], delta_info["res_end"]
    delta_slice = delta_info["delta_slice"].to(device=x.device, dtype=x.dtype)

    if category == "ffn_expand":
        y = F.linear(x[..., r_s:r_e], delta_slice)
    else:
        y = F.linear(x[..., i_s:i_e], delta_slice)

    return (y.float().norm(dim=-1) / math.sqrt(y.size(-1))).mean().item()


def apply_graft_to_model(model, art, device):
    layers = discover_ffn_layers(model, art.get("layer_range"))
    validate_graft_against_layers(art, art.get("_path", "<artifact>"), layers)

    with torch.no_grad():
        for name, graft in art["grafts"].items():
            mod = layers[name]["module"]
            category = graft["category"]
            i_s, i_e = graft["inter_start"], graft["inter_end"]
            r_s, r_e = graft["res_start"], graft["res_end"]
            delta_slice = graft["delta_slice"].to(device=device, dtype=mod.weight.dtype)

            if category == "ffn_expand":
                mod.weight.data[i_s:i_e, r_s:r_e] += delta_slice
            else:
                mod.weight.data[r_s:r_e, i_s:i_e] += delta_slice


def print_eval_result(graft_path, vanilla_ppl, grafted_ppl):
    print(
        f"  {os.path.basename(graft_path):<25} | "
        f"Vanilla: {vanilla_ppl:.4f} | "
        f"Grafted: {grafted_ppl:.4f} | "
        f"dPPL: {grafted_ppl - vanilla_ppl:+.4f}"
    )


def evaluate_graft_pair(base_model, tokenizer, device, graft_path, data_path):
    art = load_graft(graft_path)
    dataset = EvalDataset(load_jsonl(data_path), tokenizer)
    vanilla_ppl = compute_ppl(base_model, dataset, device)

    grafted_model = copy.deepcopy(base_model)
    apply_graft_to_model(grafted_model, art, device)
    grafted_ppl = compute_ppl(grafted_model, dataset, device)
    print_eval_result(graft_path, vanilla_ppl, grafted_ppl)

    del grafted_model
    clear_cuda_cache()


def cmd_eval(args):
    device = get_device(args.device)
    art = load_graft(args.graft)
    model_path = resolve_model_path(resolve_base_model([art], args.model))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base_model = load_model(model_path, device, args.fa2)
    evaluate_graft_pair(base_model, tokenizer, device, args.graft, args.data)


def cmd_compare(args):
    if len(args.grafts) % 2 != 0:
        sys.exit("compare expects pairs: graft.pt data.jsonl [graft.pt data.jsonl ...]")

    pairs = list(zip(args.grafts[0::2], args.grafts[1::2]))
    artifacts = [load_graft(graft_path) for graft_path, _ in pairs]

    device = get_device(args.device)
    model_path = resolve_model_path(resolve_base_model(artifacts, args.model))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base_model = load_model(model_path, device, args.fa2)

    print("\nAxisARW graft evaluation")
    for graft_path, data_path in pairs:
        evaluate_graft_pair(base_model, tokenizer, device, graft_path, data_path)


def cmd_stack_test(args):
    if len(args.grafts) != len(args.data):
        sys.exit("stack-test requires one data file per graft.")

    device = get_device(args.device)
    artifacts = [load_graft(g) for g in args.grafts]
    validate_non_overlapping_grafts(artifacts, args.grafts)

    model_path = resolve_model_path(resolve_base_model(artifacts, args.model))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("\nSingle-graft baselines")
    single_ppls = {}
    for art, graft_path, data_path in zip(artifacts, args.grafts, args.data):
        model = load_model(model_path, device, args.fa2)
        apply_graft_to_model(model, art, device)
        ppl = compute_ppl(model, EvalDataset(load_jsonl(data_path), tokenizer), device)
        single_ppls[graft_path] = ppl
        print(f"  {os.path.basename(graft_path):<25} | PPL: {ppl:.4f}")
        del model
        clear_cuda_cache()

    print("\nStacked PPL and bleed diagnostics")
    stacked_model = load_model(model_path, device, args.fa2)
    all_layers = discover_ffn_layers(stacked_model)
    artifact_deltas = []

    for art in artifacts:
        art_layers = discover_ffn_layers(stacked_model, art.get("layer_range"))
        validate_graft_against_layers(art, art.get("_path", "<artifact>"), art_layers)
        deltas = {}

        with torch.no_grad():
            for name, graft in art["grafts"].items():
                mod = art_layers[name]["module"]
                category = graft["category"]
                i_s, i_e = graft["inter_start"], graft["inter_end"]
                r_s, r_e = graft["res_start"], graft["res_end"]
                delta_slice = graft["delta_slice"].to(device=device, dtype=mod.weight.dtype)

                if category == "ffn_expand":
                    mod.weight.data[i_s:i_e, r_s:r_e] += delta_slice
                else:
                    mod.weight.data[r_s:r_e, i_s:i_e] += delta_slice

                deltas[name] = {
                    "category": category,
                    "inter_start": i_s,
                    "inter_end": i_e,
                    "res_start": r_s,
                    "res_end": r_e,
                    "delta_slice": delta_slice.detach(),
                }

        artifact_deltas.append(deltas)

    saved_inputs = {}
    fwd_hooks = [
        info["module"].register_forward_hook(lambda _m, inp, _out, n=name: saved_inputs.update({n: inp[0].detach()}))
        for name, info in all_layers.items()
    ]

    try:
        for i, (graft_path, data_path) in enumerate(zip(args.grafts, args.data)):
            dataset = EvalDataset(load_jsonl(data_path), tokenizer)
            ppl = compute_ppl(stacked_model, dataset, device)
            delta_ppl = ppl - single_ppls[graft_path]
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            signal_energy, bleed_energy, batches = 0.0, 0.0, 0

            with torch.inference_mode():
                for input_ids in loader:
                    if batches >= 10:
                        break

                    input_ids = input_ids.to(device)
                    saved_inputs.clear()
                    _ = stacked_model(input_ids=input_ids)
                    x_float = {n: t.float() for n, t in saved_inputs.items()}

                    for name in all_layers:
                        if name not in x_float:
                            continue
                        x = x_float[name]
                        if name in artifact_deltas[i]:
                            signal_energy += delta_output_energy(x, artifact_deltas[i][name])
                        for j, deltas in enumerate(artifact_deltas):
                            if i != j and name in deltas:
                                bleed_energy += delta_output_energy(x, deltas[name])

                    batches += 1

            avg_signal = signal_energy / max(1, batches)
            avg_bleed = bleed_energy / max(1, batches)
            snr_db = 10 * math.log10(avg_signal / avg_bleed) if avg_bleed > 1e-10 and avg_signal > 0 else float("inf")
            status = "silent" if delta_ppl < 0.5 and snr_db > 10 else "noise"

            print(
                f"  {os.path.basename(graft_path):<25} | "
                f"Stacked: {ppl:.4f} (dPPL {delta_ppl:+.4f}) | "
                f"SNR: {snr_db:.1f} dB | {status}"
            )
    finally:
        for hook in fwd_hooks:
            hook.remove()


def cmd_install(args):
    device = get_device(args.device)
    artifacts = [load_graft(g) for g in args.graft]
    validate_non_overlapping_grafts(artifacts, args.graft)

    model_path = resolve_model_path(resolve_base_model(artifacts, args.model))
    print(f"[install] Baking {len(artifacts)} grafts into {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = load_model(model_path, device, args.fa2)

    for art, graft_path in zip(artifacts, args.graft):
        apply_graft_to_model(model, art, device)
        print(f"  Domain {art['domain_index']:>2} | {os.path.basename(graft_path)}")

    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"[install] Saved {args.output}")


def add_common_model_args(parser):
    parser.add_argument("--model", default=None, help=f"Base model path or HF id. Defaults to artifact metadata, then {DEFAULT_MODEL}.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fa2", action="store_true", help="Use FlashAttention 2 when supported by the model and device.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate, stack-test, and install AxisARW grafts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("eval", help="Evaluate one graft on one data file.")
    eval_parser.add_argument("--graft", required=True)
    eval_parser.add_argument("--data", required=True)
    add_common_model_args(eval_parser)

    compare_parser = subparsers.add_parser("compare", help="Evaluate graft/data pairs.")
    compare_parser.add_argument("--grafts", nargs="+", required=True, help="Pairs: graft.pt data.jsonl [graft.pt data.jsonl ...]")
    add_common_model_args(compare_parser)

    stack_parser = subparsers.add_parser("stack-test", help="Evaluate stacked graft interference.")
    stack_parser.add_argument("--grafts", nargs="+", required=True)
    stack_parser.add_argument("--data", nargs="+", required=True)
    add_common_model_args(stack_parser)

    install_parser = subparsers.add_parser("install", help="Bake grafts into a model directory.")
    install_parser.add_argument("--graft", nargs="+", required=True)
    install_parser.add_argument("--output", required=True)
    add_common_model_args(install_parser)

    args = parser.parse_args()
    {
        "eval": cmd_eval,
        "compare": cmd_compare,
        "stack-test": cmd_stack_test,
        "install": cmd_install,
    }[args.command](args)


if __name__ == "__main__":
    main()
