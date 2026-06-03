import math
import os
import re
import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore", message=".*TF32 behavior.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torchao")

def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)

def get_amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        if getattr(torch.version, "hip", None):
            return torch.bfloat16
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return torch.bfloat16
        return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32

def get_model_dtype(device: torch.device) -> torch.dtype:
    return get_amp_dtype(device)

def resolve_model_path(model_arg: str) -> str:
    if os.path.isdir(model_arg):
        return model_arg
    print(f"[model] Using Hugging Face model or cache entry: {model_arg}")
    return model_arg

def categorize_layer(name: str) -> str:
    n = name.lower()
    if any(x in n for x in ["gate_proj", "up_proj", "w1", "w3", "fc1", "c_fc", "dense_h_to_4h"]):
        return "ffn_expand"
    if any(x in n for x in ["down_proj", "w2", "fc2", "c_proj", "dense_4h_to_h"]):
        return "ffn_contract"
    return "skip"

def discover_ffn_layers(model, layer_range: str = None) -> dict:
    found = {}
    allowed_layers = None

    if layer_range:
        spec = layer_range.strip()
        try:
            if "," in spec:
                allowed_layers = {int(x.strip()) for x in spec.split(",") if x.strip()}
            elif "-" in spec:
                min_l, max_l = (int(x.strip()) for x in spec.split("-", 1))
                if min_l > max_l:
                    raise ValueError
                allowed_layers = set(range(min_l, max_l + 1))
            else:
                allowed_layers = {int(spec)}
            if not allowed_layers:
                raise ValueError
        except ValueError:
            sys.exit("Invalid layer_range. Use a layer number, comma list, or MIN-MAX range.")

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        cat = categorize_layer(name)
        if cat == "skip":
            continue

        layer_match = re.search(r"\.(\d+)\.", name)
        if layer_match and allowed_layers is not None and int(layer_match.group(1)) not in allowed_layers:
            continue
        if hasattr(module, "weight") and module.weight.ndim == 2:
            found[name] = {"module": module, "category": cat}

    return found

def compute_axis_slices(layers: dict, domain_index: int, max_domains: int, hidden_size: int) -> dict:
    if max_domains <= 0:
        raise ValueError("max_domains must be greater than zero.")
    if domain_index < 0 or domain_index >= max_domains:
        raise ValueError("domain_index must be in [0, max_domains).")

    res_start, res_end = (hidden_size * domain_index) // max_domains, (hidden_size * (domain_index + 1)) // max_domains
    slices = {}

    for name, info in layers.items():
        mod, cat = info["module"], info["category"]
        out_features, in_features = mod.weight.shape
        inter_axis = out_features if cat == "ffn_expand" else in_features
        inter_start, inter_end = (inter_axis * domain_index) // max_domains, (inter_axis * (domain_index + 1)) // max_domains
        if inter_end <= inter_start or res_end <= res_start:
            continue

        slices[name] = {
            "inter_start": inter_start,
            "inter_end": inter_end,
            "I_inter": inter_axis,
            "res_start": res_start,
            "res_end": res_end,
            "I_res": hidden_size,
            "category": cat,
        }

    return slices

class AxisDeltaInjector:
    def __init__(self, layers: dict, slices: dict):
        self.layers = layers
        self.slices = slices
        self.deltas = nn.ParameterDict()
        self.saved_energy = {}
        self._hooks = []

        for name, info in layers.items():
            if name not in slices:
                continue
            mod, s = info["module"], slices[name]
            inter_width, res_width = s["inter_end"] - s["inter_start"], s["res_end"] - s["res_start"]
            shape = (inter_width, res_width) if s["category"] == "ffn_expand" else (res_width, inter_width)
            self.deltas[name.replace(".", "_")] = nn.Parameter(
                torch.zeros(shape, device=mod.weight.device, dtype=torch.float32)
            )

        self.attach()

    def parameters(self):
        return self.deltas.values()

    def named_parameters(self):
        return self.deltas.items()

    def clear_saved_energy(self):
        self.saved_energy.clear()

    def _inject_hook(self, name: str, safe_name: str):
        def hook(mod, inp, out):
            x, s, delta = inp[0], self.slices[name], self.deltas[safe_name]
            if s["category"] == "ffn_expand":
                x_slice = x[..., s["res_start"]:s["res_end"]]
                delta_out = F.linear(x_slice, delta)
                out = out.clone()
                out[..., s["inter_start"]:s["inter_end"]] += delta_out
            else:
                x_slice = x[..., s["inter_start"]:s["inter_end"]]
                delta_out = F.linear(x_slice, delta)
                out = out.clone()
                out[..., s["res_start"]:s["res_end"]] += delta_out
            self.saved_energy[safe_name] = delta_out.float().norm(dim=-1) / math.sqrt(delta_out.size(-1))
            return out

        return hook

    def attach(self):
        if self._hooks:
            return
        for name, info in self.layers.items():
            if name not in self.slices:
                continue
            self._hooks.append(info["module"].register_forward_hook(self._inject_hook(name, name.replace(".", "_"))))

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def delta_token_energy(self, safe_name: str):
        return self.saved_energy.get(safe_name)
