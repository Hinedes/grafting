import argparse
import math
import os

import torch

def compute_spectral_stats(w):
    """Compute effective rank and top singular values from a Gram matrix."""
    out_dim, in_dim = w.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w_dev = w.to(device)
    
    if out_dim <= in_dim:
        gram = w_dev @ w_dev.T
    else:
        gram = w_dev.T @ w_dev
        
    try:
        eigvals = torch.linalg.eigvalsh(gram)
        eigvals = torch.flip(eigvals, dims=[0])
        eigvals = torch.clamp(eigvals, min=0.0)
        
        if torch.any(torch.isnan(eigvals)):
            return None, None, "NaN eigenvalues detected (ill-conditioned Gram matrix)"
            
        singular_values = torch.sqrt(eigvals).cpu()
    except Exception as e:
        return None, None, str(e)

    if singular_values[0] == 0:
        return 0, [0.0] * 5, None
        
    sv_norm = singular_values / singular_values[0]
    eff_rank = (sv_norm > 0.1).sum().item()
    top5 = sv_norm[:min(5, len(sv_norm))].tolist()
    
    return eff_rank, top5, None

def autopsy_graft(path):
    if not os.path.exists(path):
        raise SystemExit(f"Artifact not found: {path}")

    print(f"\n{'='*80}")
    print(f" GRAFT AUTOPSY: {os.path.basename(path)}")
    print(f"{'='*80}")
    
    try:
        art = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        art = torch.load(path, map_location="cpu")
        
    meta_keys = ["version", "model", "domain_index", "max_domains", "steps", "layer_range"]
    print("\n[ Metadata ]")
    for k in meta_keys:
        if k in art:
            print(f"  {k:<15}: {art[k]}")

    grafts = art.get("grafts", {})
    if not grafts:
        print("No graft tensors found in artifact.")
        return

    total_params = 0
    total_sq_norm = 0.0
    global_max, global_min = -float('inf'), float('inf')
    sum_abs = 0.0
    
    layer_stats = []
    spectral_data = []
    
    layer_names = sorted(list(grafts.keys()))
    n_layers = len(layer_names)
    
    sample_indices = set([0, n_layers - 1])
    if n_layers > 4:
        sample_indices.update([n_layers // 4, n_layers // 2, 3 * n_layers // 4])

    print(f"\n[ Analyzing {n_layers} layers... ]")
    
    for idx, name in enumerate(layer_names):
        g = grafts[name]
        w = g["delta_slice"].float()
        
        numel = w.numel()
        total_params += numel
        total_sq_norm += w.norm().item()**2
        
        w_max, w_min = w.max().item(), w.min().item()
        global_max = max(global_max, w_max)
        global_min = min(global_min, w_min)
        sum_abs += w.abs().sum().item()
        
        sparsity = (w.abs() < 1e-5).sum().item() / numel
        
        layer_stats.append({
            "name": name,
            "shape": tuple(w.shape),
            "l2": w.norm().item(),
            "sparsity": sparsity
        })
        
        if idx in sample_indices:
            eff_rank, top5, err = compute_spectral_stats(w)
            spectral_data.append({"name": name, "eff_rank": eff_rank, "top5": top5, "err": err})

    avg_l2 = math.sqrt(total_sq_norm / max(1, total_params))
    mean_abs = sum_abs / max(1, total_params)
    avg_sparsity = sum(ls["sparsity"] for ls in layer_stats) / n_layers

    print(f"\n[ Global Weight Statistics ]")
    print(f"  Total Parameters : {total_params:,}")
    print(f"  Global L2 Norm   : {math.sqrt(total_sq_norm):.4f}")
    print(f"  Avg L2 per param : {avg_l2:.6f}")
    print(f"  Mean Abs Value   : {mean_abs:.6f}")
    print(f"  Min / Max Values : {global_min:.6f} / {global_max:.6f}")
    print(f"  Avg Sparsity     : {avg_sparsity*100:.2f}% (|w| < 1e-5)")

    print(f"\n[ Layer-wise L2 Norm & Sparsity (Sampled) ]")
    print(f"  {'Layer Name':<45} | {'Shape':<14} | {'L2 Norm':<10} | {'Sparsity':<8}")
    print(f"  " + "-"*85)
    
    display_indices = sorted(list(set([0, 1, n_layers//2, n_layers-2, n_layers-1])))
    display_indices = [i for i in display_indices if 0 <= i < n_layers]
    
    for i in display_indices:
        ls = layer_stats[i]
        shape_str = f"{ls['shape'][0]:>5} x {ls['shape'][1]:<5}"
        print(f"  {ls['name']:<45} | {shape_str:<14} | {ls['l2']:<10.4f} | {ls['sparsity']*100:>6.2f}%")

    print(f"\n[ Spectral Analysis (Effective Rank & Singular Value Decay) ]")
    print(f"  Effective Rank = # of singular values > 10% of the maximum.")
    print("  Higher = distributed/complex corrections. Lower = concentrated/simple bias.")
    print(f"  " + "-"*85)
    
    for sd in spectral_data:
        if sd["err"]:
            print(f"  {sd['name']:<45} | Decomposition failed: {sd['err']}")
        else:
            top5_str = ", ".join([f"{x:.3f}" for x in sd["top5"]])
            print(f"  {sd['name']:<45} | Rank: {sd['eff_rank']:>4} | Top 5 SVs (norm): [{top5_str}]")

    print(f"\n{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(description="Inspect AxisARW graft tensor statistics.")
    parser.add_argument("graft", help="Path to a .graft.pt artifact.")
    args = parser.parse_args()
    autopsy_graft(args.graft)

if __name__ == "__main__":
    main()
