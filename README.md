# Grafting v0.3 (Axis ARW)

This architecture started as a simple daydream (the AuDHD kind, the looping kind): expose a frozen model to novel data, let its gradients ignore what it already knows, and capture its frantic reactions to out-of-distribution tokens. Pack those raw gradient descent responses into a standalone file, and you have a graft.

It sounded too simple. I thought, "This can't work." (I have no ML background, so what do I know.) Then the telemetry came in, my simple idea, somehow worked! And then I spent two more weeks making it work. ("It would appear, daydreaming can't code, but i, Hinedes, have a dream!")

Probing the residual stream on the SmolLM3‑3B medical graft revealed a +0.97 correlation between base model ignorance and graft‑induced loss reduction. The frozen model acts as a filter for its own ignorance. When the base model is confused, the graft corrects; when the base model is confident, the graft contributes essentially zero. It is a self‑distilling, opportunistic error‑catcher.

At install time, it completely abandons the overlapping mathematical superposition of v0.1. There is no decoding key. Instead, the 300 MB graft folds directly into a hard, isolated slice of the base weights:

$$W_{\text{new}}[s_y:e_y, s_x:e_x] = W_{\text{base}}[s_y:e_y, s_x:e_x] + \Delta_{\text{graft}}$$

The base model reads the graft natively because the graft physically becomes a room inside the host's FFN geometry. The domains never touch during expansion or contraction because their index ranges have a hard wall between them.

---

## The Mixer Problem (Blender Slander, Orthogonality Must Die)

Version 0.1 relied on rotated orthogonal subspaces. Beautiful on paper. Completely broken by the harsh reality of how things actually are.

Why? Because Transformer non‑linearities don't care about linear algebra. As was explained to me later on, the SwiGLU block uses element‑wise multiplication:

$$y = W_{down} \Big( \text{SiLU}(W_{gate} x) \odot (W_{up} x) \Big)$$

Pass a mathematically orthogonal signal through the SiLU, and it acts like a cryptographic mixer. The subspace shatters. The signal scatters everywhere.

So, Version 0.3 throws out the elegant math in favor of physical walls. It chops the FFN intermediate and residual widths into hard, axis‑aligned Boolean masks.

It's crude. But it's airtight. (Maybe.)

Disjoint support forces the cross‑terms to exactly zero: $\text{SiLU}(\Delta W_{gate}^A x) \odot \Delta W_{up}^B x = 0$. The mixer has nothing to multiply, meaning cross‑domain interference inside the FFN is structurally dead.

(I didn't touch the attention heads, by the way. The Stem Cell Hypothesis holds up: attention handles syntactic, domain‑agnostic routing. Factual knowledge lives in the FFN. Messing with attention is simply architecturally wrong.)

### Additional Graveyard Discoveries:

**The EMI Limiter Trap (v0.2):** In a panic to fix cross‑talk, we added an "Energy‑Matched Injection" volume limiter that dynamically squashed the graft's output during training. The optimizer fought back by growing massive, bloated FP32 weights (L2 Norm ≈ 145) to compensate. At inference those monstrosities exploded the logits and sent the model into degenerate token loops. Moral: don't choke the signal; the signal will choke you right back. (A fight you can't win)

**The Expansion Quadratic Trap:** We tried grafting both the expansion (`gate`/`up`) and contraction (`down`) projections at the same time. At higher step counts the cross‑multiplication of the deltas ($\Delta_{Gate} \odot \Delta_{Up}$) blew up the intermediate space like a firecracker in a tin can. Lesson learned: pick one target and leave the other alone.

---

## The Residual Tax (Or: The Noisy Roommates)

Inference is free. Training is brutal. And now i have OOM PTSD.

If you want to run the four‑domain stack on SmolLM3‑3B, you'd need something like AMD MI300X and about 130 GB of VRAM. You have to force massive sequence volume through the base model just to make the silence loss effective. (Machine Learning at its best. Bruteforcing at its finest.)

And then there's the residual tax.

Axis‑aligned slicing solves the FFN problem, but it exposes another one. The domains still have to dump their outputs into a shared residual stream and route context through shared attention heads. On our stack, Finance and Coding held up fine. Legal, however, took a +2.41 PPL hit. It turned a catastrophic failure into a localized cost, it didn't eliminate the interference. It just managed it.

**What the four‑domain stack actually looked like (200 steps, pure physics, no EMI junk):**

| domain | single PPL | stacked PPL | ΔPPL |
| --- | --- | --- | --- |
| finance | 1.44 | 1.54 | +0.10 |
| medical | 1.71 | 2.42 | +0.71 |
| coding | 1.28 | 1.68 | +0.40 |
| legal | 3.81 | 6.22 | +2.41 |

For comparison, my Rotated ARW baseline stacked at **+25.8 PPL** – complete logit collapse. So yeah, we killed the big ghost. Legal still gets mugged in the hallway, but at least it's not a massacre. (Making the mother of all omelettes here, Jack - can't fret over every egg!)

---

## A Fragile Equilibrium (Crash course of mistakes, inefficiency and overfitting)

The default config is 200 steps at batch size 16. Stable enough to teach Graft and not overfit.

Don't try to push the step count. Running 10 000 steps at a low batch size is a trap. The optimizer will just inflate the delta weights to scrape fractional loss improvements. You end up training high‑energy noise that violently corrupts the shared stream when stacked. Push it to 2 000 steps at BS16, and you fall off an overfitting cliff where the graft starts fighting the base model's syntactic priors.

Two hundred high‑energy updates. That seemed to be exactly enough to shift logits and capture the vocabulary. I stopped there.

Your out‑of‑distribution (OOD) data matters just as much. Training against general web text gives a weak suppression gradient because the model already knows English. To force active suppression, you have to train the graft adversarially against its sibling domains. Medical must fight Legal and Coding to carve out its physical space. 

---

## The Receipts (Telemetry from the Trenches)
### 1. How much the graft actually learns

With 152M trainable parameters and 200 steps, the graft surgically overwrites the base model's ignorance. Vanilla PPL is the frozen base guessing blindly; Grafted PPL is base + graft.

| Domain | Vanilla Base PPL | Grafted PPL | Absolute Delta (Δ) |
| --- | --- | --- | --- |
| Finance | 92.80 | 1.44 | **‑91.35** |
| Medical | 15.65 | 1.71 | **‑13.94** |
| Legal | 8.15 | 3.81 | **‑4.34** |
| Coding | 2.87 | 1.28 | **‑1.59** |

Finance didn't even have a fighting chance. The base model was basically illiterate on that domain, and the graft just vacuumed up the whole mess.

### 2. The N=4 Stack Test (The Linear Shadow Tax)

SmolLM3‑3B, 4 domains, 200 steps, `‑‑lambda_silence 5.0`, BS=16, FlashAttention 2.

| Domain | Single PPL | Stacked PPL | ΔPPL (The Tax) |
| --- | --- | --- | --- |
| Finance | 1.44 | 1.54 | +0.10 |
| Medical | 1.71 | 2.42 | +0.71 |
| Coding | 1.28 | 1.68 | +0.40 |
| Legal | 3.81 | 6.22 | +2.41 |

I didn't eliminate the interference. Legal still got clobbered because it's dense, high‑entropy text that throws a massive shadow across the shared stream. But we turned a catastrophic failure into a measurable, localized tax. The SiLU ghost is dead; this could be just the cost of doing business in a shared hallway.

---

## Dead Ends and the Roadmap (I don't have one.)

Portability is strictly locked to the Anchor Rule. A graft only works on a model it was made from. You can't take a SmolLM3‑3B graft and drop it into Gemma.

And don't even try this on Hybrid Recurrent or State-Space Models! Grafting requires pure spatial Transformer geometry. Modifying the FFN in a recurrent block permanently poisons the rolling hidden state. The fluid dynamics of those engines will violently reject spatial isolation. I spent few days trying to make it work on Qwen 3.5 2B! Wondering why it didn't work.

What's next? Who knows. But i have an idea.

Expansion and contraction are isolated, but the `down_proj` write‑back still pollutes the common hallway. The next evolution could be the residual‑lane masking, constraining those writes and reads to specific ranges. I mean this whole idea was born from allow forward pass to both model and graft, restrict the backpropogation to graft. It could work again, or not. Until then, this is a highly effective, but fundamentally leaky, multiplexing strategy.

---

## Codebase Layout

* `dataset.py` – automated data fetching and balanced loader classes
* `engine.py` – layer discovery, tri‑partite channel slicing, and training delta hooks
* `train.py` – training loop pipeline
* `eval.py` – eval, multi‑pair compare, stack‑test, and static install
* `autopsy.py` – tensor diagnostics and weight space metrics

---

## Operations

Install dependencies (Supports CUDA, ROCm, MPS, CPU. Delta master weights stay float32 during training, saved as bfloat16 on compatible environments):

```bash
pip install -e .[gpu]

```

Download datasets:

```bash
python dataset.py --domains medical legal finance coding

```

Train a graft (MI300X‑class config). Reduce `‑‑batch_size`, `‑‑max_len`, or layer range if VRAM constrained:

```bash
python train.py \
  --model HuggingFaceTB/SmolLM3-3B \
  --domain_data medical \
  --max_domains 4 \
  --lambda_silence 5.0 \
  --steps 200 \
  --batch_size 16 \
  --max_len 512 \
  --num_workers 8 \
  --fa2

```

Evaluate standalone artifact boundaries:

```bash
python eval.py eval --graft medical.graft --data data/medical.jsonl

```

Stack artifacts and measure interference prior to installation:

```bash
python eval.py stack-test \
  --grafts medical.graft legal.graft coding.graft finance.graft \
  --data data/medical.jsonl data/legal.jsonl data/coding.jsonl data/finance.jsonl

```

Bake artifacts directly into the model weights (creates a new directory):

```bash
python eval.py install \
  --graft medical.graft legal.graft coding.graft finance.graft \
  --output smol-grafted

```

---

**Acknowledgement:** Massive thanks to the **AMD Developer Cloud** for the compute access and the **MI300X** horsepower that made this research possible.
