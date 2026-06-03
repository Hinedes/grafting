# Grafting v0.3 (Axis ARW)

This architecture started as a simple daydream (the AuDHD kind, the looping kind): expose a frozen model to novel data, let its gradients ignore what it already knows, and capture its frantic reactions to out-of-distribution tokens. Pack those raw gradient descent responses into a standalone file, and you have a graft. 

It sounded too simple. I thought, "This can't work." (I have no ML background, so what do I know.) Then the telemetry came in. My brain told me: _"Don't you dare say it!"_ But I said: **_"Jackpot!"_** And then I spent two more weeks making it work. ("It would appear, daydreaming can't code, but i, Hinedes, have a dream!")

Probing the residual stream on the SmolLM3-3B medical graft revealed a +0.97 correlation between base model ignorance and graft-induced loss reduction. The frozen model acts as a filter for its own ignorance. When the base model is confused, the graft corrects; when the base model is confident, the graft contributes essentially zero. It is a self-distilling, opportunistic error-catcher.

At install time, it completely abandons the overlapping mathematical superposition of v0.1. There is no decoding key. Instead, the 300 MB graft folds directly into a hard, isolated slice of the base weights:

$$W_{\text{new}}[s_y:e_y, s_x:e_x] = W_{\text{base}}[s_y:e_y, s_x:e_x] + \Delta_{\text{graft}}$$

The base model reads the graft natively because the graft physically becomes a room inside the host's FFN geometry. The domains never touch during expansion or contraction because their index ranges have a hard wall between them.

---

## The Mixer Problem (Blender Slander, Orthogonality Must Die)

Version 0.1 relied on rotated orthogonal subspaces. Beautiful on paper. Completely broken in silicon.

Why? Because Transformer non-linearities don't care about linear algebra. As was explained to me later on, the SwiGLU block uses element-wise multiplication:

$$y = W_{down} \Big( \text{SiLU}(W_{gate} x) \odot (W_{up} x) \Big)$$

Pass a mathematically orthogonal signal through the SiLU, and it acts like a cryptographic mixer. The subspace shatters. The signal scatters everywhere.

So, Version 0.3 throws out the elegant math in favor of physical walls. It chops the FFN intermediate and residual widths into hard, axis-aligned Boolean masks.

It’s crude. But it’s airtight. (Maybe.)

Disjoint support forces the cross-terms to exactly zero: $\text{SiLU}(\Delta W_{gate}^A x) \odot \Delta W_{up}^B x = 0$. The mixer has nothing to multiply, meaning cross-domain interference inside the FFN is structurally dead.

(We don't touch the attention heads, by the way. The Stem Cell Hypothesis holds up: attention handles syntactic, domain-agnostic routing. Factual knowledge lives in the FFN. Messing with attention is simply architecturally wrong.)

---

## The Residual Tax (Or: The Noisy Roommates)

Inference is free. Training is brutal. And now i have OOM PTSD.

If you want to run the four-domain stack on SmolLM3-3B, you'd need something like AMD MI300X and about 130 GB of VRAM. You have to force massive sequence volume through the base model just to make the silence loss effective. (Machine Learning at its best. Bruteforcing at its finest.) 

And then there’s the residual tax.

Axis-aligned slicing solves the FFN problem, but it exposes another one. The domains still have to dump their outputs into a shared residual stream and route context through shared attention heads. On our stack, Finance and Coding held up fine. Legal, however, took a +2.41 PPL hit. It turned a catastrophic failure into a localized cost, but let's not pretend it eliminated the interference. It just managed it.

---

## A Fragile Equilibrium (Crash course of mistakes, inefficiency and overfitting)

The default config is 200 steps at batch size 16. Stable enough to teach Graft and not overfit.

Don't try to push the step count. Running 10,000 steps at a low batch size is a trap. The optimizer will just inflate the delta weights to scrape fractional loss improvements. You end up training high-energy noise that violently corrupts the shared stream when stacked. Push it to 2,000 steps at BS16, and you fall off an overfitting cliff where the graft starts fighting the base model's syntactic priors.

Two hundred high-energy updates. That seemed to be exactly enough to shift logits and capture the vocabulary. I stopped there.

Your out-of-distribution (OOD) data matters just as much. Training against general web text gives a weak suppression gradient because the model already knows English. To force active suppression, you have to train the graft adversarially against its sibling domains. Medical must fight Legal and Coding to carve out its physical space.

---

## Dead Ends and the Roadmap (I don't have one.)

Portability is strictly locked to the Anchor Rule. A graft only works on a model it was made from. You can't take a SmolLM3-3B graft and drop it into Gemma.

And don't even try this on Hybrid Recurrent or State-Space Models! Grafting requires pure spatial Transformer geometry. Modifying the FFN in a recurrent block permanently poisons the rolling hidden state. The fluid dynamics of those engines will violently reject spatial isolation. I spent few days trying to make it work on Qwen 3.5 2B! Wondering why it didn’t work.

What’s next? Who knows. But i have an idea.

Expansion and contraction are isolated, but the `down_proj` write-back still pollutes the common hallway. The next evolution could be the residual-lane masking, constraining those writes and reads to specific ranges. I mean this whole idea was born from allow forward pass to both model and graft, restrict the backpropogation to graft. It could work again, or not. Until then, this is a highly effective, but fundamentally leaky, multiplexing strategy. 

---

## Operations

Install dependencies (FlashAttention 2 requires CUDA/ROCm and is highly recommended):

```bash
uv sync --extra gpu

```

Download datasets:

```bash
uv run python dataset.py --domains medical legal finance

```

Train a graft (MI300X-class config). Reduce `--batch_size`, `--max_len`, or layer range if VRAM constrained:

```bash
uv run python train.py \
uv run python train.py \
  --model HuggingFaceTB/SmolLM3-3B \
  --domain_data medical.jsonl \
  --ood_data legal.jsonl coding.jsonl finance.jsonl minipile.jsonl \
  --ood_data legal.jsonl coding.jsonl finance.jsonl minipile.jsonl \
  --domain_index 0 \
  --max_domains 4 \
  --lambda_silence 5.0 \
  --steps 200 \
  --batch_size 16 \
  --max_len 512 \
  --num_workers 8 \
  --fa2 \
  --lambda_silence 5.0 \
  --steps 200 \
  --batch_size 16 \
  --max_len 512 \
  --num_workers 8 \
  --fa2 \
  --output medical.graft.pt

```

Stack artifacts and measure interference prior to installation:

```bash
uv run python eval.py stack-test \
  --grafts medical.graft.pt legal.graft.pt coding.graft.pt finance.graft.pt \
  --data medical.jsonl legal.jsonl coding.jsonl finance.jsonl

```

Bake artifacts directly into the model weights (creates a new directory):

```bash
uv run python eval.py install \
  --graft medical.graft.pt legal.graft.pt coding.graft.pt finance.graft.pt \
uv run python eval.py install \
  --graft medical.graft.pt legal.graft.pt coding.graft.pt finance.graft.pt \
  --output smol-grafted

```
---

**Acknowledgement:** Massive thanks to the **AMD Developer Cloud** for the compute access and the **MI300X** horsepower that made this research possible.
