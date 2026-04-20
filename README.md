# Quantization experiments on nanoGPT

Experiments exploring Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT) on Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).

**Model:** character-level GPT trained on Shakespeare  
**Hardware:** NVIDIA V100 (32GB), HSE University HPC cluster  
**Framework:** PyTorch 2.4

---

## What is quantization?

Quantization replaces 32-bit floating point weights with lower-precision integers (int8), reducing memory usage and potentially speeding up inference.

Two approaches were compared:

- **PTQ (Post-Training Quantization)** — quantize a trained model directly, no additional training. Fast but the model was never exposed to quantization noise during training.
- **QAT (Quantization-Aware Training)** — fine-tune the model with simulated quantization noise in the forward pass (fake-quant). The model adapts to the noise, leading to better quality after quantization.

In both cases `torch.quantization.quantize_dynamic` was applied to all `nn.Linear` layers (Q/K/V projections, MLP layers, and the language model head). Embeddings and LayerNorm layers were left in float32.

---

## Baseline model

| Parameter     | Value          |
|---------------|----------------|
| Architecture  | GPT (nanoGPT)  |
| n_layer       | 6              |
| n_head        | 8              |
| n_embd        | 384            |
| Parameters    | 10.65M         |
| Dataset       | shakespeare_char |
| Training      | 5000 steps, lr=1e-3, bs=32 |
| Val loss      | 1.4561         |
| Perplexity    | 4.289          |

---

## Experiment 1 — PTQ vs QAT baseline

Simple comparison: take the best checkpoint, apply PTQ and QAT (1000 fine-tuning steps).

| Method           | Val loss | Perplexity | Size MB | PPL delta   |
|------------------|----------|------------|---------|-------------|
| float32 baseline | 1.4602   | 4.3070     | 41.0    | —           |
| PTQ int8 dynamic | 1.4630   | 4.3187     | 0.5     | +0.27%      |
| QAT int8 dynamic | 1.4624   | 4.3161     | 41.0    | +0.21%      |

QAT recovers ~0.06% of the PTQ quality loss. The effect is small for a 10M parameter model but the direction is consistent with theory.

> Note: Size 0.5 MB for PTQ reflects the quantized weight storage. QAT size shows 41 MB because `quantize_dynamic` is applied at inference time — weights are stored as float32 in memory and quantized on the fly per forward pass.

---

## Experiment 2 — QAT steps vs quality

How does the number of QAT fine-tuning steps affect the result?

| Steps     | Val loss | Perplexity | PPL delta  |
|-----------|----------|------------|------------|
| baseline  | 1.4597   | 4.3047     | —          |
| 0 (PTQ)   | 1.4614   | 4.3122     | +0.175%    |
| 500       | 1.4620   | 4.3144     | +0.226%    |
| 1000      | 1.4558   | 4.2877     | **-0.393%**|
| 3000      | 1.4633   | 4.3203     | +0.364%    |
| 5000      | 1.4697   | 4.3479     | +1.004%    |

![QAT steps vs PPL delta](figures/exp_a_steps.png)

**Key finding:** 1000 steps is the sweet spot. At 1000 steps, QAT actually slightly improves over the float32 baseline (-0.393%). Beyond 1000 steps the model overfits — the shakespeare_char dataset is small enough that extended fine-tuning hurts generalization.

---

## Experiment 3 — Quantization sensitivity: small vs large model

Does a larger model lose more quality when quantized?

| Model        | Method           | Val loss | PPL    | PPL delta  |
|--------------|------------------|----------|--------|------------|
| small (nl=6) | float32 baseline | 1.4603   | 4.3073 | —          |
| small (nl=6) | PTQ int8         | 1.4638   | 4.3222 | +0.348%    |
| small (nl=6) | QAT int8         | 1.4612   | 4.3113 | **+0.093%**|
| large (nl=12)| float32 baseline | 1.4706   | 4.3518 | —          |
| large (nl=12)| PTQ int8         | 1.4676   | 4.3386 | -0.302%    |
| large (nl=12)| QAT int8         | 1.4790   | 4.3886 | +0.847%    |

![Model size vs quantization sensitivity](figures/exp_b_models.png)

**Key finding:** Counterintuitively, QAT hurts the larger model (+0.847%) while helping the smaller one (+0.093% vs PTQ). The hypothesis: the larger model (nl=12) has more layers accumulating quantization noise, and 1000 fine-tuning steps on a small dataset are insufficient for it to adapt. The PTQ result for nl=12 (-0.302%) is within noise range.

---

## How to reproduce

### Requirements

```bash
module load Python/PyTorch_GPU_v2.4   # or: pip install torch==2.4.0 numpy
```

### Step 1 — train the baseline model

```bash
cd nanoGPT
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py \
    --learning_rate=1e-3 --n_layer=6 --n_head=8 --batch_size=32 \
    --out_dir=checkpoints/baseline --max_iters=5000 \
    --dtype=float16 --compile=False
```

### Step 2 — run quantization experiments

```bash
# PTQ vs QAT baseline
python scripts/qat_v2.py --checkpoint checkpoints/baseline/ckpt.pt

# Experiment A: QAT steps
python scripts/qat_ab.py --experiment a --checkpoint checkpoints/baseline/ckpt.pt

# Experiment B: model sizes
python scripts/qat_ab.py --experiment b \
    --ckpt_small checkpoints/lr0.001_nl6_nh8_bs32/ckpt.pt \
    --ckpt_large checkpoints/lr0.001_nl12_nh8_bs32/ckpt.pt
```

### Step 3 — plot results

```bash
python scripts/plot_quant.py
```

---

## Repository structure

```
.
├── scripts/
│   ├── qat_v2.py          # PTQ vs QAT baseline comparison
│   ├── qat_ab.py          # Experiments A and B
│   ├── quant_eval.py      # float32 / float16 / int8 evaluation
│   └── plot_quant.py      # generate all figures
├── results/
│   ├── qat_results.txt    # Experiment 1 raw results
│   ├── exp_a_results.txt  # Experiment 2 raw results
│   └── exp_b_results.txt  # Experiment 3 raw results
└── figures/
    ├── exp_a_steps.png    # QAT steps vs PPL delta
    └── exp_b_models.png   # model size comparison
```

---

## Notes

- All experiments run on NVIDIA V100 (sm_70). `bfloat16` is not supported on V100, so `float16` was used during training with `--compile=False` (Triton does not support sm_70).
- `torch.quantization.quantize_dynamic` dispatches to `QuantizedCPU` backend — inference runs on CPU. V100 does support `QuantizedCUDA` but requires static quantization with `QuantStub` wrappers not present in the original nanoGPT architecture.
- QAT was implemented as manual fake-quantization (rounding weights to int8 scale during forward pass) rather than `torch.quantization.prepare_qat`, which failed due to the same architecture constraint.

---

## References

- [nanoGPT](https://github.com/karpathy/nanoGPT) — Andrej Karpathy
- [PyTorch quantization docs](https://pytorch.org/docs/stable/quantization.html)
