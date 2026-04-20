#!/usr/bin/env python3
"""
Quantization experiments for nanoGPT.

Usage:
    python quant_eval.py --checkpoint /home/dashilovskiy/nanoGPT/checkpoints/lr0.001_nl6_nh8_bs32/ckpt.pt
"""

import os
import sys
import time
import argparse
import math
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, "/home/dashilovskiy/nanoGPT/nanoGPT")
from model import GPT, GPTConfig

# ── args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--data_dir",
    default="/home/dashilovskiy/nanoGPT/nanoGPT/data/shakespeare_char")
parser.add_argument("--eval_tokens", type=int, default=102400,
    help="Number of tokens to evaluate perplexity on")
parser.add_argument("--block_size", type=int, default=256)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

print(f"Device: {args.device}")
print(f"Checkpoint: {args.checkpoint}")

# ── load data ────────────────────────────────────────────────────────────────

val_data = np.memmap(os.path.join(args.data_dir, "val.bin"),
                     dtype=np.uint16, mode="r")

def get_batch(split="val"):
    data = val_data
    ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
    x = torch.stack([torch.from_numpy(
        data[i:i+args.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(
        data[i+1:i+1+args.block_size].astype(np.int64)) for i in ix])
    return x.to(args.device), y.to(args.device)

# ── load model ───────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt["model_args"]
    model_cfg = GPTConfig(**cfg)
    model = GPT(model_cfg)
    state = {k.replace("_orig_mod.", ""): v
             for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()
    return model

# ── evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_perplexity(model, device, n_tokens=None, dtype=torch.float32):
    if n_tokens is None:
        n_tokens = args.eval_tokens
    model.to(device)
    model.eval()

    total_loss  = 0.0
    total_tokens = 0
    n_batches = max(1, n_tokens // (args.batch_size * args.block_size))

    t0 = time.time()
    for _ in range(n_batches):
        x, y = get_batch()
        with torch.amp.autocast(device_type=device.split(":")[0], dtype=dtype,
                                enabled=(dtype != torch.float32)):
            logits, loss = model(x, y)
        total_loss   += loss.item() * x.numel()
        total_tokens += x.numel()
    elapsed = time.time() - t0

    avg_loss   = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    tok_per_sec = total_tokens / elapsed
    return perplexity, avg_loss, tok_per_sec

def model_size_mb(model):
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    return total / (1024 * 1024)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

# ── quantization methods ──────────────────────────────────────────────────────

def quantize_dynamic_int8(model):
    """Dynamic int8 quantization — weights quantized, activations at runtime."""
    q_model = torch.quantization.quantize_dynamic(
        model.cpu(),
        {nn.Linear},
        dtype=torch.qint8
    )
    return q_model

def quantize_float16(model):
    return model.half()

def quantize_float32(model):
    return model.float()

# ── main ──────────────────────────────────────────────────────────────────────

print("\nLoading model...")
base_model = load_model(args.checkpoint, args.device)
n_params = count_params(base_model)
print(f"Parameters: {n_params/1e6:.2f}M")

results = []

experiments = [
    ("float32",      lambda m: quantize_float32(m.cpu()), "cpu",        torch.float32),
    ("float16",      lambda m: quantize_float16(m.to(args.device)), args.device, torch.float16),
    ("int8_dynamic", lambda m: quantize_dynamic_int8(m),  "cpu",        torch.float32),
]

# Only run float16 if CUDA is available
if args.device == "cpu":
    experiments = [e for e in experiments if e[0] != "float16"]

for name, quant_fn, dev, dtype in experiments:
    print(f"\n{'='*50}")
    print(f"Experiment: {name}")

    model = load_model(args.checkpoint, "cpu")
    model = quant_fn(model)
    size_mb = model_size_mb(model)

    print(f"  Model size: {size_mb:.1f} MB")

    try:
        ppl, loss, tps = evaluate_perplexity(model, dev, dtype=dtype)
        print(f"  Perplexity: {ppl:.4f}")
        print(f"  Val loss:   {loss:.4f}")
        print(f"  Throughput: {tps:.0f} tokens/sec")
        results.append({
            "name":       name,
            "ppl":        ppl,
            "loss":       loss,
            "size_mb":    size_mb,
            "tps":        tps,
        })
    except Exception as e:
        print(f"  ERROR: {e}")

# ── summary table ─────────────────────────────────────────────────────────────

print("\n" + "="*70)
print(f"{'Method':<15} {'Val loss':>10} {'Perplexity':>12} {'Size MB':>10} {'Tok/sec':>12}")
print("-"*70)

baseline_ppl = results[0]["ppl"] if results else None
for r in results:
    delta = ""
    if baseline_ppl and r["name"] != results[0]["name"]:
        d = (r["ppl"] - baseline_ppl) / baseline_ppl * 100
        delta = f"  ({d:+.1f}%)"
    print(f"{r['name']:<15} {r['loss']:>10.4f} {r['ppl']:>12.4f} "
          f"{r['size_mb']:>10.1f} {r['tps']:>12.0f}{delta}")

print("="*70)

# Save results
out_path = os.path.join(os.path.dirname(args.checkpoint), "quant_results.txt")
with open(out_path, "w") as f:
    f.write(f"Checkpoint: {args.checkpoint}\n")
    f.write(f"Params: {n_params/1e6:.2f}M\n\n")
    f.write(f"{'Method':<15} {'Val loss':>10} {'Perplexity':>12} {'Size MB':>10} {'Tok/sec':>12}\n")
    f.write("-"*65 + "\n")
    for r in results:
        f.write(f"{r['name']:<15} {r['loss']:>10.4f} {r['ppl']:>12.4f} "
                f"{r['size_mb']:>10.1f} {r['tps']:>12.0f}\n")
print(f"\nSaved: {out_path}")
