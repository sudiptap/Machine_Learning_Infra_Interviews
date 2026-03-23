# Quantization Theory — Smaller Models, Faster Inference

**Priority:** Read in Week 2–3.
**Interview map:** Track A Round 2, Track B Round 3

---

## 1. Why Quantize?

LLM inference is memory-bandwidth-bound during decode. The bottleneck is loading model weights from HBM.

```
Llama-3 70B in FP16:   ~140 GB   → needs 2x A100 80GB
Llama-3 70B in INT8:   ~70 GB    → fits on 1x A100 80GB
Llama-3 70B in INT4:   ~35 GB    → fits on 1x A100 40GB
```

### Benefits of quantization:
1. **Less memory** → serve larger models on fewer GPUs
2. **Faster decode** → less data to load from HBM per token
3. **Lower cost** → fewer GPUs needed
4. **Higher throughput** → more concurrent requests in same memory

### The tradeoff:
- Lower precision = some quality degradation
- The question is always: how much quality do you lose for how much speed do you gain?

---

## 2. Number Formats

### FP16 (Half Precision)
- 16 bits: 1 sign, 5 exponent, 10 mantissa
- Range: ±65,504
- Standard training and inference format
- 2 bytes per parameter

### BF16 (Brain Float 16)
- 16 bits: 1 sign, 8 exponent, 7 mantissa
- Same range as FP32 but lower precision
- Better for training (fewer overflow issues)
- 2 bytes per parameter

### FP8 (H100+ only)
- 8 bits: two variants — E4M3 (4 exp, 3 mantissa) and E5M2 (5 exp, 2 mantissa)
- E4M3: more precision, less range → better for weights
- E5M2: more range, less precision → better for gradients
- 1 byte per parameter → 2x compression vs FP16
- Native Tensor Core support on H100

### INT8
- 8 bits, integer format
- Range: -128 to 127
- 1 byte per parameter → 2x compression vs FP16
- Requires careful scaling to map float weights to integer range

### INT4
- 4 bits, integer format
- Range: -8 to 7 (or 0 to 15 unsigned)
- 0.5 bytes per parameter → 4x compression vs FP16
- Significant quality risk — needs smart quantization methods

---

## 3. Quantization Methods

### Weight-Only Quantization
Quantize only the model weights, keep activations in FP16.
- During computation: dequantize weights on-the-fly, multiply with FP16 activations
- Why this works: weights are static (loaded from HBM), activations are dynamic
- Most common approach for LLM inference

### Weight + Activation Quantization
Quantize both weights AND activations.
- Harder: activations have outliers that are hard to represent in low precision
- SmoothQuant: migrates quantization difficulty from activations to weights
- Used in some INT8 deployments

---

## 4. GPTQ (Post-Training Quantization)

### What it is
- Post-training quantization method based on Optimal Brain Quantization (OBQ)
- Quantizes weights to INT4 or INT3 with minimal quality loss
- Uses a calibration dataset (small, ~128 samples)

### How it works (simplified)
1. Process the model layer by layer
2. For each layer, quantize weights one column at a time
3. After quantizing each column, adjust remaining columns to compensate for the quantization error
4. Uses Hessian information (second-order optimization) to minimize output error

### Key properties
- **One-time cost:** Quantization takes 1-4 hours on a single GPU
- **INT4 weights, FP16 computation:** Weights stored as INT4, dequantized to FP16 at runtime
- **Group quantization:** Weights quantized in groups of 128 (different scale per group)
- **Quality:** Typically <1% perplexity degradation at INT4 for large models (>7B)

### When to use
- You want INT4 quantization (maximum compression)
- Willing to run calibration step
- Model is large enough (>7B) that quantization loss is small

---

## 5. AWQ (Activation-Aware Weight Quantization)

### What it is
- Post-training weight quantization that identifies and protects "salient" weights
- Key insight: not all weights are equally important — weights corresponding to large activations matter more

### How it works (simplified)
1. Run calibration data through the model
2. Identify which weights produce large activations (these are "salient")
3. Apply per-channel scaling that makes salient weights easier to quantize
4. Quantize all weights to INT4

### Key difference from GPTQ
- **GPTQ:** Adjusts remaining weights after quantizing each one (compensate for error)
- **AWQ:** Scales weights before quantization to protect important ones
- AWQ is generally **faster to quantize** and produces **similar or better quality**

### When to use
- Same scenarios as GPTQ
- Often preferred over GPTQ for its simplicity and speed
- Good default choice for INT4 quantization

---

## 6. FP8 Quantization

### What it is
- Quantize weights (and optionally activations) to 8-bit floating point
- Native hardware support on H100 Tensor Cores
- 2x compression vs FP16

### Key advantage
- **No calibration needed** for weight-only FP8 (just cast)
- **Hardware-accelerated** on H100 — FP8 Tensor Core ops are 2x faster than FP16
- **Minimal quality loss** — FP8 preserves more dynamic range than INT8

### FP8 vs INT8
| | FP8 | INT8 |
|---|---|---|
| Hardware support | H100+ only | A100 and H100 |
| Quality | Generally better (floating point) | Good with proper calibration |
| Calibration | Optional | Usually needed |
| Speedup | 2x over FP16 on H100 | 1.5-2x over FP16 |
| Compression | 2x | 2x |

### When to use
- You have H100 GPUs
- Want simple, high-quality 2x compression
- Don't need INT4-level compression

---

## 7. Comparison Table

| Method | Precision | Compression | Quality Loss | Calibration | Hardware |
|---|---|---|---|---|---|
| FP16 | 16-bit float | 1x (baseline) | None | No | All |
| FP8 | 8-bit float | 2x | Minimal | Optional | H100+ |
| INT8 (W8A8) | 8-bit int | 2x | Small | Yes | A100+ |
| GPTQ | 4-bit int | 4x | Small-Medium | Yes (1-4 hrs) | All |
| AWQ | 4-bit int | 4x | Small-Medium | Yes (faster) | All |

### Decision framework
```
Need maximum throughput + have H100?     → FP8
Need to fit large model on fewer GPUs?   → AWQ or GPTQ (INT4)
Need best quality, cost is secondary?    → FP16
Need balance of quality and speed?       → INT8 or FP8
```

---

## 8. Practical Impact — The Numbers

For Llama-3 8B on A100 80GB (approximate):

| Format | VRAM | TPOT (batch=1) | Throughput (tokens/s) |
|---|---|---|---|
| FP16 | ~16 GB | ~35 ms | ~28 |
| INT8 | ~8 GB | ~22 ms | ~45 |
| INT4 (AWQ) | ~4 GB | ~18 ms | ~55 |

Throughput can increase further with larger batches since you have more free memory.

---

## Interview Questions This Answers

> "What is quantization and what are the tradeoffs?"

> "A customer wants to serve Llama-3 70B on a single A100 80GB. How?"
→ INT8 or INT4 (AWQ/GPTQ). Show the math: 70B × 1 byte = 70 GB for INT8, fits on A100 80GB.

> "When would you recommend FP8 over INT8?"
→ If they have H100s: FP8 is native, no calibration, better quality. If A100: INT8 is the option.

> "What's the difference between AWQ and GPTQ?"
→ Both are INT4 post-training quantization. GPTQ compensates for error after each weight. AWQ protects important weights before quantization. AWQ is simpler and often produces equal or better quality.

---

## Self-Test Questions

1. Why does quantization help LLM inference speed? (Hint: it's about bandwidth, not compute)
2. What is the difference between FP8 E4M3 and E5M2?
3. Explain GPTQ at a high level. What does it optimize?
4. Explain AWQ at a high level. What is its key insight?
5. A customer has A100 80GB and wants to serve Llama-3 70B with minimal quality loss. Recommend a quantization strategy and show the math.
6. Why is FP8 preferred over INT8 on H100?

---

## Notes
```
(Take your own notes here)


```
