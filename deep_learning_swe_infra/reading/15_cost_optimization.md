# Cost Optimization for LLM Inference

**Priority:** Week 3-4. Critical for Track A (SA must talk cost).
**Interview map:** Track A Rounds 2 & 4, Track B Round 2

---

## 1. The Cost Equation

```
Cost per token = GPU cost per hour / Tokens per hour
             = (GPU $/hr) / (Throughput in tokens/sec × 3600)
```

Everything that increases throughput or reduces GPU requirements decreases cost per token.

---

## 2. GPU Cost Landscape (2025-2026 Approximate)

### Cloud GPU pricing (on-demand, approximate)
| GPU | VRAM | $/hr (cloud) | HBM BW | Best for |
|---|---|---|---|---|
| A10 | 24 GB | $0.60-1.00 | 600 GB/s | Small models, dev/test |
| A100 40GB | 40 GB | $2.00-3.00 | 1.5 TB/s | Medium models |
| A100 80GB | 80 GB | $3.00-4.50 | 2 TB/s | Standard production |
| H100 80GB | 80 GB | $4.00-6.00 | 3.35 TB/s | High-performance |
| H200 141GB | 141 GB | $5.00-8.00 | 4.8 TB/s | Large models, long context |

### DGX pricing (purchased)
- DGX A100: ~$200K (8x A100 80GB)
- DGX H100: ~$300-400K (8x H100 80GB)
- Amortized over 3 years: DGX H100 ≈ $15/hr for 8 GPUs ≈ $1.9/GPU/hr
- **Buy vs rent breakpoint:** If >50% utilization, owning DGX is cheaper than cloud

---

## 3. Optimization Levers

### Lever 1: Quantization (Biggest Impact)

```
Llama-3 70B serving cost comparison on A100 80GB:

FP16: needs 2 GPUs (TP=2)     → 2 × $3.50/hr = $7.00/hr
INT8: needs 1 GPU              → 1 × $3.50/hr = $3.50/hr  (50% savings)
INT4: needs 1 GPU, more room   → 1 × $3.50/hr = $3.50/hr  (+ higher throughput from more KV cache room)
```

INT8/FP8 is almost always the right choice for production serving. The quality loss is minimal and the cost savings are massive.

### Lever 2: Batching Efficiency

```
Batch=1:  Load 140 GB weights → generate 1 token  → cost = X per token
Batch=32: Load 140 GB weights → generate 32 tokens → cost = X/32 per token
```

Higher batch utilization = lower cost per token. Key metrics:
- **GPU utilization:** Should be >80% during decode
- **Batch occupancy:** How full is your batch on average?
- Continuous batching helps here — keeps batch full

### Lever 3: Right-Sizing GPUs

Don't use H100 for a 7B model that runs fine on A10.

```
Llama-3 8B:
  A10 ($0.75/hr):  ~30 tokens/sec  → $0.025 per 1K tokens
  A100 ($3.50/hr): ~100 tokens/sec → $0.035 per 1K tokens
  → A10 is cheaper per token for small models!
```

### Lever 4: Spot/Preemptible Instances

- Cloud spot instances: 60-80% cheaper than on-demand
- Risk: can be preempted (killed) with short notice
- Works for: batch processing, non-real-time workloads, development
- Doesn't work for: real-time serving with SLOs (can't tolerate preemption)

### Lever 5: Request-Level Optimization

**Prompt compression:**
- Shorter prompts = faster prefill = less compute
- LLMLingua, AutoCompressor: compress prompts with minimal quality loss
- Can reduce prompt length by 50-70%

**Output length limits:**
- Set `max_tokens` appropriately — don't let models ramble
- Shorter outputs = fewer decode steps = less cost

**Caching:**
- Semantic caching: if a similar question was asked before, return cached answer
- Prefix caching: reuse KV cache for shared prefixes
- Can eliminate 20-50% of computation for repetitive workloads

### Lever 6: Model Selection

Sometimes a smaller model is the right choice:
```
Task: classification (yes/no)
  Llama-3 70B: 99% accuracy, $7.00/hr for 2 GPUs
  Llama-3 8B:  97% accuracy, $0.75/hr for 1 GPU
  → 9x cheaper for 2% accuracy difference
```

**Model routing:** Use a small model for easy queries, large model for hard ones.
- Router model or confidence-based: if small model is confident, use its answer
- If not, escalate to large model
- Can reduce average cost by 50-80%

---

## 4. Cost Comparison Example

### Scenario: Serve Llama-3 70B for 1M tokens/day

**Option A: Cloud A100 80GB × 2 (TP=2), FP16, on-demand**
- Throughput: ~50 tokens/sec
- Time to serve 1M tokens: 1M / 50 / 3600 ≈ 5.6 hrs
- Cost: 2 GPUs × $3.50/hr × 5.6 hrs = $39/day = **$1,170/month**

**Option B: Cloud A100 80GB × 1, INT8, on-demand**
- Throughput: ~60 tokens/sec (fits on 1 GPU, no TP overhead)
- Time: 1M / 60 / 3600 ≈ 4.6 hrs
- Cost: 1 GPU × $3.50/hr × 4.6 hrs = $16/day = **$480/month**

**Option C: Same as B but spot instances (70% discount)**
- Cost: $16 × 0.3 = $4.8/day = **$144/month** (if workload tolerates preemption)

**Option D: DGX H100 (owned, amortized), INT8, FP8 KV cache**
- Throughput: ~120 tokens/sec (H100 + FP8)
- Time: 1M / 120 / 3600 ≈ 2.3 hrs
- Amortized cost: 1 GPU × $1.9/hr × 2.3 hrs = $4.4/day = **$132/month**

**Summary:**
| Option | Monthly Cost | Savings |
|---|---|---|
| A: Cloud FP16 | $1,170 | Baseline |
| B: Cloud INT8 | $480 | 59% |
| C: Cloud INT8 + Spot | $144 | 88% |
| D: DGX H100 INT8 owned | $132 | 89% |

---

## 5. TCO (Total Cost of Ownership) for Track A

As an SA, you'll help customers calculate TCO:

```
TCO = Hardware + Software + Operations + Opportunity Cost

Hardware:
  - GPU purchase or cloud rental
  - Networking (InfiniBand switches)
  - Storage (model weights, logs)

Software:
  - NVIDIA AI Enterprise license (~$4,500/GPU/year)
  - Monitoring stack (Prometheus, Grafana — free/OSS)
  - Orchestration (Kubernetes — free/OSS)

Operations:
  - ML engineers to manage infrastructure
  - On-call for production serving
  - Model updates and redeployment

Opportunity Cost:
  - Time to deploy (NIM = days, custom TRT-LLM = weeks)
  - Time to optimize (NVIDIA support vs DIY debugging)
```

### The SA value proposition
"By using NIM + AI Enterprise, you deploy in days instead of months. The license costs $4,500/GPU/year, but your ML engineers spend zero time on inference optimization — NVIDIA handles it. At senior engineer salary ($200K+), freeing up even 25% of one engineer's time saves $50K/year, far exceeding the license cost."

---

## 6. Interview Connections

> "A customer is spending $50K/month on LLM inference on cloud GPUs. How do you reduce their costs?"

**A:** "I'd analyze their setup along 6 dimensions: (1) Are they using quantization? INT8/FP8 can halve GPU needs. (2) What's their GPU utilization? Low utilization means over-provisioned. (3) Right-sizing — are they using H100 for workloads that run fine on A10? (4) Are they using continuous batching? Static batching wastes 50%+ capacity. (5) Can they use spot instances for non-real-time workloads? (6) Can they cache responses or use a smaller model for simple queries? In my experience at JPMC, combining quantization + continuous batching + caching reduced our serving costs by roughly 3x."

> "When does it make sense to buy DGX vs rent cloud GPUs?"

**A:** "Crossover is roughly at 50% sustained utilization. Below that, cloud is cheaper — you only pay for what you use. Above that, the amortized cost of owned hardware wins. A DGX H100 at $350K over 3 years is ~$1.9/GPU/hr. Cloud H100 is $5-6/GPU/hr. If you're running 24/7, owning saves 60%. Also consider: compliance requirements (banking often requires on-prem), data residency, and the value of guaranteed capacity without spot preemption."

---

## Self-Test Questions

1. What is the formula for cost per token?
2. Why does INT8 quantization roughly halve serving costs?
3. A customer runs Llama-3 70B FP16 on 4x A100 cloud GPUs 24/7. Calculate monthly cost and propose an optimization plan.
4. When is buying DGX cheaper than cloud?
5. What is model routing and how does it reduce costs?
6. As an SA, how do you justify the NVIDIA AI Enterprise license cost?

---

## Notes
```


```
