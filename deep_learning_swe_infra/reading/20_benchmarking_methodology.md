# Benchmarking Methodology for LLM Inference

**Priority:** Week 4-5. Critical for InferBench project and interview credibility.
**Interview map:** Track A Round 2, Track B Round 2

---

## 1. Why Benchmarking Methodology Matters

Bad benchmarks lead to bad decisions. Common mistakes:
- Measuring throughput without controlling for latency
- Not warming up the engine → measuring cold start
- Not controlling input/output length distribution
- Cherry-picking one scenario that makes your system look good
- Not reporting confidence intervals

Your InferBench project and any interview claim about "X ms latency" must be defensible.

---

## 2. The Core Metrics

### Latency Metrics
| Metric | What it measures | How to compute |
|---|---|---|
| TTFT | Time from request to first token | timestamp(first_token) - timestamp(request_sent) |
| TPOT | Time between consecutive tokens | mean(token[i+1].time - token[i].time) for i > 0 |
| End-to-end latency | Total request time | timestamp(last_token) - timestamp(request_sent) |
| TTFT P50/P99 | Percentile latencies | Sort all TTFT values, take 50th/99th percentile |

### Throughput Metrics
| Metric | What it measures | How to compute |
|---|---|---|
| Tokens/sec (output) | Generation speed | total_output_tokens / total_time |
| Tokens/sec (total) | Input + output processing speed | (input + output tokens) / total_time |
| Requests/sec | Request throughput | total_completed_requests / total_time |
| Tokens/sec/GPU | Per-GPU efficiency | total_tokens_sec / num_gpus |

### Efficiency Metrics
| Metric | What it measures | How to compute |
|---|---|---|
| GPU utilization | % of time GPU is computing | nvidia-smi or Nsight |
| GPU memory used | VRAM consumption | nvidia-smi |
| Cost per 1M tokens | Economic efficiency | (gpu_cost_per_hour / tokens_per_hour) × 1M |
| Batch utilization | How full is the batch | mean(batch_size) / max_batch_size |

---

## 3. Benchmark Dimensions

A proper benchmark varies these independently:

### Input length distribution
```
Short prompts:  ~100 tokens  (chatbot, simple Q&A)
Medium prompts: ~500 tokens  (RAG with context)
Long prompts:   ~2000 tokens (document analysis)
Very long:      ~8000 tokens (code generation with context)
```

### Output length distribution
```
Short outputs:  ~50 tokens   (classification, yes/no)
Medium outputs: ~200 tokens  (explanations)
Long outputs:   ~1000 tokens (generation, summarization)
```

### Concurrency levels
```
Low:    1-4 concurrent requests    (development, testing)
Medium: 16-64 concurrent requests  (typical production)
High:   128-512 concurrent requests (high-traffic production)
```

### Request arrival pattern
```
Constant rate:  X requests/sec, evenly spaced
Poisson:        Random arrivals at average rate X (more realistic)
Bursty:         Quiet periods + sudden bursts (stress test)
```

---

## 4. Benchmark Design — The Right Way

### Step 1: Define the workload profile
Before running anything, define what you're measuring:
```yaml
workload:
  model: meta-llama/Llama-3.1-8B-Instruct
  input_length_distribution:
    mean: 512
    std: 128
    min: 128
    max: 2048
  output_length_distribution:
    mean: 256
    std: 64
    max: 1024
  arrival_pattern: poisson
  target_request_rate: [1, 10, 50, 100, 200]  # sweep
  duration: 300  # seconds per rate point
  warmup: 30  # seconds before measuring
```

### Step 2: Warmup
```
Phase 1: Send 50-100 requests to warm up KV cache allocator, JIT compilation
Phase 2: Wait until GPU memory stabilizes
Phase 3: Start measuring
→ Never include warmup in reported numbers
```

### Step 3: Sweep request rate
For each rate, measure latency AND throughput:
```
Rate 1 req/s:   TTFT P50=80ms,  P99=120ms, throughput=45 tok/s
Rate 10 req/s:  TTFT P50=85ms,  P99=150ms, throughput=400 tok/s
Rate 50 req/s:  TTFT P50=120ms, P99=800ms, throughput=1800 tok/s
Rate 100 req/s: TTFT P50=500ms, P99=5000ms, throughput=2200 tok/s  ← saturating
Rate 200 req/s: TTFT P50=2000ms, P99=15000ms, throughput=2100 tok/s ← overloaded
```

### Step 4: Find the saturation point
The sweet spot: highest throughput where P99 latency still meets SLO.
```
SLO: TTFT P99 < 1000ms
→ Maximum sustainable rate: ~50 req/s at throughput ~1800 tok/s
```

---

## 5. Common Benchmarking Tools

### vLLM built-in benchmark
```bash
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct

# Benchmark script
python benchmarks/benchmark_serving.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --num-prompts 1000 \
  --request-rate 10 \
  --input-len 512 \
  --output-len 256
```

### ShareGPT dataset
Real conversation data commonly used for benchmarking:
- Realistic input/output length distribution
- Industry-standard comparison point
- `--dataset ShareGPT_V3_unfiltered_cleaned_split.json`

### Custom benchmark script pattern
```python
import asyncio
import aiohttp
import time
import numpy as np

async def send_request(session, prompt, results):
    start = time.perf_counter()
    first_token_time = None
    tokens = 0

    async with session.post(url, json=payload) as resp:
        async for chunk in resp.content:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            tokens += 1

    end = time.perf_counter()
    results.append({
        'ttft': first_token_time - start,
        'total_time': end - start,
        'tokens': tokens,
        'tpot': (end - first_token_time) / max(tokens - 1, 1),
    })

# Run with controlled concurrency
# Use asyncio.Semaphore to limit concurrent requests
# Use Poisson process for realistic arrival pattern
```

---

## 6. How to Report Results

### The minimum viable benchmark report

```markdown
## Setup
- Model: Llama-3.1-8B-Instruct
- GPU: 1x A100 80GB
- Engine: vLLM 0.4.x
- Quantization: FP16 / INT8 / FP8
- Input length: mean=512 (ShareGPT distribution)
- Output length: mean=256

## Results

### Latency vs Throughput (Request Rate Sweep)
| Rate (req/s) | TTFT P50 | TTFT P99 | TPOT P50 | TPOT P99 | Throughput (tok/s) |
|---|---|---|---|---|---|
| 1 | 80ms | 120ms | 25ms | 35ms | 45 |
| 10 | 85ms | 150ms | 28ms | 40ms | 420 |
| 50 | 120ms | 800ms | 32ms | 55ms | 1,800 |
| 100 | 500ms | 5,000ms | 45ms | 120ms | 2,200 |

### Comparison: vLLM vs TRT-LLM vs SGLang (at 50 req/s)
| Engine | TTFT P50 | TTFT P99 | Throughput | GPU Mem |
|---|---|---|---|---|
| vLLM | 120ms | 800ms | 1,800 tok/s | 72 GB |
| TRT-LLM | 100ms | 600ms | 2,100 tok/s | 68 GB |
| SGLang | 110ms | 700ms | 1,950 tok/s | 70 GB |

### Charts
[Include: Latency vs Throughput curve, TTFT distribution histogram, GPU utilization over time]
```

### What makes a benchmark credible
1. **Reproducible:** Script is public, exact versions listed
2. **Controlled:** One variable at a time (don't change model AND engine AND GPU)
3. **Realistic:** Use ShareGPT or production-like distributions, not fixed lengths
4. **Complete:** Report both latency AND throughput (not just one)
5. **Statistical:** Report percentiles (P50, P99), not just mean. Report over enough requests (1000+)

---

## 7. Pitfalls to Avoid

### Pitfall 1: Measuring only throughput
"Our system does 5000 tok/s!" — at what latency? If P99 TTFT is 30 seconds, it's useless for interactive use.

### Pitfall 2: Fixed input/output lengths
"TTFT is 50ms!" — at input_len=10. Real prompts with 2000 tokens might have 500ms TTFT.

### Pitfall 3: No warmup
First few requests are slow (model loading, JIT compilation, cache warmup). Including them skews results.

### Pitfall 4: Measuring without load
Latency at 1 req/s is meaningless for production. Always sweep across request rates to find saturation.

### Pitfall 5: Ignoring tail latency
Mean latency is 100ms, P99 is 10 seconds. If 1% of your users wait 10 seconds, that's terrible UX.

---

## 8. Interview Connections

> "How would you benchmark an inference engine? What metrics would you report?"

> "You claim your system achieves 50ms TPOT. How did you measure that?"
→ They want to hear: controlled workload, warmup, multiple concurrency levels, percentile reporting.

> "A customer says vLLM is slower than TRT-LLM. How do you validate this claim?"
→ Controlled comparison: same model, same GPU, same workload, same quantization. Sweep request rates. Compare latency-throughput curves, not single numbers.

---

## Self-Test Questions

1. What's the difference between TTFT and TPOT? Which phase of inference dominates each?
2. Why must you sweep request rates rather than measure at a single rate?
3. What is the saturation point and why does it matter?
4. Name 3 common benchmarking pitfalls and how to avoid them.
5. Design a benchmark comparing vLLM vs TRT-LLM. What do you control? What do you vary?
6. Why report P99 latency instead of mean?

---

## Notes
```


```
