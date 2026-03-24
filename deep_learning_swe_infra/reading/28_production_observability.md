# Production Observability for LLM Serving

**Priority:** Week 8-10.
**Interview map:** Track A Rounds 2 & 4 (you'll discuss monitoring), Track B Round 2

---

## 1. Why LLM Observability is Different

Traditional web services: monitor CPU, memory, request latency, error rate.

LLM serving adds unique dimensions:
- **Two-phase latency:** TTFT (prefill) and TPOT (decode) are separate concerns
- **GPU metrics:** Utilization, memory, temperature, power
- **KV cache:** Occupancy, eviction rate, hit rate (prefix caching)
- **Batch dynamics:** Batch size varies continuously
- **Token-level streaming:** Responses stream over seconds, not milliseconds
- **Model quality:** Hard to monitor in production (no ground truth)

---

## 2. The Metrics Stack

### Tier 1: Request-Level Metrics (User-Facing)
| Metric | What to track | Alert threshold |
|---|---|---|
| TTFT P50/P99 | Time to first token | P99 > SLO |
| TPOT P50/P99 | Time per output token | P99 > SLO |
| End-to-end latency P50/P99 | Total request time | P99 > SLO |
| Error rate | 5xx, timeouts, OOM | > 1% |
| Request throughput | Requests/sec | Below expected baseline |
| Token throughput | Output tokens/sec | Below expected baseline |
| Queue depth | Pending requests | > threshold |
| Queue wait time | Time in queue before processing | > 500ms |

### Tier 2: Engine-Level Metrics (Operator-Facing)
| Metric | What to track | Why |
|---|---|---|
| Batch size (running) | Current batch size | Low = underutilized GPU |
| Batch size (waiting) | Queued requests | High = need to scale |
| KV cache utilization | % of blocks used | >90% = risk of OOM/preemption |
| KV cache eviction rate | Blocks evicted/sec | High = memory pressure |
| Prefix cache hit rate | % of prefill tokens cached | Low = prefix caching not helping |
| Preemption rate | Requests preempted/sec | High = too many concurrent requests |
| Num running requests | Active requests | Basic capacity metric |
| Num waiting requests | Queued requests | Scaling signal |

### Tier 3: GPU-Level Metrics (Infrastructure)
| Metric | What to track | Alert threshold |
|---|---|---|
| GPU utilization | SM activity % | <50% (wasting money) or 100% (saturated) |
| GPU memory used | VRAM GB | >95% (OOM risk) |
| GPU memory bandwidth util | HBM utilization % | >90% (memory-bound) |
| GPU temperature | Celsius | >85°C (throttling risk) |
| GPU power draw | Watts | Near TDP = max perf |
| NVLink throughput | GB/s | Low = TP communication bottleneck |
| PCIe throughput | GB/s | High = CPU-GPU transfer bottleneck |

---

## 3. Monitoring Stack

### The standard stack
```
┌─────────────────────────────────────────┐
│           Grafana Dashboards             │
│  ├─ Request latency overview             │
│  ├─ GPU utilization per node             │
│  ├─ KV cache health                      │
│  ├─ Throughput and scaling               │
│  └─ Cost tracking ($/token)              │
├─────────────────────────────────────────┤
│           Prometheus                      │
│  ├─ Scrapes metrics from all sources     │
│  ├─ Stores time-series data              │
│  └─ Evaluates alerting rules             │
├─────────────────────────────────────────┤
│       Metrics Sources                    │
│  ├─ vLLM /metrics endpoint (Prometheus)  │
│  ├─ nvidia-smi / DCGM (GPU metrics)     │
│  ├─ Node exporter (CPU, memory, disk)    │
│  └─ Custom app metrics (audit, errors)   │
└─────────────────────────────────────────┘
```

### NVIDIA DCGM (Data Center GPU Manager)
- Collects GPU metrics at scale (100s of GPUs)
- Prometheus-compatible exporter
- Tracks: utilization, memory, temperature, ECC errors, NVLink
- More comprehensive than nvidia-smi for production

### vLLM Metrics Endpoint
vLLM exposes Prometheus metrics at `/metrics`:
```
vllm:num_requests_running         — current batch size
vllm:num_requests_waiting         — queue depth
vllm:num_requests_swapped         — swapped to CPU
vllm:gpu_cache_usage_perc         — KV cache utilization
vllm:cpu_cache_usage_perc         — CPU swap usage
vllm:avg_prompt_throughput_toks   — input tokens/sec
vllm:avg_generation_throughput_toks — output tokens/sec
vllm:e2e_request_latency_seconds  — histogram
vllm:time_to_first_token_seconds  — histogram
vllm:time_per_output_token_seconds — histogram
```

---

## 4. Key Dashboards to Build

### Dashboard 1: Request Latency Overview
```
Row 1: TTFT P50 (gauge) | TTFT P99 (gauge) | TPOT P50 (gauge) | TPOT P99 (gauge)
Row 2: TTFT distribution histogram over time
Row 3: TPOT distribution histogram over time
Row 4: Error rate (%) | Timeout rate (%)
```

### Dashboard 2: GPU Health
```
Row 1: GPU utilization per GPU (line chart, all GPUs)
Row 2: GPU memory used per GPU (stacked area)
Row 3: GPU temperature (line, with threshold at 85°C)
Row 4: NVLink throughput (if TP enabled)
```

### Dashboard 3: Engine Health
```
Row 1: Running requests | Waiting requests | Swapped requests
Row 2: KV cache utilization (%) — CRITICAL dashboard
Row 3: Batch size over time
Row 4: Prefix cache hit rate (if enabled)
Row 5: Preemption rate
```

### Dashboard 4: Cost & Efficiency
```
Row 1: Tokens/sec/GPU (efficiency metric)
Row 2: Cost per 1M tokens (calculated)
Row 3: GPU hours consumed (for billing)
Row 4: Requests per dollar
```

---

## 5. Alerting Rules

### Critical alerts (page on-call)
```yaml
- alert: HighTTFTP99
  expr: histogram_quantile(0.99, vllm:time_to_first_token_seconds) > 5.0
  for: 5m
  severity: critical
  annotation: "TTFT P99 > 5s for 5 minutes — users experiencing delays"

- alert: GPUMemoryNearFull
  expr: vllm:gpu_cache_usage_perc > 0.95
  for: 2m
  severity: critical
  annotation: "KV cache >95% — OOM and preemption imminent"

- alert: HighErrorRate
  expr: rate(vllm:request_errors_total[5m]) / rate(vllm:request_total[5m]) > 0.05
  for: 3m
  severity: critical
  annotation: "Error rate >5% — investigate immediately"
```

### Warning alerts (investigate soon)
```yaml
- alert: QueueBuildingUp
  expr: vllm:num_requests_waiting > 50
  for: 5m
  severity: warning
  annotation: "50+ requests queued — consider scaling up"

- alert: LowGPUUtilization
  expr: dcgm_gpu_utilization < 30
  for: 15m
  severity: warning
  annotation: "GPU utilization <30% for 15m — consider scaling down"

- alert: HighPreemptionRate
  expr: rate(vllm:preemptions_total[5m]) > 10
  for: 5m
  severity: warning
  annotation: "High preemption rate — KV cache pressure"
```

---

## 6. Autoscaling Signals

### What to scale on
```
Primary signal:  Queue depth (num_requests_waiting)
  → Best correlation with user-perceived latency
  → Scale up when queue > threshold, scale down when queue = 0

Secondary signal: KV cache utilization
  → Scale up when >85% (approaching memory limit)

DO NOT scale on:
  → CPU utilization (irrelevant for GPU workloads)
  → GPU utilization alone (can be high with good batching)
```

### Scaling strategy
```
Scale-up: Fast (add replicas in 1-2 minutes)
  Trigger: queue_depth > 20 for 2 minutes

Scale-down: Slow (graceful drain, 10-15 minutes)
  Trigger: queue_depth = 0 for 10 minutes
  Drain: stop accepting new requests, finish in-flight, then terminate

Minimum replicas: Always keep ≥1 warm (cold start = 30s-2min for model loading)
```

---

## 7. Distributed Tracing for LLM Requests

For complex pipelines (RAG + guardrails + LLM):

```
Request trace:
├── API Gateway (2ms)
├── PII Scrubber (15ms)
├── RAG Retrieval
│   ├── Embedding (10ms)
│   └── Vector search (20ms)
├── Prompt Construction (5ms)
├── LLM Inference
│   ├── Queue wait (50ms)
│   ├── Prefill (120ms) ← TTFT
│   └── Decode (800ms, 200 tokens × 4ms each)
├── Output PII Check (10ms)
└── Response sent (total: 1032ms)
```

### Tools
- **OpenTelemetry:** Standard tracing framework
- **Jaeger:** Trace visualization
- **LangSmith:** LangChain-specific tracing (if using LangChain)

---

## 8. Interview Connections

> "How do you monitor an LLM serving system in production?"

**A:** "Three tiers: (1) Request-level: TTFT and TPOT P50/P99, error rate, queue depth — these directly reflect user experience. (2) Engine-level: KV cache utilization, batch size, preemption rate — these predict problems before users see them. (3) GPU-level: utilization, memory, temperature via DCGM. I use Prometheus + Grafana, scraping vLLM's native /metrics endpoint and DCGM exporters. Key alert: KV cache >95% triggers scale-up before OOM."

> "What signal do you use for autoscaling GPU inference?"

**A:** "Queue depth is the primary signal — it directly correlates with user-perceived latency. Scale up when queue exceeds threshold for 2+ minutes. Scale down slowly with a 10-minute cool-down to avoid thrashing. Never scale to zero — keep at least one warm replica to avoid cold start latency."

---

## Self-Test Questions

1. What 3 metrics would you put on the most critical monitoring dashboard?
2. Why is GPU utilization alone a bad autoscaling signal?
3. What is DCGM and how does it differ from nvidia-smi?
4. Describe the autoscaling strategy for an LLM serving system.
5. What does high KV cache utilization + high preemption rate tell you?
6. Design an alerting rule for TTFT P99 exceeding the SLO.

---

## Notes
```


```
