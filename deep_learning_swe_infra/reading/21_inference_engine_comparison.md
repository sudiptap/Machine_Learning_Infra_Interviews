# Inference Engine Deep Comparison: vLLM vs TRT-LLM vs SGLang

**Priority:** Week 4-5.
**Interview map:** Track A Round 2, Track B Rounds 2 & 3

---

## 1. Overview

| | vLLM | TensorRT-LLM | SGLang |
|---|---|---|---|
| **Org** | UC Berkeley (open source) | NVIDIA | UC Berkeley (open source) |
| **Language** | Python + C++ extensions | C++ core, Python wrapper | Python + C++ extensions |
| **Key innovation** | PagedAttention | Compile-time optimization | RadixAttention |
| **Setup complexity** | Easy (pip install) | Medium (build engine) | Easy (pip install) |
| **Model support** | Broad (HuggingFace) | NVIDIA-optimized models | Broad (HuggingFace) |
| **License** | Apache 2.0 | Apache 2.0 | Apache 2.0 |
| **First release** | 2023 | 2023 | 2023 |
| **Community** | Very large | NVIDIA-backed | Growing fast |

---

## 2. vLLM — Deep Dive

### Architecture
```
AsyncLLMEngine
├── Scheduler (continuous batching, priority)
├── BlockManager (PagedAttention)
├── ModelRunner (executes forward pass)
└── Workers (one per GPU, handles TP)
```

### Key strengths
1. **PagedAttention:** Near-optimal KV cache memory utilization
2. **Ease of use:** `pip install vllm` → `vllm serve model_name`
3. **Model coverage:** Supports 50+ model architectures from HuggingFace
4. **Community:** Largest open-source LLM serving community, fastest bug fixes
5. **Feature velocity:** New features land fast (chunked prefill, prefix caching, LoRA, speculative decoding)

### Key limitations
1. **Performance ceiling:** Pure Python scheduler has overhead vs compiled systems
2. **No compile-time optimization:** Kernels are general-purpose, not model-specific
3. **Single-engine:** Doesn't naturally support multi-model serving (need separate instances)

### Best for
- General-purpose LLM serving
- Rapid experimentation with different models
- When developer productivity matters more than last 10% performance
- When you need broad model support

### Performance characteristics
- TTFT: Good (efficient prefill with FlashAttention/FlashInfer)
- TPOT: Good (PagedAttention reduces memory waste → more room for batching)
- Throughput: Very good at high concurrency (continuous batching + PagedAttention)
- Typically within 10-20% of TRT-LLM, sometimes matching or exceeding

---

## 3. TensorRT-LLM — Deep Dive

### Architecture
```
Build phase (offline):
  HuggingFace model → TRT-LLM builder → optimized TRT engine
  ├── Graph optimization (layer fusion, constant folding)
  ├── Quantization (INT8/FP8/INT4)
  ├── Kernel auto-tuning (best kernel per layer)
  └── Engine serialization (save to disk)

Runtime phase:
  TRT engine → Executor → GptSession
  ├── In-flight batching (continuous batching)
  ├── Paged KV cache
  ├── Tensor parallelism
  └── Inflight decoding
```

### Key strengths
1. **Compile-time optimization:** Model-specific kernel selection and fusion → fastest kernels
2. **FP8 on H100:** Native support, 2x throughput over FP16
3. **In-flight batching:** NVIDIA's implementation of continuous batching
4. **Kernel library:** Access to NVIDIA's proprietary optimized kernels (fMHA, etc.)
5. **Enterprise support:** Part of NVIDIA AI Enterprise

### Key limitations
1. **Build step:** Must compile engine for each model + config (10-60 minutes)
2. **Less flexible:** Engine is compiled for specific batch size range, TP config
3. **Model support:** Lags behind vLLM for new/unusual architectures
4. **Steeper learning curve:** More configuration knobs
5. **Tied to NVIDIA GPUs:** No AMD/Intel support

### Best for
- Maximum performance on NVIDIA GPUs
- Production deployment where 10-20% performance gain = real cost savings
- When FP8 on H100 is available (biggest advantage)
- Enterprises already in NVIDIA ecosystem

### Performance characteristics
- TTFT: Excellent (optimized kernels for prefill)
- TPOT: Best-in-class (compile-time kernel selection)
- Throughput: Highest at scale (optimized everything)
- Typically 10-30% faster than vLLM, especially with FP8

---

## 4. SGLang — Deep Dive

### Architecture
```
FastAPI server
├── TokenizerManager (async tokenization)
├── Scheduler (continuous batching)
├── RadixAttention (radix tree KV cache)
├── ModelRunner (TorchTP or custom)
└── ZMQ-based multi-process communication
```

### Key strengths
1. **RadixAttention:** Most sophisticated prefix caching via radix tree
2. **Structured generation:** Fastest constrained decoding (JSON, regex, grammar)
3. **Multi-modal:** Strong vision-language model support
4. **Programming model:** SGLang DSL for complex LLM programs (branching, loops, multi-call)
5. **FlashInfer integration:** Cutting-edge attention kernels

### Key limitations
1. **Younger project:** Smaller community than vLLM
2. **Less enterprise adoption:** Fewer production deployments at scale
3. **Documentation:** Less comprehensive than vLLM

### Best for
- Workloads with heavy prefix sharing (chatbots with system prompts, RAG)
- Structured output (JSON mode, constrained generation)
- Complex multi-step LLM programs
- When prefix caching is critical for throughput

### Performance characteristics
- TTFT: Excellent (RadixAttention can skip most of prefill for shared prefixes)
- TPOT: Good (comparable to vLLM)
- Throughput: Excellent for prefix-heavy workloads (can exceed vLLM/TRT-LLM)
- Prefix cache hit rate is the key differentiator

---

## 5. Head-to-Head Comparison

### Scenario 1: Chatbot (heavy prefix sharing, system prompt reuse)
```
Winner: SGLang
Why: RadixAttention reuses system prompt KV cache across all requests.
     90%+ prefix cache hit → dramatically reduced prefill compute.
vLLM:    Has prefix caching, but less sophisticated than radix tree.
TRT-LLM: Has KV cache reuse, but not as flexible.
```

### Scenario 2: Maximum throughput, A100, FP16
```
Winner: TRT-LLM (usually) or vLLM (close)
Why: TRT-LLM's compile-time optimization gives 10-20% edge at scale.
     vLLM is within striking distance and much easier to deploy.
SGLang:  Competitive, sometimes faster on specific workloads.
```

### Scenario 3: Maximum throughput, H100, FP8
```
Winner: TRT-LLM
Why: Native FP8 support is TRT-LLM's biggest advantage on H100.
     2x throughput over FP16. vLLM's FP8 support is catching up.
```

### Scenario 4: New model just released on HuggingFace
```
Winner: vLLM
Why: Broadest model support, fastest to add new architectures.
     TRT-LLM may not support it for weeks/months.
     SGLang support depends on community.
```

### Scenario 5: Constrained JSON output
```
Winner: SGLang
Why: Built-in grammar-constrained decoding is fastest.
     vLLM: Has structured output via outlines, but slower.
     TRT-LLM: Limited structured generation support.
```

### Scenario 6: Enterprise with compliance requirements
```
Winner: TRT-LLM / NIM
Why: NVIDIA AI Enterprise support, security patches, SLAs.
     vLLM/SGLang are community-supported (no enterprise SLA).
```

---

## 6. Feature Matrix

| Feature | vLLM | TRT-LLM | SGLang |
|---|---|---|---|
| Continuous batching | ✅ | ✅ (in-flight) | ✅ |
| PagedAttention | ✅ | ✅ (paged KV) | ✅ |
| Prefix caching | ✅ | ✅ | ✅ (RadixAttention) |
| FP8 quantization | ✅ | ✅ (best) | ✅ |
| INT4 (AWQ/GPTQ) | ✅ | ✅ | ✅ |
| Tensor parallelism | ✅ | ✅ | ✅ |
| Pipeline parallelism | ✅ | ✅ | Partial |
| Speculative decoding | ✅ | ✅ | ✅ |
| LoRA serving | ✅ | ✅ | ✅ |
| Chunked prefill | ✅ | ✅ | ✅ |
| Structured output | ✅ (outlines) | Limited | ✅ (fastest) |
| Multi-modal | ✅ | Limited | ✅ (strong) |
| OpenAI-compatible API | ✅ | ✅ | ✅ |
| Disaggregated prefill | Experimental | ✅ | Experimental |

---

## 7. When to Recommend Each

### As Track A (SA):
```
Customer wants simplest deployment?         → NIM (TRT-LLM under hood)
Customer needs maximum performance + H100?  → TRT-LLM / NIM
Customer needs broad model support?         → vLLM
Customer has compliance requirements?       → NIM + AI Enterprise
Customer has heavy prefix sharing?          → Evaluate SGLang vs vLLM
```

### As Track B (Engineer):
```
Contributing to which codebase?  → vLLM (largest community, most impact)
                                   SGLang (growing fast, more greenfield)
Building InferBench?             → Benchmark all three
Understanding inference deeply?  → Read vLLM code (Python, accessible)
                                   Then read TRT-LLM (see compiled approach)
```

---

## 8. Interview Connections

> "What are the tradeoffs between vLLM and TensorRT-LLM?"

**A:** "vLLM is easier to deploy, supports more models, and has the largest community. TRT-LLM achieves ~10-30% better performance through compile-time optimization and has the best FP8 support on H100. I'd recommend vLLM for rapid deployment and experimentation, TRT-LLM (or NIM) for production where performance directly impacts cost. In practice, the gap is narrowing — vLLM's FlashInfer backend is closing the performance difference."

> "A customer is choosing between vLLM and SGLang. How do you help them decide?"

**A:** "It depends on their workload. If they have heavy prefix sharing (chatbot with system prompt, RAG with overlapping documents), SGLang's RadixAttention can give 50-90% prefill savings — that's a massive throughput boost. If they need structured JSON output, SGLang is also faster. For general-purpose serving without heavy prefix sharing, vLLM's larger community and broader model support usually wins."

---

## Self-Test Questions

1. What is TRT-LLM's compile-time optimization and why does it make it faster?
2. When does SGLang outperform vLLM? What's the key feature?
3. A customer has H100s and wants maximum throughput. Which engine and why?
4. A startup needs to deploy 5 different model architectures quickly. Which engine?
5. Compare the KV cache management approach in all three engines.
6. You're building InferBench. What's the fair way to compare these three engines?

---

## Notes
```


```
