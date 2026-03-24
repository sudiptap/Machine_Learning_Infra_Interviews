# Advanced Interview Q&A Bank — Batch 2

**Usage:** These are harder questions for weeks 8+. Practice after mastering Batch 1 (09_interview_qa_bank.md).
**Method:** Cover the answer, speak out loud for 2 minutes, then compare.

---

## Category 1: Advanced Systems Design

### Q1: Design a multi-tenant LLM platform for a Fortune 500 bank with 20 departments.

**A:**

**Requirements:** 20 departments, some need customized models (LoRA), compliance (SOC2, data residency), high availability, cost efficiency.

**Architecture:**
1. **Shared base model** (Llama-3 70B INT8 on 1x A100 80GB per replica) + **20 LoRA adapters** (~50 MB each = 1 GB total adapter overhead)
2. **Multi-LoRA serving** via vLLM/NIM — route requests to correct adapter based on department
3. **Request routing:** API gateway with department-level auth → LoRA adapter selection → shared inference pool
4. **KV cache isolation:** Logical isolation per department (separate cache pools) to prevent cross-contamination
5. **Guardrails:** Per-department topic restrictions (NeMo Guardrails), shared PII scrubbing, shared audit logging
6. **Observability:** Per-department dashboards (TTFT/TPOT/throughput per department), shared GPU health monitoring
7. **Scaling:** Autoscale on aggregate queue depth. Minimum 2 replicas for HA.
8. **Cost allocation:** Track tokens per department for internal chargeback

**Cost math:** Without LoRA: 20 × 70B models × 1 GPU = 20 GPUs ($70/hr). With LoRA: 2 replicas × 1 GPU = 2 GPUs ($7/hr). **10x cost reduction.**

---

### Q2: A customer's P99 TTFT is 5 seconds. They need <500ms. Diagnose and fix.

**A:**

**Step 1: Where is time spent?**
- Queue wait? Check queue depth. If >0, requests are waiting → scale up.
- Prefill computation? Check prompt length distribution. Long prompts = slow prefill.
- Model loading? Is model in GPU memory or loading from disk?

**Step 2: Isolate the cause (in order of likelihood):**
1. **Queue congestion:** Queue depth > 0 means requests wait. Solution: add replicas, autoscale on queue depth.
2. **Long prompts:** P99 likely corresponds to longest prompts. Solutions: chunked prefill (reduces max TTFT spike), prompt compression, or prefill-decode disaggregation.
3. **Prefill blocking decode:** Other requests' prefill operations block this request. Solution: chunked prefill limits max block time.
4. **Cold start:** First request after scaling triggers model load. Solution: keep warm replicas, pre-load models.
5. **Network:** Client → server latency. Solution: check with ping, colocate.

**Step 3: Fix (most likely path):**
Enable chunked prefill (cap at 512 tokens) + autoscale on queue depth + set TTFT SLO alerting.

---

### Q3: Design a system that serves 10,000 requests/second with Llama-3 8B.

**A:**

**Math first:**
- Llama-3 8B on A100 80GB INT8: ~100 output tokens/sec per request, ~60 concurrent requests
- At average 100 output tokens per request: ~60 completions/sec per GPU
- Need: 10,000 / 60 ≈ **167 GPUs** (A100 80GB)
- With A100 40GB: ~30 concurrent → 333 GPUs
- With H100 FP8: ~120 concurrent → ~83 GPUs

**Architecture:**
1. **Load balancer:** Layer 7 (can route by model version), health-check aware
2. **167 vLLM replicas** behind the load balancer, each on 1x A100 80GB
3. **Kubernetes:** GPU operator, HPA on custom queue depth metric
4. **Region:** Multi-AZ for availability, single region for latency
5. **Caching:** Semantic cache for common queries (reduce 20-30% of requests). Prefix caching for shared system prompts.
6. **Optimization:** INT8 weights + FP8 KV cache = maximum concurrent requests per GPU

**Cost:** 167 A100s × $3.50/hr = $584/hr = **$14K/day = $420K/month**
With semantic caching (30% reduction): ~117 GPUs = **$290K/month**

---

## Category 2: Advanced GPU/Kernel Questions

### Q4: Given this Nsight output, analyze the kernel:
```
DRAM throughput: 1.8 TB/s (90% of peak 2.0 TB/s)
SM throughput: 45 TFLOPS (14% of peak 312 TFLOPS)
Achieved occupancy: 75%
L2 hit rate: 12%
```

**A:** This kernel is **memory-bandwidth-bound**:
- DRAM at 90% of peak → we're saturating HBM bandwidth
- SM at 14% of peak → GPU compute is mostly idle, waiting for data
- Low L2 hit rate (12%) → most accesses go to HBM, not served from cache
- 75% occupancy is good → enough warps to hide latency

**To optimize:** We need to reduce HBM accesses. Options:
1. **Kernel fusion:** Combine with adjacent operations to avoid writing/reading intermediates
2. **Tiling into shared memory:** Reuse data from SRAM instead of re-reading from HBM
3. **Quantization:** Reduce data size → less bandwidth needed
4. This is the typical profile for LLM decode — weight loading dominates.

---

### Q5: Write pseudocode for online softmax (used in FlashAttention).

**A:**
```
function online_softmax(x_blocks):
    m = -infinity  // running max
    l = 0.0        // running sum of exp
    d = 0.0        // running weighted sum (for attention output)

    for block in x_blocks:
        m_new = max(m, max(block))        // update max

        // Rescale previous results
        correction = exp(m - m_new)
        l = l * correction + sum(exp(block - m_new))

        // If computing attention: rescale output accumulator too
        d = d * correction + sum(exp(block - m_new) * v_block)

        m = m_new

    return d / l  // final softmax-weighted output
```

**Key insight:** The correction factor `exp(m_old - m_new)` rescales all previous partial results when a new maximum is discovered. This is mathematically equivalent to standard softmax but doesn't require a first pass to find the global max.

---

### Q6: Why is FP8 inference on H100 almost 2x faster than FP16, not just 2x less memory?

**A:** Two effects compound:
1. **Memory bandwidth:** FP8 weights are 2x smaller → load from HBM in half the time. For memory-bound decode, this alone gives ~2x speedup.
2. **Compute throughput:** H100 FP8 Tensor Cores have 2x the peak FLOPS (1979 vs 989 TFLOPS). For compute-bound prefill, this gives another ~2x speedup.

During decode (memory-bound), the ~2x comes mainly from bandwidth.
During prefill (compute-bound), the ~2x comes mainly from Tensor Core throughput.
Net result: ~2x across both phases, but for different reasons.

---

## Category 3: Track A Scenario Questions

### Q7: A bank wants to move from OpenAI API to self-hosted. Walk me through the migration.

**A:**

**Phase 1: Assessment (2 weeks)**
- Inventory current OpenAI usage: models, token volume, latency requirements
- Identify which models to replace (GPT-4 → Llama-3 70B, GPT-3.5 → Llama-3 8B)
- Compliance requirements: data residency, audit logging, PII handling
- Budget: GPU costs vs OpenAI API costs (find the breakeven)

**Phase 2: Infrastructure (4 weeks)**
- Deploy NIM or TRT-LLM on A100/H100 cluster (on-prem or private cloud)
- Set up monitoring (Prometheus + Grafana)
- Implement guardrails (PII scrubbing, content filtering)
- Build API compatibility layer (OpenAI-compatible endpoint → NIM backend)

**Phase 3: Validation (2 weeks)**
- Run evaluation suite: compare self-hosted quality vs OpenAI on real use cases
- Load test: verify latency SLOs are met at production traffic
- Security audit: pen test, compliance review

**Phase 4: Migration (2 weeks)**
- Canary rollout: 5% → 25% → 50% → 100%
- Monitor quality metrics during rollout
- Maintain OpenAI as fallback during transition

**Cost comparison:**
- OpenAI GPT-4: ~$30/1M input tokens
- Self-hosted Llama-3 70B: ~$1-5/1M tokens (depends on utilization)
- **5-30x cost reduction** at sufficient scale

---

### Q8: Customer asks: "Should we use on-prem DGX or cloud GPUs?"

**A:**

**Recommend on-prem DGX when:**
- Sustained >50% GPU utilization (cost crossover)
- Strict data residency (data cannot leave premises)
- Predictable, steady workload (not bursty)
- 3-year commitment is acceptable
- Team has infrastructure expertise

**Recommend cloud when:**
- Variable/bursty workload
- Need to experiment with different GPU types
- Don't want infrastructure management overhead
- Need global distribution (multi-region)
- Short-term or uncertain demand

**Hybrid approach (often best):**
- On-prem DGX for baseline steady-state workload
- Cloud burst for peak demand
- Keep compliance-critical workloads on-prem
- Use cloud for development and testing

---

## Category 4: Research-Adjacent Questions

### Q9: Compare FlashAttention-2 to FlashAttention-3. What changed?

**A:**
- FA3 leverages **H100-specific hardware:** TMA (Tensor Memory Accelerator) for async data loading, FP8 Tensor Cores
- FA3 achieves ~75% of H100 peak FLOPS (vs FA2's ~73% on A100)
- Key innovation: **asynchronous block-level pipelining** — while one block computes on Tensor Cores, the next block loads via TMA, and the previous block's softmax rescaling runs on non-Tensor Core units
- Three operations overlap: memory load, matmul, softmax — exploits H100's ability to run these on different hardware units simultaneously

### Q10: What is FlashInfer and how does it relate to FlashAttention?

**A:**
FlashInfer is a library of high-performance attention kernels specifically designed for **LLM serving** (as opposed to FlashAttention which was designed for training). Key differences:
- **Paged KV cache support:** Works natively with PagedAttention's non-contiguous blocks
- **Variable-length batching:** Different requests in the batch have different sequence lengths
- **Decode-optimized:** Special kernels for the decode phase (1 query token, long KV cache)
- **GQA/MQA optimized:** Efficient grouped-query attention kernels
- Used by vLLM and SGLang as their attention backend

---

## How to Practice These
1. Set a 30-minute session: pick 3 questions randomly
2. Answer each out loud for 3 minutes (set timer)
3. For systems design (Q1-Q3): practice on whiteboard/paper
4. For kernel questions (Q4-Q6): make sure you can write pseudocode
5. For SA scenarios (Q7-Q8): practice with "customer" framing
6. Record yourself and review

---

## Notes
```
(Track questions you struggled with — revisit them)


```
