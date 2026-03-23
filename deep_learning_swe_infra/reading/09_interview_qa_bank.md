# Interview Q&A Bank — Both Tracks

**Usage:** Read each question, try to answer from memory FIRST, then check the answer.
**Practice:** Cover the answer, speak your response out loud, then compare.

---

## Category 1: GPU Fundamentals (Track B Round 4)

### Q1: Describe the GPU memory hierarchy.
**A:** Four levels, fastest to slowest: (1) Registers — per thread, ~1 cycle latency, ~20 TB/s effective. (2) Shared memory (SRAM) — per block, ~20-30 cycles, ~19 TB/s on H100, ~228 KB per SM. (3) L2 cache — shared across SMs, ~200 cycles, ~40 MB on A100. (4) HBM (global memory) — 80 GB on A100, ~400-600 cycles, ~2 TB/s. The key insight: there's a ~10,000x speed difference between registers and HBM. All GPU optimizations exist to minimize HBM accesses.

### Q2: What is a warp and what is warp divergence?
**A:** A warp is 32 threads that execute the same instruction in lockstep (SIMT). If threads in a warp take different branches (e.g., an if/else), both branches execute sequentially — this is warp divergence. It effectively halves (or worse) performance. In attention computation, causal masking creates a triangular pattern where some threads compute and others are masked, causing divergence. FlashAttention handles this by skipping entirely masked blocks.

### Q3: What is the roofline model?
**A:** A model to classify kernels as compute-bound or memory-bound. X-axis: arithmetic intensity (FLOPs per byte of data moved). Y-axis: achievable performance (FLOPS). The "roof" has two segments — a sloped line (memory bandwidth limit) and a flat line (peak compute). Below the ridge point: memory-bound. Above: compute-bound. LLM decode falls on the left (memory-bound), prefill on the right (compute-bound).

### Q4: Why does shared memory tiling improve matmul performance?
**A:** In naive matmul, each thread reads its row and column from global memory (HBM). For an NxN matmul, each element is read N times from HBM. With tiling, threads cooperatively load tiles into shared memory, then compute from shared memory. Each HBM element is loaded once per tile, but reused by all threads in the block. This reduces HBM reads by a factor of the tile size, shifting the kernel from memory-bound toward compute-bound.

### Q5: Write pseudocode for a tiled matmul.
**A:**
```
kernel tiled_matmul(A, B, C, N):
    // Identify this thread's output element
    row = blockIdx.y * TILE_SIZE + threadIdx.y
    col = blockIdx.x * TILE_SIZE + threadIdx.x

    shared float As[TILE_SIZE][TILE_SIZE]
    shared float Bs[TILE_SIZE][TILE_SIZE]

    float sum = 0.0

    // Loop over tiles along the K dimension
    for tile = 0 to N/TILE_SIZE:
        // Cooperatively load tile into shared memory
        As[threadIdx.y][threadIdx.x] = A[row][tile * TILE_SIZE + threadIdx.x]
        Bs[threadIdx.y][threadIdx.x] = B[tile * TILE_SIZE + threadIdx.y][col]
        __syncthreads()

        // Compute partial dot product from shared memory
        for k = 0 to TILE_SIZE:
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x]
        __syncthreads()

    C[row][col] = sum
```

---

## Category 2: LLM Inference (Both Tracks, Rounds 2 & 3)

### Q6: Why is LLM decode memory-bandwidth-bound?
**A:** During decode, we generate one token at a time. Each token requires a matrix-vector multiply against the full model weights — loading all weights from HBM but doing very little computation per weight. The arithmetic intensity is ~1 FLOP/byte (vs ~N FLOP/byte for matrix-matrix multiply). On an A100, loading 140 GB of weights (Llama-3 70B FP16) at 2 TB/s takes 70ms, but the actual multiply-add computation takes much less. The GPU is waiting for data, not for compute.

### Q7: Why does batching help throughput but not TTFT?
**A:** Batching amortizes the cost of loading model weights from HBM. With batch=1, you load 140 GB to generate 1 token. With batch=32, you load 140 GB once but generate 32 tokens — 32x better utilization of memory bandwidth. However, each individual request still waits the same time for its turn through the model, so TTFT doesn't improve (and may get slightly worse due to queue time).

### Q8: How does continuous batching work?
**A:** Instead of batching at the request level (wait for all requests to finish), batch at the iteration level. After each decode step: remove completed requests, add new requests from the queue. Every iteration has a full batch. This prevents GPU waste from short requests waiting for long ones. Typically gives 2-3x throughput improvement over static batching.

### Q9: Explain PagedAttention.
**A:** Inspired by OS virtual memory. KV cache is divided into fixed-size blocks (pages). A request's KV cache can be stored in non-contiguous blocks. A block table maps logical positions to physical memory blocks. Blocks are allocated on-demand as tokens are generated, and freed when requests complete. This eliminates internal fragmentation (unused reserved space) and external fragmentation (non-contiguous free memory). The vLLM paper showed this reduces KV cache memory waste from 60-80% to <4%.

### Q10: Walk through the full request path from HTTP to token generation.
**A:** (1) HTTP request arrives at API gateway/load balancer. (2) Enters request queue with priority. (3) Scheduler decides when to add it to the running batch — checks if enough KV cache memory is available. (4) Prefill: tokenize input, run full forward pass on all input tokens (matrix-matrix multiply), store KV cache. Output: first token. (5) Decode loop: each step, run forward pass for one new token (matrix-vector multiply), attend to all cached K,V, sample next token, stream back to client. (6) On completion, scheduler frees KV cache blocks and fills the slot with next queued request.

### Q11: What is FlashAttention and why does it reduce memory?
**A:** Standard attention materializes the N×N attention score matrix in HBM — O(N²) memory. FlashAttention tiles the computation into blocks that fit in SRAM, computes attention tile-by-tile using online softmax, and never materializes the full attention matrix. This reduces extra memory from O(N²) to O(N), and reduces HBM accesses by 5-20x. Since attention is memory-bound, fewer HBM accesses means directly faster execution: 2-4x speedup.

### Q12: What is speculative decoding?
**A:** Use a small draft model to propose K tokens quickly, then verify all K at once with the target model in a single forward pass (prefill-style, compute-bound). Accept/reject scheme guarantees identical output distribution. Helps at low batch sizes (2-3x speedup) when the draft model has high acceptance rate. Doesn't help at large batch sizes where the GPU is already well-utilized.

### Q13: A model's TPOT is 80ms. Customer needs 40ms. What do you look at?
**A:** In order: (1) Check if memory-bandwidth saturated — is HBM bandwidth the bottleneck? (2) Quantize: FP16→INT8 halves the weights, could halve TPOT. (3) Tensor parallelism: TP=2 splits weights across 2 GPUs, roughly halving per-GPU weight loading. (4) Check KV cache: is it competing for HBM bandwidth? Quantized KV cache helps. (5) Check batch size: over-batching can increase contention. (6) Speculative decoding: if batch=1, could give 2-3x improvement.

---

## Category 3: Distributed Inference (Both Tracks)

### Q14: Tensor parallelism vs pipeline parallelism?
**A:** TP splits each layer across GPUs (all GPUs process every token, on a subset of weights). Requires all-reduce after every layer — needs NVLink bandwidth. Reduces latency. PP splits layers across GPUs (each GPU processes full layers, different depth). Only point-to-point communication between stages — works over InfiniBand. Use TP within a node, PP across nodes.

### Q15: Llama-3 70B on A100 80GB — configure tensor parallelism.
**A:** FP16 = ~140 GB. Won't fit on 1 GPU. TP=2 minimum (70 GB per GPU). Each decode step loads ~70 GB at 2 TB/s = ~35ms TPOT plus all-reduce overhead (~1-2ms per layer over NVLink). For lower latency, TP=4 → ~17.5ms TPOT. Alternative: INT8 quantization → ~70 GB total → fits on 1 GPU, no TP overhead.

---

## Category 4: NVIDIA Products (Track A Rounds 1–4)

### Q16: NIM vs setting up Triton + TRT-LLM manually?
**A:** NIM when: fastest time to deploy, standard models, customer wants simplicity and enterprise support. Manual Triton + TRT-LLM when: custom models NIM doesn't support, need custom serving logic, multi-model pipelines, or maximum control. NIM uses TRT-LLM + Triton under the hood — it's the easy button.

### Q17: Why would a bank choose NIM over raw vLLM?
**A:** Enterprise support with SLAs, security patches, certified containers, compliance-friendly (NVIDIA AI Enterprise license), TRT-LLM optimization (generally faster), and NVIDIA's reputation in regulated industries. vLLM is great but community-supported — a bank needs vendor backing for production.

---

## Category 5: Track A SA Scenarios

### Q18: Design an LLM serving system for a large bank (SOC2, data residency).
**A:** (1) On-premise or private cloud (data residency). DGX cluster or private cloud with A100/H100. (2) NIM containers in hardened Kubernetes. (3) Network: PII scrubbing at ingress, TLS everywhere, audit logging on every request. (4) KMS-managed encryption at rest and in transit. (5) Multi-tenant routing with isolation — separate KV cache pools per department. (6) Observability: Prometheus/Grafana for TTFT/TPOT/throughput, audit trail for compliance. (7) Model versioning with approval workflows. (8) Autoscaling on queue depth with minimum warm replicas.

### Q19: A customer sees 800ms TTFT. Debug it.
**A:** (1) Is it queue time? Check request queue depth — if requests are waiting, TTFT includes wait time. Scale up. (2) Is the prompt long? TTFT ∝ prompt length during prefill. Suggest prompt compression or chunked prefill. (3) GPU utilization — is the GPU compute-saturated during prefill? Check with Nsight or nvidia-smi. (4) Is the model loading from disk? Check if model is cached in GPU memory. (5) Is TRT-LLM engine compiled? First request on TRT-LLM triggers compilation — subsequent requests are faster. (6) Network latency — check time from client to inference endpoint.

### Q20: Tell me about a time you drove technical adoption against resistance.
**A:** [Frame JPMC LLM Suite story]: "When I proposed building LLM Suite at JPMC, there was significant resistance from both the security team (concerned about data leakage) and engineering leadership (doubted we could meet compliance requirements). I drove adoption by: (1) building a proof-of-concept with full audit logging and PII detection, (2) presenting a threat model to the security team that addressed every concern, (3) getting early buy-in from 3 business units who had immediate use cases. Within 6 months, we went from a rejected proposal to 50K+ users and 12 production deployments."

---

## How to Practice
1. Pick 5 random questions per day
2. Set a timer: 2 minutes per answer
3. Speak out loud (not just think)
4. Record yourself if possible — listen back for filler words and gaps
5. For systems design questions (Q18), practice on a whiteboard or paper

---

## Notes
```
(Track questions you struggled with here — revisit them)


```
