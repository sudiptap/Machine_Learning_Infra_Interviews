# PagedAttention and vLLM

**Priority:** Read in Week 2–3.
**Interview map:** Track A Round 2, Track B Rounds 2 & 3
**Paper:** Efficient Memory Management for Large Language Model Serving with PagedAttention (arxiv 2309.06180)

---

## 1. The Problem PagedAttention Solves

### Memory Fragmentation in KV Cache

Without PagedAttention, each request gets a **contiguous block** of GPU memory for its KV cache. The problem:

- You don't know how long the output will be when the request arrives
- So you either:
  - **Over-allocate:** Reserve max_seq_len worth of KV cache → massive waste (60-80% of memory unused)
  - **Under-allocate:** Run out mid-generation → request fails or needs expensive reallocation

```
Traditional KV Cache Allocation:

GPU Memory:
[Request A: 2048 reserved, 500 used][Request B: 2048 reserved, 100 used][  wasted  ]
                                                                         ▲
                                                          Could fit 3 more requests here
                                                          but memory is fragmented
```

### The Numbers
The vLLM paper showed that existing systems waste **60-80% of KV cache memory** due to:
- **Internal fragmentation:** Reserved but unused space within a request
- **External fragmentation:** Free memory exists but in non-contiguous chunks
- **Reservation waste:** Pre-allocating for max possible output length

---

## 2. PagedAttention — The Core Idea

Inspired by **virtual memory and paging** in operating systems.

Instead of allocating contiguous memory for each request's KV cache:
- Divide KV cache memory into fixed-size **blocks** (pages)
- Each block holds KV cache for a fixed number of tokens (e.g., 16 tokens)
- A request's KV cache can be stored in **non-contiguous blocks**
- A **block table** (like a page table in OS) maps logical token positions to physical memory blocks
- Allocate blocks on-demand as the sequence grows

```
PagedAttention:

Physical GPU Memory Blocks:
[Block 0: Req A tokens 0-15]
[Block 1: Req B tokens 0-15]
[Block 2: Req A tokens 16-31]  ← non-contiguous! that's fine
[Block 3: Req C tokens 0-15]
[Block 4: Req B tokens 16-31]
[Block 5: FREE]
[Block 6: FREE]

Block Table for Request A:
Logical Block 0 → Physical Block 0
Logical Block 1 → Physical Block 2

Block Table for Request B:
Logical Block 0 → Physical Block 1
Logical Block 1 → Physical Block 4
```

### Benefits
1. **Near-zero waste:** Only allocate blocks as tokens are generated
2. **No external fragmentation:** Any free block can be used by any request
3. **Memory sharing:** Multiple requests with the same prefix can share KV cache blocks (copy-on-write)
4. **Flexible eviction:** Evict blocks (not entire requests) when memory is tight

### Results from the Paper
- **2-4x throughput improvement** over HuggingFace Transformers and FasterTransformer
- **Near-optimal memory utilization** (<4% waste)

---

## 3. vLLM Architecture

vLLM is the open-source inference engine built around PagedAttention.

### Key Components

```
┌──────────────────────────────────────────┐
│              API Server                   │
│         (OpenAI-compatible)              │
└────────────────┬─────────────────────────┘
                 │
┌────────────────▼─────────────────────────┐
│              LLM Engine                   │
│  ├─ Scheduler                            │
│  ├─ Block Manager (PagedAttention)       │
│  └─ Model Runner                         │
└────────────────┬─────────────────────────┘
                 │
┌────────────────▼─────────────────────────┐
│              Workers                      │
│  ├─ GPU 0 (model shard if TP)            │
│  ├─ GPU 1 (model shard if TP)            │
│  └─ ...                                  │
└──────────────────────────────────────────┘
```

### Scheduler (`vllm/core/scheduler.py`)
The scheduler is the brain. Each step it decides:
1. **Which waiting requests to admit** (enough memory for their KV cache?)
2. **Which running requests to continue** (still generating?)
3. **Which requests to preempt** (memory pressure → pause and swap to CPU)

Scheduling policies:
- **FCFS (First Come First Served):** default, simple
- **Priority-based:** higher priority requests get memory first

### Block Manager (`vllm/core/block_manager.py`)
Manages the physical memory blocks:
- Allocates blocks when a request starts or generates more tokens
- Frees blocks when a request finishes
- Handles **copy-on-write** for shared prefixes
- Tracks free blocks, used blocks, reference counts

### Continuous Batching
vLLM implements continuous batching (from Orca paper):
- After each decode step, check for:
  - Completed requests → free their blocks, remove from batch
  - New waiting requests → allocate blocks, add to batch
- Every decode iteration runs with a potentially different batch composition

---

## 4. Prefix Caching and SGLang's RadixAttention

### Prefix Caching (vLLM)
If multiple requests share the same prefix (e.g., same system prompt), they can share the same KV cache blocks. Copy-on-write: only duplicate when a request needs to modify a shared block.

### RadixAttention (SGLang)
SGLang takes prefix caching further with a **radix tree** (prefix tree) for KV cache:
- All requests' KV caches are stored in a global radix tree
- Common prefixes are automatically shared
- When a new request arrives, the engine matches its prefix against the tree
- Matched prefix → reuse cached KV, only compute the new part

**Why this matters:** In production, many requests share system prompts, few-shot examples, or RAG context. RadixAttention can eliminate 50-90% of prefill computation.

You already studied this in SGLang. Connect it to the broader concept.

---

## 5. Preemption Strategies

When GPU memory is full and a new high-priority request arrives:

### Swapping
- Move a low-priority request's KV cache from GPU to CPU memory
- Resume later by swapping back
- Slow (PCIe bandwidth ~32 GB/s vs HBM ~2 TB/s)

### Recomputation
- Discard a low-priority request's KV cache entirely
- When resuming, rerun prefill to regenerate the KV cache
- Better when KV cache is large and prefix is short

### Which to choose?
- Short sequences with large KV cache → recomputation
- Long sequences where prefill is expensive → swapping
- vLLM implements both; scheduler decides based on heuristics

---

## 6. How This Maps to the Interview

### Track A (SA) questions:
- "How does vLLM manage memory differently from a naive approach?"
- "A customer is running out of GPU memory with 100 concurrent users. What do you recommend?"
- "What's the advantage of NIM's memory management over hosting raw HuggingFace?"

### Track B (Eng) questions:
- "Explain PagedAttention. How does the block table work?"
- "Walk me through vLLM's scheduler — how does it decide what to run next?"
- "How does prefix caching work and when does it help?"
- "Compare SGLang's RadixAttention to vLLM's prefix caching."

---

## Self-Test Questions

1. What problem does PagedAttention solve? What was wasted before?
2. How does a block table map logical tokens to physical memory?
3. What is copy-on-write in the context of KV cache sharing?
4. Explain vLLM's scheduler loop: what happens at each decode step?
5. What is the difference between swapping and recomputation for preemption?
6. How does RadixAttention (SGLang) differ from vLLM's prefix caching?
7. A customer has 100 concurrent requests, each with 4096 context on an A100 80GB serving Llama-3 8B. Will it fit? Show your math.

---

## Notes
```
(Take your own notes here)


```
