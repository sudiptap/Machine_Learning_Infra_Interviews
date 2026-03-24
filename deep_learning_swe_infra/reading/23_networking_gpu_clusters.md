# Networking for GPU Clusters — NVLink, InfiniBand, GPUDirect

**Priority:** Week 6-8.
**Interview map:** Track A Round 4, Track B Round 2

---

## 1. Why Networking Matters for Inference

Distributed inference requires GPUs to communicate:
- **Tensor parallelism:** All-reduce every layer (microseconds matter)
- **Pipeline parallelism:** Send activations between stages
- **Expert parallelism:** All-to-all routing for MoE models
- **KV cache transfer:** For disaggregated prefill/decode

The network is often the bottleneck, not the GPU.

---

## 2. NVLink — GPU-to-GPU Within a Node

### What it is
Direct high-bandwidth link between GPUs, bypassing CPU and PCIe.

### Generations
| Version | Per-GPU BW (bidirectional) | Found in |
|---|---|---|
| NVLink 3.0 | 600 GB/s | A100 (DGX A100) |
| NVLink 4.0 | 900 GB/s | H100 (DGX H100) |
| NVLink 5.0 | 1,800 GB/s | B200 (DGX B200) |

### NVSwitch
- Connects ALL GPUs in a node to each other at full NVLink bandwidth
- DGX A100: NVSwitch connects 8 GPUs with 600 GB/s each
- DGX H100: 4th gen NVSwitch, 900 GB/s per GPU
- **Full bisection bandwidth:** Any GPU can talk to any other at full speed simultaneously

### Why NVLink matters for inference
```
Tensor parallelism all-reduce for Llama-3 70B:
  Data per all-reduce: batch × seq × hidden × 2 bytes ≈ 1 × 1 × 8192 × 2 = 16 KB (decode)
  But 80 layers × 2 all-reduces = 160 all-reduces per token
  Latency per all-reduce: ~5μs on NVLink
  Total: ~800μs per token for TP communication

  Same over PCIe (64 GB/s): ~50μs per all-reduce → ~8ms per token
  → NVLink is 10x faster for TP communication
```

---

## 3. PCIe — The Slower Alternative

| Version | Bandwidth (per lane × 16) | Latency |
|---|---|---|
| PCIe 4.0 | ~32 GB/s | ~1-2 μs |
| PCIe 5.0 | ~64 GB/s | ~1-2 μs |

### When PCIe is used
- Connecting GPUs to CPU (for KV cache swapping)
- Connecting NIC to GPU (for inter-node communication)
- Budget GPU servers without NVLink (e.g., 2x or 4x A100 consumer setups)

### PCIe limitations for inference
- Too slow for tensor parallelism (10x slower than NVLink)
- Acceptable for pipeline parallelism (less frequent communication)
- KV cache swap to CPU is limited by PCIe bandwidth

---

## 4. InfiniBand — Between Nodes

### What it is
High-performance networking fabric designed for HPC and AI clusters.

### Generations
| Version | Per-port bandwidth | Typical use |
|---|---|---|
| HDR | 200 Gb/s (25 GB/s) | DGX A100 clusters |
| NDR | 400 Gb/s (50 GB/s) | DGX H100 clusters |
| XDR | 800 Gb/s (100 GB/s) | Next gen (DGX B200) |

### RDMA (Remote Direct Memory Access)
- GPU on node A writes directly to GPU memory on node B
- **Bypasses CPU entirely** — no OS involvement, no copies
- Lowest possible inter-node latency (~1-2 μs)
- This is what makes multi-node inference practical

### InfiniBand topology
```
Typical DGX SuperPOD:

        ┌──── Spine InfiniBand Switch ────┐
        │                                  │
   ┌────┴────┐                       ┌────┴────┐
   │ Leaf SW  │                       │ Leaf SW  │
   ├────┬────┤                       ├────┬────┤
   │DGX │DGX │                       │DGX │DGX │
   │ 0  │ 1  │                       │ 2  │ 3  │
   └────┴────┘                       └────┴────┘

Fat-tree topology: full bisection bandwidth
Any node pair gets full InfiniBand bandwidth simultaneously
```

---

## 5. GPUDirect Technologies

### GPUDirect RDMA
GPU memory on one node is directly accessible from another node's GPU:
```
Normal path: GPU 0 → PCIe → CPU 0 → Network → CPU 1 → PCIe → GPU 1
GPUDirect:   GPU 0 → NIC → Network → NIC → GPU 1
             (bypasses CPU entirely)
```
**Impact:** ~50% lower latency, ~40% higher bandwidth for inter-node GPU communication.

### GPUDirect Storage
Read directly from NVMe SSD to GPU memory, bypassing CPU:
```
Normal: SSD → CPU memory → PCIe → GPU memory
GPUDirect Storage: SSD → GPU memory (via PCIe, CPU bypassed)
```
**Use case:** Loading model weights from disk. Can reduce model load time significantly.

### GPUDirect Peer-to-Peer (P2P)
GPU-to-GPU communication within a node via PCIe (without NVLink):
```
GPU 0 → PCIe switch → GPU 1
```
Slower than NVLink but still faster than going through CPU memory.

---

## 6. NCCL — The Communication Library

### What it is
NVIDIA Collective Communication Library — implements efficient multi-GPU communication patterns using NVLink, InfiniBand, and PCIe.

### Key collectives for inference
```
All-Reduce:      Used by tensor parallelism (sum partial results)
All-Gather:      Gather tensor shards from all GPUs to all GPUs
Reduce-Scatter:  Reduce and distribute (used in sequence parallelism)
All-to-All:      Used by expert parallelism (MoE routing)
Send/Recv:       Used by pipeline parallelism (point-to-point)
```

### Ring vs Tree All-Reduce
```
Ring All-Reduce:
  GPUs form a ring. Data flows around ring in 2 phases:
  Phase 1 (reduce-scatter): each GPU sends chunk to next, accumulates
  Phase 2 (all-gather): each GPU sends final chunk to next
  Time: 2 × (N-1)/N × data_size / bandwidth
  Good for large messages.

Tree All-Reduce:
  GPUs form a binary tree. Reduce up to root, broadcast down.
  Time: 2 × log(N) × data_size / bandwidth
  Better for small messages (lower latency).
```

NCCL automatically selects the best algorithm based on message size and topology.

---

## 7. Network Considerations for Inference Architectures

### Tensor Parallelism
```
Communication pattern: All-reduce (frequent, small messages during decode)
Required bandwidth: NVLink (>600 GB/s)
Latency sensitivity: Very high (per-layer, microseconds matter)
Recommendation: ONLY within a single NVLink-connected node
```

### Pipeline Parallelism
```
Communication pattern: Point-to-point (between adjacent stages)
Required bandwidth: InfiniBand sufficient (~50 GB/s)
Latency sensitivity: Moderate (per-microbatch, milliseconds OK)
Recommendation: Across nodes
```

### Expert Parallelism (MoE)
```
Communication pattern: All-to-all (any GPU to any GPU)
Required bandwidth: High (many small messages)
Latency sensitivity: High (per-token routing)
Recommendation: Within node (NVLink) if possible, InfiniBand if needed
```

### Disaggregated Prefill-Decode
```
Communication pattern: Point-to-point (KV cache transfer)
Data volume: Large (KV cache can be hundreds of MB per request)
Latency sensitivity: Moderate (once per request, not per token)
Recommendation: InfiniBand with GPUDirect RDMA
```

---

## 8. Ethernet vs InfiniBand

| | InfiniBand NDR | 100GbE (RoCE) | 400GbE |
|---|---|---|---|
| Bandwidth | 400 Gb/s | 100 Gb/s | 400 Gb/s |
| Latency | ~1 μs | ~5-10 μs | ~2-5 μs |
| RDMA | Native | RoCE (needs config) | RoCE v2 |
| Cost | High ($$$) | Lower | Medium |
| Ecosystem | HPC/AI dominant | Datacenter standard | Growing |

### When to use which
- **InfiniBand:** Multi-node training, multi-node inference with TP, performance-critical
- **Ethernet (RoCE):** Budget-conscious, inference-only (less latency sensitive), existing datacenter infra
- **Cloud:** AWS EFA ≈ InfiniBand-class; GCP uses custom interconnect

---

## 9. Interview Connections

> "What is the difference between NVLink and InfiniBand?"

**A:** "NVLink is GPU-to-GPU interconnect within a single node — 900 GB/s on H100, used for tensor parallelism where per-layer all-reduce requires microsecond latency. InfiniBand is the network between nodes — 50 GB/s for NDR, used for pipeline parallelism and data parallelism where communication is less frequent. The rule: TP within a node (NVLink), PP across nodes (InfiniBand)."

> "A customer is building a 64-GPU cluster for inference. What networking do you recommend?"

**A:** "8 nodes of 8 GPUs (DGX H100). Within each node: NVSwitch + NVLink 4.0 (900 GB/s) for tensor parallelism. Between nodes: NDR InfiniBand (400 Gb/s) in a fat-tree topology for pipeline parallelism and model loading. Enable GPUDirect RDMA for lowest inter-node latency. Total: ~$2-3M for the InfiniBand fabric alone, but required for multi-node inference at low latency."

---

## Self-Test Questions

1. What is NVLink bandwidth on H100? Compare to PCIe 5.0.
2. Why can't you do tensor parallelism over InfiniBand?
3. What is RDMA and why does it matter for multi-node inference?
4. What is GPUDirect RDMA? Draw the data path vs normal path.
5. When would you use Ethernet (RoCE) instead of InfiniBand?
6. What is NVSwitch and why does DGX use it?
7. Calculate all-reduce time for TP=8 on NVLink 4.0 for a 16 KB message.

---

## Notes
```


```
