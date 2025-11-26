# Module 2: ML Internals & Architecture Interview Practice

**Target Role:** ML Infra, Model Performance Engineer, AI Systems  
**Goal:** Demonstrate deep intuition for Hardware, Memory, and Communication bottlenecks.  
**Rule of Thumb:** Always connect the "Algorithm" to the "Hardware Constraint" (e.g., Memory Bandwidth, Cache Size, Network Latency).

---

## Topic 1: The "Memory Wall" & Training (ZeRO, Sharding)
*The bottleneck:* A 175B parameter model takes ~700GB VRAM. An A100 GPU has 80GB. How do we fit it?

### **Core Concepts to Master**
* **Data Parallel (DP) vs. Model Parallel (MP):** When to use which?
* **ZeRO Stages 1, 2, 3:** What gets sharded? (Optimizer State, Gradients, Parameters).
* **Mixed Precision (FP16/BF16):** Why does it save memory? (Hint: It's not just storage, it's bandwidth).

### **The Interview Drill**
1.  **The "OOM" Debugger:**
    * *Q:* "I am training Llama-3-8B on a single GPU. It runs fine for 10 steps, then crashes with Out Of Memory (OOM). Why?"
    * *Checklist:* Memory fragmentation? Is `grad_accumulation` storing too much graph history? Memory leak in the DataLoader?

2.  **The ZeRO Deep Dive:**
    * *Q:* "Explain the trade-off of ZeRO-3 (sharding everything). Why is it slower than ZeRO-1?"
    * *A:* Communication overhead. In ZeRO-3, every time you need a weight for computation, you must fetch it from another GPU across the network (NVLink/InfiniBand).

3.  **Gradient Checkpointing (Activation Recomputation):**
    * *Q:* "We are out of memory. I enable 'Gradient Checkpointing' and it fits, but training is 30% slower. Why?"
    * *A:* You deleted the intermediate activations to save RAM. Now you have to *re-calculate* them during the backward pass. CPU/GPU compute is traded for Memory.

---

## Topic 2: Efficient Attention (FlashAttention, PagedAttention)
*The bottleneck:* Standard Attention reads/writes huge $N \times N$ matrices to HBM (High Bandwidth Memory), which is slow.

### **Core Concepts to Master**
* **HBM vs. SRAM:** The GPU has "Slow/Big" memory (HBM) and "Fast/Tiny" memory (SRAM).
* **IO-Awareness:** The goal is to load data into SRAM once, do all math, and write back.
* **KV-Cache:** Why inference gets slower as the chat gets longer.

### **The Interview Drill**
4.  **The "Flash" Explanation:**
    * *Q:* "Explain FlashAttention to a Junior Engineer. Don't use math. Use hardware terms."
    * *A:* Standard attention writes the huge attention matrix to HBM (Main Memory) then reads it back for Softmax. FlashAttention calculates Softmax in "tiles" (chunks) directly on the SRAM (Chip Cache) without ever writing the full matrix to HBM. It saves I/O, not FLOPs.

5.  **The PagedAttention (vLLM) Logic:**
    * *Q:* "In standard serving, why is memory fragmentation a problem for the KV-Cache?"
    * *A:* Requests have unknown lengths. We used to reserve contiguous blocks of memory (like an array). PagedAttention allows non-contiguous blocks (like a Linked List or OS Virtual Memory), eliminating waste ("internal fragmentation").

---

## Topic 3: Distributed Communication (AllReduce, Ring)
*The bottleneck:* 1000 GPUs need to agree on the gradients. The network is the slowest part of the cluster.

### **Core Concepts to Master**
* **Collective Primitives:** Broadcast, Scatter, Gather, All-Gather, All-Reduce.
* **Ring All-Reduce:** How to sync data without a central bottleneck.
* **Parameter Server vs. All-Reduce:** Centralized vs. Decentralized.

### **The Interview Drill**
6.  **The Bandwidth Calc:**
    * *Q:* "We are doing All-Reduce on a 7B parameter model. We use FP16 (2 bytes). How much data does *each* GPU send?"
    * *A:* 7B params * 2 bytes = 14GB. In Ring All-Reduce, each node sends roughly $2 \times ModelSize$. So ~28GB per step.

7.  **The Topology Debug:**
    * *Q:* "Our training cluster is slow. `nvidia-smi` shows GPUs fluctuate between 0% and 100% utilization. What is happening?"
    * *A:* Communication Bubble. The GPUs are waiting for the network (All-Reduce) to finish before they can compute the next step.
    * *Follow-up:* How to fix? (Gradient Accumulation to compute longer between syncs, or better overlap of Comm/Compute).

---

## Topic 4: Inference & Quantization
*The bottleneck:* Serving models is expensive. We need to make them smaller and faster.

### **Core Concepts to Master**
* **Post-Training Quantization (PTQ):** FP16 -> INT8.
* **Continuous Batching (Orca/vLLM):** Grouping requests dynamically.
* **Speculative Decoding:** Using a small model to draft, big model to verify.

### **The Interview Drill**
8.  **The INT8 "Why":**
    * *Q:* "Why is INT8 faster than FP16? Is it just the math?"
    * *A:* Two reasons. 1) Math: INT8 tensor cores are faster. 2) **Memory Bandwidth:** You move 50% less data from HBM to the core. For LLMs, bandwidth is usually the bottleneck, so this is the main win.

9.  **Continuous Batching:**
    * *Q:* "Request A takes 5 seconds. Request B takes 1 second. In a standard static batch, how long do we wait?"
    * *A:* 5 seconds. We are limited by the slowest request.
    * *Q:* "How does Continuous Batching fix this?"
    * *A:* As soon as Request B finishes, we eject it and insert Request C *mid-batch*. The GPU never waits.

---

## Topic 5: Hardware "Sanity Checks"
*The bottleneck:* You. Knowing the rough numbers is mandatory.

### **Memorize These Numbers (The "Back of Napkin" Kit)**
10. **Hardware Specs (A100 / H100):**
    * **VRAM:** 80GB.
    * **Bandwidth:** ~2TB/s (HBM2e) on A100.
    * **Interconnect (NVLink):** ~600GB/s.
    * **Network (InfiniBand):** ~200Gbps (Bits, not Bytes! Divide by 8).

11. **Data Sizes:**
    * **FP32:** 4 Bytes.
    * **FP16 / BF16:** 2 Bytes.
    * **INT8:** 1 Byte.
    * **KV Cache Size:** `2 * 2 * n_layers * d_model * seq_len * batch_size`. (The two 2s are for K+V and Bytes).

12. **The "Math" Question:**
    * *Q:* "Can I fit a 70B parameter model on a single 80GB A100 for *inference*?"
    * *Calculation:* 70B * 2 bytes (FP16) = 140GB.
    * *A:* No. You need at least 2 GPUs (Tensor Parallelism) or Quantization to 4-bit (70B * 0.5 bytes = 35GB).

---

## How to Prepare (The "Paper Reading" Strategy)

Do not read papers efficiently. **Read them for the "Architecture Diagram" and the "Evaluation" section.**

* **Week 1:** Read **"ZeRO: Memory Optimizations"**. Draw the diagram of where the Optimizer State lives in ZeRO-1 vs ZeRO-2.
* **Week 2:** Read **"FlashAttention"** (just the Intro and Section 3). Understand the "Tiling" concept.
* **Week 3:** Read **"vLLM: PagedAttention"**. Understand the "Block Table" lookup (similar to OS Virtual Memory).