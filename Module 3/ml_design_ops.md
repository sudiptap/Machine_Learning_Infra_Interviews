# Module 3: ML System Design Patterns (The Blueprints)

**Target Role:** ML Infra, Platform Engineer, AI Systems  
**Goal:** Move beyond "boxes and arrows." Design systems that respect hardware limits (Bandwidth, VRAM, Latency) and data consistency.  
**Strategy:** For every component you draw, ask: "What happens if this fails?" and "Is this the bottleneck?"

---

## Pattern 1: Large Scale Distributed Training (The "Llama" Stack)
**The Prompt:** *"Design a system to train a 70B parameter model on 2TB of text data."*

### **1. Data Loading Pipeline (The "Stall" Check)**
* **The Constraint:** GPUs compute faster than disks can read. If the GPU waits for data, you are burning money.
* **The Design:**
    * **Storage:** Object Store (S3/GCS) holding sharded tar files (e.g., `WebDataset`).
    * **Network:** Dedicated network interface (NIC) for data vs. inter-node communication.
    * **Prefetching:** A dedicated CPU thread pool that downloads and decompresses the *next* batch while the GPU processes the *current* batch.
    * **Shuffling:** Don't global shuffle. Shuffle the list of shards, then use a small in-memory shuffle buffer (e.g., 10k samples) on each node.

### **2. Checkpointing (The "Time-Loss" Check)**
* **The Constraint:** Saving a 70B model (140GB) to S3 takes time. If you block training to save, efficiency drops.
* **The Design:**
    * **Asynchronous Checkpointing:** Copy the model state to CPU RAM (fast), then let a background process upload from RAM to S3 (slow) while the GPU resumes training immediately.

### **3. Fault Tolerance (The "Straggler" Check)**
* **The Constraint:** In a 1000-GPU cluster, failures are guaranteed. Standard `AllReduce` halts if one node dies.
* **The Design:**
    * **Warm Spares:** Keep a few nodes idle and provisioned.
    * **Heartbeat Monitor:** If a node misses a heartbeat, the orchestrator (K8s/Slurm) kills the job, blacklists the bad node, swaps in a spare, and restarts from the last checkpoint.

---

## Pattern 2: High-Throughput Inference (The "ChatGPT" Stack)
**The Prompt:** *"Design a serving layer for an LLM. We need low latency for the first token, but high throughput for the rest."*

### **1. The Decoupling (Prefill vs. Decode)**
* **The Constraint:**
    * **Prefill (Processing the user prompt):** Compute-bound.
    * **Decode (Generating tokens):** Memory-bandwidth bound.
* **The Design:**
    * **Chunked Prefill:** Break long prompts into chunks so they can be processed alongside decoding steps (prevents the "convoy effect" where one long prompt blocks everyone).

### **2. The KV-Cache Manager**
* **The Constraint:** Storing the history (KV-Cache) for every active user consumes massive VRAM.
* **The Design:**
    * **PagedAttention (vLLM):** Allocate memory in non-contiguous blocks (pages) just like an OS manages RAM. Eliminates memory fragmentation.
    * **Offloading:** If a user is inactive for 10 seconds, move their KV-cache to CPU RAM. Move it back to GPU when they send the next message.

### **3. Autoscaling Metrics**
* **The Trap:** Scaling on CPU usage is useless for LLMs.
* **The Design:** Scale based on **"Request Queue Depth"** (how many users are waiting) or **"KV-Cache Utilization"** (how full is the GPU memory).

---

## Pattern 3: The Feature Store (The "Data Leakage" Stack)
**The Prompt:** *"Design a fraud detection system. It needs real-time features like 'Number of transactions in last 10 minutes'."*

### **1. The Architecture (Lambda Architecture)**
* **Offline Store (The Warehouse):** S3/Parquet. Optimized for scanning huge datasets for training.
* **Online Store (The Cache):** Redis/DynamoDB/Cassandra. Optimized for <5ms key-value lookups during inference.

### **2. The Sync Mechanism**
* **The Constraint:** Code definitions for features must be identical in training and serving.
* **The Design:**
    * Write feature logic once (e.g., in Python or SQL).
    * **Materialization Engine:** A stream processor (Flink) calculates the feature in real-time and pushes it to Redis. A batch job (Spark) calculates the history and pushes it to S3.

### **3. Point-in-Time Correctness**
* **The Constraint:** "Time Travel." When training on past data, you must not use "future" knowledge.
* **The Design:**
    * Store features as a **Change Log** (Event Sourcing).
    * When creating training data, perform an **"As-Of Join"**: `Select value from Features where Timestamp <= EventTime ORDER BY Timestamp DESC LIMIT 1`.

---

## How to Practice Module 3
1.  **Draw it out:** You must be able to draw these architectures on a whiteboard/Excalidraw. Label the data flow arrows.
2.  **Memorize the Bottlenecks:**
    * Training -> Network Bandwidth.
    * Inference -> Memory Bandwidth.
    * Feature Store -> Data Consistency (Drift).