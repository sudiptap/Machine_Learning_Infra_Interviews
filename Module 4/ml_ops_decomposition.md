# Module 4: Decomposition & Ambiguity (The "Sherlock" Round)

**Target Role:** Senior/Staff ML Infra, Production Engineering  
**Goal:** You are given a vague, broken, or impossible problem. You must break it down, isolate variables, and propose a plan.  
**Mental Model:** The "Binary Search" for debugging. (Is it Network? No. Is it Disk? No. Is it Compute? Yes.)

---

## Scenario A: The "Silent" Performance Regression
**The Prompt:** *"The training job for Model X is 30% slower today than it was last week. No code was changed. Fix it."*

### **The Investigation Strategy**
**Phase 1: Isolate the Layer**
1.  **Check the Environment (The "Noise"):**
    * Are we on the same hardware? (Did we silently get downgraded from A100 to A10G?)
    * Check "Steal Time" (CPU) or network bandwidth saturation. Are "noisy neighbors" on the cloud sharing our switch?
2.  **Check the Data (The "Bloat"):**
    * Did the dataset change? Check average file size or sequence length.
    * *Physics:* If average sequence length doubles ($N \to 2N$), Attention compute quadruples ($O(N^2)$).
3.  **Check the Storage (The "IO"):**
    * Check `iowait`. Are GPUs at 0% utilization ("starved") waiting for disk? S3 throughput limits often trigger silently.

**Phase 2: The Profiler**
* If basic metrics pass, run the PyTorch Profiler.
* Look for **"CUDA Kernel Gaps"** (white space on the timeline). This means the GPU is idle. Find out what the CPU is doing during those gaps.

---

## Scenario B: The Reliability Nightmare
**The Prompt:** *"We are training on 10,000 GPUs. A GPU fails every 4 hours. The restart takes 20 minutes. We are making zero progress."*

### **The Mitigation Plan**
**1. The "Band-Aid" (Immediate Fix)**
* **Redundant Workers:** Keep ~50 nodes warm/idle. If a rank fails, re-map the rank to a spare immediately. Do not wait for a full node reboot.

**2. The "Code Change" (Optimization)**
* **In-Memory Checkpointing:** Writing 140GB to disk takes time. Instead, replicate the model weights to a neighbor node's CPU RAM. If a node dies, fetch weights from the neighbor (Network is faster than Disk).

**3. The "Architecture Shift" (Long Term)**
* **Elastic Training (TorchElastic):**
    * Current State: Job needs exactly 10,000 GPUs. If 1 dies, job dies.
    * New State: Job can run on $N$ GPUs. If 1 dies, the job pauses briefly, resizes to 9,999 GPUs, and continues.

---

## Scenario C: The "Buy vs. Build" (The CTO Round)
**The Prompt:** *"We spend $5M/year on Databricks. Should we build our own Kubernetes ML Platform on raw EC2 instances to save money?"*

### **The Decomposition Framework**
**1. Calculate the Hard Costs (COGS)**
* Raw EC2/GCP instances are usually 20-30% cheaper than managed services.
* *Potential Savings:* $1M - $1.5M/year.

**2. Calculate the Hidden Costs (Engineering)**
* To build a reliable scheduler, feature store, and notebook environment, we need a platform team.
* *Headcount:* 3 Senior Engineers @ $500k/yr = $1.5M/year.
* *Net Result:* $1.5M savings - $1.5M salary = **$0 profit.**

**3. The "Opportunity Cost" (Risk)**
* "What is the cost of downtime?"
* If Databricks goes down, they fix it. If *our* K8s cluster breaks, our Data Scientists stop working for 2 days while we debug `etcd`.
* *Verdict:* **Do not build.** (Unless the scale is >$20M/year or the workload is so custom that Databricks cannot support it).

---

## How to Practice Module 4
1.  **The "5 Whys" Game:** Take a vague statement like "My model is hallucinating" or "The dashboard is slow" and ask "Why?" 5 times until you hit a hardware or infrastructure root cause.
2.  **Learn Linux Tools:** You should know what these do: `htop`, `nvtop`, `iostat`, `strace`, `tcpdump`.