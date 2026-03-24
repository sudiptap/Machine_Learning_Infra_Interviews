# NCA-AIIO Certification Study Guide

**Priority:** Week 2-3 (exam by Week 3-4).
**Interview map:** Track A resume keyword, credential signal

---

## 1. What is NCA-AIIO?

**NVIDIA Certified Associate — AI Infrastructure and Operations**

- Entry-level NVIDIA certification
- Validates understanding of AI infrastructure concepts
- ATS keyword match for NVIDIA SA roles
- Fast to get with your background (~1-2 weeks of study)

---

## 2. Exam Details

| | Details |
|---|---|
| Format | Multiple choice, ~50 questions |
| Duration | 60-90 minutes |
| Passing score | ~70% |
| Cost | ~$135 |
| Proctored | Yes (online or test center) |
| Validity | 2 years |
| Prep resources | NVIDIA DLI courses, NVIDIA documentation |

---

## 3. Exam Domains

### Domain 1: AI/ML Fundamentals (~15%)
- Types of ML: supervised, unsupervised, reinforcement learning
- Neural network basics: layers, activation functions, backpropagation
- Training concepts: loss functions, optimizers, overfitting, regularization
- Common architectures: CNNs, RNNs, Transformers

**You already know this.** Quick review only.

### Domain 2: GPU Computing Fundamentals (~20%)
- CPU vs GPU architecture
- CUDA programming model (threads, blocks, grids)
- GPU memory hierarchy
- Tensor Cores and their role
- Parallel computing concepts

**Covered in your reading:** 01_gpu_architecture.md + 10_cuda_programming_concepts.md

### Domain 3: AI Infrastructure (~25%)
- DGX systems and their components
- GPU cluster architecture
- NVLink, NVSwitch, InfiniBand
- GPU memory (HBM)
- Networking for AI workloads
- Storage for AI (GPU Direct Storage)

**Covered in:** 03_nvidia_product_stack.md + 23_networking_gpu_clusters.md

### Domain 4: AI Software Stack (~20%)
- NVIDIA AI Enterprise
- TensorRT / TensorRT-LLM
- Triton Inference Server
- CUDA toolkit
- Container technologies (NGC, Docker)
- Kubernetes for AI

**Covered in:** 03_nvidia_product_stack.md + 21_inference_engine_comparison.md

### Domain 5: AI Operations (~20%)
- Model deployment lifecycle
- Monitoring and observability
- Multi-tenancy
- Performance optimization
- Security and compliance
- Scaling strategies

**Covered in:** 15_cost_optimization.md + 16_systems_design_patterns.md

---

## 4. Key Facts to Memorize

### GPU Hardware Specs
```
A100:
- 80 GB HBM2e, 2 TB/s bandwidth
- 312 TFLOPS FP16 Tensor Core
- 108 SMs, NVLink 3.0 (600 GB/s)
- PCIe and SXM form factors

H100:
- 80 GB HBM3, 3.35 TB/s bandwidth
- 989 TFLOPS FP16, 1979 TFLOPS FP8
- 132 SMs, NVLink 4.0 (900 GB/s)
- 4th gen Tensor Cores

DGX A100: 8x A100, 5 petaFLOPS, NVSwitch
DGX H100: 8x H100, 32 petaFLOPS, NVSwitch
```

### Software Stack
```
NVIDIA AI Enterprise: Software platform license for production AI
NGC: NVIDIA GPU Cloud — container registry for AI containers
NIM: Prepackaged inference microservices
TensorRT-LLM: LLM optimization and serving engine
Triton Inference Server: Multi-model serving platform
CUDA Toolkit: Compiler + runtime + libraries
cuDNN: GPU-accelerated deep learning primitives
NCCL: Multi-GPU communication library
```

### Networking
```
NVLink: GPU-to-GPU within node (600-900 GB/s)
NVSwitch: Connects all GPUs in DGX at full NVLink bandwidth
InfiniBand: Between nodes (200-400 Gb/s)
GPUDirect RDMA: Direct GPU-to-GPU across network (bypasses CPU)
GPUDirect Storage: SSD-to-GPU direct transfer
```

### Kubernetes for AI
```
NVIDIA GPU Operator: Automates GPU driver and toolkit deployment on K8s
NVIDIA Device Plugin: Exposes GPUs as schedulable resources
MIG (Multi-Instance GPU): Partition A100/H100 into up to 7 isolated instances
Time-slicing: Share GPU among pods via time division
```

---

## 5. NVIDIA DLI Courses (Free/Low Cost)

These NVIDIA Deep Learning Institute courses map to exam domains:

1. **Fundamentals of Deep Learning** — Domain 1
2. **Getting Started with AI on Jetson Nano** — Domain 2 (GPU basics)
3. **Building Intelligent Recommendations** — practical ML
4. **Accelerating CUDA C++ Applications** — Domain 2

Check: https://www.nvidia.com/en-us/training/

---

## 6. Study Plan (1-2 Weeks)

### Week 1: Review + Practice
| Day | Focus | Hours |
|---|---|---|
| Day 1 | Review reading materials 01, 03, 10 (GPU, NVIDIA stack, CUDA) | 2 |
| Day 2 | Review reading materials 23, 15 (networking, cost/operations) | 2 |
| Day 3 | Memorize key specs (GPU, networking, software stack) | 1.5 |
| Day 4 | NVIDIA DLI: Fundamentals of Deep Learning (if needed) | 2 |
| Day 5 | Practice questions (NVIDIA sample questions) | 2 |

### Week 2: Practice + Exam
| Day | Focus | Hours |
|---|---|---|
| Day 1 | Review weak areas from practice questions | 2 |
| Day 2 | Final review of all 5 domains | 1.5 |
| Day 3 | **Take the exam** | 1.5 |

---

## 7. Tips for Exam Day

1. **Read questions carefully** — NVIDIA exams often have "choose the BEST answer" (multiple could be partially correct)
2. **DGX questions:** Know the exact GPU counts and interconnects
3. **When in doubt:** Think "what would NVIDIA recommend?" — the answer often favors NVIDIA products
4. **Time management:** ~1.5 min per question, flag and revisit uncertain ones
5. **MIG questions:** Know that MIG partitions an A100/H100 into isolated GPU instances (up to 7)

---

## 8. After NCA-AIIO: Next Certifications

| Cert | When | Value |
|---|---|---|
| NCA-AIIO | Week 3-4 | Foundation, ATS keyword |
| NVIDIA LLM Certification | Week 7-8 | Direct relevance to both tracks |
| NCP-AII (Professional) | Month 4+ | Shows operational depth |
| AWS SA Pro | Optional | Validates cloud infra credibility |

---

## Self-Test: Practice Questions

1. What interconnect technology connects GPUs within a DGX H100 system?
   a) PCIe 5.0  b) InfiniBand NDR  c) NVLink 4.0 + NVSwitch  d) Ethernet

2. Which NVIDIA product provides prepackaged, optimized containers for serving AI models?
   a) Triton Inference Server  b) TensorRT-LLM  c) NIM  d) CUDA Toolkit

3. What is MIG (Multi-Instance GPU)?
   a) Running multiple models on one GPU via time-sharing
   b) Partitioning a GPU into isolated instances with dedicated compute and memory
   c) Distributing one model across multiple GPUs
   d) A GPU virtualization technology that requires a hypervisor

4. Which memory type provides the highest bandwidth in a GPU?
   a) CPU DDR5  b) GPU shared memory (SRAM)  c) GPU HBM  d) L3 cache

5. NCCL is primarily used for:
   a) GPU memory management  b) Model compilation  c) Multi-GPU communication  d) Container orchestration

**Answers:** 1-c, 2-c, 3-b, 4-b (SRAM is faster but smaller; HBM is the highest bandwidth *bulk* memory), 5-c

Note: Q4 is tricky — SRAM (shared memory) has higher bandwidth but is tiny. Exam likely means "main GPU memory" = HBM. Read questions carefully.

---

## Notes
```


```
