# NVIDIA Product Stack — Know the Full Map

**Priority:** Read in Week 1–2.
**Interview map:** Track A Rounds 1-4 (you MUST know these products), Track B Round 2

---

## 1. The Full Stack — Top to Bottom

```
┌────────────────────────────────────────────────────────┐
│  NVIDIA AI Enterprise (Software Platform License)      │
│  Enterprise support, security patches, SLAs            │
├────────────────────────────────────────────────────────┤
│  NIM (Inference Microservices)                         │
│  Prepackaged containers, one-click deploy              │
│  OpenAI-compatible API                                 │
├────────────────────────────────────────────────────────┤
│  NeMo (Training + Customization Framework)             │
│  Fine-tuning, RLHF, PEFT, data curation               │
├────────────────────────────────────────────────────────┤
│  Triton Inference Server                               │
│  Model serving, dynamic batching, model pipelines      │
├────────────────────────────────────────────────────────┤
│  TensorRT-LLM                                         │
│  LLM compilation, optimization, quantization           │
├────────────────────────────────────────────────────────┤
│  TensorRT (General)                                    │
│  DNN optimizer and runtime (non-LLM models too)        │
├────────────────────────────────────────────────────────┤
│  cuDNN / cuBLAS / NCCL / CUTLASS                       │
│  Low-level GPU libraries                               │
├────────────────────────────────────────────────────────┤
│  CUDA Toolkit                                          │
│  Compiler (nvcc), runtime, driver                      │
├────────────────────────────────────────────────────────┤
│  GPU Hardware: A100 / H100 / H200 / B100 / B200       │
│  DGX Systems, HGX Boards                               │
│  NVLink (intra-node), InfiniBand (inter-node)          │
└────────────────────────────────────────────────────────┘
```

---

## 2. NIM (NVIDIA Inference Microservices)

### What it is
- Docker containers with a fully optimized model ready to serve
- Pull, run, query — no compilation or optimization steps needed
- Uses TRT-LLM + Triton under the hood (but you don't see them)
- OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`)

### When to recommend NIM
- Customer wants fastest time-to-deploy
- Standard models (Llama, Mistral, etc.) without heavy customization
- Enterprise environments where simplicity and support matter
- When you don't need to customize the serving logic

### When NOT to recommend NIM
- Heavily customized models that NIM doesn't support yet
- Customer needs custom batching logic, custom preprocessing
- Customer wants maximum control over every layer of the stack
- Cost-sensitive customers who don't need enterprise support

### Key details
- Requires NGC API key
- Runs on NVIDIA GPUs (A100, H100, L40S, etc.)
- Supports multi-GPU tensor parallelism automatically
- Includes health checks, metrics endpoints
- Part of NVIDIA AI Enterprise license (paid for production use)

---

## 3. Triton Inference Server

### What it is
- Open-source model serving platform (GitHub: triton-inference-server)
- Supports multiple backends: TensorRT, PyTorch, TensorFlow, ONNX, Python
- Handles: dynamic batching, model ensembles, model pipelines, concurrent model execution

### Key features
- **Dynamic batching:** Automatically combines incoming requests into batches
- **Model repository:** Serve multiple models from a directory structure
- **Ensemble models:** Chain models together (e.g., tokenizer → LLM → detokenizer)
- **Metrics:** Prometheus-compatible `/metrics` endpoint
- **Model management:** Load/unload models dynamically

### When to recommend Triton
- Serving non-LLM models (vision, speech, recommender systems)
- Multi-model serving (several different models on the same GPU)
- Custom model pipelines (pre/post-processing chains)
- When customer needs more control than NIM provides

### Triton vs NIM
| | NIM | Triton |
|---|---|---|
| Setup complexity | Minimal | Medium |
| LLM optimization | Built-in (TRT-LLM) | Manual (configure backend) |
| Model support | Curated list | Anything |
| Customization | Limited | Full |
| Use case | LLM serving | Any model serving |

---

## 4. TensorRT-LLM

### What it is
- Open-source library for optimizing and running LLMs on NVIDIA GPUs
- Compiles a model (e.g., from HuggingFace) into an optimized TensorRT engine
- Written in C++ with Python API

### What it does
- **Kernel fusion:** Combines multiple operations into single GPU kernels (fewer HBM reads)
- **Quantization:** INT8, FP8, INT4 (AWQ, GPTQ) weight quantization
- **KV cache management:** Paged KV cache (like PagedAttention)
- **In-flight batching:** Continuous batching (add/remove requests each iteration)
- **Tensor parallelism:** Split model across multiple GPUs
- **Pipeline parallelism:** Split model layers across nodes
- **Custom attention kernels:** Optimized MHA, GQA, MQA implementations

### TRT-LLM vs vLLM
| | TRT-LLM | vLLM |
|---|---|---|
| Language | C++ core, Python wrapper | Python (with C++ extensions) |
| Optimization | Compile-time (builds engine) | Runtime |
| Setup | Heavier (build step) | Lighter (pip install) |
| Performance | Generally faster at high scale | Competitive, easier to customize |
| Community | NVIDIA-maintained | Open-source community |
| Flexibility | Less (compiled engine) | More (Python-native) |

### When to recommend TRT-LLM
- Maximum performance on NVIDIA GPUs is the priority
- Production deployment where compile time is acceptable
- Customer is already in the NVIDIA ecosystem
- Large-scale deployment where 10% performance gain = real cost savings

---

## 5. NeMo Framework

### What it is
- End-to-end framework for training, customizing, and deploying LLMs
- Covers: pretraining, fine-tuning, RLHF, PEFT (LoRA), data curation

### Key components
- **NeMo Curator:** Data curation and preprocessing pipelines
- **NeMo Customizer:** Fine-tuning (full, LoRA, P-tuning)
- **NeMo Guardrails:** Safety and control for LLM applications
- **NeMo Evaluator:** Model evaluation and benchmarking

### Relevance to your roles
- Track A (SA): Help customers fine-tune models on their data using NeMo
- Track B: NeMo Microservices team is a target role — understanding the serving layer

---

## 6. Hardware: DGX, NVLink, InfiniBand

### DGX Systems
- **DGX A100:** 8x A100 GPUs, 640 GB total GPU memory, NVLink interconnect
- **DGX H100:** 8x H100 GPUs, 640 GB total GPU memory, NVSwitch + NVLink 4.0
- **DGX SuperPOD:** Multiple DGX nodes connected via InfiniBand

### NVLink
- High-bandwidth GPU-to-GPU interconnect **within a single node**
- NVLink 4.0 (H100): 900 GB/s bidirectional per GPU
- Used for tensor parallelism — GPUs need to exchange activations every layer
- Much faster than PCIe (which is ~64 GB/s)

### InfiniBand
- High-bandwidth network **between nodes**
- NDR InfiniBand: 400 Gb/s per port
- Used for pipeline parallelism, data parallelism across nodes
- NVIDIA acquired Mellanox (2020) for this technology

### NVSwitch
- Connects all GPUs in a DGX node via NVLink
- Full bisection bandwidth — any GPU can talk to any other at full speed
- Critical for tensor parallelism where all GPUs need to all-reduce gradients/activations

### When this matters
- Serving Llama-3 70B: needs at least 2x A100 80GB → NVLink for tensor parallelism
- Serving Llama-3 405B: needs multiple nodes → InfiniBand for pipeline parallelism
- A customer asks about infrastructure → you need to explain NVLink vs InfiniBand

---

## 7. NVIDIA AI Enterprise

### What it is
- Software license + support for production AI deployments
- Includes: NIM, NeMo, Triton, TRT-LLM, curated containers, enterprise support
- Annual subscription per GPU

### Why it exists (the business model)
NVIDIA makes money from:
1. GPU hardware sales
2. AI Enterprise software subscriptions
3. DGX system sales

As an SA, you'd help customers justify AI Enterprise by demonstrating value (easier deployment, enterprise support, security patches).

### What customers get
- Certified, tested container images
- Security vulnerability scanning and patches
- Enterprise support with SLAs
- Long-term support branches (stability)

---

## Self-Test Questions

1. A customer asks: "Should I use NIM or set up Triton + TRT-LLM myself?" How do you answer?
2. What does TRT-LLM do that raw PyTorch inference doesn't?
3. When would you recommend Triton over NIM?
4. What is the difference between NVLink and InfiniBand? When is each used?
5. A customer has a DGX A100 (8x A100 80GB). They want to serve Llama-3 405B. Can they? What's the architecture?
6. What is NVIDIA AI Enterprise and why would a bank buy it?
7. Draw the NVIDIA stack from hardware to NIM. Name each layer.

---

## Notes
```
(Take your own notes here)


```
