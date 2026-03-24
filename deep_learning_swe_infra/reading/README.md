# Reading List — No GPU Required
**Use this at office or anywhere without GPU access.**

All reading is ordered by priority. Each section maps to a specific interview question.

## How to Use This
- Read in order within each week
- Take notes in the notes files provided
- After each section, try the self-test questions WITHOUT looking back
- Mark items done in the checklist below

---

## Week 1–2: Foundations (Batch 1)

| File | Topic | Time | Status |
|---|---|---|---|
| [01_gpu_architecture.md](01_gpu_architecture.md) | GPU memory hierarchy, warps, compute vs memory bound | 3-4 hrs | |
| [02_llm_inference_fundamentals.md](02_llm_inference_fundamentals.md) | Prefill vs decode, KV cache, TTFT/TPOT, why inference is hard | 3-4 hrs | |
| [03_nvidia_product_stack.md](03_nvidia_product_stack.md) | NIM, Triton, TRT-LLM, DGX, InfiniBand — the full NVIDIA map | 2-3 hrs | |
| [04_paged_attention_and_vllm.md](04_paged_attention_and_vllm.md) | PagedAttention paper summary, vLLM architecture, continuous batching | 3-4 hrs | |
| [05_flash_attention.md](05_flash_attention.md) | FlashAttention 1 & 2 — tiling, online softmax, why it matters | 2-3 hrs | |
| [06_quantization_theory.md](06_quantization_theory.md) | INT8, FP8, AWQ, GPTQ — what they do and when to use each | 2-3 hrs | |
| [07_distributed_inference.md](07_distributed_inference.md) | Tensor parallelism, pipeline parallelism, when to use which | 2-3 hrs | |
| [08_speculative_decoding.md](08_speculative_decoding.md) | How it works, when it helps, when it doesn't | 1-2 hrs | |
| [09_interview_qa_bank.md](09_interview_qa_bank.md) | 20+ questions with detailed answers for both tracks | Ongoing | |

**Batch 1 total: ~20-25 hours**

---

## Week 2–4: Deep Dives (Batch 2)

| File | Topic | Time | Status |
|---|---|---|---|
| [10_cuda_programming_concepts.md](10_cuda_programming_concepts.md) | Memory coalescing, bank conflicts, occupancy, Nsight profiling | 3-4 hrs | |
| [11_attention_variants.md](11_attention_variants.md) | MHA, MQA, GQA, MLA (DeepSeek) — KV cache size tradeoffs | 2-3 hrs | |
| [12_prefill_decode_disaggregation.md](12_prefill_decode_disaggregation.md) | Splitwise, DistServe, chunked prefill — cutting-edge arch | 2-3 hrs | |
| [13_kv_cache_optimization.md](13_kv_cache_optimization.md) | KV quantization, sliding window, token eviction, prefix caching | 2-3 hrs | |
| [14_lora_serving.md](14_lora_serving.md) | Multi-LoRA serving, S-LoRA, multi-tenant fine-tuned models | 2-3 hrs | |
| [15_cost_optimization.md](15_cost_optimization.md) | GPU pricing, cost per token, TCO, optimization levers | 2-3 hrs | |
| [16_systems_design_patterns.md](16_systems_design_patterns.md) | 7 architecture patterns + complete interview answer template | 3-4 hrs | |

**Batch 2 total: ~16-23 hours**

---

**Grand total: ~36-48 hours of reading covering all theory for Month 1–2**
