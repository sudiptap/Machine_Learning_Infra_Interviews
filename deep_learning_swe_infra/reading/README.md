# Reading List — No GPU Required
**Use this at office or anywhere without GPU access.**
**Reading schedule:** Mon-Thu at office. Fri-Sun hands-on with RunPod.

---

## How to Use This
- Read in order within each batch
- After each section, try the self-test questions WITHOUT looking back
- Take notes in the notes section of each file
- Mark items done in the Status column below

---

## Batch 1: Foundations (Week 1-2) — ~20-25 hrs

| # | Topic | Time | Status |
|---|---|---|---|
| [01](01_gpu_architecture.md) | GPU memory hierarchy, warps, compute vs memory bound | 3-4 hrs | |
| [02](02_llm_inference_fundamentals.md) | Prefill vs decode, KV cache, TTFT/TPOT | 3-4 hrs | |
| [03](03_nvidia_product_stack.md) | NIM, Triton, TRT-LLM, DGX, InfiniBand | 2-3 hrs | |
| [04](04_paged_attention_and_vllm.md) | PagedAttention, vLLM architecture, continuous batching | 3-4 hrs | |
| [05](05_flash_attention.md) | FlashAttention 1 & 2 — tiling, online softmax | 2-3 hrs | |
| [06](06_quantization_theory.md) | INT8, FP8, AWQ, GPTQ tradeoffs | 2-3 hrs | |
| [07](07_distributed_inference.md) | Tensor parallelism, pipeline parallelism | 2-3 hrs | |
| [08](08_speculative_decoding.md) | Draft-verify paradigm, Medusa, Eagle | 1-2 hrs | |
| [09](09_interview_qa_bank.md) | 20 Q&As with detailed answers | Ongoing | |

---

## Batch 2: Deep Dives (Week 2-4) — ~16-23 hrs

| # | Topic | Time | Status |
|---|---|---|---|
| [10](10_cuda_programming_concepts.md) | Memory coalescing, bank conflicts, occupancy, Nsight | 3-4 hrs | |
| [11](11_attention_variants.md) | MHA, MQA, GQA, MLA — KV cache size tradeoffs | 2-3 hrs | |
| [12](12_prefill_decode_disaggregation.md) | Splitwise, DistServe, chunked prefill | 2-3 hrs | |
| [13](13_kv_cache_optimization.md) | KV quantization, sliding window, token eviction, prefix caching | 2-3 hrs | |
| [14](14_lora_serving.md) | Multi-LoRA serving, S-LoRA, multi-tenant fine-tuned models | 2-3 hrs | |
| [15](15_cost_optimization.md) | GPU pricing, cost per token, TCO, optimization levers | 2-3 hrs | |
| [16](16_systems_design_patterns.md) | 7 architecture patterns + interview answer template | 3-4 hrs | |

---

## Batch 3: Intermediate Topics (Week 4-6) — ~15-20 hrs

| # | Topic | Time | Status |
|---|---|---|---|
| [17](17_triton_gpu_language.md) | Triton programming model, vector add, matmul, fused softmax | 3-4 hrs | |
| [18](18_megatron_lm.md) | Megatron TP/PP/SP, 3D parallelism, interleaved pipeline | 3-4 hrs | |
| [19](19_mixture_of_experts.md) | MoE architecture, expert parallelism, Mixtral, DeepSeek | 2-3 hrs | |
| [20](20_benchmarking_methodology.md) | How to run proper benchmarks, metrics, pitfalls | 3-4 hrs | |
| [21](21_inference_engine_comparison.md) | vLLM vs TRT-LLM vs SGLang deep comparison | 3-4 hrs | |

---

## Batch 4: Advanced Topics (Week 6-10) — ~18-24 hrs

| # | Topic | Time | Status |
|---|---|---|---|
| [22](22_advanced_cuda_optimization.md) | Kernel fusion, warp primitives, async ops, quantized compute | 3-4 hrs | |
| [23](23_networking_gpu_clusters.md) | NVLink, InfiniBand, RDMA, GPUDirect, NCCL | 3-4 hrs | |
| [24](24_model_compilation.md) | torch.compile, TRT-LLM build, CUDA Graphs, XLA | 2-3 hrs | |
| [25](25_certification_study.md) | NCA-AIIO exam prep, key facts, practice questions | 3-4 hrs | |
| [26](26_behavioral_interview.md) | STAR framework, 5 story bank, Track A/B framing | 3-4 hrs | |

---

## Batch 5: Production & Interview Polish (Week 8-12) — ~15-20 hrs

| # | Topic | Time | Status |
|---|---|---|---|
| [27](27_llm_safety_guardrails.md) | Guardrails, NeMo Guardrails, Llama Guard, prompt injection | 2-3 hrs | |
| [28](28_production_observability.md) | Prometheus, Grafana, DCGM, alerting, autoscaling | 3-4 hrs | |
| [29](29_oss_contribution_strategy.md) | vLLM/SGLang contribution guide, PR strategy, timeline | 2-3 hrs | |
| [30](30_emerging_research.md) | Ring attention, tree speculation, disaggregated serving, hardware trends | 3-4 hrs | |
| [31](31_advanced_interview_qa.md) | 10 advanced Q&As: systems design, kernel analysis, SA scenarios | 3-4 hrs | |

---

## Summary

| Batch | Topics | Hours | Weeks |
|---|---|---|---|
| Batch 1 | Foundations | 20-25 hrs | Week 1-2 |
| Batch 2 | Deep dives | 16-23 hrs | Week 2-4 |
| Batch 3 | Intermediate | 15-20 hrs | Week 4-6 |
| Batch 4 | Advanced | 18-24 hrs | Week 6-10 |
| Batch 5 | Production & polish | 15-20 hrs | Week 8-12 |
| **Total** | **31 reading modules** | **~84-112 hrs** | **~12 weeks** |

At ~16-20 hrs/week of office reading: **~5-7 weeks to complete all reading.**
This gives you buffer to re-read and practice weak areas.
