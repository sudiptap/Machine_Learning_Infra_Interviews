# LLM Serving Systems Design Patterns

**Priority:** Week 3-4. The synthesis — combines everything into interview-ready designs.
**Interview map:** Track A Round 4, Track B Round 2

---

## 1. Pattern: Basic Single-Model Serving

**When:** Small model, moderate traffic, simple requirements.

```
Client → Load Balancer → [vLLM/NIM on 1-2 GPUs]
                          ├─ Continuous batching
                          ├─ PagedAttention
                          └─ Prometheus metrics
```

**Key decisions:**
- vLLM for flexibility, NIM for simplicity
- INT8 quantization for cost efficiency
- Autoscale replicas on queue depth

---

## 2. Pattern: Multi-Model Gateway

**When:** Multiple models (different sizes/tasks), route by request.

```
Client → API Gateway → Model Router → ┬─ [Llama-3 70B: complex tasks]
                       (routing logic)  ├─ [Llama-3 8B: simple tasks]
                                        ├─ [Code model: code generation]
                                        └─ [Embedding model: RAG]
```

**Key decisions:**
- Router logic: rule-based (URL path), ML-based (complexity classifier), or client-specified
- Each model pool autoscales independently
- Shared Prometheus/Grafana for all models
- Cost optimization: route simple tasks to cheap models

---

## 3. Pattern: RAG-Augmented Serving

**When:** Enterprise knowledge base, document Q&A, grounded responses.

```
Client → API Gateway → ┬─ Embedding Model → Vector DB (OpenSearch/Pinecone)
                        │                      ↓ retrieved docs
                        └─ LLM ← augmented prompt (query + retrieved context)
                           ↓
                        Response + citations
```

**Key decisions:**
- **Chunking strategy:** How to split documents (fixed-size, semantic, sliding window)
- **Embedding model:** Colocate on same GPU or separate? (usually separate — different compute profile)
- **Retrieval:** Top-K documents, reranking, hybrid search (BM25 + vector)
- **Context window:** Retrieved chunks + system prompt + query must fit in model's context
- **PII handling:** Scrub retrieved documents before sending to LLM
- **Caching:** Prefix caching for shared system prompt + common retrieved docs

### JPMC connection
This is your LLM Suite architecture. Frame it in these terms.

---

## 4. Pattern: Regulated Environment (Financial Services)

**When:** Banking, healthcare, government — compliance requirements.

```
┌─────────────────────── VPC / Private Network ───────────────────────┐
│                                                                      │
│  Client → WAF → API Gateway → PII Scrubber → Rate Limiter           │
│                                      ↓                               │
│                              ┌── Request Router ──┐                  │
│                              │                    │                  │
│                    ┌─────────▼──────┐  ┌─────────▼──────┐          │
│                    │ LLM Inference  │  │ LLM Inference  │          │
│                    │ (NIM/TRT-LLM)  │  │ (NIM/TRT-LLM)  │          │
│                    │ Dept A LoRA    │  │ Dept B LoRA    │          │
│                    └────────────────┘  └────────────────┘          │
│                              │                                      │
│                    ┌─────────▼──────┐                               │
│                    │ PII Re-check   │ ← verify output has no PII    │
│                    │ Audit Logger   │ ← log everything              │
│                    │ Guardrails     │ ← content filtering           │
│                    └────────────────┘                               │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐          │
│  │ KMS (CMK)    │  │ Audit Store  │  │ Monitoring       │          │
│  │ Encrypt keys │  │ Immutable log│  │ Prometheus/      │          │
│  │ Key rotation │  │ Retention 7yr│  │ Grafana          │          │
│  └──────────────┘  └──────────────┘  └──────────────────┘          │
└──────────────────────────────────────────────────────────────────────┘
```

**Compliance elements:**
- **Data residency:** All processing within designated region/VPC
- **PII scrubbing:** Pre-inference and post-inference
- **Audit logging:** Every request/response logged immutably (7-year retention for financial)
- **Encryption:** KMS-CMK for data at rest and in transit
- **Access control:** Per-department/per-team authorization
- **Model governance:** Approval workflow for model updates, A/B testing before rollout
- **Content guardrails:** Block harmful outputs, check for hallucinated financial advice

**This is your FinServLLM project and JPMC experience.**

---

## 5. Pattern: High-Throughput Batch Processing

**When:** Offline document processing, bulk classification, embedding generation.

```
Input Queue (S3/SQS) → Batch Scheduler → GPU Pool (spot instances)
                                          ├─ Continuous batching
                                          ├─ Max batch size for throughput
                                          └─ Results → Output Store
```

**Key decisions:**
- Use spot instances (60-80% cheaper) — batch jobs tolerate preemption
- Maximize batch size for throughput (latency doesn't matter)
- Checkpoint progress so preempted jobs can resume
- Consider offline quantization (INT4) since quality is evaluated in bulk

---

## 6. Pattern: Real-Time Streaming with Strict SLOs

**When:** Customer-facing chatbot, financial trading assistant, real-time.

```
Client ←→ WebSocket → API Gateway → Priority Scheduler → GPU Pool
                                      ├─ Chunked prefill (TPOT stability)
                                      ├─ Reserved capacity (no queue)
                                      ├─ SLO monitoring (alert on P99 > threshold)
                                      └─ Fallback: smaller model if overloaded
```

**Key decisions:**
- **TPOT SLO:** <50ms for streaming feel
- **TTFT SLO:** <500ms for responsiveness
- **Chunked prefill** to prevent decode stalls
- **Reserved capacity** (no autoscale-to-zero — cold starts break SLOs)
- **Circuit breaker:** If GPU pool overloaded, route to smaller model or return cached response
- **Speculative decoding** at low batch sizes for faster TPOT

---

## 7. Pattern: Multi-Region / Disaster Recovery

**When:** Global enterprise, availability requirements.

```
DNS (Route53/CloudFlare) → Closest Region

Region US-East:                    Region EU-West:
┌─────────────────┐               ┌─────────────────┐
│ Full LLM stack  │               │ Full LLM stack  │
│ (independent)   │               │ (independent)   │
└─────────────────┘               └─────────────────┘

Failover: if US-East unhealthy → DNS routes to EU-West
Model sync: Same model version deployed to all regions
Data: Region-local (comply with data residency)
```

**Key decisions:**
- Active-active (both serve traffic) vs active-passive (standby)
- Model consistency: ensure same version in all regions
- Data residency: some data CAN'T cross region boundaries
- Cost: running full stack in N regions is N× cost → active-passive is cheaper

---

## 8. The Complete Interview Answer Template

When asked "Design an LLM serving system for X":

```
1. REQUIREMENTS (2 min)
   - Scale: users, requests/sec, models
   - Latency SLOs: TTFT, TPOT targets
   - Compliance: data residency, audit, PII
   - Budget constraints

2. HIGH-LEVEL ARCHITECTURE (3 min)
   - Draw the request flow (client → gateway → router → inference → response)
   - Identify GPU requirements (model size → quantization → GPU count)
   - Choose inference engine (NIM for simplicity, vLLM for flexibility)

3. DEEP DIVES (5 min)
   - KV cache management: PagedAttention, quantized KV cache
   - Batching: continuous batching, chunked prefill for SLOs
   - Scaling: autoscaling on queue depth, TP/PP configuration
   - Monitoring: TTFT/TPOT/throughput metrics, alerting

4. COST OPTIMIZATION (2 min)
   - Quantization (INT8/FP8)
   - Right-sizing GPUs
   - Spot instances for non-critical workloads
   - Caching (prefix, semantic)

5. TRADE-OFFS (3 min)
   - Latency vs throughput (batch size)
   - Quality vs cost (quantization level)
   - Simplicity vs control (NIM vs custom stack)
   - Availability vs cost (multi-region)
```

---

## Self-Test: Design Exercises

Practice these on paper or whiteboard:

1. "Design an LLM-powered customer support chatbot for a bank with 10M users."
2. "Design a document processing pipeline that classifies 1M financial documents per day."
3. "Design a multi-model serving platform for an enterprise with 20 teams, each needing different LLM capabilities."
4. "Your company wants to offer LLM-as-a-Service to external customers. Design the infrastructure."

For each: identify the pattern(s) from this document, draw the architecture, specify GPU/model choices, and discuss trade-offs.

---

## Notes
```


```
