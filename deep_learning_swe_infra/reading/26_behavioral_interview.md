# Behavioral Interview — STAR Framework + Story Bank

**Priority:** Week 8-10 (but start reading now, refine stories over time).
**Interview map:** Track A Round 5, Track B Round 5

---

## 1. The STAR Framework

Every behavioral answer follows this structure:

```
S — Situation: Context (1-2 sentences). Where, when, what was happening.
T — Task:      Your specific responsibility. What was expected of you.
A — Action:    What YOU did (not the team). Specific steps, decisions, trade-offs.
R — Result:    Quantified outcome. Numbers, impact, what changed.
```

### Common mistakes
- **Too much Situation:** 3 minutes of context, 30 seconds of action → flip the ratio
- **"We" instead of "I":** Interviewers want YOUR contribution, not the team's
- **No numbers:** "It was faster" vs "P99 latency dropped from 800ms to 200ms" — always quantify
- **No trade-off:** Good answers explain what you sacrificed for what you gained

---

## 2. NVIDIA's Behavioral Questions — Both Tracks

### Question 1: "Tell me about a system you designed end-to-end where you owned the architecture decision"

**Map to:** LLM Suite at JPMC

```
S: JPMC had no internal LLM platform. 50K+ employees needed AI capabilities
   but couldn't use public APIs due to compliance requirements.

T: I was responsible for designing and building the entire LLM serving
   infrastructure from scratch — architecture, technology selection,
   deployment, and scaling.

A: I designed a multi-tenant architecture on AWS ECS with:
   - NLB/ALB routing for multi-model serving
   - KMS-CMK encryption for data at rest and in transit
   - PII scrubbing layer before and after inference
   - Audit logging with 7-year retention for compliance
   - RAG pipeline with OpenSearch for knowledge grounding
   - Evaluated TensorRT-LLM, vLLM, and SGLang as inference backends
   I made the decision to use [engine] because [specific reason].
   I pushed back on the initial proposal to use [alternative] because
   [specific technical reason].

R: Platform launched serving 50K+ employees across 12 production deployments.
   P99 TTFT <800ms at 200 concurrent users. Passed SOC2 audit.
   Became the standard AI platform for the firm.
```

### Question 2: "Describe a time you had to make a hard performance tradeoff"

**Map to:** LLM Suite scaling decisions

```
S: During load testing, we hit GPU memory limits at 150 concurrent users.
   The target was 200 concurrent.

T: I needed to find a way to serve 200 concurrent users without adding
   more GPUs (budget was fixed for the quarter).

A: I analyzed the bottleneck: KV cache was consuming 85% of available GPU
   memory. I evaluated three options:
   1. Reduce max context length (limits functionality)
   2. Enable FP8 KV cache quantization (small quality impact)
   3. Add request queuing with priority (increases latency for low-priority)

   I chose option 2 (FP8 KV cache) + option 3 (priority queuing) together.
   FP8 KV cache gave us 2x more concurrent capacity with <0.5% quality
   degradation. Priority queuing ensured executive-facing apps got low
   latency while batch workloads tolerated higher queue times.

   I specifically rejected option 1 because our RAG pipeline required
   long contexts (4096+ tokens) for document grounding.

R: Achieved 250 concurrent users on the same GPU budget. Executive apps
   maintained P99 TTFT <500ms. Batch workloads P99 TTFT <3s (acceptable).
   Saved ~$X/month in additional GPU costs.
```

### Question 3: "Tell me about a time you pushed back on a technical direction and were right"

```
S: Early in LLM Suite development, the team lead proposed using a simple
   round-robin load balancer across model replicas.

T: I believed this approach would cause hot-spotting because LLM requests
   have wildly different compute costs (short prompt vs long prompt).

A: I built a prototype of both approaches:
   1. Round-robin (their proposal)
   2. Queue-depth-aware routing (my proposal) — route to the replica with
      the shortest queue

   I ran a benchmark with realistic request distributions (mix of short
   and long prompts). Round-robin showed 3x P99 latency variance because
   one replica would get 3 long prompts in a row while another sat idle.

   I presented the data to the team, not as "you're wrong" but as
   "here's what I found — the data suggests queue-depth routing." The
   team lead agreed after seeing the benchmark.

R: Queue-depth routing reduced P99 TTFT from 2.5s to 800ms under load.
   This approach became the standard for all new model deployments.
   The team lead later thanked me for the push and cited it as an
   example of good engineering culture.
```

### Question 4: "Tell me about a time you drove technical adoption against resistance"

```
S: When I proposed adding a PII scrubbing layer to the LLM pipeline,
   engineering leadership pushed back — they said it would add latency
   and complexity, and the legal team's requirements were "theoretical."

T: I needed to convince both engineering and legal that PII protection
   was non-negotiable for a regulated financial institution.

A: I took three steps:
   1. Built a lightweight PII detector prototype using [tool/approach]
      that added only 15ms of latency (<5% of total request time).
   2. Ran it on 1000 real user queries (anonymized) and found that 12%
      contained PII (account numbers, names, SSNs). Presented this to
      leadership — the risk was real, not theoretical.
   3. Worked with legal to draft a threat model showing regulatory
      exposure without PII scrubbing.

   The data made the decision obvious. I didn't argue opinions —
   I showed evidence.

R: PII scrubbing was added to the standard pipeline. The 15ms latency
   was invisible to users. During a compliance audit 6 months later,
   the auditors specifically praised the PII handling as "best in class"
   for an internal AI platform. This became a mandatory component for
   all new AI services at the firm.
```

### Question 5: "Describe a project where you owned the outcome end to end"

**Map to:** The same LLM Suite story, but emphasize the full lifecycle:
- Proposal and buy-in
- Architecture design
- Technology selection
- Implementation
- Deployment and operations
- Scaling and optimization
- Onboarding internal teams (50K users)

---

## 3. Track A-Specific Behavioral Angles

For SA roles, emphasize:
- **Customer-facing:** Frame internal teams as "customers" you onboarded
- **Workshops:** "I conducted technical workshops to drive adoption"
- **Trade studies:** "I evaluated TRT-LLM vs vLLM for the team"
- **Enablement:** "I created documentation and runbooks that enabled self-service"
- **Stakeholder management:** "I aligned engineering, security, and business teams"

### SA-specific reframes:
```
Engineer framing: "I built the LLM platform"
SA framing: "I architected the platform AND drove adoption across 12 teams,
             conducting workshops and supporting each team's unique use case"
```

---

## 4. Track B-Specific Behavioral Angles

For engineering roles, emphasize:
- **Technical depth:** Specific optimizations, profiling results, kernel-level decisions
- **Open source:** "I contributed to vLLM because I found a bug in the scheduler"
- **Performance:** Numbers, benchmarks, before/after comparisons
- **Architecture decisions:** Why you chose one approach over another

### Eng-specific reframes:
```
General: "I optimized the platform"
Eng: "I profiled the inference pipeline, identified KV cache memory as the
      bottleneck, and implemented FP8 quantization that doubled concurrent
      capacity while maintaining <0.5% quality degradation"
```

---

## 5. Story Preparation Checklist

Prepare 5 stories, each mappable to multiple questions:

| # | Story | Maps to questions |
|---|---|---|
| 1 | LLM Suite architecture (end-to-end ownership) | Owned architecture, end-to-end project |
| 2 | Performance optimization (KV cache, scaling) | Hard tradeoff, performance optimization |
| 3 | PII scrubbing adoption (pushed back on resistance) | Drove adoption, pushed back |
| 4 | SGLang deep-dive (technical investigation) | Technical depth, learning new system |
| 5 | Cross-team collaboration (12 deployments) | Stakeholder management, customer-facing |

### For each story, prepare:
- [ ] 2-minute version (initial answer)
- [ ] 5-minute version (with follow-up details)
- [ ] Key numbers (quantified results)
- [ ] The trade-off you made
- [ ] What you'd do differently (shows self-awareness)
- [ ] Dual framing (SA version + Eng version)

---

## 6. Practice Method

1. **Write out** each story in STAR format (this document)
2. **Speak it** out loud — timing yourself (2 minutes for initial answer)
3. **Record yourself** on phone — listen back for:
   - Filler words (um, uh, like, so)
   - "We" vs "I" ratio (should be mostly "I")
   - Missing numbers (add quantification)
   - Rambling situation (shorten context, lengthen action)
4. **Practice with someone** — have them ask follow-ups
5. **Refine** based on feedback

### Common follow-ups to prepare for:
- "What would you do differently if you could do it again?"
- "How did you measure success?"
- "What was the biggest risk and how did you mitigate it?"
- "How did you get buy-in from stakeholders?"
- "What did you learn from this experience?"

---

## Notes
```
(Draft your STAR stories here)


```
