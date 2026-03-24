# Open Source Contribution Strategy — vLLM & SGLang

**Priority:** Week 6-8 (start reading, contribute weeks 8-16).
**Interview map:** Track B resume (highest-signal item for Track B)

---

## 1. Why OSS Contributions Matter for Track B

NVIDIA's Track B job postings explicitly mention:
> "Experience contributing to vLLM, SGLang, FlashInfer, or similar projects"

**3-5 merged PRs** before applying is the single highest-signal item. NVIDIA engineers WILL check your GitHub.

---

## 2. The Contribution Ladder

### Level 1: Documentation and Typos (Week 1-2)
- Fix typos in README, docs
- Add missing docstrings to functions you've traced
- Document undocumented configuration options
- **Value:** Gets you familiar with the contribution process (CLA, CI, review)
- **Signal:** Low, but establishes your GitHub presence on the project

### Level 2: Benchmarks and Tests (Week 2-4)
- Add a missing benchmark (new model, new scenario)
- Add test cases for untested edge cases
- Improve test coverage for a module you studied
- **Value:** Shows you understand the codebase enough to test it
- **Signal:** Medium — demonstrates competence

### Level 3: Bug Fixes (Week 4-8)
- Filter issues by `good first issue`, `bug`, or `help wanted`
- Look for bugs in areas you've studied (scheduler, attention, KV cache)
- Reproduce → diagnose → fix → test → PR
- **Value:** Shows you can read and modify the codebase
- **Signal:** High — this is real engineering work

### Level 4: Features (Week 8-16)
- Propose a small enhancement based on your understanding
- Ideas: new profiling hook, new metric, new quantization test, new CLI option
- Discuss in an issue first before coding
- **Value:** Shows initiative and deep understanding
- **Signal:** Very high — you're contributing at engineer level

---

## 3. vLLM Contribution Guide

### Repository structure
```
vllm/
├── vllm/
│   ├── core/
│   │   ├── scheduler.py         ← Scheduling logic (continuous batching)
│   │   ├── block_manager.py     ← PagedAttention block management
│   │   └── policy.py            ← Scheduling policies
│   ├── attention/
│   │   ├── backends/            ← Attention kernel backends
│   │   └── ops/                 ← Attention operations
│   ├── model_executor/
│   │   ├── models/              ← Model implementations
│   │   └── layers/              ← Quantization, TP layers
│   ├── worker/                  ← GPU worker processes
│   ├── engine/                  ← LLM engine, async engine
│   └── entrypoints/             ← API servers, CLI
├── benchmarks/                   ← Benchmark scripts
├── tests/                        ← Test suite
└── docs/                         ← Documentation
```

### Good areas for first contributions
1. **benchmarks/**: Add a benchmark for a model/scenario not yet covered
2. **tests/**: Add tests for edge cases in scheduler or block manager
3. **docs/**: Document architecture decisions or configuration options
4. **vllm/model_executor/models/**: Add support for a new model architecture
5. **vllm/core/scheduler.py**: Bug fixes in scheduling logic

### Finding issues
- GitHub Issues → filter by `good first issue`
- GitHub Issues → filter by `help wanted`
- GitHub Issues → search for keywords: "scheduler", "kv cache", "memory", "benchmark"
- Also: read recent PRs to understand what changes are being made

### Contribution process
```
1. Fork the repo
2. Create a branch: git checkout -b fix/scheduler-edge-case
3. Make changes
4. Run tests: pytest tests/ -x
5. Run linting: make lint (or pre-commit hooks)
6. Push and create PR
7. Respond to review comments promptly
8. Once approved → squash and merge
```

---

## 4. SGLang Contribution Guide

### Repository structure
```
sglang/
├── python/sglang/
│   ├── srt/                     ← SRT (SGLang Runtime)
│   │   ├── managers/            ← Scheduler, token manager
│   │   ├── model_executor/      ← Model runners
│   │   ├── layers/              ← Attention, TP layers
│   │   └── server.py            ← HTTP server
│   ├── lang/                    ← SGLang DSL
│   └── bench/                   ← Benchmarks
├── test/                        ← Tests
└── docs/                        ← Documentation
```

### Good areas for first contributions
1. **Benchmarks:** Add comparative benchmarks
2. **RadixAttention:** Bug fixes or optimizations in the radix tree
3. **Structured generation:** Improve grammar/JSON constrained decoding
4. **Documentation:** SGLang's docs are less mature than vLLM — more opportunities
5. **FlashInfer integration:** Bugs or improvements in attention kernels

### SGLang advantages for contributors
- Smaller community → easier to get noticed by maintainers
- More greenfield → more opportunities for meaningful contributions
- Growing fast → your early contributions have outsized impact

---

## 5. PR Strategy for Maximum Impact

### PR title format
```
[Component] Brief description of change

Examples:
[Scheduler] Fix race condition in continuous batching with chunked prefill
[Benchmark] Add Llama-3-70B throughput benchmark with FP8 quantization
[Docs] Document KV cache block allocation algorithm
[Attention] Fix GQA head count validation for Mistral models
```

### PR description template
```markdown
## Summary
One paragraph describing what this PR does and why.

## Changes
- Bullet list of specific changes

## Testing
- How you tested this (unit tests, benchmarks, manual testing)
- Include benchmark results if applicable

## Related Issues
Fixes #123 (link to the issue)
```

### Tips for getting PRs merged
1. **Start small:** First PR should be simple (typo, test, small bug fix)
2. **Follow style:** Match the codebase's coding style exactly
3. **Test thoroughly:** Maintainers won't merge untested code
4. **Respond quickly:** Reply to review comments within 24 hours
5. **Be respectful:** These are volunteers. Thank them for reviews.
6. **Open an issue first:** For features, discuss before coding
7. **Keep PRs focused:** One change per PR. Don't bundle unrelated fixes.

---

## 6. Contribution Ideas (Concrete)

### For vLLM
1. Add benchmark: Llama-3 8B with FP8 on H100, varying context lengths
2. Add test: scheduler behavior when all KV cache blocks are exhausted
3. Fix: any `good first issue` related to memory management
4. Feature: add a metric for prefix cache hit rate to the Prometheus endpoint
5. Docs: write an architecture overview of the block manager

### For SGLang
1. Add benchmark: SGLang vs vLLM on ShareGPT dataset
2. Fix: any open bug in the radix attention tree
3. Feature: add Grafana dashboard template for SGLang metrics
4. Docs: document the ZMQ multi-process architecture
5. Feature: add support for a new model architecture

---

## 7. Building Your GitHub Profile

### What NVIDIA recruiters will look at
1. **Merged PRs:** Number and complexity
2. **Discussion quality:** Your comments on issues and PRs
3. **Consistency:** Contributions over time, not a burst before applying
4. **Relevance:** Contributions to inference-related areas (not just docs)

### GitHub profile optimization
- Pin vLLM/SGLang contributions to your profile
- Add InferBench as a pinned repository
- Write a clear bio: "ML Infrastructure Engineer | LLM Inference | Contributing to vLLM/SGLang"
- Keep your contribution graph green (regular activity)

---

## 8. Timeline

| Week | Target |
|---|---|
| Week 6-7 | Set up dev environment. Browse issues. Understand CI pipeline. |
| Week 8 | Submit first PR (documentation or benchmark). |
| Week 9-10 | Submit second PR (bug fix or test). |
| Week 12-14 | Third PR (more substantial bug fix or small feature). |
| Week 15-16 | Fourth + fifth PR (feature or optimization). |
| **By Week 16** | **3-5 merged PRs** |

---

## Self-Test Questions

1. Why do NVIDIA interviewers check your GitHub for vLLM/SGLang contributions?
2. What's the best first contribution to make? Why not start with a feature?
3. Describe the vLLM codebase structure — where is the scheduler? The attention kernels?
4. What makes a PR more likely to get merged quickly?
5. How would you find a good first issue to work on?

---

## Notes
```


```
