# Model Compilation — torch.compile, Graph Optimization, XLA

**Priority:** Week 7-8.
**Interview map:** Track B Rounds 2 & 3

---

## 1. Why Model Compilation Matters

PyTorch default: **eager mode** — operations execute one at a time as Python runs.
- Easy to debug
- Flexible (dynamic shapes, control flow)
- Slow: Python overhead, no cross-operation optimization, no kernel fusion

Compilation: Convert the model into an optimized graph, then execute the graph.
- Harder to debug
- Less flexible (some operations can't be compiled)
- Fast: no Python overhead, automatic kernel fusion, optimized memory

---

## 2. torch.compile (PyTorch 2.0+)

### What it is
PyTorch's built-in compilation system. Traces your model, optimizes the graph, generates efficient kernels.

```python
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3-8B")

# One line to compile
compiled_model = torch.compile(model)

# Use normally — first call is slow (compilation), subsequent calls are fast
output = compiled_model(input_ids)
```

### How it works
```
Python model code
    ↓ (TorchDynamo traces)
FX Graph (intermediate representation)
    ↓ (TorchInductor optimizes)
Triton kernels + C++ code
    ↓ (compiled and cached)
Optimized execution
```

### Components
1. **TorchDynamo:** Captures Python code into a graph (handles control flow, dynamic shapes)
2. **AOTAutograd:** Traces backward pass ahead of time (for training)
3. **TorchInductor:** Backend that generates optimized Triton kernels
4. **Triton:** Generates GPU machine code from Inductor's output

### What gets optimized
- **Kernel fusion:** Multiple PyTorch ops → single Triton kernel
- **Memory planning:** Reuse memory buffers, reduce allocations
- **Operator scheduling:** Reorder operations for better memory access
- **Dead code elimination:** Remove unused computations

### Compilation modes
```python
# Default: balanced speed/compile time
torch.compile(model)

# Maximum optimization (slower compile, faster runtime)
torch.compile(model, mode="max-autotune")

# Faster compile, less optimization
torch.compile(model, mode="reduce-overhead")
```

---

## 3. How TRT-LLM Compilation Differs

### TRT-LLM build process
```
HuggingFace model (PyTorch)
    ↓ (TRT-LLM builder script)
TRT-LLM model definition (C++/Python)
    ↓ (TensorRT builder)
Optimized TensorRT engine (.engine file)
    ↓ (serialized to disk)
Runtime loads engine → serves inference
```

### What TRT-LLM optimizes that torch.compile doesn't
1. **Model-specific kernel selection:** Chooses from a library of hand-tuned kernels per layer type
2. **INT4/INT8/FP8 quantization integration:** Baked into the engine at compile time
3. **Custom attention kernels:** fMHA variants optimized for specific head dimensions
4. **Weight layout optimization:** Reorders weights for optimal Tensor Core access patterns
5. **Plugin system:** Custom C++ kernels for special operations

### Trade-off
```
torch.compile:  Minutes to compile. General purpose. Works with any PyTorch model.
                ~10-30% speedup over eager mode.

TRT-LLM:       10-60 minutes to build. Model-specific. Best performance.
                ~30-50% speedup over eager mode. ~10-20% over torch.compile.
```

---

## 4. Graph-Level Optimizations

### Operator Fusion
```
Before fusion:
  op1: x = linear(input)      → HBM write
  op2: x = gelu(x)            → HBM read + write
  op3: x = dropout(x)         → HBM read + write

After fusion:
  fused_op: x = dropout(gelu(linear(input)))  → one HBM read/write
```

### Constant Folding
```
Before: output = input * 2.0 * 3.14159
After:  output = input * 6.28318  (computed at compile time)
```

### Common Subexpression Elimination
```
Before: a = x * w;  b = x * w + bias  (x * w computed twice)
After:  t = x * w;  a = t;  b = t + bias  (x * w computed once)
```

### Memory Planning
```
Before: Each intermediate tensor gets its own allocation
After:  Tensors that don't overlap in lifetime share memory
        e.g., layer 1 output and layer 3 output can use same buffer
        (layer 2 consumes layer 1's output before layer 3 produces)
```

---

## 5. CUDA Graphs

### The problem
Launching a CUDA kernel has overhead (~5-10 μs per launch). LLM decode launches hundreds of kernels per token. For small operations, launch overhead dominates.

### What CUDA Graphs do
Record a sequence of kernel launches, then replay the entire sequence with a single launch:
```python
# Record
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model(static_input)

# Replay (single kernel launch overhead)
static_input.copy_(new_input)
g.replay()
```

### Impact
- Eliminates per-kernel launch overhead
- 10-20% speedup for decode (many small kernels)
- Used by vLLM, TRT-LLM, SGLang for decode phase

### Limitations
- Input shapes must be static (can't change batch size dynamically)
- No Python control flow during graph replay
- Must pre-allocate memory for all intermediate tensors
- vLLM works around this with "padding" — pads batch to next power of 2

---

## 6. XLA and JAX Compilation (Bonus)

### XLA (Accelerated Linear Algebra)
- Google's ML compiler (used by JAX and TensorFlow)
- Whole-program optimization: sees the entire computation graph
- Aggressive fusion and memory optimization
- TPU-native, but also works on GPU

### JAX + XLA for inference
- Some teams use JAX for inference (Alpa, Pathways)
- Whole-program compilation can exceed PyTorch performance
- But ecosystem is smaller, less community support for LLM serving

### Why you should know about it
- Demonstrates awareness of the broader landscape beyond PyTorch
- Google's TPU inference stack uses XLA
- If asked "what compilation approaches exist beyond TensorRT?", XLA is the answer

---

## 7. Practical Impact on Inference Engines

| Engine | Compilation Approach | Impact |
|---|---|---|
| vLLM | torch.compile (optional) + CUDA Graphs | 10-20% decode speedup |
| TRT-LLM | Full TensorRT compilation | 30-50% speedup, best at scale |
| SGLang | torch.compile + CUDA Graphs + FlashInfer | Competitive performance |
| NIM | TRT-LLM engine (pre-compiled per model) | Best single-model performance |

---

## 8. Interview Connections

> "How does TRT-LLM's compilation differ from torch.compile?"

**A:** "torch.compile works with any PyTorch model, compiles in minutes, and generates Triton kernels via TorchInductor. It gives 10-30% speedup through operator fusion and memory optimization. TRT-LLM goes further: it selects from a library of hand-tuned kernels per layer, integrates quantization at compile time, optimizes weight layouts for Tensor Cores, and produces a fully serialized engine. This takes 10-60 minutes but gives 30-50% speedup. The trade-off is flexibility — TRT-LLM engines are fixed for specific configs, while torch.compile handles dynamic shapes."

> "What are CUDA Graphs and why do inference engines use them?"

**A:** "CUDA Graphs record a sequence of kernel launches and replay them with a single API call, eliminating per-kernel launch overhead (~5-10μs each). During decode, models launch hundreds of small kernels per token — the launch overhead can be 10-20% of total time. CUDA Graphs eliminate this. The constraint is that input shapes must be static, so engines pad batches to fixed sizes."

---

## Self-Test Questions

1. What is the difference between eager mode and compiled mode in PyTorch?
2. Name 3 optimizations that graph compilation enables.
3. What is kernel fusion and why does it help inference performance?
4. How do CUDA Graphs reduce decode latency? What's the trade-off?
5. Compare torch.compile vs TRT-LLM compilation: speed, flexibility, optimization depth.
6. What is TorchInductor and what does it generate?

---

## Notes
```


```
