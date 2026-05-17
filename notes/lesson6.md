# Lesson 6: AWQ Quantized Model Inference

This note summarizes the article at https://gogongxt.com/posts/12d23d8d.html and maps it to the AWQ implementation in this repo.

Lessons 1 to 5 were mostly about runtime control flow: processes, streaming, router scheduling, memory pools, and chunked prefill. Lesson 6 changes the focus.

It asks:

How does the inference framework actually recognize a quantized model and route linear layers onto the right quantized weight format and kernel?

The article answers that with AWQ as the concrete example.

## 1. The four questions this lesson is answering

The article frames quantized inference around four practical questions:

1. How do we know the model is quantized?
2. How do we know which layers are quantized?
3. How is quantization integrated polymorphically into normal model code?
4. What actual compute path runs the quantized matrix multiplication?

This repo answers those questions with:

- `ModelRunner.load_model(...)`
- `AWQConfig` and `AWQLinearMethod`
- linear layer classes in `layers/linear.py`
- the Triton kernel wrapper in `layers/quantization/awq_triton.py`

## 2. How the runtime detects an AWQ model

Detection happens inside `ModelRunner.load_model()`:

```python
hf_quant_config = getattr(
    self.model_config.hf_config, "quantization_config", None
)
if hf_quant_config is not None:
    from sglang.srt.layers.quantization.awq import AWQConfig

    quant_config = AWQConfig.from_config(hf_quant_config)
    linear_method = quant_config.get_linear_method()
model = model_class(
    config=self.model_config.hf_config, linear_method=linear_method
)
```

This is the first key insight.

The runtime does not detect AWQ by filename patterns or by special-case model classes.

It looks at the Hugging Face config object and checks whether `quantization_config` exists.

If it does, this repo currently assumes the quantization path is AWQ and constructs an `AWQConfig`.

## 3. What `quantization_config` tells the runtime

The article uses config values like:

```json
"quantization_config": {
  "bits": 4,
  "group_size": 128,
  "quant_method": "awq",
  "zero_point": true
}
```

The code extracts the same information through `AWQConfig.from_config(...)`:

```python
@classmethod
def from_config(cls, config):
    weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
    group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
    zero_point = cls.get_from_keys(config, ["zero_point"])
    return cls(weight_bits, group_size, zero_point)
```

For this repo, those values determine:

- weight bit width
- group size for shared scale and zero-point
- whether zero-point quantization is used
- derived pack factor, `32 // weight_bits`

For 4-bit AWQ, the pack factor is 8 because one `int32` stores eight 4-bit weights.

## 4. How the repo represents AWQ as a strategy, not a special model

The most important design decision in this lesson is the use of `linear_method`.

The abstract interface is in `layers/linear.py`:

```python
class LinearMethodBase(ABC):
    @abstractmethod
    def create_weights(...):
        ...

    @abstractmethod
    def apply_weights(...):
        ...
```

Two concrete implementations matter here:

- `UnquantizedLinearMethod`
- `AWQLinearMethod`

That means the model code does not need separate AWQ-specific attention or MLP classes.

Instead, the same high-level layers can call a polymorphic linear backend.

This is the core design pattern behind the implementation.

## 5. How `linear_method` gets threaded through the model

`ModelRunner.load_model()` creates the model with:

```python
model = model_class(
    config=self.model_config.hf_config, linear_method=linear_method
)
```

Then the LLaMA model passes that method down through the stack:

- `LlamaForCausalLM`
- `LlamaModel`
- `LlamaDecoderLayer`
- `LlamaAttention` and `LlamaMLP`
- `QKVParallelLinear`, `MergedColumnParallelLinear`, and `RowParallelLinear`

For example, in `LlamaAttention`:

```python
self.qkv_proj = QKVParallelLinear(
    hidden_size,
    self.head_dim,
    self.total_num_heads,
    self.total_num_kv_heads,
    bias=False,
    linear_method=linear_method,
)
self.o_proj = RowParallelLinear(
    self.total_num_heads * self.head_dim,
    hidden_size,
    bias=False,
    linear_method=linear_method,
)
```

So quantization is injected at model-construction time, not hardcoded in the forward path.

## 6. Which layers actually become quantized in practice

The article points out that not every layer in a model is quantized.

That is also true in this repo.

Only layers built on top of the generic linear abstractions receive `linear_method`.

That means:

- linear projections such as QKV, output projection, and MLP projections can use AWQ
- embeddings, norms, and other non-linear layers do not automatically become AWQ just because the model is quantized

So in practice, the code structure itself determines much of the quantized coverage.

## 7. How AWQ weights are represented in memory

`AWQLinearMethod.create_weights(...)` defines the actual parameter layout:

```python
qweight = Parameter(
    torch.empty(
        input_size,
        output_size // self.quant_config.pack_factor,
        device="cuda",
        dtype=torch.int32,
    ),
    requires_grad=False,
)

qzeros = Parameter(
    torch.empty(
        input_size // self.quant_config.group_size,
        output_size // self.quant_config.pack_factor,
        device="cuda",
        dtype=torch.int32,
    ),
    requires_grad=False,
)

scales = Parameter(
    torch.empty(
        input_size // self.quant_config.group_size,
        output_size,
        device="cuda",
        dtype=params_dtype,
    ),
    requires_grad=False,
)
```

This is the second major insight of the lesson.

An AWQ linear layer does not store a single `weight` tensor.

It stores three tensors:

1. `qweight`
2. `qzeros`
3. `scales`

That matches the article's explanation:

- the original weight matrix is packed into 4-bit integers inside `int32`
- zero-points are also packed
- scales stay in floating-point

## 8. Why `set_weight_attrs(...)` matters so much

When AWQ weights are created, the code also annotates them:

```python
set_weight_attrs(
    qweight,
    {
        "input_dim": 0,
        "output_dim": 1,
        "packed_dim": 1,
        "pack_factor": self.quant_config.pack_factor,
    },
)
```

The same pattern is applied to `qzeros` and `scales`.

These attributes are not cosmetic.

They tell the weight-loading code how tensor-parallel sharding should work, especially when one dimension is packed.

That is the bridge between abstract quantization format and real checkpoint loading.

## 9. How checkpoint loading still works with quantized tensors

Model loading still goes through the same high-level path:

```python
model.load_weights(
    self.model_config.path,
    cache_dir=None,
    load_format=self.load_format,
    revision=None,
)
```

For LLaMA, `load_weights()` iterates over checkpoint tensors and dispatches through each parameter's `weight_loader`.

The important part is that the linear layer classes know how to slice weights correctly for tensor parallelism.

For example, `MergedColumnParallelLinear.weight_loader(...)` and `QKVParallelLinear.weight_loader(...)` both contain logic like:

```python
packed_dim = getattr(param, "packed_dim", None)
if packed_dim == output_dim:
    shard_size = shard_size // param.pack_factor
    shard_offset = shard_offset // param.pack_factor
```

This is exactly the kind of detail the article is trying to teach.

Because AWQ packs eight 4-bit values into one `int32`, sharding cannot use the same offsets as an unquantized FP16 weight tensor.

The loader has to adjust offsets and sizes for the packed representation.

## 10. How the same forward call dispatches to different math

The runtime forward path stays beautifully simple because of the strategy interface.

In `RowParallelLinear.forward(...)`:

```python
output_parallel = self.linear_method.apply_weights(
    self.linear_weights, input_parallel
)
```

That single call site is polymorphic.

If the model is unquantized, it resolves to:

```python
return F.linear(x, weight, bias)
```

If the model is AWQ, it resolves to:

```python
out = awq_gemm_triton(reshaped_x, qweight, scales, qzeros, pack_factor)
```

So the model code does not branch on "if awq" during every attention or MLP call.

The branching already happened earlier when the model was constructed.

## 11. What `AWQLinearMethod.apply_weights(...)` really does

The AWQ forward path is:

```python
def apply_weights(self, weights, x, bias=None):
    qweight = weights["qweight"]
    qzeros = weights["qzeros"]
    scales = weights["scales"]
    pack_factor = self.quant_config.pack_factor
    out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)
    reshaped_x = x.reshape(-1, x.shape[-1])
    out = awq_gemm_triton(reshaped_x, qweight, scales, qzeros, pack_factor)
    if bias is not None:
        out = out + bias
    return out.reshape(out_shape)
```

This code is compact, but conceptually it means:

1. unpack the AWQ weight representation logically
2. combine packed weights with scales and zero-points
3. run the fused quantized GEMM kernel
4. reshape the result back to the expected output shape

The actual unpacking and dequantization happen inside the Triton kernel path rather than as a separate explicit Python step.

That is the performance-critical point.

## 12. What the Triton kernel wrapper is responsible for

`awq_gemm_triton(...)` in `awq_triton.py` validates shapes and launches the Triton kernel.

Its signature makes the data model clear:

```python
def awq_gemm_triton(
    input,
    qweight,
    scales,
    qzeros,
    split_k_iters,
    block_size_m=32,
    block_size_n=32,
    block_size_k=32,
):
```

The wrapper checks:

- input shape `[M, K]`
- packed weight shape `[K, N // 8]`
- grouped zero-point shape `[K // G, N // 8]`
- grouped scale shape `[K // G, N]`

Then it launches the Triton kernel and reduces over `split_k_iters`.

The important lesson is that the AWQ path avoids a slow two-step process like:

1. fully dequantize to FP16
2. run ordinary `torch.linear`

Instead, it fuses the quantization-aware math into the custom kernel path.

## 13. How this fits into tensor parallelism

The article mostly focuses on quantization, but in this repo quantization is still layered on top of tensor-parallel linear abstractions.

That means:

- packed AWQ tensors are sharded across ranks when needed
- the same row-parallel or column-parallel wrappers still own TP collectives
- quantization changes weight format and GEMM implementation, not the overall TP architecture

That is why the linear wrappers are the right insertion point for quantization strategy.

## 14. Why this is a clean example of strategy pattern

Lesson 6 is one of the clearest examples of strategy pattern in the codebase.

The structure is:

```python
class LinearMethodBase:
    create_weights(...)
    apply_weights(...)

class UnquantizedLinearMethod(LinearMethodBase):
    ...

class AWQLinearMethod(LinearMethodBase):
    ...
```

The rest of the model uses the common interface.

That gives the implementation two big advantages:

1. model architecture code stays mostly unchanged
2. new quantization methods can be added by implementing the same interface

So Lesson 6 is not only about AWQ. It is also about how to structure a runtime so quantization is extensible.

## 15. The full AWQ inference path in one story

Here is the whole path end to end.

### Step 1: read model config

`ModelRunner.load_model()` inspects `hf_config.quantization_config`.

### Step 2: build quantization strategy

`AWQConfig.from_config(...)` parses bits, group size, and zero-point information.

### Step 3: construct the model with `linear_method=AWQLinearMethod`

That method is threaded down into attention and MLP projection layers.

### Step 4: create quantized parameter tensors

Each AWQ-capable linear layer creates `qweight`, `qzeros`, and `scales` instead of a plain `weight` tensor.

### Step 5: load checkpoint tensors

Weight loaders use parameter attributes such as `packed_dim` and `pack_factor` so packed quantized tensors are sharded correctly.

### Step 6: run forward

Linear layers call `self.linear_method.apply_weights(...)`.

### Step 7: dispatch to AWQ kernel

`AWQLinearMethod.apply_weights(...)` calls `awq_gemm_triton(...)`, which launches the fused Triton GEMM path.

That is the complete runtime story for AWQ in this repo.

## 16. What to remember after Lesson 6

If you forget the details, remember these six facts:

1. AWQ detection starts from `quantization_config` in the model config.
2. This repo expresses quantization through `linear_method`, not separate model classes.
3. AWQ linear layers store `qweight`, `qzeros`, and `scales`, not a single FP16 weight matrix.
4. Packed weight loading works because parameters carry shape metadata like `packed_dim` and `pack_factor`.
5. The forward call site in linear layers stays the same; only the strategy implementation changes.
6. The AWQ math path ends in a fused Triton kernel, not in `torch.linear`.

## 17. Best files to reread for this lesson

If you want to trace AWQ directly in code, reread these in this order:

1. `python/sglang/srt/managers/router/model_runner.py`
2. `python/sglang/srt/layers/quantization/base_config.py`
3. `python/sglang/srt/layers/quantization/awq.py`
4. `python/sglang/srt/layers/linear.py`
5. `python/sglang/srt/models/llama2.py`
6. `python/sglang/srt/layers/quantization/awq_triton.py`

That order follows the real control path from model loading to kernel dispatch.