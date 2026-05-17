# Lesson 7: AWQ Kernel Internals

This note summarizes the article at https://gogongxt.com/posts/6393f8f0.html and maps it to the Triton kernel implementation in this repo.

Lesson 6 explained how the runtime detects AWQ models and dispatches linear layers to `AWQLinearMethod`. Lesson 7 zooms into the next layer down:

What actually happens inside the AWQ matrix multiplication kernel?

The answer lives in `python/sglang/srt/layers/quantization/awq_triton.py`.

## 1. The kernel's job in one sentence

The AWQ kernel computes a matrix multiply where:

- activations are ordinary FP16 input matrix `A[M, K]`
- weights are stored in packed int4 form instead of dense FP16
- zero-points and scales are stored separately per quantization group
- output is an ordinary FP16 matrix `C[M, N]`

So the kernel has to do more than GEMM.

It must:

1. load a block of packed weights
2. unpack the 4-bit values into their logical positions
3. look up the matching zero-points and scales
4. dequantize the block
5. multiply it with the activation block

That is the whole purpose of the AWQ kernel.

## 2. The data layout the kernel expects

The wrapper `awq_gemm_triton(...)` documents the expected tensor layout:

```python
# input   - [M, K]
# qweight - [K, N // 8]
# qzeros  - [K // G, N // 8]
# scales  - [K // G, N]
```

This layout is the first thing to internalize.

### Activations

`input` is ordinary dense activation data:

```text
[M, K]
```

### Packed quantized weights

`qweight` is not `[K, N]`.

It is:

```text
[K, N / 8]
```

because each `int32` stores eight 4-bit weights.

### Grouped zero-points and scales

`qzeros` and `scales` are indexed by `K // group_size` rather than the full `K` dimension.

That means the kernel must translate the current K-block into the correct quantization group before it can dequantize the weights.

## 3. The host-side wrapper does three things

The Python wrapper `awq_gemm_triton(...)` is not doing the actual dequantization itself. Its job is to:

1. validate shapes and constraints
2. configure the Triton launch grid
3. allocate temporary output for split-K accumulation

The key setup code is:

```python
M, K = input.shape
N = qweight.shape[1] * 8
group_size = qweight.shape[0] // qzeros.shape[0]

grid = lambda META: (
    triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    split_k_iters,
)

result = torch.zeros((split_k_iters, M, N), dtype=scales.dtype, device=input.device)
```

This already tells you something important:

- `N` is reconstructed from packed storage as `qweight.shape[1] * 8`
- `group_size` is inferred from the ratio between `qweight` and `qzeros`
- split-K parallelism is implemented by storing temporary partial outputs with an extra leading dimension

## 4. How Triton maps work across the output matrix

Inside the kernel, Triton identifies each program instance with:

```python
pid = tl.program_id(axis=0)
pid_z = tl.program_id(1)
```

Then it maps `pid` onto an `(M, N)` tile:

```python
num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
pid_m = pid // num_pid_n
pid_n = pid % num_pid_n
```

So one program instance is responsible for:

- one tile along M
- one tile along N
- one split-K slice along K

That is the kernel's work decomposition.

## 5. The accumulator starts as a dense output tile

The kernel begins with:

```python
accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)
```

This is the local output tile the program will accumulate into across the K loop.

That part is standard block GEMM structure.

The unusual part is how the weight tile is reconstructed before the dot product.

## 6. Why the AWQ reorder pattern exists

The article emphasizes the special AWQ packing order. The code expresses it as:

```python
reverse_awq_order_tensor = (
    (tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]
).reshape(8)

shifts = reverse_awq_order_tensor * 4
shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_K * (BLOCK_SIZE_N // 8), 8))
shifts = tl.reshape(shifts, (BLOCK_SIZE_K, BLOCK_SIZE_N))
```

The resulting logical order is:

```text
[0, 4, 1, 5, 2, 6, 3, 7]
```

This is not arbitrary.

It matches the way AWQ packed the 4-bit weights into each `int32`, so the kernel must use the matching shift pattern to reconstruct the correct logical column order.

If you skip this reorder logic, you would unpack bits into the wrong positions and the matrix multiply would compute nonsense.

## 7. Loading one tile of activations and packed weights

For each K-loop iteration, the kernel loads:

```python
a = tl.load(a_ptrs, mask=masks_a, other=0.0)
b = tl.load(b_ptrs, mask=masks_b, other=0.0)
```

Here:

- `a` is a dense activation tile of shape `[BLOCK_SIZE_M, BLOCK_SIZE_K]`
- `b` is a packed weight tile of shape `[BLOCK_SIZE_K, BLOCK_SIZE_N / 8]`

Then `b` is expanded with repeated interleave operations:

```python
b = tl.interleave(b, b)
b = tl.interleave(b, b)
b = tl.interleave(b, b)
```

This transforms the packed storage into a layout wide enough to apply the per-nibble shift extraction across the logical N dimension.

## 8. How the kernel finds the right scale/zero-point group

The quantization metadata is grouped along K, so the kernel computes the current group index with:

```python
offsets_szk = (
    BLOCK_SIZE_K * SPLIT_K * k + pid_z * BLOCK_SIZE_K
) // group_size + tl.arange(0, 1)
```

This is one of the most important lines in the kernel.

It says:

- locate the current K tile
- determine which quantization group that tile belongs to
- use that group index to load `qzeros` and `scales`

Then the kernel loads both metadata tensors:

```python
zeros = tl.load(zeros_ptrs, mask=masks_z, other=0.0)
scales = tl.load(scales_ptrs, mask=masks_s, other=0.0)
```

and broadcasts them to match the current weight tile.

## 9. The actual unpacking and dequantization math

The core AWQ logic is just a few lines:

```python
b = (b >> shifts) & 0xF
zeros = (zeros >> shifts) & 0xF
b = (b - zeros) * scales
b = b.to(c_ptr.type.element_ty)
```

This is the center of the lesson.

What each line does:

1. `b >> shifts` aligns the desired 4-bit nibble with the low bits
2. `& 0xF` extracts that nibble
3. the same happens for packed zero-points
4. `(b - zeros) * scales` performs dequantization
5. the result is cast to the output/accumulator dtype, usually FP16

So the kernel reconstructs only the current weight tile, only when it is needed.

That is much cheaper than fully materializing a dense FP16 `[K, N]` weight matrix first.

### Small worked example: one packed `int32` becomes eight logical weights

Suppose one packed value inside `qweight` is:

```text
packed = 0x76543210
```

If you read its nibbles in plain low-to-high order, you would get:

```text
[0, 1, 2, 3, 4, 5, 6, 7]
```

But AWQ does not use that plain order.

The kernel builds this shift pattern:

```text
[0, 16, 4, 20, 8, 24, 12, 28]
```

which corresponds to the logical AWQ order:

```text
[0, 4, 1, 5, 2, 6, 3, 7]
```

So the unpack step effectively computes:

```python
packed = 0x76543210
shifts = [0, 16, 4, 20, 8, 24, 12, 28]
values = [(packed >> shift) & 0xF for shift in shifts]
# values == [0, 4, 1, 5, 2, 6, 3, 7]
```

That is why the reorder logic matters.

The kernel is not just extracting eight nibbles.

It is extracting them in the exact order that AWQ used when packing the weight matrix.

Then the same shift pattern is applied to packed `qzeros`, and the kernel finishes with:

```python
dequantized = (values - zeros) * scales
```

So one packed `int32` does not directly mean "columns 0 to 7 in order".

It means "eight 4-bit values that must be reordered before they become the logical dense weight tile."

## 10. Why dequantization is done inside the K loop

The article highlights that dequantization is performed block by block inside the kernel, not as a separate preprocessing pass.

The code confirms that.

This has two big advantages:

1. lower memory traffic because dense dequantized weights never need to be fully stored
2. better locality because unpacking, dequantization, and GEMM happen on the same tile while it is already in fast storage/registers

This is the whole performance point of the AWQ kernel.

## 11. The dot product is still a normal tile GEMM after dequantization

Once the weight tile has been reconstructed into dense FP16 form, the kernel does:

```python
accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)
```

So the unusual AWQ-specific work happens before the dot product.

After that, the kernel is back in ordinary dense matrix multiplication territory for the current tile.

That is a useful mental split:

- AWQ-specific part: unpack and dequantize the weight tile
- generic GEMM part: multiply dense activation tile and dense reconstructed weight tile

## 12. What split-K is doing here

The wrapper allocates:

```python
result = torch.zeros((split_k_iters, M, N), ...)
```

and the kernel writes each partial result to:

```python
c_ptrs = c_ptr + pid_z * N * M + N * offs_cm[:, None] + offs_cn[None, :]
```

Then the wrapper finishes with:

```python
result = result.sum(0)
```

This is split-K reduction.

Instead of one program handling the whole K dimension serially, multiple programs handle different K slices, write partial outputs, and then the host-side wrapper reduces them afterward.

That can improve parallelism when K is large.

## 13. The full kernel flow in one sequence

Here is the complete path in one story.

### Step 1: Python wrapper validates shapes

It checks the packed tensor shapes, supported group size, and split-K constraints.

### Step 2: Triton launch grid is built

The grid tiles M and N and adds an extra split-K axis.

### Step 3: Each kernel instance chooses one `(M, N, K-slice)` tile

`pid_m`, `pid_n`, and `pid_z` determine which tile this program owns.

### Step 4: The kernel loads one activation tile and one packed weight tile

`a` is dense FP16, `b` is packed int4 inside `int32`.

### Step 5: The kernel loads matching grouped zero-points and scales

These come from the quantization group associated with the current K tile.

### Step 6: The kernel unpacks and dequantizes the weight tile

It uses shifts, masks, zero-points, and scales to reconstruct a dense tile.

### Step 7: The kernel performs `tl.dot(a, b)`

This accumulates into the local output tile.

### Step 8: The kernel stores one split-K partial output

Partial outputs go to the temporary `[SPLIT_K, M, N]` buffer.

### Step 9: The wrapper sums across split-K

That produces the final dense `[M, N]` output matrix.

## 14. Why Lesson 7 matters after Lesson 6

Lesson 6 told you that AWQ eventually calls `awq_gemm_triton(...)`.

Lesson 7 explains why that path is necessary.

Without this kernel, the runtime would need a much slower fallback like:

1. unpack all int4 weights into dense FP16
2. apply zero-point and scale globally
3. run ordinary GEMM

That would lose much of the benefit of quantized inference.

So Lesson 7 is the mechanism that makes Lesson 6 practical.

## 15. What to remember after Lesson 7

If you forget the details, remember these six facts:

1. `qweight` stores eight 4-bit weights per `int32`, so the logical N dimension is physically compressed by 8.
2. The kernel restores AWQ's packing order with a fixed shift pattern before extracting nibbles.
3. `qzeros` and `scales` are indexed by quantization group along K.
4. Dequantization is done tile by tile inside the kernel, not as a separate global preprocessing step.
5. After dequantization, the core math is still a standard tile GEMM using `tl.dot(...)`.
6. Split-K parallelism writes temporary partial results and reduces them afterward.

## 16. Best files to reread for this lesson

If you want to trace the AWQ kernel path directly, reread these in this order:

1. `python/sglang/srt/layers/quantization/awq.py`
2. `python/sglang/srt/layers/quantization/awq_triton.py`
3. `python/sglang/srt/layers/linear.py`
4. `python/sglang/srt/managers/router/model_runner.py`

That order follows the real call path from AWQ strategy selection to kernel execution.