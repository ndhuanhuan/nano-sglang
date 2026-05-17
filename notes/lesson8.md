# Lesson 8: Multimodal Model Loading And Adaptation

This note summarizes the article at https://gogongxt.com/posts/1960588d.html and maps it to the actual multimodal code path in this repo.

In this codebase, multimodal support means one specific thing:

Take a request that contains both text and an image, turn the image into vision embeddings during `EXTEND`, splice those embeddings into the text embedding stream, and then let the normal text-only decode path continue from the KV cache.

The main implementation lives in:

- `python/sglang/srt/managers/tokenizer_manager.py`
- `python/sglang/srt/managers/router/model_rpc.py`
- `python/sglang/srt/managers/router/infer_batch.py`
- `python/sglang/srt/managers/router/model_runner.py`
- `python/sglang/srt/models/llava.py`

## 1. The core idea in one sentence

Multimodal inference adds extra work before the first forward pass, but after the image has been fused into embeddings and written into KV cache, generation goes back to the ordinary text decode path.

That is the highest-level mental model for this lesson.

## 2. Where this repo decides a model is multimodal

The first gate is `is_multimodal_model(...)` in `python/sglang/srt/utils.py`:

```python
def is_multimodal_model(model):
    if isinstance(model, str):
        return "llava" in model
    from sglang.srt.model_config import ModelConfig

    if isinstance(model, ModelConfig):
        return "llava" in model.path.lower()
```

This is intentionally simple.

In this repo, multimodal currently means LLaVA-style models.

That decision is used in both the frontend request path and the backend model execution path.

The second gate is `ModelRunner.load_model()`:

```python
for arch in architectures:
    if arch == "LlavaLlamaForCausalLM":
        model_class = LlavaLlamaForCausalLM
        break
```

So there are two layers of recognition:

1. string/path-based routing to decide whether to use a processor
2. architecture-based routing to decide which model class to instantiate

## 3. The first major fork is `processor` versus `tokenizer`

For plain text models, `TokenizerManager` loads only a tokenizer.

For multimodal models, it loads a Hugging Face processor instead:

```python
if is_multimodal_model(self.model_path):
    self.processor = get_processor(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    self.tokenizer = self.processor.tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    self.executor = concurrent.futures.ProcessPoolExecutor(
        initializer=init_global_processor, initargs=(server_args,)
    )
else:
    self.tokenizer = get_tokenizer(...)
```

This is the first real multimodal divergence in the runtime.

Why it matters:

- text-only models only need tokenization
- multimodal models need tokenization plus image preprocessing
- image preprocessing is pushed into a process pool so it does not block the main event loop

That process-pool detail is important because image decoding and preprocessing are heavier than plain tokenization.

## 4. What image preprocessing actually produces

The helper `get_pixel_values(...)` shows the exact output:

```python
def get_pixel_values(image_data, processor=None):
    image = load_image(image_data)
    image_hash = hash(image_data)
    pixel_values = processor.image_processor(image)["pixel_values"][0]
    pixel_values = pixel_values.astype(np.float16)
    return pixel_values, image_hash
```

So one image input becomes two things:

1. `pixel_values`: the tensor-like numeric image representation used by the vision tower
2. `image_hash`: a cheap ID that this repo later uses to derive placeholder token values

Then `TokenizerManager.generate_request(...)` packages those fields into the transport object:

```python
tokenized_obj = TokenizedGenerateReqInput(
    rid=rid,
    input_ids=input_ids,
    pixel_values=pixel_values,
    image_hash=image_hash,
    sampling_params=sampling_params,
    ...
)
```

So multimodal input is still one request message, but now that message carries both text tokens and preprocessed image data.

## 5. The router expands the image placeholder before scheduling

When the request reaches `ModelRpcServer.handle_generate_request(...)`, the router copies the image fields onto the internal `Req` object:

```python
req = Req(recv_req.rid)
req.input_ids = recv_req.input_ids
req.pixel_values = recv_req.pixel_values
```

If the request actually has an image, the router creates a special placeholder pattern:

```python
if req.pixel_values is not None:
    pad_value = [
        (recv_req.image_hash) % self.model_config.vocab_size,
        (recv_req.image_hash >> 16) % self.model_config.vocab_size,
        (recv_req.image_hash >> 32) % self.model_config.vocab_size,
        (recv_req.image_hash >> 64) % self.model_config.vocab_size,
    ]
    req.input_ids, req.image_offset = self.model_runner.model.pad_input_ids(
        req.input_ids, pad_value
    )
```

This is a useful repo-specific detail.

The article explains the idea as expanding `<image>` into many placeholder positions. In this repo, those placeholders are not a fixed synthetic token like `PAD0`, `PAD1`, and so on.

Instead, the router derives a short repeating pattern from `image_hash` and uses that as filler token IDs.

Conceptually, though, the purpose is the same:

- remove the single `<image>` token
- reserve `image_feature_len` consecutive positions in the sequence
- remember where those positions begin via `image_offset`

## 6. What `pad_input_ids(...)` does in the LLaVA model

The actual expansion logic lives in `python/sglang/srt/models/llava.py`:

```python
def pad_input_ids(self, input_ids, pad_value):
    pad_ids = pad_value * (
        (self.image_feature_len + len(pad_value)) // len(pad_value)
    )
    offset = input_ids.index(self.config.image_token_index)
    new_input_ids = (
        input_ids[:offset]
        + pad_ids[: self.image_feature_len]
        + input_ids[offset + 1 :]
    )
    return new_input_ids, offset
```

This is the exact mechanism behind the article's sequence-expansion diagram.

One image token becomes many placeholder positions, because the image will eventually contribute many patch embeddings rather than a single text embedding.

## 7. The LLaVA model is three components glued together

The model class makes the architecture explicit:

```python
class LlavaLlamaForCausalLM(nn.Module):
    def __init__(self, config, linear_method=None):
        super().__init__()
        self.vision_tower = None
        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.language_model = LlamaForCausalLM(config, linear_method)
```

So the multimodal stack is:

1. a vision encoder
2. a projector that maps vision features into the text hidden dimension
3. a normal language model

That decomposition is the whole design trick.

The language model does not need a brand-new decode loop. It only needs the right embeddings at the right positions.

## 8. Weight loading happens in three stages

`LlavaLlamaForCausalLM.load_weights(...)` shows the initialization sequence:

```python
self.vision_tower = CLIPVisionModel.from_pretrained(
    vision_path, torch_dtype=torch.float16
).cuda()
self.vision_tower.eval()

self.image_size = self.vision_tower.config.image_size
self.patch_size = self.vision_tower.config.patch_size
self.image_feature_len = int((self.image_size / self.patch_size) ** 2)
```

Then it loads projector weights by name remapping and finally calls:

```python
self.language_model.load_weights(...)
```

So the loading order is:

1. load the CLIP vision tower
2. determine how many patch features each image will produce
3. load the multimodal projector
4. load the base language model

That is exactly why `pad_input_ids(...)` can know how many placeholder positions to reserve.

## 9. The batch object carries image tensors only during `EXTEND`

The normal batch builder already has the fields needed for multimodal execution.

Inside `Batch.init_extend_batch(...)`:

```python
self.pixel_values = [r.pixel_values for r in reqs]
self.image_offsets = [
    r.image_offset - p_len for r, p_len in zip(reqs, prefix_lens)
]
```

This is subtle but important.

The model does not only need `pixel_values`. It also needs to know where, inside the uncached suffix that is about to run through `EXTEND`, the image placeholder region begins.

That is what `image_offsets` tracks.

## 10. `ModelRunner.forward(...)` has a multimodal-only branch for `EXTEND`

The execution branch is very narrow:

```python
if self.is_multimodal_model and forward_mode == ForwardMode.EXTEND:
    kwargs = {
        "input_ids": batch.input_ids,
        "pixel_values": batch.pixel_values,
        "image_offsets": batch.image_offsets,
        "req_pool_indices": batch.req_pool_indices,
        "seq_lens": batch.seq_lens,
        "prefix_lens": batch.prefix_lens,
        "out_cache_loc": batch.out_cache_loc,
    }
    return self.forward_extend_multi_modal(**kwargs)
```

That means multimodal logic is not spread through every execution path.

Only `EXTEND` is special.

That matches the article's main point: fuse modalities once, then reuse the ordinary decode machinery.

## 11. The real fusion happens inside `LlavaLlamaForCausalLM.forward(...)`

In `EXTEND` mode, the model first builds ordinary text embeddings:

```python
input_embeds = self.language_model.model.embed_tokens(input_ids)
```

Then it decides which requests in the batch still need image work:

```python
need_vision = (
    (positions[input_metadata.extend_start_loc] < self.image_feature_len)
    .cpu()
    .numpy()
)
has_pixel = np.array([pixel_values[i] is not None for i in range(bs)])
need_vision = need_vision & has_pixel
```

If vision work is needed, it runs the vision tower and projector:

```python
image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
selected_image_feature = image_outputs.hidden_states[self.vision_feature_layer]
image_features = self.multi_modal_projector(selected_image_feature)
```

Then comes the key fusion step:

```python
input_embeds[
    start_idx + image_offsets[i] : start_idx + image_offsets[i] + pad_len
] = image_features[pt]
```

This is the single most important line in the whole lesson.

The model is literally overwriting the placeholder token embeddings with projected vision features.

After that, it simply calls the language model:

```python
return self.language_model(
    input_embeds, positions, input_metadata, skip_embed=True
)
```

That is why the design feels elegant.

The language model still just consumes an embedding sequence. It does not need to know whether those embeddings came from text or images.

## 12. `DECODE` mode goes back to the plain text path

The decode branch is tiny:

```python
elif input_metadata.forward_mode == ForwardMode.DECODE:
    return self.language_model(
        input_ids, positions, input_metadata, skip_embed=False
    )
```

This is the exact code reason the article says:

Image handling is an `EXTEND` concern, not a `DECODE` concern.

By the time decode starts, the multimodal prompt has already been turned into KV cache entries.

So later token generation can reuse the normal language-model path.

## 13. A concrete request example

Suppose the user sends:

```text
USER: <image> What's this?
```

The runtime roughly transforms it like this:

### Step 1: text tokenization

```text
[USER, :, <image>, What, 's, this, ?]
```

### Step 2: image preprocessing

```text
pixel_values = processor.image_processor(image)["pixel_values"][0]
image_hash = hash(image_data)
```

### Step 3: image token expansion

```text
[USER, :, H0, H1, H2, H3, H0, H1, ..., What, 's, this, ?]
```

Here `H0..H3` stands for the repeating hash-derived placeholder pattern. The real repo computes them from `image_hash` modulo `vocab_size`.

### One worked example with a short placeholder span

The real model may reserve hundreds of positions, but the mechanism is easier to see with a tiny made-up example.

Suppose:

- the original token sequence is

```text
[USER, :, <image>, What, 's, this, ?]
```

- `image_hash` produces a repeating placeholder pattern

```text
pad_value = [H0, H1, H2, H3]
```

- `image_feature_len = 6`

Then `pad_input_ids(...)` turns the single `<image>` token into six placeholder positions:

```text
before: [USER, :, <image>, What, 's, this, ?]
after:  [USER, :, H0, H1, H2, H3, H0, H1, What, 's, this, ?]
                 ^^^^^^^^^^^^^^^^^^
                 6 reserved positions
```

In other words, the runtime removes one logical image marker and expands it into a span large enough to hold all projected vision features for that image.

Now the model computes ordinary text embeddings first:

```text
[E(USER), E(:), E(H0), E(H1), E(H2), E(H3), E(H0), E(H1), E(What), ...]
```

Then the vision tower and projector produce six image feature vectors:

```text
[V0, V1, V2, V3, V4, V5]
```

The fusion step overwrites exactly the placeholder span:

```text
before fusion:
[E(USER), E(:), E(H0), E(H1), E(H2), E(H3), E(H0), E(H1), E(What), ...]

after fusion:
[E(USER), E(:), V0, V1, V2, V3, V4, V5, E(What), ...]
```

That is the most concrete way to think about multimodal `EXTEND` in this repo.

The model is not appending image features to the end of the prompt.

It is replacing a reserved placeholder region inside the embedding sequence.

### Step 4: `EXTEND` embedding fusion

The model builds text embeddings for the whole expanded sequence, then replaces the placeholder region with projected CLIP features.

### Step 5: normal autoregressive decode

After the first multimodal forward pass, later generation steps are ordinary text decode steps.

That is the whole flow.

## 14. What "support a new multimodal model" means in this repo

The article gives a general checklist. In this repo, the most concrete code changes would be:

1. extend `is_multimodal_model(...)` so the runtime recognizes the new model family
2. add architecture selection in `ModelRunner.load_model()`
3. implement a model class like `LlavaLlamaForCausalLM`
4. make sure the model class provides `pad_input_ids(...)`, `load_weights(...)`, and multimodal `forward(...)`
5. ensure `AutoProcessor` or an equivalent processor can produce the needed `pixel_values`

Notice what does not need a full rewrite:

- the FastAPI handler
- the async streaming path
- the router's overall `EXTEND` versus `DECODE` scheduler structure

Those pieces are already generic enough.

The repo-specific control points are model recognition, placeholder expansion, and embedding fusion.

## 15. Why Lesson 8 matters after the earlier lessons

Lesson 8 is easier to understand if you connect it to the first seven lessons:

- Lesson 1: the process boundaries do not change
- Lesson 2: the streaming mechanism does not change
- Lesson 3: the router still schedules `EXTEND` and `DECODE`
- Lesson 4: KV-cache allocation still works the same way
- Lesson 5: multimodal prompt work still lands in the prefill/extend budget
- Lesson 6 and 7: the language-model linear layers can still be quantized underneath all of this

So multimodal support is not a separate inference engine.

It is a carefully placed extension to the existing inference engine.

## 16. What to remember after Lesson 8

If you forget the details, remember these six facts:

1. This repo currently treats LLaVA-style models as the multimodal path.
2. Multimodal requests use a processor and image preprocessing, not just a tokenizer.
3. The router expands one image token into many placeholder positions before scheduling.
4. `EXTEND` replaces those placeholder embeddings with projected vision features.
5. `DECODE` then reuses the ordinary text-only language-model path.
6. Supporting a new multimodal model mostly means implementing recognition, placeholder expansion, weight loading, and fusion logic at the model boundary.

## 17. Best files to reread for this lesson

If you want to trace the multimodal path directly, reread these in this order:

1. `python/sglang/srt/managers/tokenizer_manager.py`
2. `python/sglang/srt/managers/router/model_rpc.py`
3. `python/sglang/srt/managers/router/infer_batch.py`
4. `python/sglang/srt/managers/router/model_runner.py`
5. `python/sglang/srt/models/llava.py`
6. `python/sglang/srt/hf_transformers_utils.py`

That order follows the exact path of a multimodal request from HTTP-side preprocessing to the final embedding fusion inside the model.