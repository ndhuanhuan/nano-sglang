# Lesson 3: Router Scheduling And Inference

This note summarizes the article at https://gogongxt.com/posts/1538c67e.html and maps it to the actual router implementation in nano-sglang.

Lesson 1 explained the process layout. Lesson 2 explained how streamed outputs get back to the client. This lesson focuses on the middle of the system: how the Router decides what can run on the model right now.

## 1. What the Router is responsible for

The Router is the scheduling core of nano-sglang. It sits between the Tokenizer and the model execution path.

Its job is not just to "run inference." It must do four things together:

1. Accept tokenized requests from the Tokenizer process.
2. Convert them into Router-internal request objects.
3. Decide which requests can fit into memory and should run next.
4. Execute inference in two phases: prefill first, then decode.

In the codebase, the main implementation lives in `python/sglang/srt/managers/router/model_rpc.py`.

## 2. The high-level control flow

The article describes Router as a step-based scheduler. In code, the main path is:

```python
exposed_step(recv_reqs)
	-> handle_generate_request(recv_req)
	-> forward_step()
		-> get_new_fill_batch()
		-> forward_fill_batch(...) or forward_decode_batch(...)
		-> handle_finished_requests(...)
```

That path is the core of Router.

You can think of it like this:

- `exposed_step` is the outer driver.
- `handle_generate_request` converts incoming requests into Router state.
- `get_new_fill_batch` decides what new work can be admitted.
- `forward_fill_batch` handles prompt processing.
- `forward_decode_batch` handles next-token generation for running requests.
- `handle_finished_requests` emits outputs and frees memory.

## 3. Entry point: `exposed_step`

The Router receives requests from the Tokenizer in `exposed_step`:

```python
def exposed_step(self, recv_reqs):
	if self.tp_size != 1:
		recv_reqs = obtain(recv_reqs)

	for recv_req in recv_reqs:
		if isinstance(recv_req, TokenizedGenerateReqInput):
			self.handle_generate_request(recv_req)

	self.forward_step()

	ret = self.out_pyobjs
	self.out_pyobjs = []
	return ret
```

Important detail: Router does not immediately run each request one by one. It first converts incoming requests and appends them to internal queues. Then one scheduler step decides what to do next.

That means Router behaves like a small operating system scheduler, not a direct function call from input to output.

## 4. Data structure conversion: external request to internal request

The Tokenizer sends `TokenizedGenerateReqInput`. Router converts it to its internal `Req` type in `handle_generate_request`.

The external form contains things like:

- `rid`
- `input_ids`
- `pixel_values`
- `sampling_params`
- `stream`

The internal `Req` adds scheduler-specific state such as:

- `output_ids`
- `finished`
- `prefix_indices`
- `adjust_input_len`
- `last_node`
- `finish_reason`

Conceptually, the conversion looks like this:

```text
TokenizedGenerateReqInput
	-> Req
	-> Batch
	-> BatchTokenIDOut
```

This is an important design choice: the Router does not want to keep using the public transport object directly. It needs a richer internal object that can store scheduling and cache-related metadata.

## 5. The two Router queues you should remember

Inside `ModelRpcServer`, two containers matter most:

- `forward_queue`: requests waiting to be admitted into execution
- `running_batch`: requests that are already active and are in decode

This gives Router two distinct states for work:

1. Waiting to enter the model.
2. Already occupying model and KV-cache resources.

That separation is why Router can make better decisions than a naive FIFO loop.

## 6. Why inference is split into Prefill and Decode

The article emphasizes that LLM inference has two phases:

1. `Prefill`
   The prompt is processed and the first new token is produced.
2. `Decode`
   The model keeps generating one token at a time for active requests.

In Router, this split appears directly in `forward_step`:

```python
def forward_step(self):
	new_batch = self.get_new_fill_batch()

	if new_batch is not None:
		self.forward_fill_batch(new_batch)
		...
	else:
		if self.running_batch is not None:
			self.forward_decode_batch(self.running_batch)
```

The key policy is: new prefill work gets priority when it can be admitted. Otherwise Router continues decoding the currently running batch.

## 7. `forward_step` is the scheduler heart

`forward_step` decides between three cases:

1. There is a new prefill batch to run.
2. There is no new prefill batch, but there is an active decode batch.
3. There is no runnable work, so Router checks whether KV-cache accounting still looks healthy.

One subtle point in this implementation is that decode can run repeatedly in a short loop:

```python
for _ in range(10):
	self.forward_decode_batch(self.running_batch)
	if self.running_batch.is_empty():
		self.running_batch = None
		break
```

That is a throughput optimization. When there is already active decode work, Router can keep decoding several steps in a row instead of bouncing back through more scheduler overhead every single token.

## 8. Admission control: `get_new_fill_batch`

This is the most important function in the lesson.

`get_new_fill_batch` decides which waiting requests can enter execution right now. It is not enough for a request to merely exist in `forward_queue`. It must also fit within memory and batch constraints.

The function does several things in order.

### Step 1: estimate prefix reuse

For each request, Router asks the radix cache whether part of the prompt has already been computed:

```python
prefix_indices, last_node = self.tree_cache.match_prefix(req.input_ids)
req.adjust_input_len = len(req.input_ids) - len(prefix_indices)
req.prefix_indices = prefix_indices
req.last_node = last_node
```

This means:

- `prefix_indices` tells Router which prefix tokens already have KV-cache entries.
- `adjust_input_len` is how many prompt tokens still need fresh computation.

This is the first major optimization. If a request shares a long prompt prefix with previous work, Router does less new compute.

### Step 2: reorder the queue by scheduling policy

After prefix matching, Router sorts the queue using the scheduler:

```python
self.forward_queue = self.scheduler.get_priority_queue(self.forward_queue)
```

The scheduler supports several heuristics in `python/sglang/srt/managers/router/scheduler.py`:

- `lpm`: longest prefix match first
- `random`
- `fcfs`
- `weight`

The default idea behind `lpm` is simple: requests with more reusable prefix cache should often be more efficient to run first.

### Step 3: estimate available memory

Router computes how much KV-cache space is available:

```python
available_size = (
	self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size()
)
```

If there is already a running batch, Router reduces that estimate further because decode requests will continue to need future tokens:

```python
available_size -= sum(
	(r.max_new_tokens() - len(r.output_ids)) * new_ratio
	for r in self.running_batch.reqs
)
```

This is a conservative reservation mechanism. Router is not only thinking about current tokens; it is also trying to avoid overcommitting future decode capacity.

### Step 4: filter requests that can actually run

Each request must satisfy both:

1. Total token growth must fit available memory.
2. Total fresh prompt work must stay under `max_prefill_num_token`.

Only requests that pass those checks are appended to `can_run_list` and turned into a new `Batch`.

This is why Router is a scheduler, not just a queue consumer.

## 9. Why prefix matching matters so much

If two users send prompts with a shared prefix, the radix cache can reuse previous KV-cache entries. That reduces prompt-side recomputation and makes admission more efficient.

The article highlights this because it changes scheduling behavior in a practical way:

- without prefix reuse, every request looks expensive
- with prefix reuse, some requests become cheaper to admit
- the scheduler can exploit that by prioritizing requests with larger matched prefixes

That is why `lpm` exists and why `prefix_indices` is computed before sorting.

## 10. Prefill execution: `forward_fill_batch`

Once Router has chosen a new batch, `forward_fill_batch` executes the prompt phase.

Its job is:

1. Build the tensors needed for the extend/prefill pass.
2. Run the model once in `ForwardMode.EXTEND`.
3. Sample the first new token.
4. Check whether the request is already finished.

In code:

```python
batch.init_extend_batch(...)
logits, normalized_logprobs = self.model_runner.forward(
	batch, ForwardMode.EXTEND, batch.return_normalized_logprob
)
next_token_ids, next_token_probs = batch.sample(logits)
```

After sampling, Router stores the first generated token into each request's `output_ids` and calls `check_finished()`.

This is the transition point from "new request" to "active generation."

## 11. Decode execution: `forward_decode_batch`

After prefill, unfinished requests stay alive in `running_batch`. Decode then continues one token at a time.

The decode path is simpler:

```python
batch.update_for_decode()
logits = self.model_runner.forward(batch, ForwardMode.DECODE)
next_token_ids, next_token_probs = batch.sample(logits)
```

Then Router appends one token to each active request:

```python
reqs[i].output_ids.append(next_token_ids[i])
reqs[i].check_finished()
```

So the mental difference is:

- prefill: process the remaining prompt and generate the first token
- decode: repeatedly consume the last token and generate the next token

## 12. What a `Batch` really is

The article describes batch lifecycle, and the code in `infer_batch.py` makes that concrete.

A `Batch` is not just a list of requests. It is the execution package Router gives to the model runner. It contains:

- `reqs`
- pooled request-to-token mappings
- KV-cache locations
- GPU tensors such as `input_ids`, `seq_lens`, `prefix_lens`, `temperatures`, `top_ps`, and `top_ks`

So the lifecycle is:

1. Build a `Batch` from admitted `Req`s.
2. Materialize GPU-side tensors.
3. Run model forward.
4. Update request state.
5. Filter out finished requests.
6. Reuse the batch for decode if requests remain.

## 13. How finished requests are emitted

`handle_finished_requests` is where Router packages outputs for the next stage.

It builds a `BatchTokenIDOut` object containing:

- request ids
- generated token ids
- stop-string information
- whether special tokens should be skipped
- metadata such as prompt/completion token counts
- finished flags

That object is appended to `self.out_pyobjs`, which `exposed_step` returns to the caller.

One useful detail: Router can emit outputs either when a request finishes or, for streaming requests, at a configured interval:

```python
if req.finished or (
	req.stream and self.decode_forward_ct % self.stream_interval == 0
):
	...
```

So streaming is not necessarily every single decode step. It is interval-based here.

## 14. How cleanup works

When a request finishes, Router does not just drop it from the batch. It must also clean up cache and memory state correctly.

The cleanup path does three important things:

1. Insert the completed sequence into the radix cache.
2. Free or decrement KV-cache references.
3. Remove the request from the live batch.

This is why `handle_finished_requests` is both an output function and a resource-management function.

If cleanup were wrong, you would either leak memory or lose reusable cache entries.

## 15. The complete mental model

If you only keep one picture in your head, use this one:

```text
Tokenizer sends tokenized requests
	-> Router stores them in forward_queue
	-> Router matches reusable prefixes
	-> Router sorts waiting requests by scheduling policy
	-> Router admits only the requests that fit memory limits
	-> Router runs prefill to generate first tokens
	-> Router keeps unfinished work in running_batch
	-> Router runs decode repeatedly for active requests
	-> Router emits BatchTokenIDOut to Detokenizer
	-> Router frees memory and updates radix cache for finished requests
```

This is the center of the whole inference pipeline.

## 16. Why this design is better than naive FIFO inference

A naive implementation would do something like this:

1. take one request
2. run it immediately
3. finish it completely
4. move to the next request

That would waste batching opportunities, ignore prefix reuse, and make poor use of KV-cache memory.

Router's design is better because it can:

- batch compatible work together
- prioritize requests that reuse more cached prefix state
- limit admission based on memory pressure
- keep active decode requests moving efficiently
- stream partial outputs while preserving batch execution

## 17. Questions you should be able to answer now

After this lesson, you should be able to answer these:

1. Why does Router need both `forward_queue` and `running_batch`?
2. Why is inference split into prefill and decode?
3. Why does Router compute `prefix_indices` before sorting the queue?
4. What does `get_new_fill_batch` decide that a normal queue would not?
5. Why does cleanup have to touch both the radix cache and the memory pool?

If those answers are clear, you understand the scheduler at a useful systems level.

## 18. Summary in one paragraph

This article explains Router as the scheduling core of nano-sglang. Incoming tokenized requests are converted into internal `Req` objects, placed into a waiting queue, checked for prefix-cache reuse, reordered by scheduler policy, filtered by memory and token-budget constraints, and then executed in two stages: prefill for first-token generation and decode for ongoing token generation. Finished or stream-ready outputs are packaged as `BatchTokenIDOut`, while completed requests update the radix cache and release memory. In short, Router is the component that turns queued requests into memory-aware, batched, incremental model execution.