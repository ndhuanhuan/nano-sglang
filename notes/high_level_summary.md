# How Inference Works In Nano SGLang

This note is the "connect the dots" version of Lessons 1 to 3.

If the first three lessons felt disconnected, the simplest way to recover is to stop thinking about them as three separate topics:

- Lesson 1 explains where each runtime role lives.
- Lesson 2 explains how results get back to the client without blocking.
- Lesson 3 explains how the router decides what the GPU should do next.

They are all describing one request pipeline.

## The one mental model to keep in your head

When a user sends a prompt, nano-sglang does this:

1. FastAPI receives the HTTP request.
2. `TokenizerManager` turns text into token IDs.
3. The router converts that request into internal scheduler state.
4. The router chooses between `EXTEND` and `DECODE` work.
5. `ModelRunner` executes the actual GPU forward pass.
6. The router sends generated token IDs to the detokenizer.
7. The detokenizer turns token IDs into text.
8. `TokenizerManager` wakes the waiting HTTP coroutine and streams the text back.

So the full inference loop is:

```text
HTTP -> TokenizerManager -> Router -> ModelRunner -> Detokenizer -> TokenizerManager -> HTTP stream
```

That is the runtime.

## 1. Lesson 1 in one picture: who owns what

The entrypoint is `python/sglang/srt/server.py`.

This snippet is the first thing to internalize:

```python
def launch_server(server_args):
	can_use_ports = alloc_usable_network_port(
		num=4 + server_args.tp_size, used_list=(server_args.port,)
	)
	port_args = PortArgs(
		tokenizer_port=can_use_ports[0],
		router_port=can_use_ports[1],
		detokenizer_port=can_use_ports[2],
		nccl_port=can_use_ports[3],
		model_rpc_ports=can_use_ports[4:],
	)

	tokenizer_manager = TokenizerManager(server_args, port_args)

	proc_router = mp.Process(
		target=start_router_process,
		args=(server_args, port_args, pipe_router_writer),
	)
	proc_detoken = mp.Process(
		target=start_detokenizer_process,
		args=(server_args, port_args, pipe_detoken_writer),
	)
```

This tells you the runtime is not a single monolithic server.

It is split into:

- main process: FastAPI + `TokenizerManager`
- router process: scheduling + model stepping
- detokenizer process: token IDs back to strings
- optional model RPC workers when `tp_size > 1`

That is why Lesson 1 focused so much on processes and ports. It was not setup trivia. It was telling you the boundaries of the inference pipeline.

## 2. The HTTP handler is thin on purpose

The `/generate` endpoint does almost nothing by itself:

```python
@app.post("/generate")
async def generate_request(obj: GenerateReqInput):
	obj.post_init()
	result_generator = tokenizer_manager.generate_request(obj)

	if obj.stream:
		async def stream_results():
			async for out in result_generator:
				yield (json.dumps(out) + "\0").encode("utf-8")

		return StreamingResponse(stream_results(), media_type="text/event-stream")
	else:
		ret = await result_generator.__anext__()
		return ret
```

Important point: the FastAPI layer does not run inference.

It just:

- normalizes the input
- asks `TokenizerManager` for an async generator
- either streams chunks from it or waits for one final result

So if you want to understand inference, do not stare at `server.py` for too long. The real behavior starts in `TokenizerManager`.

## 3. Lesson 2 in code: TokenizerManager is the bridge between HTTP and the runtime

`TokenizerManager.generate_request(...)` is where user input becomes runtime work.

This is the key part:

```python
async def generate_request(self, obj: GenerateReqInput):
	if self.to_create_loop:
		await self.create_handle_loop()

	rid = obj.rid
	input_ids = self.tokenizer.encode(obj.text)
	sampling_params = SamplingParams(**obj.sampling_params)

	tokenized_obj = TokenizedGenerateReqInput(
		rid=rid,
		input_ids=input_ids,
		pixel_values=pixel_values,
		image_hash=image_hash,
		sampling_params=sampling_params,
		return_normalized_logprob=obj.return_normalized_logprob,
		normalized_logprob_start_len=obj.normalized_logprob_start_len,
		stream=obj.stream,
	)
	self.send_to_router.send_pyobj(tokenized_obj)

	event = asyncio.Event()
	state = ReqState([], False, event, lock)
	self.rid_to_state[rid] = state

	while True:
		await event.wait()
		yield state.out_list[-1]
		state.out_list = []
		if state.finished:
			del self.rid_to_state[rid]
			break
		event.clear()
```

This one snippet explains most of Lesson 2.

What changes here?

- text becomes `input_ids`
- request parameters become `SamplingParams`
- user-facing input becomes `TokenizedGenerateReqInput`
- the request gets a per-request wait point: `ReqState`
- the caller then suspends on `await event.wait()`

That last part is the streaming trick.

The HTTP coroutine is not polling the model. It is sleeping until some other part of the system says, "new text for request `rid` is ready."

More concretely, `await event.wait()` means:

- this request coroutine pauses immediately
- the asyncio event loop is free to run other tasks
- this coroutine will not continue until somebody calls `event.set()` for this request

So the request path does not look like this:

```python
while not finished:
	check_model_status()
```

That would be polling.

Instead, it looks like this:

```python
send_request_to_router(rid)
await event.wait()
yield newest_text_chunk
```

That is why I called it "sleeping." The coroutine is suspended at `await event.wait()` and does no work until it is woken up.

The wake-up path is:

1. Router generates token IDs.
2. Detokenizer turns token IDs into text.
3. `TokenizerManager.handle_loop()` receives that text.
4. It looks up `rid` in `rid_to_state`.
5. It appends the new output chunk.
6. It calls `state.event.set()`.
7. The sleeping request coroutine resumes and yields the chunk to FastAPI.

Here is the same idea as a small sequence diagram:

```text
HTTP request coroutine                 background handle_loop
----------------------                ----------------------
send tokenized request  ----------->  runtime keeps working elsewhere
await event.wait()      -- sleep -->  recv text for rid from detokenizer
									  rid_to_state[rid].out_list.append(...)
									  rid_to_state[rid].event.set()
resume after wait       <----------   event wakes this coroutine
yield text chunk        ----------->  StreamingResponse sends bytes to client
```

So the important distinction is:

- the router side is actively advancing inference
- the HTTP coroutine is passively waiting for a signal

Only the waiting HTTP coroutine is "not polling." The overall system is still making progress in the background.

### If `asyncio.Event` still feels abstract, use this mental model

Think of `asyncio.Event` as a doorbell, not a mailbox.

- `state.out_list` stores the actual text chunk
- `state.event` only answers: should this coroutine wake up now?

So `Event` does not carry the result itself. It only coordinates when the waiting coroutine should continue.

The pattern in this file is:

```python
state.out_list.append(out_dict)
state.event.set()
```

That means:

1. put the new result somewhere the request can read it
2. ring the bell so the waiting coroutine wakes up

Then the request coroutine does:

```python
await event.wait()
yield state.out_list[-1]
event.clear()
```

That means:

1. sleep until the bell rings
2. read the newest result
3. reset the bell for the next chunk

### What `await` actually does here

The subtle point is that `await` does not freeze the whole Python process.

It only pauses this one coroutine.

So when execution reaches:

```python
await event.wait()
```

the event loop can still run:

- `TokenizerManager.handle_loop()`
- other HTTP requests
- socket I/O
- other background async tasks

That is why the server can keep working while this request is waiting for more tokens.

### Exact timeline for one request

```text
1. generate_request(rid) sends the tokenized request to the router
2. generate_request(rid) reaches await event.wait() and pauses
3. the event loop runs other work while this coroutine is paused
4. router and detokenizer finish another chunk for rid
5. handle_loop() receives BatchStrOut for rid
6. handle_loop() writes the text into rid_to_state[rid].out_list
7. handle_loop() calls rid_to_state[rid].event.set()
8. the event loop marks generate_request(rid) runnable again
9. generate_request(rid) resumes after await event.wait()
10. generate_request(rid) yields the chunk to StreamingResponse
```

The key phrase is "marks runnable again."

`event.set()` does not directly jump into the other coroutine. It tells the event loop that this waiting coroutine is now allowed to continue on the next scheduling turn.

### Why `rid_to_state` matters so much

Each request gets its own state object:

- its own output buffer
- its own `Event`
- its own finished flag

That is why the code uses `rid_to_state[rid]`.

Without that mapping, the runtime would not know which sleeping request coroutine should wake up when a particular output chunk arrives.

## 4. The return path is another loop, not a callback

The background producer is `TokenizerManager.handle_loop()`:

```python
async def handle_loop(self):
	while True:
		recv_obj = await self.recv_from_detokenizer.recv_pyobj()

		if isinstance(recv_obj, BatchStrOut):
			for i, rid in enumerate(recv_obj.rids):
				out_dict = {
					"text": recv_obj.output_str[i],
					"meta_info": recv_obj.meta_info[i],
				}
				state = self.rid_to_state[rid]
				state.out_list.append(out_dict)
				state.finished = recv_obj.finished[i]
				state.event.set()
```

This is the other half of the Lesson 2 design.

- `generate_request(...)` is the consumer loop for one request.
- `handle_loop()` is the producer loop for all requests.

The shared rendezvous point is `rid_to_state`.

That is why Lesson 2 kept repeating the producer-consumer idea. It is the exact mechanism that converts background inference progress into streaming HTTP output.

## 5. The router is the scheduling brain

The router process has two loops in `python/sglang/srt/managers/router/manager.py`:

```python
async def loop_for_recv_requests(self):
	while True:
		recv_req = await self.recv_from_tokenizer.recv_pyobj()
		self.recv_reqs.append(recv_req)

async def loop_for_forward(self):
	while True:
		next_step_input = list(self.recv_reqs)
		self.recv_reqs = []
		out_pyobjs = await self.model_client.step(next_step_input)

		for obj in out_pyobjs:
			self.send_to_detokenizer.send_pyobj(obj)
```

This is the control handoff:

- receive loop: collect newly tokenized requests
- forward loop: ask the model side to advance inference by one scheduler step

The router is not doing string decoding and it is not serving HTTP. It is the middle coordinator that keeps the model busy and forwards generated token IDs downstream.

## 6. Lesson 3 starts here: external request becomes internal scheduler state

Inside `ModelRpcServer.exposed_step(...)`, the router converts transport objects into internal request objects:

```python
def exposed_step(self, recv_reqs):
	for recv_req in recv_reqs:
		if isinstance(recv_req, TokenizedGenerateReqInput):
			self.handle_generate_request(recv_req)

	self.forward_step()

	ret = self.out_pyobjs
	self.out_pyobjs = []
	return ret
```

Then `handle_generate_request(...)` turns the message into a `Req`:

```python
def handle_generate_request(self, recv_req):
	req = Req(recv_req.rid)
	req.input_ids = recv_req.input_ids
	req.pixel_values = recv_req.pixel_values
	req.sampling_params = recv_req.sampling_params
	req.return_normalized_logprob = recv_req.return_normalized_logprob
	req.normalized_logprob_start_len = recv_req.normalized_logprob_start_len
	req.stream = recv_req.stream
	req.tokenizer = self.tokenizer

	req.input_ids = req.input_ids[: self.model_config.context_len - 1]
	req.sampling_params.max_new_tokens = min(
		req.sampling_params.max_new_tokens,
		self.model_config.context_len - 1 - len(req.input_ids),
	)
	self.forward_queue.append(req)
```

This is an important boundary.

`TokenizedGenerateReqInput` is the message sent between processes.

`Req` is the router's internal working object.

That is where scheduling metadata starts to live, such as:

- `output_ids`
- `prefix_indices`
- `finished`
- `finish_reason`
- regex FSM state for constrained decoding

## 7. The most important function: `forward_step()`

If you only memorize one scheduler function, memorize this one:

```python
def forward_step(self):
	new_batch = self.get_new_fill_batch()

	if new_batch is not None:
		self.forward_fill_batch(new_batch)

		if not new_batch.is_empty():
			if self.running_batch is None:
				self.running_batch = new_batch
			else:
				self.running_batch.merge(new_batch)
	else:
		if self.running_batch is not None:
			for _ in range(10):
				self.forward_decode_batch(self.running_batch)
				if self.running_batch.is_empty():
					self.running_batch = None
					break
```

This is Lesson 3 in one block.

The router is always deciding between two kinds of work:

- `EXTEND` work for new requests entering the system
- `DECODE` work for requests already generating tokens

That is the real meaning of the prefill/decode split.

## 8. Why prefix cache and scheduling are discussed together

`get_new_fill_batch()` is where the router estimates how expensive each waiting request is.

```python
for req in self.forward_queue:
	prefix_indices, last_node = self.tree_cache.match_prefix(req.input_ids)
	req.adjust_input_len = len(req.input_ids) - len(prefix_indices)
	req.prefix_indices = prefix_indices
	req.last_node = last_node

self.forward_queue = self.scheduler.get_priority_queue(self.forward_queue)

available_size = (
	self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size()
)
```

This is why Lesson 3 spent so much time on the radix cache and scheduler.

The router does not just ask, "which request came first?"

It also asks:

- how much prefix can I reuse?
- how many fresh prompt tokens do I need to compute?
- how much KV space is still available?
- which requests are cheapest or best to admit right now?

The scheduler implementation is tiny, but the policy matters:

```python
def get_priority_queue(self, forward_queue):
	if self.schedule_heuristic == "lpm":
		forward_queue.sort(key=lambda x: -len(x.prefix_indices))
		return forward_queue
```

`lpm` means longest-prefix-match first.

So cache reuse is not just a memory optimization. It directly influences admission order.

## 9. What `EXTEND` really does

For a new batch, the router builds the prompt-side tensors and runs one forward pass:

```python
def forward_fill_batch(self, batch: Batch):
	batch.init_extend_batch(self.model_config.vocab_size, self.int_token_logit_bias)

	logits, normalized_logprobs = self.model_runner.forward(
		batch, ForwardMode.EXTEND, batch.return_normalized_logprob
	)

	next_token_ids, next_token_probs = batch.sample(logits)

	for i in range(len(batch.reqs)):
		batch.reqs[i].output_ids = [next_token_ids[i]]
		batch.reqs[i].check_finished()
```

In plain English:

- build the uncached prompt portion into a batch
- run the model in `EXTEND` mode
- sample the first generated token
- initialize each request's `output_ids`

This is why I prefer the word `extend` over `prefill` when reading this repo. The code is literally extending the already-cached prefix with the uncached suffix.

The batch builder in `infer_batch.py` makes this explicit:

```python
input_ids = [r.input_ids[len(r.prefix_indices):] for r in reqs]
prefix_indices = [r.prefix_indices for r in reqs]

extend_num_tokens = seq_lens.sum() - prefix_lens.sum()
out_cache_loc = self.token_to_kv_pool.alloc(extend_num_tokens)
```

That is the concrete meaning of prefix reuse.

The batch only allocates fresh KV slots for tokens that are not already covered by `prefix_indices`.

## 10. What `DECODE` really does

Once a request has already produced at least one token, the router moves into decode mode:

```python
def forward_decode_batch(self, batch: Batch):
	self.decode_forward_ct += 1
	batch.update_for_decode()

	logits = self.model_runner.forward(batch, ForwardMode.DECODE)
	next_token_ids, next_token_probs = batch.sample(logits)

	for i in range(len(batch.reqs)):
		batch.reqs[i].output_ids.append(next_token_ids[i])
		batch.reqs[i].check_finished()
```

And `update_for_decode()` shows the key difference from `EXTEND`:

```python
def update_for_decode(self, input_ids=None):
	if input_ids is None:
		input_ids = [
			r.output_ids[-1] if r.output_ids else r.input_ids[-1] for r in self.reqs
		]
	self.input_ids = torch.tensor(input_ids, dtype=torch.int32, device="cuda")
	self.seq_lens.add_(1)
```

In decode mode, the model no longer reprocesses the whole prompt.

It feeds the last token, advances sequence length, allocates one more KV slot, and predicts one more token.

That is the central efficiency idea behind autoregressive serving.

## 11. Where the real GPU forward call happens

`ModelRunner` is the closest thing to the raw model execution layer in these lessons:

```python
def forward(self, batch: Batch, forward_mode: ForwardMode, return_normalized_logprob=False):
	kwargs = {
		"input_ids": batch.input_ids,
		"req_pool_indices": batch.req_pool_indices,
		"seq_lens": batch.seq_lens,
		"prefix_lens": batch.prefix_lens,
		"out_cache_loc": batch.out_cache_loc,
	}

	if forward_mode == ForwardMode.DECODE:
		kwargs["out_cache_cont_start"] = batch.out_cache_cont_start
		kwargs["out_cache_cont_end"] = batch.out_cache_cont_end
		return self.forward_decode(**kwargs)
	elif forward_mode == ForwardMode.EXTEND:
		kwargs["return_normalized_logprob"] = return_normalized_logprob
		return self.forward_extend(**kwargs)
```

This is the moment where scheduler state becomes actual tensors for the model.

Everything before this prepares work.

Everything after this interprets the result.

## 12. How tokens become streamed text again

After each scheduler step, the router decides whether a request should emit output now:

```python
if req.finished or (
	req.stream and self.decode_forward_ct % self.stream_interval == 0
):
	output_rids.append(req.rid)
	output_tokens.append(req.output_ids)
	output_finished.append(req.finished)
```

Those token IDs are wrapped into `BatchTokenIDOut` and sent to the detokenizer.

Then `DetokenizerManager` converts them into strings:

```python
if isinstance(recv_obj, BatchTokenIDOut):
	output_strs = self.tokenizer.batch_decode(
		output_tokens,
		skip_special_tokens=recv_obj.skip_special_tokens[0],
	)

	self.send_to_tokenizer.send_pyobj(
		BatchStrOut(
			recv_obj.rids,
			output_strs,
			recv_obj.meta_info,
			recv_obj.finished,
		)
	)
```

That `BatchStrOut` goes back to `TokenizerManager.handle_loop()`, which calls `state.event.set()`, which wakes the waiting `generate_request(...)`, which yields a chunk to `StreamingResponse`.

That closes the loop.

## 13. The full request lifecycle, now as one connected story

Here is the same path again, but this time tied directly to code behavior.

### Step 1: FastAPI receives a request

`server.py` normalizes the input and calls `tokenizer_manager.generate_request(obj)`.

### Step 2: TokenizerManager turns text into runtime input

`TokenizerManager.generate_request(...)`:

- tokenizes the text
- builds `SamplingParams`
- creates `TokenizedGenerateReqInput`
- sends it to the router with ZeroMQ
- registers `ReqState` in `rid_to_state`
- waits on `event`

### Step 3: Router receives the tokenized request

`RouterManager.loop_for_recv_requests()` appends it into `recv_reqs`.

### Step 4: Router advances the model by one step

`RouterManager.loop_for_forward()` calls `model_client.step(next_step_input)`.

### Step 5: ModelRpcServer converts transport objects into internal `Req` objects

`handle_generate_request(...)` truncates prompt length if needed, stores sampling config, and pushes the request into `forward_queue`.

### Step 6: Scheduler decides between extend and decode

`forward_step()`:

- tries to create a new fill batch
- if successful, runs `EXTEND`
- otherwise continues `DECODE` on `running_batch`

### Step 7: Prefix cache reduces repeated prompt work

`get_new_fill_batch()` uses `tree_cache.match_prefix(req.input_ids)` to find reusable KV cache.

### Step 8: ModelRunner executes the forward pass

`ModelRunner.forward(...)` dispatches to `forward_extend(...)` or `forward_decode(...)`.

### Step 9: Router samples tokens and checks finish conditions

`Req.check_finished()` stops on:

- max token length
- EOS token
- stop string

### Step 10: Router emits token IDs for streaming or final output

`handle_finished_requests(...)` builds `BatchTokenIDOut`.

### Step 11: Detokenizer turns token IDs into strings

`DetokenizerManager.handle_loop()` calls `batch_decode(...)` and sends `BatchStrOut` back.

### Step 12: TokenizerManager wakes the waiting request coroutine

`handle_loop()` writes into `rid_to_state[rid]` and calls `state.event.set()`.

### Step 13: FastAPI streams the text chunk to the client

The async generator yields the newest output chunk, and `StreamingResponse` sends it immediately.

## 14. Why the first three lessons were ordered this way

The lesson order is actually good. It only feels confusing if you do not yet see the single pipeline.

The dependency chain is:

1. First understand the process boundaries.
2. Then understand how one request waits for results without blocking.
3. Then understand how the router decides what the model should run next.

If you reverse that order, the router code looks much more abstract than it really is.

## 15. What to remember after reading this note

If you forget details, keep these five facts:

1. Inference is a pipeline, not a single function call.
2. `TokenizerManager` is the bridge between HTTP and runtime messages.
3. The router's core decision is always `EXTEND` versus `DECODE`.
4. Prefix caching affects both memory reuse and scheduling priority.
5. Streaming works because one coroutine waits on `asyncio.Event` while another coroutine fills results and signals it.

## 16. Best next files to reread

If you want to revisit the first three lessons with much less confusion, reread these in this order:

1. `python/sglang/srt/server.py`
2. `python/sglang/srt/managers/tokenizer_manager.py`
3. `python/sglang/srt/managers/router/manager.py`
4. `python/sglang/srt/managers/router/model_rpc.py`
5. `python/sglang/srt/managers/router/infer_batch.py`
6. `python/sglang/srt/managers/detokenizer_manager.py`

That order follows the same direction as an actual inference request.

## 17. Lesson 4 in one pass: memory pools make prefix reuse real

Lesson 4 adds the missing memory model behind Lesson 3.

The scheduler is not just choosing requests. It is also reasoning about whether the KV-cache system can represent those requests safely and efficiently.

The runtime uses three cooperating structures:

1. `RadixCache`: remembers reusable prompt prefixes.
2. `ReqToTokenPool`: maps each live request row to the KV-slot index of each token position.
3. `TokenToKVPool`: owns the actual KV tensors on GPU and tracks slot reference counts.

The pools are created in `ModelRunner.init_memory_pool(...)`:

```python
self.req_to_token_pool = ReqToTokenPool(...)
self.token_to_kv_pool = TokenToKVPool(...)
```

The flow for one request is:

### Step 1: match cached prefix

The router asks the radix tree how much of the prompt is already cached:

```python
prefix_indices, last_node = self.tree_cache.match_prefix(req.input_ids)
req.adjust_input_len = len(req.input_ids) - len(prefix_indices)
req.prefix_indices = prefix_indices
req.last_node = last_node
```

So `prefix_indices` is the concrete list of reusable KV slots for that prompt prefix.

### Step 2: extend allocates only the uncached suffix

During prefill/extend, the batch builder reuses the cached prefix and allocates new slots only for the remaining prompt suffix:

```python
input_ids = [r.input_ids[len(r.prefix_indices):] for r in reqs]
req_pool_indices = self.req_to_token_pool.alloc(bs)
out_cache_loc = self.token_to_kv_pool.alloc(extend_num_tokens)
```

That is the whole trick: shared prefix stays shared, only new tokens consume new KV memory.

### Step 3: decode appends one KV slot per token step

For active requests, decode grows the sequence incrementally:

```python
self.seq_lens.add_(1)
self.out_cache_loc = self.token_to_kv_pool.alloc(bs)
self.req_to_token_pool.req_to_token[
	self.req_pool_indices, self.seq_lens - 1
] = self.out_cache_loc
```

So decode is just "add one more token slot per live request."

### Step 4: finishing a request does not blindly delete the cache

When a request finishes, the runtime inserts its token sequence into the radix cache, frees the request row, and decreases reference counts:

```python
prefix_len = self.tree_cache.insert(token_ids[:seq_len], indices.clone())
self.token_to_kv_pool.free(indices[:prefix_len])
self.req_to_token_pool.free(req_pool_idx)
self.tree_cache.dec_ref_counter(req.last_node)
```

This is the key Lesson 4 idea.

Completion means "drop request-specific ownership" rather than "erase all KV memory immediately."

### Step 5: eviction reclaims only unused cache

If new allocation fails, the runtime asks the radix tree to evict old leaves with ref count 0:

```python
out_cache_loc = self.token_to_kv_pool.alloc(extend_num_tokens)
if out_cache_loc is None:
	self.tree_cache.evict(extend_num_tokens, self.token_to_kv_pool.free)
```

So eviction is guided by cache structure and reference counts, not by blindly scanning tensors.

### A concrete example: `Write a poem about cats`

Suppose a previous request already cached the prefix `Write a poem about`, and that prefix maps to KV slots `[101, 102, 103, 104]`.

Now a new prompt arrives:

```text
Write a poem about cats
-> [11, 12, 13, 14, 15]   # illustrative token IDs only
```

Then the three structures cooperate like this:

1. `RadixCache.match_prefix(...)` returns the reusable prefix slots `[101, 102, 103, 104]`.
2. `ReqToTokenPool` allocates one request row, for example row `7`.
3. `TokenToKVPool` allocates only one new slot for the uncached token `cats`, for example slot `220`.

So the logical request row becomes:

```text
row 7 = [101, 102, 103, 104, 220]
```

Then decode appends more slots, one token at a time, such as `221`, `222`, and so on.

The important point is that the shared prefix is reused, not duplicated. Only the uncached suffix and later generated tokens consume new KV memory.

## 18. What to remember from Lesson 4

If you want the shortest review version, remember these five points:

1. `RadixCache` tells the router what prefix KV memory can be reused.
2. `ReqToTokenPool` is the per-request index table.
3. `TokenToKVPool` is the actual GPU KV memory with reference counts.
4. Extend allocates only uncached prompt tokens; decode allocates one new token at a time.
5. Finished requests free their own ownership, but reusable cache can stay alive until eviction is needed.

## 19. Lesson 5 in one pass: chunked prefill splits long prompts into smaller forward passes

Lesson 5 is about a limitation of the earlier design.

Even if the KV-cache accounting is correct, a very long prefill can still be expensive because the prompt-side forward pass has large activation memory cost. Chunked prefill solves that by processing one long prompt in multiple smaller prefill rounds instead of one huge one.

The article explains two related controls in full sglang:

1. a total per-forward prefill budget
2. a per-request chunk size

In the article's minisglang implementation, the key idea is simplified into one budget-like control, `max_extend_tokens`, and the scheduler may return either:

- a normal request that finishes prefill this round
- a chunked request that still has prompt tokens left for a later prefill round

The important behavioral change is this:

- without chunked prefill, a request is admitted whole or not at all
- with chunked prefill, one long request can make partial progress and come back next round

This repo does not yet implement that full chunked-prefill mechanism.

What it has today is a batch-wide prefill cap:

```python
self.max_prefill_num_token = max(
	self.model_config.context_len, self.max_total_num_token // 6
)
```

and an admission check in `get_new_fill_batch()`:

```python
if (
	req.adjust_input_len + req.max_new_tokens() + new_batch_total_tokens
	< available_size
	and req.adjust_input_len + new_batch_input_tokens
	< self.max_prefill_num_token
):
	can_run_list.append(req)
```

That limits total prompt work in one batch, but it does not split a single request into multiple prefill chunks. So the current repo has a prefill budget guard, not true chunked prefill.

## 20. What to remember from Lesson 5

If you want the shortest review version, remember these five points:

1. Chunked prefill exists to reduce activation-memory spikes from very long prompt prefill.
2. It can also improve utilization by filling leftover prefill budget with partial work from long requests.
3. A chunked request carries unfinished prompt state into the next scheduler round.
4. The article's implementation uses dedicated chunk-aware request handling.
5. This repo currently has only a coarse prefill-token cap, not the full chunked-prefill flow.
