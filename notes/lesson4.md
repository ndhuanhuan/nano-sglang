# Lesson 4: Memory Pools And KV Cache Lifecycle

This note summarizes the article at https://gogongxt.com/posts/cd2448d1.html and maps it to the actual memory-management code in nano-sglang.

Lesson 3 explained how the Router decides what to run next. Lesson 4 explains the other half of that decision: where the KV cache lives, how cached prefixes are reused, and how memory is released safely when requests finish.

If Lesson 3 felt like "the scheduler picks requests," Lesson 4 adds the missing constraint:

The scheduler can only admit work that the KV-cache memory system can represent.

## 1. The big idea of this lesson

In nano-sglang, memory management is built out of three cooperating components:

1. `RadixCache`
2. `ReqToTokenPool`
3. `TokenToKVPool`

They live in:

- `python/sglang/srt/managers/router/radix_cache.py`
- `python/sglang/srt/memory_pool.py`

The article's main idea is correct: you should not think of KV cache as just one big tensor.

The runtime needs three different views of the same problem:

1. a prefix tree view for shared prompt reuse
2. a request-to-token-index mapping view
3. a token-index-to-actual-KV-buffer view

Inference only works efficiently because these three views stay in sync.

## 2. Where the memory pools are created

The memory system is initialized in `ModelRunner.init_memory_pool(...)`:

```python
def init_memory_pool(self, total_gpu_memory):
	self.max_total_num_token = self.profile_max_num_token(total_gpu_memory)
	self.req_to_token_pool = ReqToTokenPool(
		int(self.max_total_num_token / self.model_config.context_len * 256),
		self.model_config.context_len + 8,
	)
	self.token_to_kv_pool = TokenToKVPool(
		self.max_total_num_token,
		dtype=torch.float16,
		head_num=self.model_config.num_key_value_heads // self.tp_size,
		head_dim=self.model_config.hidden_size
		// self.model_config.num_attention_heads,
		layer_num=self.model_config.num_hidden_layers,
	)
```

This gives you two useful facts immediately:

1. KV capacity is bounded by available GPU memory.
2. Request bookkeeping and KV storage are separate structures.

That separation is deliberate. The runtime does not want to store scheduling structure inside the raw KV tensor.

## 3. Component 1: `RadixCache` stores reusable prefixes

`RadixCache` is a prefix tree over token sequences.

The key public APIs are:

```python
class RadixCache:
	def match_prefix(self, key):
		...

	def insert(self, key, value=None):
		...

	def evict(self, num_tokens, evict_callback):
		...

	def inc_ref_counter(self, node):
		...

	def dec_ref_counter(self, node):
		...
```

The mental model is:

- tree edges represent token prefixes
- node `value` stores token-to-KV indices for that prefix fragment
- `ref_counter` means whether some live request is still depending on that cached prefix
- `last_access_time` supports LRU-style eviction

The single most important method is prefix matching:

```python
def match_prefix(self, key):
	if self.disable:
		return [], self.root_node

	value = []
	last_node = [self.root_node]
	self._match_prefix_helper(self.root_node, key, value, last_node)
	if value:
		value = torch.concat(value)
	return value, last_node[0]
```

This returns two things:

1. the already-cached KV indices for the matched prefix
2. the last tree node reached by that prefix

That is exactly what the scheduler needs to know before deciding whether a request is cheap enough to admit.

## 4. Component 2: `ReqToTokenPool` maps requests to token slots

`ReqToTokenPool` is the per-request table.

It is simpler than `RadixCache`, but it is crucial:

```python
class ReqToTokenPool:
	def __init__(self, size, max_context_len):
		self.mem_state = torch.ones((size,), dtype=torch.bool, device="cuda")
		self.can_use_mem_size = size
		self.req_to_token = torch.empty(
			(size, max_context_len), dtype=torch.int32, device="cuda"
		)
```

You can think of this as a 2D table:

- each row is one active request slot
- each column is the position of the k-th token in that request
- the stored value is the index inside `TokenToKVPool`

This is not the KV cache itself.

It is the lookup table that says:

For request row `r`, token position `k` lives in KV slot `x`.

Allocation and free are straightforward:

```python
def alloc(self, need_size):
	if need_size > self.can_use_mem_size:
		return None
	available_indices = torch.nonzero(self.mem_state)
	...

def free(self, free_index):
	...
	self.mem_state[free_index] = 1
```

One subtle point: in this pool, `mem_state == 1` means free.

That is the opposite of `TokenToKVPool`, so it is worth keeping in mind.

## 5. Component 3: `TokenToKVPool` stores the actual KV cache

`TokenToKVPool` is the GPU memory owner.

Its constructor makes that explicit:

```python
class TokenToKVPool:
	def __init__(self, size, dtype, head_num, head_dim, layer_num):
		self.mem_state = torch.zeros((size,), dtype=torch.int16, device="cuda")
		self.alloc_ct = 0

		self.kv_data = [
			torch.empty((size, 2, head_num, head_dim), dtype=dtype, device="cuda")
			for _ in range(layer_num)
		]
```

This pool owns the actual key/value tensors for every layer.

Important difference from `ReqToTokenPool`:

- here `mem_state == 0` means free
- `mem_state > 0` is a reference count, not a boolean occupied flag

That means a KV slot can be shared by multiple logical users if they refer to the same cached prefix.

This is the core of prefix reuse.

The public methods show the intended usage:

```python
def alloc(self, need_size):
	available_indices = torch.nonzero(self.mem_state == 0)
	...
	self.add_refs(select_index)
	return select_index.to(torch.int32)

def add_refs(self, token_index):
	self.mem_state[token_index] += 1

def free(self, free_index):
	return self.decrease_refs(free_index)
```

So "free" does not necessarily mean immediate deletion.

It means decrease the reference count. Only when the count reaches 0 does that slot become allocatable again.

## 6. Why the three structures are all needed

It is tempting to ask whether one of these structures could be removed.

In practice, no:

- `RadixCache` answers: which prompt prefix is reusable?
- `ReqToTokenPool` answers: for this live request, where is token position `k` stored?
- `TokenToKVPool` answers: which KV slots are physically free or shared?

Those are different questions.

The full memory picture for one request looks like this:

```text
token sequence prefix
	-> RadixCache node/value
	-> KV slot indices
	-> ReqToTokenPool row stores those indices for this request
	-> TokenToKVPool owns the real per-layer KV tensors at those indices
```

## 6.5 A concrete worked example: `Write a poem about cats`

The abstractions become much easier to follow if you walk one fake request through them.

Assume the tokenizer turns this prompt into the illustrative token IDs:

```text
Write a poem about cats
-> [11, 12, 13, 14, 15]
```

Also assume the system has already finished another request whose cached prefix is:

```text
Write a poem about
-> [11, 12, 13, 14]
```

and that prefix is already stored in `RadixCache` with KV-slot indices:

```text
[101, 102, 103, 104]
```

These numbers are only for explanation. They are not real token IDs from the model.

### Step 1: `RadixCache` matches the shared prefix

When the new prompt arrives, the router runs:

```python
prefix_indices, last_node = self.tree_cache.match_prefix(req.input_ids)
```

Conceptually the result is:

```text
prefix_indices = [101, 102, 103, 104]
last_node = node("Write a poem about")
adjust_input_len = 1
```

Meaning:

- the first four prompt tokens are already cached
- only the last token, `cats`, still needs fresh computation

So `RadixCache` answers the question:

Which part of this prompt can I reuse immediately?

### Step 2: `ReqToTokenPool` allocates a row for this live request

Suppose `ReqToTokenPool.alloc(1)` returns request row `7`.

At this point, the runtime starts building the request-to-token mapping:

```text
ReqToTokenPool row 7
position 0 -> 101
position 1 -> 102
position 2 -> 103
position 3 -> 104
position 4 -> ?
```

The first four positions are already known because the cached prefix is being reused.

### Step 3: `TokenToKVPool` allocates only the uncached suffix

The uncached suffix is only one token long, so the runtime asks `TokenToKVPool` for one fresh slot.

Suppose allocation returns:

```text
out_cache_loc = [220]
```

Now the request row becomes:

```text
ReqToTokenPool row 7
position 0 -> 101   # Write
position 1 -> 102   # a
position 2 -> 103   # poem
position 3 -> 104   # about
position 4 -> 220   # cats
```

So the three structures now play different roles:

- `RadixCache` said positions 0 to 3 were reusable
- `ReqToTokenPool` gave this request a logical row
- `TokenToKVPool` provided the new physical KV slot for `cats`

### Step 4: decode appends new generated tokens

Suppose the model now generates:

```text
" in"
" moonlight"
```

During decode, the runtime allocates one new KV slot per new token, for example:

```text
221 for " in"
222 for " moonlight"
```

Then row 7 grows again:

```text
ReqToTokenPool row 7
[101, 102, 103, 104, 220, 221, 222]
```

Notice what did not happen:

- the shared prefix did not get recomputed
- the old prefix slots did not get duplicated

Only the new suffix and generated tokens consumed new KV memory.

### Step 5: when the request finishes

At completion, the runtime inserts the finished sequence into `RadixCache`, frees request row 7, and decreases request-specific references.

That means:

- the row in `ReqToTokenPool` disappears
- useful cached prefixes may remain in `RadixCache`
- KV slots like `220`, `221`, `222` may stay reusable if they are now part of cached content

So after the request is gone, the cache may now know a longer prefix such as:

```text
Write a poem about cats
```

If the next user asks for `Write a poem about cats and stars`, the system may reuse an even longer prefix than before.

This is the main payoff of the three-structure design.

## 7. Lifecycle step 1: prefix matching before admission

Before a request can run, the router asks the radix cache how much work can be reused.

That happens in `ModelRpcServer.get_new_fill_batch()`:

```python
for req in self.forward_queue:
	prefix_indices, last_node = self.tree_cache.match_prefix(req.input_ids)
	if req.return_normalized_logprob:
		prefix_indices = prefix_indices[: req.normalized_logprob_start_len]
	req.adjust_input_len = len(req.input_ids) - len(prefix_indices)
	req.prefix_indices = prefix_indices
	req.last_node = last_node
```

This gives each request three important pieces of metadata:

- `prefix_indices`: already cached KV slots that can be reused
- `adjust_input_len`: prompt tokens that still need fresh computation
- `last_node`: the radix-tree node whose reference count will be updated

This is the point where prefix reuse becomes concrete. It stops being a theory about prompts and becomes a list of real KV indices.

## 8. Reference counting starts before the batch runs

Still inside `get_new_fill_batch()`, the router tentatively claims the matched prefix:

```python
delta = self.tree_cache.inc_ref_counter(req.last_node)
available_size += delta

...

self.token_to_kv_pool.add_refs(req.prefix_indices)
can_run_list.append(req)
```

Two things are happening together:

1. the radix-tree path is marked as actively used by a live request
2. the reused KV slots also have their reference counts increased

That pairing matters.

If the router reused cached prefix tokens without increasing these ref counters, eviction could free memory that an active request still depends on.

## 9. Lifecycle step 2: extend/prefill allocates only the uncached suffix

Once a request is admitted, `Batch.init_extend_batch()` builds the prompt-side tensors.

The most important lines are:

```python
input_ids = [r.input_ids[len(r.prefix_indices) :] for r in reqs]
prefix_indices = [r.prefix_indices for r in reqs]

req_pool_indices = self.req_to_token_pool.alloc(bs)

...

extend_num_tokens = seq_lens.sum() - prefix_lens.sum()
out_cache_loc = self.token_to_kv_pool.alloc(extend_num_tokens)
```

This is the heart of the lesson.

What it means:

- the cached prefix is reused as-is
- only the uncached suffix is placed into `flatten_input_ids`
- each request gets a row in `ReqToTokenPool`
- only the uncached suffix asks `TokenToKVPool` for fresh slots

Then the request row is filled in two pieces:

```python
self.req_to_token_pool.req_to_token[req_pool_indices_cpu[i]][
	: len(prefix_indices[i])
] = prefix_indices[i]

...

self.req_to_token_pool.req_to_token[req_pool_indices_cpu[i]][
	prefix_lens[i] : prefix_lens[i] + extend_lens[i]
] = out_cache_loc[pt : pt + extend_lens[i]]
```

So after extend-batch construction, each request row contains:

1. reused prefix KV indices
2. newly allocated suffix KV indices

That row becomes the request's full logical sequence view.

## 10. What happens if extend allocation runs out of memory

The code does not immediately fail when no free KV slots exist.

It first tries eviction:

```python
out_cache_loc = self.token_to_kv_pool.alloc(extend_num_tokens)
if out_cache_loc is None:
	self.tree_cache.evict(extend_num_tokens, self.token_to_kv_pool.free)
	out_cache_loc = self.token_to_kv_pool.alloc(extend_num_tokens)

	if out_cache_loc is None:
		print("Prefill out of memory.")
		self.tree_cache.pretty_print()
		exit()
```

This is a very important design detail.

Eviction is controlled by the radix tree, not by scanning the raw KV pool directly.

Why?

Because the radix tree knows which cached prefixes are both:

- no longer actively referenced
- least recently used

When `tree_cache.evict(...)` removes such nodes, it calls `token_to_kv_pool.free(...)` on the node's KV indices, which decreases their reference counts and makes some slots reusable.

## 11. Lifecycle step 3: decode allocates one new KV slot per active request

Once a request is already running, decode grows it token by token.

That happens in `Batch.update_for_decode()`:

```python
def update_for_decode(self, input_ids=None):
	if input_ids is None:
		input_ids = [
			r.output_ids[-1] if r.output_ids else r.input_ids[-1] for r in self.reqs
		]
	self.input_ids = torch.tensor(input_ids, dtype=torch.int32, device="cuda")
	self.seq_lens.add_(1)

	bs = len(self.reqs)
	alloc_res = self.token_to_kv_pool.alloc_contiguous(bs)
	if alloc_res is None:
		self.out_cache_loc = self.token_to_kv_pool.alloc(bs)
		if self.out_cache_loc is None:
			self.tree_cache.evict(bs, self.token_to_kv_pool.free)
			self.out_cache_loc = self.token_to_kv_pool.alloc(bs)

	self.req_to_token_pool.req_to_token[
		self.req_pool_indices, self.seq_lens - 1
	] = self.out_cache_loc
```

The decode memory pattern is simpler than extend:

- one live request needs one more token slot
- the request row grows by one column
- that new column points to one newly allocated KV slot

So decode is just incremental growth of the same mapping structure.

## 12. Why `available_size` counts both free KV slots and evictable cache

Back in `get_new_fill_batch()`, the router estimates capacity like this:

```python
available_size = (
	self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size()
)
```

This line is easy to gloss over, but it encodes the whole policy.

The router treats two kinds of capacity as usable:

1. slots that are free right now
2. slots hidden behind cache entries whose radix-tree ref count is 0 and can therefore be evicted

That is why `evictable_size()` exists.

The scheduler is not just reasoning about current free space. It is reasoning about reclaimable space.

## 13. Lifecycle step 4: finished requests are converted into cache entries

One of the trickiest parts of the code is the cleanup path in `handle_finished_requests()`.

Here is the core:

```python
indices = self.req_to_token_pool.req_to_token[req_pool_idx, :seq_len]
prefix_len = self.tree_cache.insert(
	token_ids[:seq_len], indices.clone()
)

self.token_to_kv_pool.free(indices[:prefix_len])
self.req_to_token_pool.free(req_pool_idx)
self.tree_cache.dec_ref_counter(req.last_node)
```

This is worth reading carefully.

When a request finishes:

1. its full sequence is inserted into the radix cache
2. the request row in `ReqToTokenPool` is released
3. the matched prefix reference on `last_node` is decreased
4. the overlapping prefix part in `TokenToKVPool` has its extra request reference removed

That means the request is gone, but the useful cache may remain.

This is the real purpose of the memory system: keep reusable prefixes alive without keeping finished requests alive.

## 14. Why cleanup is "decrease refs," not "delete everything"

The article emphasized that completed requests do not simply zero out all KV memory. The code confirms that.

What really happens is reference accounting.

For example:

- a prefix still kept in `RadixCache` may stay reusable with radix ref count 0
- its corresponding KV slots may also remain allocated if their token ref counts are still nonzero
- once those counts drop to 0, they become evictable or directly reusable

So completion means:

- remove the request-specific ownership
- keep the reusable cache structure around if possible

That distinction is the whole point of cache-aware serving.

## 15. Understanding the two different ref-count systems

This lesson is much easier if you separate these two counters mentally.

### Radix tree ref count

`TreeNode.ref_counter` means:

How many live requests currently depend on this cached prefix node?

If the count is 0, the node is evictable.

### KV pool ref count

`TokenToKVPool.mem_state[index]` means:

How many active references still point at this KV slot?

If the count is 0, the slot is allocatable.

These are related but not identical.

- one is tracking cache-node liveness in the tree
- the other is tracking physical slot liveness in the KV pool

That distinction is why both structures exist.

## 16. The full memory lifecycle of one request

Here is the entire story in one sequence.

### Step 1: request arrives

The router receives `input_ids` and asks `RadixCache.match_prefix(...)` how much can be reused.

### Step 2: scheduler decides admission

If the request fits, the router increases radix-tree refs on the matched node and increases KV refs on reused prefix slots.

### Step 3: extend allocates new suffix memory

`ReqToTokenPool.alloc(bs)` assigns a request row.

`TokenToKVPool.alloc(extend_num_tokens)` allocates KV slots for only the uncached suffix.

### Step 4: decode grows the request one token at a time

Each decode step allocates one more KV slot per request and writes it into the next column of that request's row.

### Step 5: request finishes

The full sequence is inserted into `RadixCache`, so future requests can match it.

### Step 6: request-specific ownership is dropped

The request row is freed, prefix refs are decreased, and KV slot refs are decreased where appropriate.

### Step 7: eviction may reclaim old cache later

If memory pressure arrives, the tree evicts LRU leaf nodes with radix ref count 0 and decreases the corresponding KV-slot ref counts.

This is the entire memory lifecycle.

## 17. The most important insight from the lesson

The key insight is not just "there are three components."

It is this:

Nano-sglang treats cached prefixes as shared resources with explicit lifetime accounting.

That is why it can do all of the following at once:

- reuse prompt prefixes across requests
- grow active requests token by token
- keep useful cache after a request finishes
- evict only memory that is both unused and old

Without the combination of `RadixCache`, `ReqToTokenPool`, and `TokenToKVPool`, those goals would interfere with each other.

## 18. What to remember after Lesson 4

If you forget most of the details, remember these six facts:

1. `RadixCache` tracks reusable token prefixes, not raw tensors.
2. `ReqToTokenPool` maps request positions to KV-slot indices.
3. `TokenToKVPool` owns the actual per-layer KV buffers on GPU.
4. Extend allocates only the uncached suffix.
5. Decode appends one more KV slot per token step.
6. Finishing a request usually decreases references; it does not blindly delete all cache.

## 19. Best files to reread for this lesson

If you want to trace Lesson 4 directly in code, use this order:

1. `python/sglang/srt/memory_pool.py`
2. `python/sglang/srt/managers/router/radix_cache.py`
3. `python/sglang/srt/managers/router/model_rpc.py`
4. `python/sglang/srt/managers/router/infer_batch.py`
5. `python/sglang/srt/managers/router/model_runner.py`

This order moves from the core data structures to the runtime code that uses them.