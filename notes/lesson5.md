# Lesson 5: Chunked Prefill

This note summarizes the article at https://gogongxt.com/posts/9829ed85.html and compares it to the current nano-sglang codebase in this repo.

Lessons 3 and 4 explained how requests are admitted and how KV cache memory is managed. Lesson 5 adds a new question:

What should the runtime do when one prompt is so long that running its full prefill at once is expensive or risks OOM?

The answer is chunked prefill.

## 1. What chunked prefill is solving

Chunked prefill exists because prefill cost is not only about KV-cache capacity.

Even if the system can eventually store all prompt tokens in KV cache, running a very long prompt in one forward pass can still create large activation memory pressure.

The article highlights two benefits of splitting long prefill into chunks:

1. reduce activation-memory spikes and lower OOM risk
2. use prefill budget more efficiently when long and short prompts are mixed together

That second point is subtle but important.

Without chunked prefill, one large request may either monopolize a batch or be delayed entirely.

With chunked prefill, the scheduler can make partial progress on the long request while still leaving room for other work in later rounds.

## 2. The article's parameter model

The article says full sglang has two related parameters:

1. `--max-prefill-tokens`
2. `--chunked-prefill-size`

The first is a batch-level cap on how much prefill work one forward pass should do.

The second is the per-request chunk length that limits how much of one long request is processed in one prefill round.

In the article's minisglang implementation, this is simplified into one budget-like parameter:

```text
max_extend_tokens
```

The article explicitly says you can roughly think of that as the chunk size control in the simplified implementation.

## 3. The key scheduling difference

Chunked prefill changes the scheduler's choices.

Without chunked prefill, a scheduler usually behaves like this:

1. inspect a waiting request
2. decide whether the whole uncached prompt can be admitted now
3. if yes, run it
4. if no, leave it for later

With chunked prefill, the scheduler gains a third option:

Run only part of the prompt now, and keep the rest pending.

That is the core conceptual change of Lesson 5.

## 4. The article's minisglang control flow

The article centers around a function called `schedule_next_batch(prefill_budget)`.

Its stated responsibilities are:

1. create a prefill adder with a token budget
2. iterate over the pending queue
3. try to add requests into the current prefill batch
4. split long requests into chunks when needed
5. put unfinished chunked requests back into the pending queue with high priority

The article's key idea is that scheduling no longer returns just one kind of prefill request object.

It may return either:

- `Req`: prefill is complete for this request
- `ChunkedReq`: only part of the prompt has been prefetched, and more prompt tokens remain

That extra request state is what makes chunked prefill possible.

## 5. Why unfinished chunked requests get priority next round

The article emphasizes that once a request has already prefetched its front part, the next scheduler round should usually continue that same request first.

That policy makes sense for two reasons:

1. the request already owns intermediate scheduling state
2. the runtime wants to finish converting that long prompt into a normal decode-ready request instead of leaving it half-prefilled for too long

So the pending queue is updated so that unfinished chunked requests are placed before untouched later requests.

This is not only a fairness detail. It is also a state-management simplification.

## 6. The article's `PrefillAdder` idea

The article spends most of its code walkthrough on `PrefillAdder`, which does the real work of deciding whether a request can join the current prefill batch.

Its responsibilities are:

1. manage the batch token budget
2. account for already reserved space from in-flight decode work
3. allocate or reuse cache and table resources
4. decide whether this request finishes prefill now or becomes chunked

The important conceptual split is:

### New request

If the request has never been chunked before, the runtime:

1. checks for available table space
2. matches any reusable radix-cache prefix
3. estimates total space needs
4. locks the cache handle
5. allocates storage
6. copies the cached prefix mapping if needed

### Continued chunked request

If the request was already chunked in a previous round, the runtime skips re-allocation and simply reuses the previously assigned scheduling state.

That reuse is the main point of having a dedicated `ChunkedReq` path.

## 7. Why this connects directly to Lesson 4

Chunked prefill is not a separate memory system. It is a different policy layered on top of the same memory ideas from Lesson 4.

The key connection is:

- Lesson 4 explained how prefixes, request rows, and KV slots are tracked
- Lesson 5 explains how a long prompt may advance through those structures incrementally instead of all at once

So chunked prefill is best understood as a scheduling strategy built on top of cache-aware memory bookkeeping.

## 8. What this repo currently implements

This repo does not yet implement the full chunked-prefill machinery described in the article.

That is important to say clearly.

There is no dedicated `ChunkedReq` type, no `schedule_next_batch(prefill_budget)` function, and no logic that splits one request into multiple prefill rounds while preserving explicit chunk state.

Instead, the current code has a coarse prefill budget cap inside `ModelRpcServer`:

```python
self.max_prefill_num_token = max(
    self.model_config.context_len, self.max_total_num_token // 6
)
```

and uses it during admission in `get_new_fill_batch()`:

```python
if (
    req.adjust_input_len + req.max_new_tokens() + new_batch_total_tokens
    < available_size
    and req.adjust_input_len + new_batch_input_tokens
    < self.max_prefill_num_token
):
    can_run_list.append(req)
```

This means the current repo can limit total prompt work in one extend batch, but it does not split one request's uncached suffix into multiple chunks.

So the current behavior is:

- a request is admitted whole
- or it waits in `forward_queue`

There is no intermediate "partially-prefilled request" representation.

## 9. Why that difference matters

This difference is not cosmetic.

Without chunked prefill, a very long prompt has to fit the current admission logic as one unit of work.

With chunked prefill, a very long prompt could be processed like:

```text
round 1: prompt tokens 0..2047
round 2: prompt tokens 2048..4095
round 3: prompt tokens 4096..6143
...
```

That reduces prefill activation spikes and lets the scheduler use fixed-sized work units instead of all-or-nothing admission.

So the article is showing a real capability increase, not just a refactor.

## 10. A concrete example

Suppose one request has a 5000-token prompt and the chunked prefill budget is 2000.

Without chunked prefill:

- the scheduler tries to admit the whole remaining uncached prompt
- if it does not fit the current batch policy, the request waits

With chunked prefill:

- round 1 processes the first 2000 prompt tokens
- round 2 processes the next 2000 prompt tokens
- round 3 processes the final 1000 prompt tokens
- only then does the request become a normal decode-ready request

That is the operational meaning of `ChunkedReq`.

The request is not finished, but it is no longer "untouched" either. It has partially completed prefill state that the scheduler should continue.

## 11. How Lesson 5 changes your mental model of prefill

Before Lesson 5, it is natural to think of prefill as one big step:

```text
prompt arrives -> prefill once -> decode
```

After Lesson 5, the better mental model is:

```text
prompt arrives -> maybe several prefill chunks -> decode
```

That is the conceptual upgrade.

Prefill becomes schedulable work, not just a fixed one-shot phase.

## 12. If chunked prefill were added to this repo, what would change?

The current repo would likely need at least three structural additions:

1. a request representation for partially-prefilled prompts
2. a scheduler path that can requeue unfinished prefill chunks ahead of untouched requests
3. admission logic that slices one request's `adjust_input_len` into chunk-sized units rather than admitting the whole remainder at once

In other words, the current `Req` and `forward_queue` logic would need one more state: not just waiting or running, but waiting-with-partial-prefill-progress.

## 13. How chunked prefill would modify the current `model_rpc.py` flow

The easiest way to see the design change is to compare the current `model_rpc.py` flow with a chunk-aware version.

Right now the path is basically:

```text
handle_generate_request
	-> push Req into forward_queue
get_new_fill_batch
	-> either admit the whole request or leave it queued
forward_fill_batch
	-> finish prompt prefill for that request in one go
	-> produce first token
	-> move request into running_batch for decode
```

Chunked prefill would change the meaning of the fill path.

Instead of treating prefill as all-or-nothing, the router would have to treat it as resumable progress.

### 13.1 `Req` would need explicit prefill progress

Today `Req` mainly tracks:

- full `input_ids`
- reusable `prefix_indices`
- `adjust_input_len`
- generated `output_ids`

For chunked prefill, it would need one more concept: how much of the uncached prompt has already been prefetched.

A pseudo-diff would look like:

```python
class Req:
    def __init__(self, rid):
        self.rid = rid
        self.input_ids = []
        self.output_ids = []
        ...
        self.prefix_indices = []
        self.adjust_input_len = 0

+       # new: chunked-prefill progress inside the uncached suffix
+       self.prefill_cursor = 0
+       self.in_prefill = True
```

The purpose of `prefill_cursor` would be:

- `prefix_indices` tracks already-cached old prefix
- `prefill_cursor` tracks newly-prefilled suffix progress for this request

That distinction matters because chunked prefill creates a new kind of prefix: not only radix-cache hits from old requests, but also prompt progress made by this same request in earlier rounds.

### 13.2 `get_new_fill_batch()` would choose a chunk, not the whole suffix

Today the current code computes:

```python
req.adjust_input_len = len(req.input_ids) - len(prefix_indices)
```

and then reasons about the whole remaining uncached prompt.

With chunked prefill, the router would instead slice that uncached remainder into a per-round chunk:

```python
remaining_prefill = req.adjust_input_len - req.prefill_cursor
chunk_len = min(
    remaining_prefill,
    chunked_prefill_size,
    self.max_prefill_num_token - new_batch_input_tokens,
)
```

Then the admission decision would no longer mean "can the whole request enter?"

It would mean:

Can the next prefill chunk of this request enter?

That is the deepest scheduler change.

### 13.3 The batch builder would need to consume only the next chunk

Today `Batch.init_extend_batch()` effectively uses:

```python
input_ids = [r.input_ids[len(r.prefix_indices):] for r in reqs]
```

That assumes one fill batch consumes the whole uncached suffix.

With chunked prefill, the batch builder would instead need a bounded slice:

```python
start = len(r.prefix_indices) + r.prefill_cursor
end = start + r.chunk_len
input_ids = [r.input_ids[start:end] for r in reqs]
```

So the extend batch would no longer mean:

- all uncached prompt tokens

It would mean:

- only the next chunk of uncached prompt tokens

### 13.4 `forward_fill_batch()` could no longer always produce the first decode token

Today the code assumes one extend pass reaches decode readiness:

```python
next_token_ids, next_token_probs = batch.sample(logits)

for i in range(len(reqs)):
    reqs[i].output_ids = [next_token_ids[i]]
    reqs[i].check_finished()
```

That is only valid because current prefill finishes the request's whole prompt before sampling the first new token.

With chunked prefill, there would be two cases:

1. this chunk does not finish prompt prefill yet
2. this chunk completes the final prefill segment and can now sample the first output token

So the logic would need a branch like this:

```python
for req in batch.reqs:
    req.prefill_cursor += req.chunk_len

    if req.prefill_cursor < req.adjust_input_len:
        # prompt is still incomplete; requeue for next prefill round
        req.in_prefill = True
        chunked_requeue.append(req)
    else:
        # prompt is now complete; sample first decode token
        req.in_prefill = False
        req.output_ids = [next_token_id]
        decode_ready.append(req)
```

This is the control-flow point where a chunked request stops being "prefill-in-progress" and becomes a normal decode request.

### 13.5 Partially-prefilled requests would need to re-enter the queue ahead of untouched requests

Today requests are removed from `forward_queue` when admitted and then either:

- move to `running_batch` for decode
- or finish

Chunked prefill would add a third destination:

- go back to the front of the fill queue because prompt prefill is only partially complete

Pseudo-diff:

```python
new_batch = Batch(can_run_list, ...)
remaining_queue = [x for x in self.forward_queue if x not in can_run_list]

+self.forward_queue = chunked_requeue + remaining_queue
```

That is how the article's `ChunkedReq` priority behavior would show up in this repo's current structure.

### 13.6 `handle_finished_requests()` would stay mostly a decode/finalization path

One useful simplification is that partially-prefilled requests should not go through normal "finished request" cleanup yet.

They still own active request rows and newly-allocated KV slots.

So the likely design would be:

- partial prefill progress updates request state in-place
- only decode-ready or truly finished requests go through the existing output and cleanup logic

That means chunked prefill is mostly a change to:

- queue state
- extend-batch slicing
- transition conditions between prefill and decode

rather than a complete rewrite of the decode path.

### 13.7 The shortest pseudo-diff summary

If you want the shortest possible summary, chunked prefill would change `model_rpc.py` like this:

```text
current:
    waiting request -> one full extend -> first token -> decode

chunked:
    waiting request -> extend chunk 1 -> extend chunk 2 -> ...
    -> final extend chunk -> first token -> decode
```

So the essential modification is not in the math kernel.

It is in the request state machine.

`Req` would no longer be only "waiting for first fill" or "running decode." It would also need a middle state: "partially-prefilled prompt, needs another fill chunk."

## 14. What to remember after Lesson 5

If you forget the details, remember these six facts:

1. Chunked prefill exists mainly to reduce activation-memory pressure from long prompt prefill.
2. It can also improve utilization when long and short requests are mixed.
3. The article's implementation introduces an explicit partially-prefilled request state.
4. A chunked request is usually prioritized in the next prefill round.
5. This repo currently enforces only a coarse prefill-token cap.
6. So the article describes a more advanced scheduling strategy than what this repo implements today.

## 15. Best files to reread after this lesson

For the current repo, reread these to understand the non-chunked baseline:

1. `python/sglang/srt/managers/router/model_rpc.py`
2. `python/sglang/srt/managers/router/infer_batch.py`
3. `python/sglang/srt/managers/router/scheduler.py`
4. `python/sglang/srt/memory_pool.py`

If you keep the article beside these files, the main comparison becomes clear:

- current repo: prefill is admitted as a whole request under a token cap
- article's design: prefill itself is broken into scheduler-managed chunks