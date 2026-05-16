# Lesson 2: Streaming Response Architecture

This note summarizes the article at https://gogongxt.com/posts/6e1733b3.html and ties it to the streaming response implementation in TokenizerManager.

If you finished Lesson 1, you understand that nano-sglang is a multi-process pipeline. This lesson shows the final piece: how results flow back to the user incrementally, token by token, without blocking.

## 1. The core problem this solves

When a user sends a request to `/generate`, they expect to see tokens appear one at a time, not all at once after the whole sequence is done. If the server batched up all 100 tokens before responding, the user would wait a long time with no feedback.

The solution is **streaming**: the server sends partial results as they become available, and the HTTP connection stays open until the full sequence is done.

## 2. The dual-loop architecture

TokenizerManager implements streaming using two async loops that run in parallel:

1. **Global producer loop** (`handle_loop`): runs once per TokenizerManager instance, receives completed tokens from the detokenizer process, and distributes them to waiting requests.
2. **Per-request consumer loop** (`generate_request`): runs once per user request, waits for results to arrive, and yields them to the HTTP response stream.

These loops communicate via:
- `rid_to_state`: a dictionary mapping request IDs to request state
- `asyncio.Event`: signals when new results are ready
- `out_list`: a buffer holding output chunks for that request

## 3. Request entry point

When a user sends a request to `/generate`, here's what happens:

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

The key insight: `tokenizer_manager.generate_request(obj)` returns an **async generator**. If streaming is enabled, the FastAPI `StreamingResponse` iterates over it and sends each result back to the client as soon as it's available.

## 4. Inside generate_request: the consumer loop

When `generate_request` is called, it first checks if the global loop exists:

```python
async def generate_request(self, obj: GenerateReqInput):
    if self.to_create_loop:
        await self.create_handle_loop()
```

This is **lazy initialization**: the global loop is only created on the first request. Subsequent requests reuse it.

Then the method tokenizes the input:

```python
    is_single = isinstance(obj.text, str)
    
    if is_single:
        rid = obj.rid
        input_ids = self.tokenizer.encode(obj.text)
        sampling_params = SamplingParams(**obj.sampling_params)
        # ... validate and preprocess ...
        tokenized_obj = TokenizedGenerateReqInput(
            rid=rid,
            input_ids=input_ids,
            # ...
        )
        self.send_to_router.send_pyobj(tokenized_obj)
```

Then it registers this request with the global state dictionary:

```python
        lock = asyncio.Lock()
        event = asyncio.Event()
        state = ReqState([], False, event, lock)
        self.rid_to_state[rid] = state
```

`ReqState` is the crucial data structure:

```python
@dataclasses.dataclass
class ReqState:
    out_list: List          # Buffer of output chunks
    finished: bool          # Whether inference is done
    event: asyncio.Event    # Signal: "new output is ready"
    lock: asyncio.Lock      # Protects concurrent access
```

Then `generate_request` enters its consumer loop:

```python
        while True:
            await event.wait()                 # Block until new output
            yield state.out_list[-1]           # Return the latest output
            state.out_list = []                # Clear the buffer
            if state.finished:
                del self.rid_to_state[rid]    # Clean up
                break
            event.clear()                      # Reset signal for next iteration
```

This loop repeatedly:
1. Waits for the signal that new data has arrived
2. Yields that data to the caller (FastAPI, which sends it to the HTTP client)
3. Clears the buffer
4. Checks if inference is done; if so, exits
5. Resets the signal for the next round

## 5. The global producer loop: handle_loop

Meanwhile, the global `handle_loop` runs in the background:

```python
async def handle_loop(self):
    while True:
        recv_obj = await self.recv_from_detokenizer.recv_pyobj()
        
        if isinstance(recv_obj, BatchStrOut):
            for i, rid in enumerate(recv_obj.rids):
                recv_obj.meta_info[i]["id"] = rid
                out_dict = {
                    "text": recv_obj.output_str[i],
                    "meta_info": recv_obj.meta_info[i],
                }
                state = self.rid_to_state[rid]
                state.out_list.append(out_dict)          # Add output
                state.finished = recv_obj.finished[i]    # Update finish status
                state.event.set()                        # Signal: "wake up!"
```

This loop:
1. Blocks waiting for a batch of results from the detokenizer process
2. Unpacks the batch (multiple requests may have completed tokens at once)
3. For each request, appends the output to its `out_list`
4. Sets the `event` to wake up the corresponding `generate_request` consumer loop

## 6. How synchronization works

This is a classic **producer-consumer pattern** using asyncio primitives:

- **Producer** (handle_loop): adds data to `out_list`, then calls `event.set()`
- **Consumer** (generate_request): calls `await event.wait()` (blocks), gets data, then calls `event.clear()` to reset

The flow for a single output:

```
detokenizer process sends BatchStrOut via ZeroMQ
         ↓
handle_loop receives it
         ↓
handle_loop does state.event.set()
         ↓
generate_request's await event.wait() returns
         ↓
generate_request yields state.out_list[-1]
         ↓
HTTP response stream receives the chunk
         ↓
client sees the partial text
         ↓
generate_request calls event.clear()
         ↓
loop repeats
```

## 7. Why this design is efficient

1. **Non-blocking waits**: the consumer doesn't spin in a busy loop; it sleeps until data arrives
2. **Lazy loop creation**: the global loop is only created on first request, not on every request
3. **Batch awareness**: the producer can send multiple requests' outputs in one message; the loop distributes them all at once
4. **Lazy finalization**: request state is deleted as soon as inference finishes, freeing memory immediately

## 8. Supporting both single and batch requests

The code above shows the single-request path. The batch path is similar but loops over all requests:

```python
        else:
            assert obj.stream is False  # Batches are not streamed
            bs = len(obj.text)
            for i in range(bs):
                rid = obj.rid[i]
                # ... tokenize each request ...
                self.rid_to_state[rid] = state
            
            output_list = []
            for i in range(bs):
                rid = obj.rid[i]
                state = self.rid_to_state[rid]
                await state.event.wait()  # Wait for ALL requests
                output_list.append(state.out_list[-1])
                assert state.finished
                del self.rid_to_state[rid]
            
            yield output_list  # Return all at once
```

Batches are not streamed (stream=False), so the API waits for all results and returns them together.

## 9. The complete request lifecycle

Here's the full flow from HTTP to output:

1. User sends POST `/generate` with `stream=true`
2. FastAPI calls `generate_request(obj)`
3. `generate_request` calls `obj.post_init()` to normalize input, then `tokenizer_manager.generate_request(obj)`
4. Inside TokenizerManager:
   - On first request: create the global `handle_loop` background task
   - Tokenize the text into `input_ids`
   - Create a `ReqState` and register it in `rid_to_state`
   - Send the tokenized request to the router process via ZeroMQ
   - Enter the consumer loop: wait for signals and yield outputs
5. Meanwhile, the router and model processes are inferring in parallel
6. When tokens are generated, the model sends them to the detokenizer process
7. The detokenizer process converts tokens to text and sends back a `BatchStrOut` message
8. The global `handle_loop` receives it, looks up the request in `rid_to_state`, and signals the consumer
9. The consumer wakes up, yields the output, and the HTTP stream sends it to the client
10. This repeats until `finished=true`
11. The consumer cleans up `rid_to_state[rid]` and exits
12. The HTTP stream closes

## 10. Key concepts for beginners

### Async generators

`generate_request` is an **async generator**:

```python
async def generate_request(self, obj):
    # ...setup...
    while True:
        await event.wait()
        yield state.out_list[-1]  # ← This makes it a generator
        # ...
```

Async generators support `async for` loops and are perfect for producing results incrementally over time.

### asyncio.Event

`asyncio.Event` is a synchronization primitive:

```python
event = asyncio.Event()
await event.wait()      # Block until set()
event.set()             # Unblock all waiters
event.clear()           # Reset for next wait()
```

It's like a flag that multiple coroutines can wait on.

### Producer-consumer pattern

This is a classic concurrency pattern:
- Producer: generates data, signals consumer
- Consumer: waits for signal, processes data

Without asyncio, this would use threading locks and condition variables. Here it's done cleanly with async/await.

## 11. Streaming is not required

The same architecture works for non-streaming requests:

```python
@app.post("/v1/completions")
async def v1_completions(obj: CompletionRequest):
    # ...create GenerateReqInput with stream=False...
    ret = await result_generator.__anext__()  # ← Get just the first (only) result
    return {"choices": [{"text": ret["text"]}]}
```

The consumer loop still runs, but instead of streaming results, the FastAPI handler just waits for one result and returns it.

## 12. What you should understand after Lesson 2

After this lesson, you should be able to answer these:

1. **Why two loops?** The producer handles results from the background process; the consumer handles user requests. Separating them avoids blocking.
2. **Why asyncio.Event?** It lets the consumer sleep until the producer has data, avoiding busy-wait loops.
3. **Why lazy loop creation?** To avoid creating a background task for every TokenizerManager instance if no requests arrive.
4. **How does streaming work?** Each token is yielded as soon as it arrives, so the HTTP connection sends partial results to the client in real time.
5. **Why rid_to_state?** It's the rendezvous point: the producer puts data there, the consumer retrieves it, and the mapping by request ID ensures each request gets its own output.

## 13. Next reading order

To deepen your understanding:

1. Trace through one streaming request in a debugger
2. Read the detokenizer process code to understand what BatchStrOut looks like
3. Read the router process code to understand what generates tokens
4. Study how HTTPResponse and streaming works in FastAPI/Starlette
5. Understand the performance implications: when is the global loop a bottleneck? When does batching help?

## 14. Summary in one paragraph

The streaming response architecture in TokenizerManager uses a dual-loop pattern: a global producer loop (`handle_loop`) receives completed token outputs from the detokenizer process and distributes them to per-request consumer loops (`generate_request`), which yield them incrementally to HTTP clients. Synchronization is achieved via `asyncio.Event` and a shared `rid_to_state` dictionary that maps request IDs to their output buffers. This design avoids blocking, supports both streaming and batch modes, and keeps the system responsive even when many requests are in flight.
