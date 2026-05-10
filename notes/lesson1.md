# Lesson 1: Start with Processes and Ports

This note is a beginner-oriented summary of the article at https://gogongxt.com/posts/90da297.html, rewritten in my own words and tied back to the code in this repo.

If you have never used nano-sglang before, the single most useful mental model is this:

The system is not "one server that runs a model." It is a small pipeline of roles:

1. an HTTP entrypoint that receives requests
2. a tokenizer side that turns text into token IDs
3. a router side that batches and schedules work
4. one or more model workers that do GPU forward passes
5. a detokenizer side that turns generated token IDs back into text

The article focuses on that process-level architecture first, which is the right place to start. If you do not understand which process owns which responsibility, the rest of the runtime will feel much more complicated than it really is.

## 1. What the article is teaching

The article explains nano-sglang by answering two concrete questions:

1. Which processes exist when the server starts?
2. Which ports and communication channels connect them?

That is a strong teaching strategy because modern LLM serving systems are mostly about coordination:

- tokenization happens on the CPU side
- scheduling decides who gets GPU time
- GPU workers run prefill and decode
- detokenization prepares user-visible text
- multiple requests share cache and batch state

The article's main idea is that tensor parallelism changes the process layout.

### When tp_size = 1

There are effectively 3 processes:

1. Main process: FastAPI server + TokenizerManager
2. Router process: router logic plus model execution in-process
3. Detokenizer process

The key point is that when tp_size is 1, the router process does not need separate RPC model workers. It can instantiate the model runtime locally and call it directly.

### When tp_size > 1

There are effectively 3 + tp_size processes.

For tp_size = 2, that becomes 5 processes:

1. Main process: FastAPI server + TokenizerManager
2. Router process
3. Detokenizer process
4. Model RPC worker rank 0
5. Model RPC worker rank 1

Now the router becomes an orchestrator. It no longer owns the model execution directly. Instead, it issues RPC calls to model worker processes, and those workers cooperate through NCCL for tensor-parallel inference.

That is the core message of the article.

## 2. Where that architecture lives in this repo

You can verify the article directly from the code.

### Entry point

The launch entry point is:

- python/sglang/launch_server.py

It only parses CLI args and forwards them into launch_server(server_args).

### Process creation

The real startup logic is in:

- python/sglang/srt/server.py

This file does three important things:

1. allocates ports
2. creates the tokenizer manager in the main process
3. spawns the router process and the detokenizer process

The dynamic port allocation is:

```python
can_use_ports = alloc_usable_network_port(
		num=4 + server_args.tp_size, used_list=(server_args.port,)
)
```

Then those ports are assigned to:

- tokenizer_port
- router_port
- detokenizer_port
- nccl_port
- model_rpc_ports

This matches the article's explanation that the runtime needs a small cluster of internal ports in addition to the public HTTP port.

## 3. The actual communication map

Inside this repo, the communication stack is split by role.

### Public HTTP API

The main process exposes:

- /generate
- /v1/completions
- /get_model_info

These are defined in python/sglang/srt/server.py.

### ZeroMQ channels

ZeroMQ connects the tokenizer, router, and detokenizer stages.

The flow is:

1. TokenizerManager sends tokenized requests to the router
2. Router sends generated token IDs to the detokenizer
3. Detokenizer sends decoded text back to the tokenizer manager

This is implemented with PUSH/PULL sockets:

- TokenizerManager binds a PULL socket for results and connects a PUSH socket to the router
- Router binds a PULL socket for tokenized requests and connects a PUSH socket to the detokenizer
- Detokenizer binds a PULL socket for token IDs and connects a PUSH socket back to the tokenizer manager

### RPyC between router and model workers

When tp_size > 1, the router does not invoke the model directly. Instead, it uses RPyC to talk to model worker processes.

That logic lives in:

- python/sglang/srt/managers/router/model_rpc.py

The important split is:

- tp_size == 1: ModelRpcClient builds a local ModelRpcServer object and wraps it as an async step
- tp_size > 1: ModelRpcClient starts one RPC process per rank and calls step remotely on all of them

This is the key bridge between the article's process diagram and the implementation.

## 4. Request lifecycle: what happens when one prompt arrives

Here is the simplest end-to-end path.

### Step 1: HTTP request enters FastAPI

The user sends a request to /generate or /v1/completions.

In server.py, the request becomes a GenerateReqInput object. Then the main process hands it to:

- TokenizerManager.generate_request(...)

### Step 2: text becomes token IDs

TokenizerManager lives in:

- python/sglang/srt/managers/tokenizer_manager.py

Its job is to:

- tokenize text into input_ids
- normalize and validate sampling parameters
- optionally preprocess image input for multimodal models
- assign a request ID
- send a TokenizedGenerateReqInput object to the router

At this point, the request has crossed from "user-facing API payload" into "runtime work item."

That is a useful conceptual boundary.

### Step 3: router collects requests

RouterManager lives in:

- python/sglang/srt/managers/router/manager.py

It has two loops:

1. receive tokenized requests from the tokenizer side
2. repeatedly call model_client.step(...) to advance inference

The router is intentionally small. It is mostly a coordinator. The heavy inference policy sits in ModelRpcServer.

### Step 4: model side turns requests into batches

The real runtime heart is:

- python/sglang/srt/managers/router/model_rpc.py

Each incoming request becomes a Req object. That object stores:

- input_ids
- output_ids
- sampling params
- streaming flag
- regex constraints if used
- cache-related metadata

Then the model side does two broad kinds of work:

1. prefill or extend: process prompt tokens that are not already cached
2. decode: generate one new token at a time for active requests

The code names for those phases are:

- ForwardMode.EXTEND
- ForwardMode.DECODE

That split is foundational for understanding LLM serving.

## 5. Why batching and caching matter so much

If you only remember one performance lesson, remember this:

Serving is not just "run the model." Serving is "reuse old work, batch new work, and avoid wasting KV memory."

This repo makes those ideas visible instead of hiding them.

### Radix cache

The radix cache lives in:

- python/sglang/srt/managers/router/radix_cache.py

Its role is to detect shared prefixes across requests.

Example:

- request A: "Write a poem about cats"
- request B: "Write a poem about dogs"

Both requests share the prefix "Write a poem about". If the KV cache for those earlier tokens already exists, the runtime should avoid recomputing them.

That is exactly the point of match_prefix(...):

- find how much of a new request is already represented in cached prefix nodes
- reuse the existing KV indices for that prefix
- only run forward passes for the uncached suffix

This is one of the main reasons SGLang-style runtimes can be much more efficient than naive one-request-at-a-time inference.

### Memory pools

The GPU-side token bookkeeping lives in:

- python/sglang/srt/memory_pool.py

There are two pools:

1. ReqToTokenPool: maps each request to token positions
2. TokenToKVPool: stores the actual KV cache slots

Think of them as two layers of indexing:

- one layer tracks which logical request owns which token sequence positions
- one layer tracks where those token states live in GPU KV memory

When batches finish, the runtime updates the radix cache and reference counts so old KV entries can be reused or evicted safely.

## 6. How scheduling works

Scheduling logic lives in:

- python/sglang/srt/managers/router/scheduler.py

The supported heuristics are:

- lpm: longest prefix match
- weight
- random
- fcfs

The most interesting beginner choice is lpm.

Why?

Because it prefers requests that share more cached prefix, which tends to improve cache reuse and reduce repeated work.

This is an example of a broader serving principle:

The scheduler is not only deciding fairness. It is also deciding efficiency.

## 7. What actually runs on the GPU

GPU execution lives behind:

- python/sglang/srt/managers/router/model_runner.py

ModelRunner is responsible for:

- loading the model
- initializing torch distributed and NCCL when tp_size > 1
- profiling available memory to size the KV cache
- allocating memory pools
- calling the model forward path for extend and decode

This is where architecture turns into hardware reality.

Important detail:

- when tp_size > 1, each rank is a separate process
- torch.distributed.init_process_group(..., backend="nccl") is used for tensor parallelism
- nccl_port is the internal bootstrap port for those ranks

So the article's process diagram is not just about software organization. It also reflects how multi-GPU inference actually has to be wired.

## 8. Why detokenization is a separate stage

DetokenizerManager lives in:

- python/sglang/srt/managers/detokenizer_manager.py

Its job is simple but important:

- convert generated token IDs back to strings
- trim stop strings when needed
- respect skip_special_tokens behavior
- send text chunks back to the tokenizer manager for API responses

Keeping detokenization separate has two benefits:

1. the router/model side can stay focused on scheduling and generation
2. the user-facing side can stream text incrementally without entangling string formatting with GPU execution logic

This separation is small, but it makes the system easier to reason about.

## 9. A clean mental model for beginners

If you are learning this system for the first time, think about it in layers.

### Layer A: API layer

Receives HTTP requests and returns text responses.

Main files:

- python/sglang/srt/server.py
- python/sglang/srt/managers/openai_protocol.py
- python/sglang/srt/managers/io_struct.py

### Layer B: orchestration layer

Moves work between components and decides what runs next.

Main files:

- python/sglang/srt/managers/tokenizer_manager.py
- python/sglang/srt/managers/router/manager.py
- python/sglang/srt/managers/detokenizer_manager.py

### Layer C: inference policy layer

Turns requests into batches, reuses cache, schedules work, and decides when requests finish.

Main files:

- python/sglang/srt/managers/router/model_rpc.py
- python/sglang/srt/managers/router/infer_batch.py
- python/sglang/srt/managers/router/scheduler.py
- python/sglang/srt/managers/router/radix_cache.py

### Layer D: hardware execution layer

Loads the model, owns KV memory, and runs the real forward passes on GPU.

Main files:

- python/sglang/srt/managers/router/model_runner.py
- python/sglang/srt/memory_pool.py
- python/sglang/srt/models/
- python/sglang/srt/layers/

That layered view is the fastest way to stop feeling lost.

## 10. The most important behind-the-scenes concepts

Here are the concepts I would focus on first.

### 1. Request state is explicit

The repo does not hide request lifecycle inside a giant framework abstraction. Req and Batch objects make the runtime state visible.

### 2. Prefill and decode are different workloads

Prefill processes prompt tokens. Decode extends active sequences token by token. They have different memory and batching behavior.

### 3. Cache reuse is a first-class optimization

The radix tree is not an extra feature. It is central to avoiding repeated prompt computation.

### 4. Scheduling affects throughput

The scheduler is partly a cache-optimization policy, not just a queue discipline.

### 5. Tensor parallelism changes process topology

The tp_size setting is not just a numeric performance knob. It changes how many processes exist and how they communicate.

## 11. A good reading order for this repo

If you want to learn the system without drowning in detail, read files in this order:

1. python/sglang/launch_server.py
2. python/sglang/srt/server.py
3. python/sglang/srt/managers/tokenizer_manager.py
4. python/sglang/srt/managers/router/manager.py
5. python/sglang/srt/managers/router/model_rpc.py
6. python/sglang/srt/managers/router/infer_batch.py
7. python/sglang/srt/managers/router/radix_cache.py
8. python/sglang/srt/memory_pool.py
9. python/sglang/srt/managers/router/model_runner.py

That order moves from the outside in:

- API
- orchestration
- batching and scheduling
- cache and memory
- actual model execution

## 12. One practical note about this repo

The README examples show --tp in some places, but the actual CLI argument defined in python/sglang/srt/server_args.py is --tp-size.

So if you run the code directly from this repo, think in terms of:

```bash
python3 -m sglang.launch_server \
	--model-path /path/to/model \
	--port 30000 \
	--tp-size 1
```

and for tensor parallel inference:

```bash
python3 -m sglang.launch_server \
	--model-path /path/to/model \
	--port 30000 \
	--tp-size 2
```

## 13. What you should understand after Lesson 1

After this lesson, you do not need to know every Triton kernel or every model detail.

You should understand these three things clearly:

1. nano-sglang is a multi-process serving pipeline, not a single monolithic loop
2. routing, caching, and batching are as important as the model forward pass itself
3. tp_size changes the architecture from local model execution to router-plus-RPC-worker execution

If those three ideas are solid, the next lessons on streaming, scheduling, memory pools, and chunked prefill will make much more sense.

## 14. My short summary in one paragraph

The article uses process layout and port wiring as the entry point for understanding nano-sglang, and this repo confirms that design exactly: the main process serves HTTP and tokenization, the router coordinates scheduling and batching, model workers run the actual inference, and the detokenizer converts token IDs back into streamed text. The real lesson is that LLM serving is a coordination problem built on request state, cache reuse, memory management, and GPU execution, and nano-sglang is small enough that you can actually see each of those pieces in code.
