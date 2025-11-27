# GPU Optimization Strategies

Simulating city traffic on a GPU can involve tens or hundreds of thousands of agents. To ensure performance and scalability, we apply several optimization strategies.

## Structure of Arrays (SoA)

FLAMEGPU2 stores agent variables in a structure-of-arrays format internally, meaning each variable is a contiguous array across agents. This is optimal for GPU memory coalescing when all agents execute the same instruction.

**Design considerations:**
- Keep frequently accessed variables (like `remaining_time`, `curr_edge`) as simple types (int/float) rather than complicated structures
- Avoid unnecessary large arrays per agent. For Packet, we did include a route array of length up to 32. This means 32 * number_of_packets integers in memory. If number_of_packets is huge, this could be heavy.

## Minimize Divergence

When writing agent functions, divergent branching can harm warp efficiency. For example, in `move_and_request`, only some agents finish edges and take the branch to output a message. Others simply continue. This will cause divergence within a warp if some packets in the warp finish and others don't.

**Mitigation strategies:**
- Sort agents by similar state or progress. In FLAMEGPU, we can sort agents by a variable via `AgentVector.sort()`. For instance, we might sort Packet agents by `remaining_time` or by state so that all waiting ones are processed in contiguous memory separate from traveling ones.
- Use separate agent functions for drastically different behaviors (we did that with states). That means warps executing `move_and_request` are all traveling agents.

## Parallel Message Processing

The use of Bucket messages allows O(N) filtering in parallel rather than each edge scanning all messages. Because each EdgeQueue agent only iterates its own bucket messages, and FLAMEGPU2 likely uses parallel scan behind scenes for bucket distribution, this is efficient.

**Note:** We must ensure keys (edge IDs) are nicely distributed to avoid warp threads all contending on one key if possible. However, if one edge gets 100 requests and others none, those 100 will be processed by one agent sequentially in its loop – but that's unavoidable because one edge agent does that work.

## Memory Coalescing

Accesses like `pyflamegpu.getVariableInt("size")` are presumably coalesced because all threads call it for their own agent. Under the hood, it likely translates to reading an array at index=threadId for that variable. That's coalesced automatically if threads are contiguous.

When reading messages, if using BruteForce, all agents iterate potentially all messages – that's bad (O(N²)). But we only do brute force for acceptance where waiting packets (which are typically fewer than total moving packets) iterate accept messages (which are fewer than requests normally). And each waiting only checks messages until it finds its id, then breaks.

## Agent Population Size

We intentionally aggregate vehicles into packets to reduce the agent count. If we had simulated each of those 15 vehicles individually, that's 15 agents vs 2 packet agents in the example. On a city scale with e.g. 100k vehicles, using packets might reduce agent count by an order of magnitude (depending on grouping aggressiveness).

**Trade-off:** Too large packets reduce fidelity (they all behave identically and cannot, for example, partially get through a yellow light). But large groups also mean less parallelism if we have very few packets – not an issue until extreme grouping.

A good strategy is group vehicles that naturally form a platoon in reality (e.g., vehicles arriving one after another at a green light).

## Memory Usage vs Occupancy

We should consider GPU memory limits. Each EdgeQueue agent is small (a handful of ints/floats). Even for, say, 10,000 edges, that's trivial (< 1e5 variables).

Packet agents could be more numerous. If we simulate a big city with e.g. 1 million vehicles over an hour, but if each packet averages 5 vehicles, that's 200k packets lifetime, but not simultaneous. At any instant maybe 20k on network concurrently. 20k Packet agents with route arrays of 32 ints – that's 20k*32*4 bytes ≈ 2.56 million bytes (~2.5 MB) just for routes, plus other vars, say total 4 MB, which is fine on modern GPUs.

## Kernel Launch Overhead

FLAMEGPU2 will launch a GPU kernel for each layer each step. We have to ensure that overhead isn't too high. Typically, tens of layers times thousands of steps is okay. If step count is large (e.g. 3600 steps for an hour with 1s step), and if number of layers ~5, that's 18k kernel launches. That's acceptable on GPU.

We can combine some layers if safe: For example, if we were not concerned about strict order, we could run signals and edges in one kernel if independent, but here independence is not the case (we need order).

## Use of Constant Memory

Some data like `edge_length` array is read-only for agents. We could store it in environment properties which might be cached or constant memory on GPU. If we suspect repeated reads, marking environment props as const can help. For instance, `env.newPropertyArrayFloat("edge_lengths", lengths, const=True)`.

## Compaction and Memory Reuse

FLAMEGPU2 automatically handles agent birth/death by reusing slots (the agent vectors can shrink or grow). We kill Packet agents on destination, freeing those slots. This prevents memory leak.

## Concurrent Execution

FLAMEGPU2 supports concurrent execution of different agent functions if no dependencies. We define explicit layers to enforce order, but maybe some layers could run concurrently if independent. We rely on the internal parallelism for that.

There's also the possibility of using multiple GPUs (FLAMEGPU2 ensembles) for multiple independent simulations. For a single simulation on one city, splitting across GPUs is not straightforward because agents on different GPUs would need to communicate at boundaries (not supported out-of-the-box). So we stick to one GPU per simulation.

## Summary

Our approach (packets + edges as agents with message-based interactions) is well-suited to GPU: it maps naturally to parallel kernels, avoids sequential bottlenecks, and keeps memory contiguous. The design choices like bucket messaging and grouping help ensure we exploit parallelism while not overwhelming the GPU with too many tiny agents.

We should be able to simulate large networks faster than real-time depending on GPU power. Actual performance would be measured, but research shows GPUs can handle millions of agent interactions per second, making city-scale (with perhaps 10^4 edges and 10^5 vehicles aggregated into say 10^4 packets) quite feasible.

