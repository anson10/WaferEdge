# ADR-0013: Threads hand wafers over through bounded lock-free SPSC rings

- **Status:** Accepted
- **Date:** 2026-10-10

## Context
The edge host (phase 4) splits work across threads: the network thread runs HSMS and GEM,
detector threads run the signatures (CPU or GPU), a decision thread holds lots. Every wafer
crosses at least two of these boundaries. A hand-off must not block the network thread (a
stalled network thread delays every later wafer and, past T3, the tool's transactions), must
not allocate (the project's hot-path rule), and should copy a wafer map at most once. Each
hand-off has exactly one producing and one consuming thread.

## Options considered
1. **`std::mutex` + `std::condition_variable` around a queue**: simple and obviously correct.
   Each operation takes a lock the other side may hold; a blocked consumer is woken with a
   futex system call; a preempted lock holder stalls the other side for a whole time slice.
2. **A lock-free multi-producer / multi-consumer queue** (e.g. a CAS-based ring, or
   moodycamel's ConcurrentQueue): more general than needed, and paid for with compare-and-swap
   loops on shared counters.
3. **A bounded single-producer / single-consumer ring**: two counters, each written by one
   thread only, so plain atomic loads and stores with acquire / release ordering suffice.
   Bounded, so it never allocates after construction and its fullness is backpressure.

## Decision
Option 3, `pipeline::SpscRing<T, N>`: N a power of two, slots constructed once, counters that
only grow (64-bit, so full and empty never look alike) and are mapped to slots with a mask.
Each counter lives on its own 64-byte cache line, and each side keeps a private copy of the
other side's counter, reloading it only when the ring looks full or empty (as in Erik
Rigtorp's SPSCQueue), so most operations touch no shared line. Besides `try_push` /
`try_pop`, the ring hands out slots in place (`begin_push` / `commit_push`, `front` / `pop`)
so a wafer map can be written straight into its slot. The ring itself never blocks; how a
thread waits when the ring is empty (spin, yield, sleep) is the edge host's choice.

## Consequences
- Measured (`bench-ring`, 4-thread cloud container, median of 3 runs): 16-byte messages
  29 M/s vs 4.4 M/s through the mutex queue (7x); 1,600-byte maps 4.6 vs 1.4 M/s (3x);
  round trip 740 ns vs 27 µs at the median (36x), 19 µs vs 217 µs at p99.9.
- On the laptop (Ryzen 5 7535HS, 6 cores / 12 threads, WSL2, release build): 16-byte messages
  22.7–23.7 M/s vs 6.8–6.9 M/s through the mutex queue (3x); 1,600-byte maps 7.4–7.9 vs
  2.8–3.2 M/s; round trip 320–340 ns vs ~100 µs at the median (~300x), 1.9–2.7 µs vs
  0.5–0.6 ms at p99.9. Under WSL2 the futex wake-up of the mutex queue costs ~4x what it
  costs in the cloud container; the ring doesn't make that system call.
- The cached counters help throughput when the ring holds a backlog: 23.7 vs 17.2 M/s for
  16-byte messages on the laptop (1.4x; at CPUs 0 and 1 the run ranges don't overlap). They
  don't help a round trip (320 ns either way): in ping-pong the ring holds at most one message,
  so the cached copy is always stale and every operation reloads it anyway. At 1,600 bytes the
  copy dominates and the uncached ring is slightly ahead (8.2–8.5 vs 7.4–7.9 M/s). The design
  keeps them: they cost two words, and the edge host's rings back up exactly when it matters.
- Hyperthread siblings (CPUs 0 and 1, sharing L1 and L2) and separate cores (0 and 2) measured
  the same within run-to-run noise: 22.7 vs 23.7 M/s, 320 vs 340 ns. The guess that siblings
  explained the first laptop run (no gain from caching) was wrong; the round trip simply
  can't show the gain, as above. A likely reason the two pinnings match: the 7535HS's six
  cores share one 16 MB L3, so a cache line crosses between cores in tens of nanoseconds. On
  a machine with several L3 domains or sockets, placement would matter more.
- Correctness rests on two acquire / release pairs that x86 hides: with the producer's
  release store made relaxed, the stress test still passed on this machine, but
  ThreadSanitizer reported the data race on the slot at once. The ring's tests run under
  TSan in CI for that reason.
- One producer and one consumer per ring, by contract; nothing checks it at run time. A
  second producer would corrupt the ring silently: the edge host's thread layout (one ring
  per thread pair) is where that rule is kept.
- Bounded: a full ring is visible backpressure (`begin_push` returns nullptr). What the
  network thread does then (drop, wait, or hold the HSMS window shut) is decided with the
  edge host.
