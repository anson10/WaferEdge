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
- The cached counters matter less than their reputation here: 29 vs 12 M/s for 16-byte
  messages at the median, but with run-to-run ranges that overlap (10.7–25.1 uncached), and
  740 vs 850 ns per round trip. On this VM two "cores" may share caches; the laptop's run,
  with real separate L2 caches, will tell. The design keeps them: they cost two words.
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
