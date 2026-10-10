# ADR-0010: HSMS as a sans-I/O state machine, driven by Asio coroutines

- **Status:** Accepted
- **Date:** 2026-10-10

## Context
HSMS (SEMI E37) puts SECS-II on TCP: length-prefixed frames, a select handshake, linktests,
and five timers (T3 reply, T5 reconnect, T6 control transaction, T7 not selected, T8 stalled
message) that decide when a connection is broken. The edge host (active) and the tool
emulator (passive) both need it, with no allocation per message (phase 4's rule) and with
every timer's behaviour tested. Timer logic is where protocol code usually breaks, and in
most implementations it can only be tested by sleeping through real timeouts, because the
rules sit inside socket callbacks. The standard is paid; behaviour comes from public
descriptions and open-source implementations, so tests are the specification here.

Asio (standalone) is the project's networking library (ADR-0003). It offers callbacks,
stackful coroutines and C++20 coroutines (`asio::awaitable`).

## Options considered
1. **Protocol inside Asio callbacks or coroutines**: the usual design; each timer a
   `steady_timer` next to the socket code. Least code; testable only over real sockets with
   real sleeps, so T3/T5/T7 tests take seconds each and race with the scheduler.
2. **Sans-I/O engine plus a thin driver**: `hsms::Protocol` takes received bytes and "now",
   and gives back bytes to send, events, and its next deadline; it has no socket and no
   clock. `hsms::Session` moves bytes between it and a socket and sleeps until the deadline.
   (The pattern of h11, quinn and other protocol libraries.)
3. **An existing HSMS library** (secsgem in Python, secs4net in C#, a few C++ ones): wrong
   language, or callback-based with allocation per message; and writing the protocol is
   part of what this project is for.

For the driver: callbacks (verbose, lifetimes by `shared_ptr`), stackful coroutines (an
extra dependency on Boost.Context), or C++20 coroutines, which read as straight-line code and
which GCC 13 and Clang 18 both support.

## Decision
Option 2, with a C++20 coroutine driver. `Protocol` is an explicit state machine (NOT
CONNECTED → NOT SELECTED → SELECTED) with every rule in one place, tested on a fake clock:
each timer at its deadline and one nanosecond before. Its receive buffer, output buffer and
the table of open transactions (64 slots) are reused, so steady-state messaging allocates
nothing. Output is handed to the driver by swapping two vectors, so a write in flight is never
invalidated by the handler sending more. A bad or oversized length field closes the
connection as soon as its 4 bytes arrive, before any memory is reserved for it.

`Session` runs three coroutines per connection, joined with Asio's `&&`: a reader, a
writer, and a deadline sleeper. When the connection ends (EOF, a protocol close, a write
error), `finish()` closes the socket, which aborts the pending read and write, and wakes the
timers; each coroutine sees the flag and returns. Errors are values (`asio::as_tuple`), not
exceptions. A Session is single-threaded (one `io_context` thread); other threads post to it.

The first version joined them with `||`, which cancels the other two when one ends. That
works, but it makes Asio attach a cancellation handler to every operation. With libc++ a
socket operation's handler is 40 bytes and a timer wait's 32; the writer alternates between
the two, so Asio freed and reallocated that memory on every transaction: 2 allocations per
transaction with Clang, 0 with GCC. The allocation test caught it in the `clang` preset.
Every operation is now bound to an empty cancellation slot, and ending a connection is
explicit.

## Consequences
- 21 protocol tests run in milliseconds without a socket; the TCP tests only check the wiring
  (select, a transaction, reconnect after T5, T7 on a silent client, linktests).
- Ending a connection is the driver's own job (`finish()`), not Asio cancellation: one more
  flag to check after each `co_await`, in exchange for no allocation on either standard
  library.
- The same engine is the HSMS fuzz target: arbitrary bytes, split arbitrarily, on a moving
  fake clock, with an oracle on its own output.
- Measured 0 allocations per transaction for the engine and for the TCP session after
  warm-up, with GCC + libstdc++ and Clang + libc++ (Asio recycles coroutine frames per
  thread): `waferedge-alloc-tests`, `bench-hsms`.
- Event spans point into the receive buffer and live until the next read; the handler must
  copy what it keeps (phase 4's ring buffer will own them).
- One control transaction at a time, single session (HSMS-SS). The general HSMS multi-session
  mode and the T5/T6/T7 interplay of a passive side refusing connections are not
  implemented; docs/secs.md lists the subset.
- GCC's TSan build warns that `std::atomic_thread_fence` (used inside Asio) isn't modelled;
  the `tsan` preset silences that warning (`-Wno-tsan`) rather than dropping Asio from it.
