# SECS-II and HSMS

How the tool emulator and the edge host talk: the equipment protocol stack fabs use, from the
bytes up: the **SECS-II item codec** (code: `include/waferedge/secs/`, `src/secs/`;
ADR-0009), the **HSMS transport** (`include/waferedge/hsms/`, `src/hsms/`; ADR-0010) and the
**GEM subset** (`include/waferedge/gem/`, `src/gem/`; ADR-0011).

SEMI E5 (SECS-II), E37 (HSMS) and E30 (GEM) are paid standards. This implementation works from
public descriptions and open-source implementations (secsgem, secs4net), covers a subset, and
is not certified against the standards.

## The stack in one picture

```
 GEM (E30)       which messages a tool must support and what they mean:
                 S1F13 establish communication, S6F11 event report, S2F41 host command, ...
                 communication and control state machines             <- implemented (subset)
 SECS-II (E5)    the message body: one item, a self-describing tree    <- implemented
 HSMS (E37)      TCP: 4-byte length, 10-byte header (session, stream, function, system bytes)
                 select, linktest, timers                              <- implemented
```

A message is named **SxFy**: stream x (a topic: 1 equipment status, 2 control, 5 alarms, 6
data collection) and function y. Odd functions are primary messages, and the reply is
function y+1: S6F11 "event report" is answered by S6F12 "acknowledge".

## Items

The body of a message is one **item**. An item is a list of child items, or an array of
values of one format. Each item starts with a header:

```
 format byte     bits 7..2: format code    bits 1..0: number of length bytes (1, 2 or 3)
 length          1-3 bytes, big-endian: data bytes for an array, children for a list
 data            the values, big-endian; a list has none (its children follow)
```

| Format | Code (octal) | Format byte¹ | Value | SML |
|---|---|---|---|---|
| List | 00 | 0x01 | child items | `<L [n] ...>` |
| Binary | 10 | 0x21 | `uint8_t` | `<B 0x00>` |
| Boolean | 11 | 0x25 | `bool`, one byte | `<BOOLEAN T>` |
| ASCII | 20 | 0x41 | `char` | `<A "LOT-1">` |
| JIS-8 | 21 | 0x45 | `char` | `<J "ab">` |
| I8 / I1 / I2 / I4 | 30 / 31 / 32 / 34 | 0x61 / 0x65 / 0x69 / 0x71 | signed integers | `<I4 -1>` |
| F8 / F4 | 40 / 44 | 0x81 / 0x91 | IEEE 754 | `<F4 1.5>` |
| U8 / U1 / U2 / U4 | 50 / 51 / 52 / 54 | 0xA1 / 0xA5 / 0xA9 / 0xB1 | unsigned integers | `<U4 17>` |

¹ With one length byte; add 1 or 2 for two or three length bytes (`<A>` of 256 characters
starts `42 01 00`).

So `<U4 1>` is `B1 04 00 00 00 01`, and an empty list is `01 00`. **Every leaf is an array**:
`<U4 1 2 3>` is one item of 12 data bytes, and a 40×40 wafer map is one `<U1 ...>` of 1,600
bytes. A list's length is its number of children, not bytes, so a list's size in bytes is only
known by walking its children's headers. Three length bytes cap an item at 16,777,215 data
bytes or children.

SML (SECS Message Language) is the text form logs and tool manuals use, e.g. the S2F41 the
edge host sends to hold a lot:

```
<L [2]
  <A "HOLD">                RCMD, the remote command
  <L [1]
    <L [2]
      <A "LOTID">           CPNAME
      <A "LOT-0042">        CPVAL
    >
  >
>
```

## The codec

**Decode** (`secs::decode`, `include/waferedge/secs/item.hpp`) validates the whole body in one
pass over its headers and returns an `ItemView`: a pointer into the receive buffer, the format
and the length. Nothing is copied or allocated. Children and values are read on access:
`ListView` walks the children, `ArrayView<T>` loads each value big-endian as it is read, and
for one-byte formats `bytes()` is the data itself, so the wafer map becomes a `WaferMapView`
over the receive buffer:

```cpp
auto body = secs::decode(bytes);   // Result<ItemView>: std::expected<ItemView, secs::Error>
if (!body) { log(secs::describe(body.error())); return; }  // e.g. "truncated at byte 52"
auto fields = body->list();        // Result<ListView>: an error if the body is not a list
auto bins = fields->at(4)->as<secs::Format::u1>();   // Result<ArrayView<std::uint8_t>>
WaferMapView map(bins->bytes(), rows, cols);         // the receive buffer's bytes, no copy
```

Validation is iterative, with an explicit stack of at most `kMaxDepth` (64) open lists, so a
malicious `01 01 01 01 ...` a million levels deep returns `too_deep` instead of overflowing
the call stack, and everything that later recurses over a decoded tree is bounded by 64
levels. Because only `decode()` creates views, a view always points at validated bytes and
the accessors only check that the format is the one asked for.

**Encode** (`secs::Encoder`, `encoder.hpp`) appends to a caller-owned
`std::vector<std::uint8_t>`. It clears the vector but keeps its capacity, so once the buffer
has grown to the largest message, encoding allocates nothing. A list declares its child count
up front (`list(n)`), as the wire format does; the encoder counts the children and reports a
mismatch. Lengths always use the fewest length bytes. `item(view)` copies a decoded item into
a new body without a tree in between (forwarding, and the fuzzer's oracle).

**Errors** are values, never exceptions: `Result<T>` is `std::expected<T, Error>`, and an
`Error` is a code plus the byte offset of the item at fault.

| Code | When |
|---|---|
| `truncated` | the input ends inside a header or an array's data, or before a list's last child |
| `zero_length_bytes` | a format byte with 0 length bytes (E5 allows 1–3) |
| `unknown_format` | a 6-bit code that is not one of the formats above |
| `bad_array_length` | data bytes not a multiple of the element size (`<U4>` of 3 bytes) |
| `too_deep` | more than 64 nested lists |
| `trailing_bytes` | bytes after the body's single item (`decode_prefix` allows them) |
| `too_long` | encode: more than 0xFFFFFF data bytes or children |
| `count_mismatch` | encode: a list with more or fewer children than it declared, or two top-level items |
| `wrong_format`, `out_of_range` | access: not the format asked for; no such child or value |

The encoder's errors are sticky: after the first, every call is a no-op and `finish()` returns
the error, so a builder chain is checked once at the end.

**Out of scope:** C2 (code 22, two-byte localized strings) is rejected as `unknown_format`;
GEM tools rarely use it and it has its own character-set prefix. ASCII and JIS-8 text is not
validated (tools put 8-bit bytes in `<A>` items) or converted. Non-minimal length bytes are
accepted on decode, so re-encoding a decoded body can be shorter than the original while
equal as a tree.

## How it is tested

- **Golden byte vectors** (`tests/test_secs.cpp`): one item of every format and bodies shaped
  like S1F14, S2F41, S2F42, S5F1 and S6F11, as bytes produced by **secsgem 0.3.0**, an
  independent Python implementation (`secsgem.secs.variables`, e.g. `U4(1).encode()`). Our
  encoder must write exactly those bytes, and the decoder must read them back to the expected
  SML. Length-byte boundaries (255 / 256 / 65,535 / 65,536 bytes) come from secsgem too.
- **Round-trip properties** (`tests/test_secs_roundtrip.cpp`): 3,000 random item trees (every
  format, up to 6 levels, arrays crossing into two length bytes, random bit patterns for
  floats, NaN payloads included): encode → decode gives the same tree, compared bit for bit,
  and re-encoding the decoded view gives the same bytes. Each child of a decoded list
  re-encodes on its own to its own bytes.
- **Errors**: every error code has a case with its expected offset; every strict prefix of
  every golden body is rejected as `truncated`.
- **Allocations** (`tests/test_secs_alloc.cpp`, its own binary because it replaces the global
  `operator new`): reading 100 wafer reports and encoding 100 more into a grown buffer makes
  0 heap allocations.
- **Fuzzing** (`fuzz/`): a libFuzzer target decodes any input; when it is valid, it reads
  every value (SML), copies it through the encoder, decodes the copy and checks the same tree
  comes back and that copying again is byte-identical. It also checks `decode` and
  `decode_prefix` agree on every input. The corpus in `fuzz/corpus/secs_item` (the golden
  bodies, edge cases, and what the fuzzer kept after a minimising merge) is replayed by a
  unit test in every build, so any crash file added there stays a regression test. CI runs
  the fuzzer for 60 s on every push.

The sanitizer builds (`asan`, `tsan`, `fuzz`) run all of it. ASan caught one real bug while
writing the codec: `for (auto v : *item.as<F>())` iterates a view that lives inside a
temporary `std::expected` destroyed before the loop body runs (C++23's P2718 extends that
lifetime; GCC 13 doesn't implement it). The dev build's output looked right. The fix, and
the rule: bind the view to a name first.

## Measurements

The codec, `build/release/bench/bench-secs --benchmark_repetitions=3` (medians; Release,
GCC 13.3). **Measured in a 4-thread cloud container** (Intel Xeon @ 2.10 GHz, AVX2, per
`waferedge-info`), not on the project laptop: the numbers show orders of magnitude and the
allocation count, and the laptop's run replaces them.

| Message | Bytes | Decode (validate) | Read every field | Encode | Allocations |
|---|---|---|---|---|---|
| Wafer report, 24×24 map | 629 | 54 ns | 120 ns | 111 ns | 0 |
| Wafer report, 40×40 map | 1,653 | 51 ns | 118 ns | 120 ns | 0 |
| Wafer report, 64×64 map | 4,149 | 53 ns | 118 ns | 165 ns | 0 |
| Wafer report, 200×200 map | 40,053 | 62 ns | 120 ns | 1,463 ns | 0 |
| S2F41 HOLD | 29 | — | 40 ns | 59 ns | 0 |
| (control: SML text of the 40×40 report) | 1,653 | | 62 µs | | 8 |

- **Decoding costs the same for every map size**: it reads the 12 headers and skips the bins.
  That is the zero-copy design at work: reading every field is ~8 M wafer reports/s on one
  core, whatever the map size; the "bytes/s" the benchmark also prints means nothing for it.
- **Encoding grows with the map**, because the bins have to be written. Copying one-byte
  values with one `memcpy` instead of value by value took the 64×64 report from 218 to 165
  ns and the 200×200 from 1.72 to 1.46 µs. What remains at 200×200 is the vector's
  zero-fill on `resize` plus the 40 KB copy.
- The allocation column counts the global `operator new` over the timed loop
  (`tests/support/alloc_counter.cpp`). The SML row shows the counter is live.
- For scale: the three classical signatures take ~71 µs for one map on the laptop's CPU
  (README). Even allowing for the different machines, the codec is two to three orders of
  magnitude below that, so it won't be the edge host's bottleneck.

Fuzzing, same container: 6 workers × 600 s, 49 M executions under ASan + UBSan, no crash;
coverage stopped growing (469 edges) after the first minutes. The minimising merge kept 473
inputs (93 KB).

## HSMS: the transport

### Frames

TCP has no message boundaries, so HSMS adds them. Every message is a 4-byte big-endian
length (the bytes that follow), a 10-byte header, and the body:

| Header bytes | Data message | Control message |
|---|---|---|
| 0–1 | session id (device id) | 0xFFFF (Select, Deselect, Linktest, Separate) |
| 2 | W-bit (reply expected) and stream | depends on the SType; Reject: the rejected SType |
| 3 | function | status (Select.rsp, Deselect.rsp) or reason (Reject) |
| 4 | PType: 0 (SECS-II) | 0 |
| 5 | SType: 0 | 1 Select.req, 2 .rsp, 3 Deselect.req, 4 .rsp, 5 Linktest.req, 6 .rsp, 7 Reject, 9 Separate |
| 6–9 | system bytes: the transaction id, copied by the reply | the same, per control transaction |

So the Select.req the edge host opens with is 14 bytes:
`00 00 00 0A | FF FF 00 00 00 01 00 00 00 01` (length 10; session 0xFFFF, PType 0, SType 1,
system bytes 1). The reply to S6F11 (stream 6, function 11, W-bit set: byte 2 = 0x86) is
S6F12 with the same system bytes; that, not the order of arrival, pairs them.

### States and timers

```
 NOT CONNECTED --TCP up--> NOT SELECTED --Select.req / .rsp ok--> SELECTED
       ^                      |      ^                                |
       +------ Separate, a timer, the TCP connection lost ------+  +-- Deselect
```

The **active** side (the host) connects and sends Select.req; the **passive** side (the tool)
listens and answers. Data messages are only accepted in SELECTED; anything else gets a
Reject.req (entity not selected). Timers, all configurable (`hsms::Config`):

| Timer | Default | Implemented as |
|---|---|---|
| T3 reply | 45 s | per open transaction (a primary with the W-bit); expiry reports `ReplyTimeout` and closes the transaction, not the connection; a later reply is `unmatched_reply` |
| T5 connect separation | 10 s | the active side waits T5 after a close or a failed connect |
| T6 control transaction | 5 s | no Select / Deselect / Linktest response: close |
| T7 not selected | 10 s | connected for T7 without SELECTED: close |
| T8 network intercharacter | 5 s | a frame half-received with no new byte for T8: close |
| Linktest period | off | Linktest.req every period while SELECTED; no answer in T6 closes |

Other rules: a length below 10 or above `max_message` (16 MiB + header by default) closes the
connection as soon as the 4 length bytes arrive. An unknown SType or a PType other than 0
gets a Reject.req. A response with no matching request gets "transaction not open". A
second Select.req gets "already active", and a Reject.req is never rejected back. Both
sides may select at the same time.

### Design: an engine without I/O, a driver with coroutines

The rules above live in `hsms::Protocol`, which has no socket and no clock. It takes received
bytes (`receive_buffer`, `on_received`) and the current time, and gives back events
(`poll`: `Selected`, `DataMessage`, `ReplyTimeout`, `Rejected`, `Closed`, ...), bytes to send
(`take_output`) and its next deadline. That makes every timer testable on a fake clock, at
its deadline and one nanosecond before, without sleeping.

`hsms::Session` drives it over TCP with three Asio C++20 coroutines per connection: a reader
(socket → engine → events to the handler), a writer (engine output → socket), and a sleeper
woken at the engine's next deadline. They are joined with Asio's `&&`. When the connection
ends (EOF, a protocol close), `finish()` closes the socket and wakes the timers, and all three
return. `run_active` reconnects after T5; `run_passive` serves one connection at a time. A
Session is single-threaded: other threads post to its executor.

No allocation once warm: the engine's receive and output buffers and its table of 64 open
transactions are reused, and output reaches the writer by swapping two vectors (so a write in
flight is never invalidated by the handler sending more). Asio recycles coroutine frames per
thread. One trap, found by the allocation test in the Clang build: joining the coroutines
with `||` (cancel the others when one ends) makes Asio keep a cancellation handler per
operation, and with libc++ a socket write's (40 bytes) and a timer wait's (32) differ, so
the writer's slot was freed and reallocated on every transaction: 2 allocations with Clang, 0
with GCC. Operations now carry no cancellation slot and the driver ends connections itself.

**Implemented subset:** HSMS-SS (one session per connection), one control transaction at a
time. **Out of scope:** HSMS-GS (several sessions on one connection), a passive side that
refuses connections (Select.rsp "not ready" is understood when received but never sent), and
TLS (the SEMI standards don't define it either).

### How HSMS is tested

- **Protocol engine** (`tests/test_hsms_protocol.cpp`, 21 cases, fake clock): the select
  handshake with its exact bytes, transactions and their replies, every timer at its deadline
  and one tick before, linktests, Deselect, Separate, every Reject reason, select refused by
  status or by Reject, simultaneous select, bad and oversized lengths, send errors, a full
  transaction table, and 50 random splits of a 40-message stream giving the same messages.
- **TCP** (`tests/test_hsms_session.cpp`, on 127.0.0.1 with short timers): select, an
  S1F1/S1F2 transaction and Separate; the active side reconnecting T5 after the passive side
  separates; a silent client closed after T7; periodic linktests keeping a quiet link up;
  1,000 back-to-back transactions with every reply matched.
- **Allocations** (`tests/test_hsms_alloc.cpp`): 0 per transaction for the engine and for
  two Sessions over TCP, after warm-up, in the GCC and the Clang + libc++ builds.
- **Fuzzing** (`fuzz/hsms_frames_target.cpp`): arbitrary bytes fed to a passive engine in
  chunks of 1–128 bytes on a moving fake clock, answering every primary. Its own output must
  parse as frames, data must only arrive in SELECTED, and after an hour of fake time an
  unselected connection must be closed. The corpus in `fuzz/corpus/hsms_frames` is replayed
  in every build. CI fuzzes it for 60 s next to the codec.

### HSMS measurements

`build/release/bench/bench-hsms --benchmark_repetitions=3` (medians; Release, GCC 13.3), in
the **same 4-thread cloud container** as the codec numbers, not the laptop. A transaction is a
primary with the W-bit and its empty reply, closed loop. Both Sessions share one thread and
one `io_context`, so the TCP times include both ends' work and two trips through the
kernel's loopback.

| Transaction body | Engine only, in memory | TCP loopback: mean | p50 | p99 | p99.9 | Allocations |
|---|---|---|---|---|---|---|
| empty | 180 ns | 7.2 µs | 6 µs | 20 µs | 46 µs | 0 |
| 1,600 B (40×40 map) | 217 ns | 7.2 µs | 6 µs | 21 µs | 42 µs | 0 |
| 40,000 B (200×200 map) | 2.8 µs | 11.6 µs | 10 µs | 32 µs | 76 µs | 0 |

- **The protocol costs ~0.2 µs per transaction; TCP costs ~7 µs.** The tails (p99.9) move
  by tens of µs between runs on a shared cloud machine; the medians are stable. The engine's share is
  the frame copy (into the output buffer, then into the peer's receive buffer); the rest is
  system calls and wake-ups. That gap is why the hot path's design effort goes into not
  allocating and not copying, rather than into the protocol logic itself.
- Percentiles come from a 1 µs histogram over every iteration. The loop is closed (the next
  primary waits for the reply), so these are round-trip service times on an idle link, not
  latency under load; phase 4 measures the pipeline at a constant offered rate, where
  coordinated omission matters.
- 0 allocations per transaction over TCP, Asio coroutines included, after 100 warm-up
  transactions (`tests/test_hsms_alloc.cpp` checks the same).

Fuzzing, same container: 3 workers × 600 s, 34 M executions under ASan + UBSan, no crash;
the minimising merge kept 213 inputs (56 KB).

## GEM: what the messages mean

GEM (SEMI E30) says which messages a tool serves and when. WaferEdge implements the subset
its closed loop needs, as `gem::Equipment` (the tool emulator's side) and `gem::Host` (the
edge host's side). Both are sans-I/O like `hsms::Protocol`: they take HSMS events and the
time, send through a `gem::Link`, and report GEM events from `poll()`.

### Communication state

```
 NOT COMMUNICATING --link selected: send S1F13--> WAIT CRA --S1F14 COMMACK 0--> COMMUNICATING
        ^                                           | denied, S1F0, T3             |
        |                                           v                              |
        |                       WAIT DELAY --establish delay (10 s)--> S1F13 again |
        +-------------------------------- HSMS link lost --------------------------+
```

Both sides send S1F13 when the HSMS link is selected; receiving one and answering COMMACK 0
also establishes communication, so the two requests crossing is fine. Until then every
other primary is answered with SxF0 (abort).

### Control state (equipment)

| State | Entered by | Serves |
|---|---|---|
| EQUIPMENT OFF-LINE | the operator (`go_offline`), the default at start | S1F13, S1F17 (refused: ONLACK 1); everything else SxF0 |
| ATTEMPT ON-LINE | the operator (`go_online`): S1F1 sent | as off-line, until S1F2 (→ ON-LINE) or S1F0 / T3 (→ HOST OFF-LINE) |
| HOST OFF-LINE | S1F15 from the host, a failed attempt | S1F13, S1F17 (accepted → ON-LINE) |
| ON-LINE LOCAL | the operator's local/remote switch | everything below; S2F41 gets HCACK 2 "cannot do now" |
| ON-LINE REMOTE | the switch | everything, and S2F41 HOLD / RELEASE reaches the app |

The equipment sends S6F11 and S5F1 only ON-LINE, and S1F13 / S1F1 off-line.

### Messages

| Message | Direction | Body | WaferEdge's use |
|---|---|---|---|
| S1F1 / S1F2 | either | — / `<L [2] <A MDLN> <A SOFTREV>>` (host: `<L [0]>`) | are you there; the on-line attempt |
| S1F13 / S1F14 | either | identity / `<L [2] <B COMMACK> identity>` | establish communication |
| S1F15 / S1F16 | host → equipment | — / `<B OFLACK>` | request off-line |
| S1F17 / S1F18 | host → equipment | — / `<B ONLACK>` (0 ok, 1 refused, 2 already) | request on-line |
| S6F11 / S6F12 | equipment → host | see below / `<B ACKC6>` | the wafer report |
| S5F1 / S5F2 | equipment → host | `<L [3] <B ALCD> <U4 ALID> <A ALTX>>` / `<B ACKC5>` | alarms (ALCD bit 7: set) |
| S2F41 / S2F42 | host → equipment | `<L [2] <A RCMD> <L [n] <L [2] <A CPNAME> CPVAL>>>` / `<L [2] <B HCACK> <L [n] <L [2] <A CPNAME> <B CPACK>>>>` | HOLD / RELEASE with LOTID |
| S9F3 / F5 / F7 / F9 | equipment → host | `<B MHEAD>`: the 10-byte header at fault | unknown stream / function, illegal data, T3 timeout |
| SxF0 | either | — | abort: not communicating, or off-line |

The wafer report, CEID 100 with the predefined report 10 (ADR-0011: no dynamic report
definition, S2F33/35/37):

```
<L [3]
  <U4 DATAID>                 a counter
  <U4 CEID 100>               wafer sorted
  <L [1]
    <L [2]
      <U4 RPTID 10>
      <L [5]
        <A LOTID> <U4 WAFERID> <U2 ROWS> <U2 COLS>
        <U1 BINS...>          rows x cols bin codes, row-major: 0 off wafer, 1 pass, >= 2 fail
      >
    >
  >
>
```

ID items are read in any unsigned width (GEM lets a tool choose); `decode_wafer_report`
returns the lot as a `string_view` and the map as a `WaferMapView` over the receive buffer.

S2F41 HCACK, decided by the equipment unless noted: 1 unknown RCMD, 2 ON-LINE LOCAL, 3
LOTID missing or not `<A>` (CPACK 3), and from the app: 0 done, 5 already in that condition,
6 no such lot.

### How GEM is tested

- **Messages** (`tests/test_gem_messages.cpp`): the encoders write secsgem's bytes; the
  decoders read them back as views; ids in any width; wrong shapes refused.
- **State machines** (`tests/test_gem.cpp`, fake clock, Equipment and Host on two
  `hsms::Protocol`s): communication from both sides, the operator putting the tool on-line,
  a wafer report with its map, HOLD and its HCACK, every HCACK the equipment decides alone,
  S9F3 / F5 / F7, off-line aborts, S1F15 / S1F17, WAIT DELAY after a denied or unanswered
  S1F13 (at the delay and one tick before), S9F9 after T3 on a report, alarms, link loss.
- **TCP** (`tests/test_gem_session.cpp`): communication, on-line, a wafer report and a HOLD
  answered, end to end over two `hsms::Session`s.
- **Allocations** (`tests/test_gem_alloc.cpp`): report, ack, hold and answer: 0 once warm.
- **Fuzzing** (`fuzz/gem_messages_target.cpp`): arbitrary records fed to a communicating,
  on-line Equipment and Host as primaries, replies to their own transactions, T3 timeouts
  and lost links; everything they send back must be a valid SECS-II body.

### GEM measurements

`build/release/bench/bench-hsms --benchmark_filter=gem` (median of 3; Release, GCC 13.3), same
4-thread cloud container as above, not the laptop. One cycle is the closed loop's protocol
work without the network: the equipment encodes and sends a 40×40 wafer report, the host
decodes it, acknowledges it and sends a HOLD, the equipment decodes the HOLD, the app answers
it, and the host reads the HCACK. Four messages through both state machines and two HSMS
engines.

| Cycle | Time | Allocations |
|---|---|---|
| report + S6F12 + HOLD + S2F42, 40×40 map | 1.4 µs | 0 |

Over TCP each of those messages adds a loopback trip (~7 µs per transaction above), so the
protocol stack costs a few µs between a sorted wafer and its lot hold. Phase 4 measures the
whole path, detectors included, under load.

Fuzzing, same container: 3 workers × 600 s, 17 M executions under ASan + UBSan, no crash;
the minimising merge kept 665 inputs (300 KB).

## Reproduce

```sh
cmake --workflow --preset dev                       # all tests, codec included
cmake --workflow --preset asan                      # the same under ASan + UBSan
build/dev/tests/waferedge-tests "[hsms]"            # HSMS: fake-clock and TCP tests
build/dev/tests/waferedge-tests "[gem]"             # GEM: messages, state machines, TCP
build/dev/tests/waferedge-alloc-tests               # 0 allocations on the hot path
cmake --workflow --preset release && build/release/bench/bench-secs
build/release/bench/bench-hsms                      # transactions/s, round trips, allocations

# Fuzzing (Clang 18 + libFuzzer; needs libclang-rt-18-dev and libc++-18-dev):
cmake --preset fuzz && cmake --build --preset fuzz
ctest --preset fuzz                                 # includes replaying the corpus
mkdir -p /tmp/secs-corpus
build/fuzz/fuzz/fuzz-secs-item -max_total_time=600 -jobs=6 -workers=6 \
    /tmp/secs-corpus fuzz/corpus/secs_item
# Keep what is new, minimised, in the checked-in corpus:
build/fuzz/fuzz/fuzz-secs-item -merge=1 fuzz/corpus/secs_item /tmp/secs-corpus
# The same for HSMS framing:
mkdir -p /tmp/hsms-corpus
build/fuzz/fuzz/fuzz-hsms-frames -max_total_time=600 /tmp/hsms-corpus fuzz/corpus/hsms_frames
# and for the GEM layer:
mkdir -p /tmp/gem-corpus
build/fuzz/fuzz/fuzz-gem-messages -max_total_time=600 /tmp/gem-corpus fuzz/corpus/gem_messages
```

New inputs go to the first directory given; the checked-in corpus is only read, except by
`-merge=1`. A crash writes `crash-<sha1>` to the working directory: fix the bug and copy the
file into `fuzz/corpus/secs_item` (or `hsms_frames`, `gem_messages`), which makes it a
regression test.

The golden vectors came from secsgem in a throwaway virtualenv
(`pip install secsgem==0.3.0`); e.g. `secsgem.secs.variables.String("LOT-1").encode()` gives
`41 05 4C 4F 54 2D 31`. Lists are built from `List`'s item header plus the children's bytes.
