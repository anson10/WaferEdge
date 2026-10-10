# SECS-II and HSMS

How the tool emulator and the edge host talk: the equipment protocol stack fabs use, from the
bytes up. This page covers what is implemented so far, the **SECS-II item codec** (phase 3,
first part). HSMS framing, the connection state machine and the GEM messages are added as
they land. Code: `include/waferedge/secs/`, `src/secs/`; design decision: ADR-0009.

SEMI E5 (SECS-II), E37 (HSMS) and E30 (GEM) are paid standards. This implementation works from
public descriptions and open-source implementations (secsgem, secs4net), covers a subset, and
is not certified against the standards.

## The stack in one picture

```
 GEM (E30)       which messages a tool must support and what they mean:
                 S1F13 establish communication, S6F11 event report, S2F41 host command, ...
 SECS-II (E5)    the message body: one item, a self-describing tree    <- this page
 HSMS (E37)      TCP: 4-byte length, 10-byte header (session, stream, function, system bytes)
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

## Reproduce

```sh
cmake --workflow --preset dev                       # all tests, codec included
cmake --workflow --preset asan                      # the same under ASan + UBSan
build/dev/tests/waferedge-alloc-tests               # 0 allocations on the hot path
cmake --workflow --preset release && build/release/bench/bench-secs

# Fuzzing (Clang 18 + libFuzzer; needs libclang-rt-18-dev and libc++-18-dev):
cmake --preset fuzz && cmake --build --preset fuzz
ctest --preset fuzz                                 # includes replaying the corpus
mkdir -p /tmp/secs-corpus
build/fuzz/fuzz/fuzz-secs-item -max_total_time=600 -jobs=6 -workers=6 \
    /tmp/secs-corpus fuzz/corpus/secs_item
# Keep what is new, minimised, in the checked-in corpus:
build/fuzz/fuzz/fuzz-secs-item -merge=1 fuzz/corpus/secs_item /tmp/secs-corpus
```

New inputs go to the first directory given; the checked-in corpus is only read, except by
`-merge=1`. A crash writes `crash-<sha1>` to the working directory: fix the bug and copy the
file into `fuzz/corpus/secs_item`, which makes it a regression test.

The golden vectors came from secsgem in a throwaway virtualenv
(`pip install secsgem==0.3.0`); e.g. `secsgem.secs.variables.String("LOT-1").encode()` gives
`41 05 4C 4F 54 2D 31`. Lists are built from `List`'s item header plus the children's bytes.
