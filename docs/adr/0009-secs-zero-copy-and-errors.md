# ADR-0009: SECS-II as validated views over the receive buffer, errors as values

- **Status:** Accepted
- **Date:** 2026-10-10

## Context
Every wafer the tool sorts arrives as an S6F11 event report: a SECS-II item tree of a few ids
and one array of 600–40,000 bin bytes (docs/secs.md). The edge host decodes it on the network
thread, and anything that thread spends delays every wafer behind it; the pipeline (phase 4)
must not allocate per message at all. The bytes come from a peer on a TCP socket, so
malformed input is ordinary: a buggy tool, a half-implemented emulator, a fuzzer. The parser
must reject any input with an error and never read out of bounds, recurse without bound or
throw through the network loop. GEM code (phase 3) then reads fields by position and format,
and needs every "wrong format here" case to be a checkable value too.

## Options considered
1. **Decode into an owning tree** (`std::variant` of vectors per item, as secsgem and secs4net
   do): easy to use; one allocation per item, and the bin array is copied, on every wafer.
2. **Pull parser / SAX-style callbacks**: no allocation and no tree, but GEM code turns into
   a state machine over events, and validation of the whole body is mixed into the field
   reading.
3. **Index tape** (simdjson's approach): one pass writes a flat array of node offsets into a
   reusable buffer; O(1) navigation. Needs that buffer sized for the worst message, and its
   speed matters for documents with thousands of nodes, not for bodies of ten items.
4. **Validate once, then lazy views**: one pass over the headers checks the whole body
   (lengths, formats, depth, nothing past the end), then `ItemView`s are pointers into the
   buffer; children and values are read on access, byte-swapped as they are read.

For errors: exceptions (the codebase builds with them, but a throw per bad message on the
network thread costs microseconds and an unwinding path nobody tests), error codes with out
parameters, or `std::expected<T, Error>` (GCC 13 and libc++ 18 both ship it; ADR-0001).

## Decision
Option 4 with `std::expected`. `decode()` validates iteratively with an explicit stack of
`kMaxDepth` (64) open lists, so a hostile million-level nesting is a `too_deep` error, not a
stack overflow, and every later recursion over a decoded tree is bounded by 64. Only
`decode()` constructs views, so a view's existence proves its bytes were validated, and
accessors need no bounds checks beyond "is this the format you asked for". Errors are an
`Error{Errc, byte offset}` in a `std::expected`; there are no exceptions in the codec.

The encoder writes into a caller-owned `std::vector<std::uint8_t>` that it clears but never
shrinks, so after the first message of the largest size it allocates nothing. Its errors are
sticky: the first error stops all writing and `finish()` reports it, so a chain of builder
calls needs one check at the end instead of one per call. Format codes, element sizes and SML
names are one `constexpr` table indexed by the 6-bit code, with `static_assert`s tying it to
the C++ value types.

## Consequences
- Decoding a 40×40 wafer report touches its 12 headers and none of the 1,600 bin bytes; the
  bins reach the detectors as a `WaferMapView` over the receive buffer. Measured: 0
  allocations per message decoded or encoded (`bench-secs`, the alloc test).
- Views dangle if the buffer goes away first, as `std::string_view` does. The HSMS layer
  must keep a message's buffer alive until the analytics are done with it (phase 4's rings
  will own the buffers).
- Moving to a list's next child walks the current child's headers; indexing a long list is
  O(n) per index. Fine for GEM's short lists, wrong for a list of 10,000 items. If one shows
  up, iterate instead of indexing, or add the index tape of option 3 for that message.
- `std::expected` with views has a C++ trap: `for (auto v : *item.as<F>())` iterates a view
  inside a temporary that dies before the loop body (fixed by C++23's P2718, which GCC 13
  doesn't implement). ASan found it in our own code; the rule is to bind the view to a name
  first.
- Non-minimal length bytes (a 3-byte length for a 5-byte item) are accepted on decode and
  never produced on encode, so re-encoding a decoded body is not always byte-identical to
  it, only equal as a tree. The fuzz target checks exactly that.
