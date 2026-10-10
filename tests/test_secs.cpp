// SECS-II codec: golden byte vectors, header boundaries, every decode / encode error, typed
// access. The golden bytes were produced by secsgem 0.3.0 (an independent Python SECS
// implementation), not by this encoder; docs/secs.md says how to regenerate them.
#include "waferedge/secs/encoder.hpp"
#include "waferedge/secs/item.hpp"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

using namespace waferedge::secs;
using Bytes = std::vector<std::uint8_t>;

namespace {

struct Golden {
    const char* name;
    Bytes bytes;
    void (*encode)(Encoder&);
    const char* sml;
};

// Each case: secsgem's bytes, the same item built with our encoder, and its SML.
const std::vector<Golden>& goldens() {
    static const std::vector<Golden> cases = {
        {"empty list", {0x01, 0x00}, [](Encoder& e) { e.list(0); }, "<L [0]>"},
        {"binary",
         {0x21, 0x02, 0x00, 0xFF},
         [](Encoder& e) { e.binary(Bytes{0x00, 0xFF}); },
         "<B 0x00 0xFF>"},
        {"boolean",
         {0x25, 0x02, 0x01, 0x00},
         [](Encoder& e) {
             const std::array<bool, 2> values = {true, false};
             e.array<Format::boolean>(values);
         },
         "<BOOLEAN T F>"},
        {"ascii",
         {0x41, 0x05, 0x4C, 0x4F, 0x54, 0x2D, 0x31},
         [](Encoder& e) { e.ascii("LOT-1"); },
         "<A \"LOT-1\">"},
        {"empty ascii", {0x41, 0x00}, [](Encoder& e) { e.ascii(""); }, "<A>"},
        {"jis8", {0x45, 0x02, 0x61, 0x62}, [](Encoder& e) { e.jis8("ab"); }, "<J \"ab\">"},
        {"i1",
         {0x65, 0x02, 0xFF, 0x7F},
         [](Encoder& e) {
             const std::array<std::int8_t, 2> values = {-1, 127};
             e.array<Format::i1>(values);
         },
         "<I1 -1 127>"},
        {"i2", {0x69, 0x02, 0xFF, 0xFE}, [](Encoder& e) { e.i2(-2); }, "<I2 -2>"},
        {"i4",
         {0x71, 0x04, 0xFF, 0xFE, 0x79, 0x60},
         [](Encoder& e) { e.i4(-100000); },
         "<I4 -100000>"},
        {"i8",
         {0x61, 0x08, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF},
         [](Encoder& e) { e.i8(-1); },
         "<I8 -1>"},
        {"u1",
         {0xA5, 0x02, 0x00, 0xFF},
         [](Encoder& e) {
             const std::array<std::uint8_t, 2> values = {0, 255};
             e.array<Format::u1>(values);
         },
         "<U1 0 255>"},
        {"u2", {0xA9, 0x02, 0xFF, 0xFF}, [](Encoder& e) { e.u2(65535); }, "<U2 65535>"},
        {"u4", {0xB1, 0x04, 0x00, 0x00, 0x00, 0x01}, [](Encoder& e) { e.u4(1); }, "<U4 1>"},
        {"u8",
         {0xA1, 0x08, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
         [](Encoder& e) { e.u8(std::uint64_t{1} << 63U); },
         "<U8 9223372036854775808>"},
        {"f4",
         {0x91, 0x08, 0x3F, 0x80, 0x00, 0x00, 0xC0, 0x20, 0x00, 0x00},
         [](Encoder& e) {
             const std::array<float, 2> values = {1.0F, -2.5F};
             e.array<Format::f4>(values);
         },
         "<F4 1 -2.5>"},
        {"f8",
         {0x81, 0x08, 0x3F, 0xB9, 0x99, 0x99, 0x99, 0x99, 0x99, 0x9A},
         [](Encoder& e) { e.f8(0.1); },
         "<F8 0.1>"},
        // Bodies shaped like the GEM messages phase 3 uses (docs/secs.md).
        {"S1F14 establish communication ack",
         {0x01, 0x02, 0x21, 0x01, 0x00, 0x01, 0x02, 0x41, 0x0D, 0x57, 0x41, 0x46, 0x45, 0x52, 0x45,
          0x44, 0x47, 0x45, 0x2D, 0x45, 0x4D, 0x55, 0x41, 0x05, 0x31, 0x2E, 0x30, 0x2E, 0x30},
         [](Encoder& e) {
             e.list(2).binary(Bytes{0x00}).list(2).ascii("WAFEREDGE-EMU").ascii("1.0.0");
         },
         "<L [2]\n  <B 0x00>\n  <L [2]\n    <A \"WAFEREDGE-EMU\">\n    <A \"1.0.0\">\n  >\n>"},
        {"S2F41 hold lot",
         {0x01, 0x02, 0x41, 0x04, 0x48, 0x4F, 0x4C, 0x44, 0x01, 0x01, 0x01, 0x02, 0x41, 0x05, 0x4C,
          0x4F, 0x54, 0x49, 0x44, 0x41, 0x08, 0x4C, 0x4F, 0x54, 0x2D, 0x30, 0x30, 0x34, 0x32},
         [](Encoder& e) {
             e.list(2).ascii("HOLD").list(1).list(2).ascii("LOTID").ascii("LOT-0042");
         },
         "<L [2]\n  <A \"HOLD\">\n  <L [1]\n    <L [2]\n      <A \"LOTID\">\n"
         "      <A \"LOT-0042\">\n    >\n  >\n>"},
        {"S2F42 host command ack",
         {0x01, 0x02, 0x21, 0x01, 0x00, 0x01, 0x00},
         [](Encoder& e) { e.list(2).binary(Bytes{0x00}).list(0); },
         "<L [2]\n  <B 0x00>\n  <L [0]>\n>"},
        {"S5F1 alarm",
         {0x01, 0x03, 0x21, 0x01, 0x80, 0xB1, 0x04, 0x00, 0x00, 0x00, 0x07,
          0x41, 0x09, 0x45, 0x44, 0x47, 0x45, 0x2D, 0x52, 0x49, 0x4E, 0x47},
         [](Encoder& e) { e.list(3).binary(Bytes{0x80}).u4(7).ascii("EDGE-RING"); },
         "<L [3]\n  <B 0x80>\n  <U4 7>\n  <A \"EDGE-RING\">\n>"},
        {"S6F11 wafer map event",
         {0x01, 0x03, 0xB1, 0x04, 0x00, 0x00, 0x00, 0x01, 0xB1, 0x04, 0x00, 0x00, 0x00,
          0x64, 0x01, 0x01, 0x01, 0x02, 0xB1, 0x04, 0x00, 0x00, 0x00, 0x0A, 0x01, 0x05,
          0x41, 0x08, 0x4C, 0x4F, 0x54, 0x2D, 0x30, 0x30, 0x34, 0x32, 0xB1, 0x04, 0x00,
          0x00, 0x00, 0x11, 0xA9, 0x02, 0x00, 0x03, 0xA9, 0x02, 0x00, 0x03, 0xA5, 0x09,
          0x00, 0x01, 0x00, 0x01, 0x02, 0x01, 0x00, 0x01, 0x00},
         [](Encoder& e) {
             const std::array<std::uint8_t, 9> bins = {0, 1, 0, 1, 2, 1, 0, 1, 0};
             e.list(3).u4(1).u4(100).list(1).list(2).u4(10).list(5);
             e.ascii("LOT-0042").u4(17).u2(3).u2(3).array<Format::u1>(bins);
         },
         "<L [3]\n  <U4 1>\n  <U4 100>\n  <L [1]\n    <L [2]\n      <U4 10>\n      <L [5]\n"
         "        <A \"LOT-0042\">\n        <U4 17>\n        <U2 3>\n        <U2 3>\n"
         "        <U1 0 1 0 1 2 1 0 1 0>\n      >\n    >\n  >\n>"},
    };
    return cases;
}

Bytes encode_with(void (*build)(Encoder&)) {
    Bytes out;
    Encoder e(out);
    build(e);
    auto body = e.finish();
    REQUIRE(body.has_value());
    return {body->begin(), body->end()};
}

} // namespace

TEST_CASE("the format table matches E5's codes") {
    using enum Format;
    const auto one_length_byte = std::to_array<std::pair<Format, std::uint8_t>>({
        {list, 0x01},
        {binary, 0x21},
        {boolean, 0x25},
        {ascii, 0x41},
        {jis8, 0x45},
        {i8, 0x61},
        {i1, 0x65},
        {i2, 0x69},
        {i4, 0x71},
        {f8, 0x81},
        {f4, 0x91},
        {u8, 0xA1},
        {u1, 0xA5},
        {u2, 0xA9},
        {u4, 0xB1},
    });
    for (const auto& [f, byte] : one_length_byte) {
        CHECK(format_byte(f, 1) == byte);
        CHECK(format_from_code(static_cast<std::uint8_t>(byte >> 2U)) == f);
    }
    std::size_t valid = 0;
    for (std::uint8_t code = 0; code < 64; ++code) {
        valid += format_from_code(code).has_value() ? 1U : 0U;
    }
    CHECK(valid == kAllFormats.size());
}

TEST_CASE("golden vectors: our encoder writes secsgem's bytes") {
    for (const auto& g : goldens()) {
        INFO(g.name);
        CHECK(encode_with(g.encode) == g.bytes);
    }
}

TEST_CASE("golden vectors decode to the expected tree") {
    for (const auto& g : goldens()) {
        INFO(g.name);
        auto item = decode(g.bytes);
        REQUIRE(item.has_value());
        CHECK(to_sml(*item) == g.sml);
        CHECK(item->encoded().size() == g.bytes.size());
    }
}

TEST_CASE("decoded views point into the receive buffer") {
    const auto& g = goldens().back(); // the S6F11 wafer map event
    auto body = decode(g.bytes);
    REQUIRE(body.has_value());
    // DATAID, CEID, then the report list: L[1] L[2] RPTID L[5] ... bins.
    auto top = body->list();
    REQUIRE(top.has_value());
    REQUIRE(top->size() == 3);
    CHECK(top->at(0)->unsigned_scalar() == 1U);
    CHECK(top->at(1)->scalar<Format::u4>() == 100U);
    auto report = top->at(2)->list()->at(0)->list();
    REQUIRE(report.has_value());
    CHECK(report->at(0)->unsigned_scalar() == 10U);
    auto values = report->at(1)->list();
    REQUIRE(values.has_value());
    CHECK(values->at(0)->text() == "LOT-0042");
    auto bins = values->at(4)->as<Format::u1>();
    REQUIRE(bins.has_value());
    CHECK(bins->size() == 9);
    CHECK(bins->bytes().data() == g.bytes.data() + g.bytes.size() - 9); // no copy
    CHECK((*bins)[4] == 2);
}

TEST_CASE("typed access checks the format and the count") {
    const Bytes u4s = {0xB1, 0x08, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00};
    auto item = decode(u4s);
    REQUIRE(item.has_value());
    CHECK(item->size() == 2);
    CHECK(item->as<Format::u2>().error().code == Errc::wrong_format);
    CHECK(item->list().error().code == Errc::wrong_format);
    CHECK(item->text().error().code == Errc::wrong_format);
    CHECK(item->scalar<Format::u4>().error().code == Errc::out_of_range); // two values
    auto values = item->as<Format::u4>();
    REQUIRE(values.has_value());
    CHECK(std::vector<std::uint32_t>(values->begin(), values->end()) ==
          std::vector<std::uint32_t>{1, 256});

    const Bytes list = {0x01, 0x01, 0x41, 0x00};
    auto l = decode(list)->list();
    REQUIRE(l.has_value());
    CHECK(l->at(1).error().code == Errc::out_of_range);
    CHECK(l->at(0)->unsigned_scalar().error().code == Errc::wrong_format);
}

TEST_CASE("length bytes: the fewest that fit, at every boundary") {
    struct Case {
        std::size_t length;
        Bytes header; // from secsgem for A items of this length
    };
    const auto cases = std::to_array<Case>({
        {0, {0x41, 0x00}},
        {255, {0x41, 0xFF}},
        {256, {0x42, 0x01, 0x00}},
        {65535, {0x42, 0xFF, 0xFF}},
        {65536, {0x43, 0x01, 0x00, 0x00}},
    });
    for (const auto& c : cases) {
        INFO(c.length);
        const std::string text(c.length, 'x');
        Bytes out;
        Encoder e(out);
        e.ascii(text);
        REQUIRE(e.finish().has_value());
        CHECK(Bytes(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(c.header.size())) ==
              c.header);
        auto item = decode(out);
        REQUIRE(item.has_value());
        CHECK(item->text()->size() == c.length);
    }
    // The decoder also accepts more length bytes than needed (E5 allows it); the encoder
    // writes the fewest when it copies such an item.
    const Bytes padded = {0x43, 0x00, 0x00, 0x01, 0x41};
    auto item = decode(padded);
    REQUIRE(item.has_value());
    CHECK(item->text() == "A");
    Bytes out;
    Encoder e(out);
    e.item(*item);
    CHECK(Bytes(e.finish()->begin(), e.finish()->end()) == Bytes{0x41, 0x01, 0x41});
}

TEST_CASE("decode errors carry a code and the offset of the item at fault") {
    struct Case {
        const char* name;
        Bytes bytes;
        Error error;
    };
    const auto cases = std::to_array<Case>({
        {"empty input", {}, {Errc::truncated, 0}},
        {"format byte only", {0x41}, {Errc::truncated, 0}},
        {"2 length bytes, 1 present", {0x42, 0x00}, {Errc::truncated, 0}},
        {"data cut short", {0x41, 0x03, 0x41, 0x42}, {Errc::truncated, 0}},
        {"zero length bytes", {0x40, 0x00}, {Errc::zero_length_bytes, 0}},
        {"C2 (out of scope)", {0x49, 0x00}, {Errc::unknown_format, 0}},
        {"code 077", {0xFD, 0x00}, {Errc::unknown_format, 0}},
        {"U4 of 3 bytes", {0xB1, 0x03, 0x00, 0x00, 0x01}, {Errc::bad_array_length, 0}},
        {"list missing a child", {0x01, 0x02, 0xA5, 0x01, 0x07}, {Errc::truncated, 5}},
        {"bad child",
         {0x01, 0x02, 0xA5, 0x01, 0x07, 0x69, 0x01, 0x00},
         {Errc::bad_array_length, 5}},
        {"two top-level items", {0xA5, 0x01, 0x07, 0x01, 0x00}, {Errc::trailing_bytes, 3}},
    });
    for (const auto& c : cases) {
        INFO(c.name);
        auto item = decode(c.bytes);
        REQUIRE_FALSE(item.has_value());
        CHECK(item.error() == c.error);
        CHECK_FALSE(describe(item.error()).empty());
    }
    // decode_prefix takes the first item and leaves the rest to the caller.
    const Bytes two = {0xA5, 0x01, 0x07, 0x01, 0x00};
    auto first = decode_prefix(two);
    REQUIRE(first.has_value());
    CHECK(first->encoded().size() == 3);
}

TEST_CASE("nesting is limited to kMaxDepth lists, without recursion") {
    const auto nested = [](std::size_t depth) {
        Bytes b;
        for (std::size_t i = 0; i < depth; ++i) {
            b.insert(b.end(), {0x01, 0x01});
        }
        b.insert(b.end(), {0x01, 0x00});
        return b;
    };
    // kMaxDepth open lists, the innermost an empty list (it opens nothing).
    CHECK(decode(nested(kMaxDepth)).has_value());
    auto deep = decode(nested(kMaxDepth + 1));
    REQUIRE_FALSE(deep.has_value());
    CHECK(deep.error() == Error{Errc::too_deep, 2 * kMaxDepth});
    // A million levels: an error, not a stack overflow.
    CHECK(decode(nested(1'000'000)).error().code == Errc::too_deep);
}

TEST_CASE("every strict prefix of a valid body is rejected") {
    for (const auto& g : goldens()) {
        INFO(g.name);
        for (std::size_t n = 0; n < g.bytes.size(); ++n) {
            auto item = decode(std::span(g.bytes).first(n));
            REQUIRE_FALSE(item.has_value());
            CHECK(item.error().code == Errc::truncated);
        }
    }
}

TEST_CASE("encoder errors are sticky and reported by finish") {
    Bytes out;
    SECTION("a list short of children") {
        Encoder e(out);
        e.list(2).u1(1);
        CHECK(e.finish().error().code == Errc::count_mismatch);
    }
    SECTION("a second top-level item") {
        Encoder e(out);
        e.u1(1).u1(2);
        CHECK(e.finish().error().code == Errc::count_mismatch);
    }
    SECTION("more than three length bytes can say") {
        Encoder e(out);
        e.list(kMaxLength + 1).u1(1);
        CHECK_FALSE(e.ok());
        CHECK(e.finish().error().code == Errc::too_long);
    }
    SECTION("the largest list header is fine") {
        Encoder e(out);
        e.list(kMaxLength);
        CHECK(out == Bytes{0x03, 0xFF, 0xFF, 0xFF});
        CHECK(e.finish().error().code == Errc::count_mismatch); // no children yet
    }
    SECTION("too deep") {
        Encoder e(out);
        for (std::size_t i = 0; i <= kMaxDepth; ++i) {
            e.list(1);
        }
        CHECK(e.finish().error().code == Errc::too_deep);
    }
    SECTION("nothing written is an empty body") {
        Encoder e(out);
        CHECK(e.finish()->empty());
    }
}

TEST_CASE("the encoder reuses its buffer") {
    Bytes out;
    {
        Encoder e(out);
        e.ascii(std::string(1000, 'x'));
    }
    const auto capacity = out.capacity();
    const auto* data = out.data();
    Encoder e(out);
    e.list(2).u4(1).ascii("short");
    REQUIRE(e.finish().has_value());
    CHECK(out.capacity() == capacity);
    CHECK(out.data() == data);
}

TEST_CASE("SML shows unprintable text bytes as hex") {
    Bytes out;
    Encoder e(out);
    e.ascii(std::string("AB\nC\"", 5));
    CHECK(to_sml(*decode(*e.finish())) == "<A \"AB\" 0x0A \"C\" 0x22>");
}
