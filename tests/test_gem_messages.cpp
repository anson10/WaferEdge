// GEM message layouts: the typed encoders write the golden bytes (from secsgem, as in
// test_secs.cpp), the decoders read them back as views, and wrong shapes are refused.
#include "support/synth.hpp"
#include "waferedge/gem/messages.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <vector>

using namespace waferedge;
using namespace waferedge::gem;
using Bytes = std::vector<std::uint8_t>;

namespace {

template <typename Build>
Bytes encode(Build&& build) {
    Bytes out;
    secs::Encoder e(out);
    build(e);
    REQUIRE(e.finish().has_value());
    return out;
}

secs::ItemView item(const Bytes& bytes) {
    auto v = secs::decode(bytes);
    REQUIRE(v.has_value());
    return *v;
}

} // namespace

TEST_CASE("S1F14 and S2F41 encode to secsgem's bytes", "[gem]") {
    const Identity emu{"WAFEREDGE-EMU", "1.0.0"};
    CHECK(encode([&](secs::Encoder& e) { encode_s1f14(e, Commack::accepted, &emu); }) ==
          Bytes{0x01, 0x02, 0x21, 0x01, 0x00, 0x01, 0x02, 0x41, 0x0D, 0x57,
                0x41, 0x46, 0x45, 0x52, 0x45, 0x44, 0x47, 0x45, 0x2D, 0x45,
                0x4D, 0x55, 0x41, 0x05, 0x31, 0x2E, 0x30, 0x2E, 0x30});
    CHECK(encode([](secs::Encoder& e) { encode_lot_command(e, {LotAction::hold, "LOT-0042"}); }) ==
          Bytes{0x01, 0x02, 0x41, 0x04, 0x48, 0x4F, 0x4C, 0x44, 0x01, 0x01,
                0x01, 0x02, 0x41, 0x05, 0x4C, 0x4F, 0x54, 0x49, 0x44, 0x41,
                0x08, 0x4C, 0x4F, 0x54, 0x2D, 0x30, 0x30, 0x34, 0x32});
    CHECK(encode([](secs::Encoder& e) { encode_host_command_ack(e, Hcack::done); }) ==
          Bytes{0x01, 0x02, 0x21, 0x01, 0x00, 0x01, 0x00});
    CHECK(encode([](secs::Encoder& e) { encode_alarm(e, {true, 0, 7, "EDGE-RING"}); }) ==
          Bytes{0x01, 0x03, 0x21, 0x01, 0x80, 0xB1, 0x04, 0x00, 0x00, 0x00, 0x07,
                0x41, 0x09, 0x45, 0x44, 0x47, 0x45, 0x2D, 0x52, 0x49, 0x4E, 0x47});
}

TEST_CASE("the wafer report: secsgem's bytes, and the map as a view", "[gem]") {
    const std::array<std::uint8_t, 9> bins = {0, 1, 0, 1, 2, 1, 0, 1, 0};
    const Bytes golden = {0x01, 0x03, 0xB1, 0x04, 0x00, 0x00, 0x00, 0x01, 0xB1, 0x04, 0x00,
                          0x00, 0x00, 0x64, 0x01, 0x01, 0x01, 0x02, 0xB1, 0x04, 0x00, 0x00,
                          0x00, 0x0A, 0x01, 0x05, 0x41, 0x08, 0x4C, 0x4F, 0x54, 0x2D, 0x30,
                          0x30, 0x34, 0x32, 0xB1, 0x04, 0x00, 0x00, 0x00, 0x11, 0xA9, 0x02,
                          0x00, 0x03, 0xA9, 0x02, 0x00, 0x03, 0xA5, 0x09, 0x00, 0x01, 0x00,
                          0x01, 0x02, 0x01, 0x00, 0x01, 0x00};
    CHECK(encode([&](secs::Encoder& e) {
              encode_wafer_report(e, 1, "LOT-0042", 17, WaferMapView(bins, 3, 3));
          }) == golden);
    auto report = decode_wafer_report(std::span<const std::uint8_t>(golden));
    REQUIRE(report.has_value());
    CHECK(report->data_id == 1);
    CHECK(report->lot == "LOT-0042");
    CHECK(report->wafer == 17);
    CHECK(report->map.rows() == 3);
    CHECK(report->map.cols() == 3);
    CHECK(std::ranges::equal(report->map.bins(), bins));
    CHECK(report->map.bins().data() == golden.data() + golden.size() - 9); // no copy

    const WaferMap big = synth::random_map(200, 200, 100, 3); // two length bytes for the bins
    const Bytes body = encode([&](secs::Encoder& e) { encode_wafer_report(e, 9, "L", 1, big); });
    auto decoded = decode_wafer_report(std::span<const std::uint8_t>(body));
    REQUIRE(decoded.has_value());
    CHECK(std::ranges::equal(decoded->map.bins(), big.view().bins()));
}

TEST_CASE("ids may come in any unsigned width", "[gem]") {
    // DATAID as U1, CEID as U2, RPTID as U8, WAFERID as U1, ROWS / COLS as U4.
    const std::array<std::uint8_t, 4> bins = {1, 2, 1, 1};
    const Bytes body = encode([&](secs::Encoder& e) {
        e.list(3).u1(5).u2(100).list(1).list(2).u8(10).list(5);
        e.ascii("LOT").u1(3).u4(2).u4(2).array<secs::Format::u1>(bins);
    });
    auto r = decode_wafer_report(std::span<const std::uint8_t>(body));
    REQUIRE(r.has_value());
    CHECK(r->data_id == 5);
    CHECK(r->wafer == 3);
    CHECK(r->map.rows() == 2);
}

TEST_CASE("wafer reports of the wrong shape are refused", "[gem]") {
    const std::array<std::uint8_t, 4> bins = {1, 1, 1, 1};
    const auto report = [&](std::uint32_t ceid, std::uint32_t rptid, std::uint16_t rows,
                            bool bins_as_binary) {
        return encode([&](secs::Encoder& e) {
            e.list(3).u4(1).u4(ceid).list(1).list(2).u4(rptid).list(5);
            e.ascii("LOT").u4(1).u2(rows).u2(2);
            if (bins_as_binary) {
                e.binary(bins);
            } else {
                e.array<secs::Format::u1>(bins);
            }
        });
    };
    CHECK(
        decode_wafer_report(std::span<const std::uint8_t>(report(100, 10, 2, false))).has_value());
    CHECK_FALSE(decode_wafer_report(std::span<const std::uint8_t>(report(101, 10, 2, false))));
    CHECK_FALSE(decode_wafer_report(std::span<const std::uint8_t>(report(100, 11, 2, false))));
    CHECK_FALSE(decode_wafer_report(std::span<const std::uint8_t>(report(100, 10, 3, false))));
    CHECK_FALSE(decode_wafer_report(std::span<const std::uint8_t>(report(100, 10, 2, true))));
    // Not even an event report.
    CHECK_FALSE(decode_event_report(item(encode([](secs::Encoder& e) { e.list(0); }))));
    // Another event decodes as an event report, not as a wafer report.
    const Bytes other = encode([](secs::Encoder& e) { e.list(3).u4(1).u4(200).list(0); });
    CHECK(decode_event_report(item(other))->ceid == 200);
    CHECK_FALSE(decode_wafer_report(item(other)));
}

TEST_CASE("host commands: parameters by name, formats checked", "[gem]") {
    const Bytes hold =
        encode([](secs::Encoder& e) { encode_lot_command(e, {LotAction::release, "LOT-7"}); });
    auto command = decode_host_command(item(hold));
    REQUIRE(command.has_value());
    CHECK(command->rcmd == "RELEASE");
    auto lot = find_parameter(*command, kCpLotId);
    REQUIRE(lot.has_value());
    CHECK(lot->text() == "LOT-7");
    CHECK(find_parameter(*command, "PPID").error().code == secs::Errc::out_of_range);

    // A parameter that isn't <L [2] <A> value> makes the whole command malformed.
    const Bytes bad = encode(
        [](secs::Encoder& e) { e.list(2).ascii("HOLD").list(1).list(2).u4(1).ascii("LOT"); });
    CHECK_FALSE(decode_host_command(item(bad)));

    const std::array errors = {ParameterError{kCpLotId, Cpack::illegal_format}};
    const Bytes ack = encode(
        [&](secs::Encoder& e) { encode_host_command_ack(e, Hcack::parameter_invalid, errors); });
    auto decoded = decode_host_command_ack(item(ack));
    REQUIRE(decoded.has_value());
    CHECK(decoded->ack == Hcack::parameter_invalid);
    CHECK(decoded->errors.size() == 1);
}

TEST_CASE("alarms, identities and one-byte acks round-trip", "[gem]") {
    const Bytes alarm = encode([](secs::Encoder& e) { encode_alarm(e, {false, 5, 42, "X"}); });
    auto a = decode_alarm(item(alarm));
    REQUIRE(a.has_value());
    CHECK_FALSE(a->set);
    CHECK(a->category == 5);
    CHECK(a->id == 42);
    CHECK(a->text == "X");

    CHECK(decode_identity(item(encode([](secs::Encoder& e) {
              encode_identity(e, nullptr);
          })))->model.empty());
    const Identity id{"M", "S"};
    // Named: the decoded views point into these bytes (a temporary would dangle).
    const Bytes s1f14_bytes =
        encode([&](secs::Encoder& e) { encode_s1f14(e, Commack::denied, &id); });
    auto s1f14 = decode_s1f14(item(s1f14_bytes));
    REQUIRE(s1f14.has_value());
    CHECK(s1f14->ack == Commack::denied);
    CHECK(s1f14->identity.software == "S");

    CHECK(decode_ack(item(encode([](secs::Encoder& e) { encode_ack(e, 2); }))) == 2);
    CHECK_FALSE(decode_ack(item(encode([](secs::Encoder& e) { e.u1(2); })))); // not <B>
}
