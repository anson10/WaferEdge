#pragma once

// A wafer-map event report body shaped like the S6F11 the tool emulator will send (phase 3's
// GEM part defines the real one): the test and benchmark workload for the codec.
//
//   <L [3]
//     <U4 DATAID>
//     <U4 CEID>                  100: wafer sorted
//     <L [1]
//       <L [2]
//         <U4 RPTID>             10
//         <L [5]
//           <A LOTID> <U4 WAFERID> <U2 ROWS> <U2 COLS> <U1 BINS...>
#include "waferedge/secs/encoder.hpp"
#include "waferedge/secs/item.hpp"
#include "waferedge/wafer_map.hpp"

#include <cstdint>
#include <string_view>

namespace waferedge::secs_messages {

inline constexpr std::uint32_t kWaferSorted = 100;
inline constexpr std::uint32_t kWaferReport = 10;

inline void wafer_report(secs::Encoder& e, std::uint32_t data_id, std::string_view lot,
                         std::uint32_t wafer, WaferMapView map) {
    e.list(3).u4(data_id).u4(kWaferSorted).list(1).list(2).u4(kWaferReport).list(5);
    e.ascii(lot)
        .u4(wafer)
        .u2(static_cast<std::uint16_t>(map.rows()))
        .u2(static_cast<std::uint16_t>(map.cols()))
        .array<secs::Format::u1>(map.bins());
}

struct WaferReport {
    std::uint64_t data_id = 0;
    std::string_view lot; // views into the received body
    std::uint64_t wafer = 0;
    WaferMapView map;
};

// Reads the report without copying: the map's bins are the U1 item's bytes in `body`.
inline secs::Result<WaferReport> read_wafer_report(std::span<const std::uint8_t> body) {
    using secs::Errc;
    using secs::Error;
    const auto fail = [](Errc code) { return std::unexpected(Error{code}); };
    auto top = secs::decode(body).and_then([](secs::ItemView v) { return v.list(); });
    if (!top || top->size() != 3) {
        return fail(top ? Errc::count_mismatch : top.error().code);
    }
    auto it = top->begin();
    WaferReport r;
    auto data_id = (*it++).unsigned_scalar();
    auto ceid = (*it++).unsigned_scalar();
    auto reports = (*it).list();
    if (!data_id || !ceid || !reports || reports->size() != 1) {
        return fail(Errc::wrong_format);
    }
    r.data_id = *data_id;
    auto report = (*reports->begin()).list();
    if (!report || report->size() != 2) {
        return fail(Errc::wrong_format);
    }
    auto values = report->at(1).and_then([](secs::ItemView v) { return v.list(); });
    if (!values || values->size() != 5) {
        return fail(Errc::wrong_format);
    }
    auto v = values->begin();
    auto lot = (*v++).text();
    auto wafer = (*v++).unsigned_scalar();
    auto rows = (*v++).unsigned_scalar();
    auto cols = (*v++).unsigned_scalar();
    auto bins = (*v).as<secs::Format::u1>();
    if (!lot || !wafer || !rows || !cols || !bins || bins->size() != *rows * *cols) {
        return fail(Errc::wrong_format);
    }
    r.lot = *lot;
    r.wafer = *wafer;
    r.map = WaferMapView(bins->bytes(), static_cast<int>(*rows), static_cast<int>(*cols));
    return r;
}

} // namespace waferedge::secs_messages
