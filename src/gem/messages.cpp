#include "waferedge/gem/messages.hpp"

namespace waferedge::gem {

namespace {

using secs::Errc;
using secs::Error;
using secs::Format;
using secs::ItemView;
using secs::ListView;

std::unexpected<Error> bad() {
    return std::unexpected(Error{Errc::wrong_format});
}

// The children of a list of exactly n items.
Result<ListView> list_of(ItemView item, std::size_t n) {
    auto list = item.list();
    if (!list || list->size() != n) {
        return bad();
    }
    return list;
}

// A <B> of exactly one byte.
Result<std::uint8_t> single_byte(ItemView item) {
    auto b = item.as<Format::binary>();
    if (!b || b->size() != 1) {
        return bad();
    }
    return (*b)[0];
}

} // namespace

// --- S1 ---------------------------------------------------------------------------------------

void encode_identity(secs::Encoder& e, const Identity* identity) {
    if (identity == nullptr) {
        e.list(0);
        return;
    }
    e.list(2).ascii(identity->model).ascii(identity->software);
}

Result<Identity> decode_identity(ItemView body) {
    auto list = body.list();
    if (!list) {
        return bad();
    }
    if (list->empty()) {
        return Identity{};
    }
    if (list->size() != 2) {
        return bad();
    }
    auto it = list->begin();
    auto model = (*it++).text();
    auto software = (*it).text();
    if (!model || !software) {
        return bad();
    }
    return Identity{*model, *software};
}

void encode_s1f14(secs::Encoder& e, Commack ack, const Identity* identity) {
    e.list(2).value<Format::binary>(static_cast<std::uint8_t>(ack));
    encode_identity(e, identity);
}

Result<EstablishAck> decode_s1f14(ItemView body) {
    auto list = list_of(body, 2);
    if (!list) {
        return bad();
    }
    auto it = list->begin();
    auto ack = single_byte(*it++);
    auto identity = decode_identity(*it);
    if (!ack || !identity) {
        return bad();
    }
    return EstablishAck{static_cast<Commack>(*ack), *identity};
}

void encode_ack(secs::Encoder& e, std::uint8_t code) {
    e.value<Format::binary>(code);
}

Result<std::uint8_t> decode_ack(ItemView body) {
    return single_byte(body);
}

// --- S6F11 ------------------------------------------------------------------------------------

void encode_wafer_report(secs::Encoder& e, std::uint32_t data_id, std::string_view lot,
                         std::uint32_t wafer, WaferMapView map) {
    e.list(3).u4(data_id).u4(kCeidWaferSorted).list(1).list(2).u4(kRptidWafer).list(5);
    e.ascii(lot)
        .u4(wafer)
        .u2(static_cast<std::uint16_t>(map.rows()))
        .u2(static_cast<std::uint16_t>(map.cols()))
        .array<Format::u1>(map.bins());
}

Result<EventReport> decode_event_report(ItemView body) {
    auto top = list_of(body, 3);
    if (!top) {
        return bad();
    }
    auto it = top->begin();
    auto data_id = (*it++).unsigned_scalar();
    auto ceid = (*it++).unsigned_scalar();
    auto reports = (*it).list();
    if (!data_id || !ceid || !reports) {
        return bad();
    }
    return EventReport{*data_id, *ceid, *reports};
}

Result<WaferReport> decode_wafer_report(ItemView body) {
    auto event = decode_event_report(body);
    if (!event || event->ceid != kCeidWaferSorted || event->reports.size() != 1) {
        return bad();
    }
    auto report = list_of(*event->reports.begin(), 2);
    if (!report) {
        return bad();
    }
    auto r = report->begin();
    auto rptid = (*r++).unsigned_scalar();
    auto values = list_of(*r, 5);
    if (!rptid || *rptid != kRptidWafer || !values) {
        return bad();
    }
    auto v = values->begin();
    auto lot = (*v++).text();
    auto wafer = (*v++).unsigned_scalar();
    auto rows = (*v++).unsigned_scalar();
    auto cols = (*v++).unsigned_scalar();
    auto bins = (*v).as<Format::u1>();
    if (!lot || !wafer || !rows || !cols || !bins || *rows > 0xFFFF || *cols > 0xFFFF ||
        bins->size() != *rows * *cols) {
        return bad();
    }
    return WaferReport{
        event->data_id, *lot, *wafer,
        WaferMapView(bins->bytes(), static_cast<int>(*rows), static_cast<int>(*cols))};
}

Result<WaferReport> decode_wafer_report(std::span<const std::uint8_t> body) {
    return secs::decode(body).and_then([](ItemView item) { return decode_wafer_report(item); });
}

// --- S5F1 -------------------------------------------------------------------------------------

void encode_alarm(secs::Encoder& e, const Alarm& alarm) {
    const auto alcd =
        static_cast<std::uint8_t>((alarm.set ? 0x80U : 0U) | (alarm.category & 0x7FU));
    e.list(3)
        .value<Format::binary>(alcd)
        .u4(static_cast<std::uint32_t>(alarm.id))
        .ascii(alarm.text);
}

Result<Alarm> decode_alarm(ItemView body) {
    auto list = list_of(body, 3);
    if (!list) {
        return bad();
    }
    auto it = list->begin();
    auto alcd = single_byte(*it++);
    auto id = (*it++).unsigned_scalar();
    auto text = (*it).text();
    if (!alcd || !id || !text) {
        return bad();
    }
    return Alarm{(*alcd & 0x80U) != 0, static_cast<std::uint8_t>(*alcd & 0x7FU), *id, *text};
}

// --- S2F41 / S2F42 ----------------------------------------------------------------------------

void encode_lot_command(secs::Encoder& e, const LotCommand& command) {
    e.list(2).ascii(command.action == LotAction::hold ? kRcmdHold : kRcmdRelease);
    e.list(1).list(2).ascii(kCpLotId).ascii(command.lot);
}

Result<HostCommand> decode_host_command(ItemView body) {
    auto list = list_of(body, 2);
    if (!list) {
        return bad();
    }
    auto it = list->begin();
    auto rcmd = (*it++).text();
    auto parameters = (*it).list();
    if (!rcmd || !parameters) {
        return bad();
    }
    for (const ItemView p : *parameters) { // every parameter is <L [2] <A CPNAME> CPVAL>
        auto pair = list_of(p, 2);
        if (!pair || !(*pair->begin()).text()) {
            return bad();
        }
    }
    return HostCommand{*rcmd, *parameters};
}

Result<ItemView> find_parameter(const HostCommand& command, std::string_view name) {
    for (const ItemView p : command.parameters) {
        const ListView pair = *p.list(); // shape checked by decode_host_command
        auto it = pair.begin();
        if (*(*it++).text() == name) {
            return *it;
        }
    }
    return std::unexpected(Error{Errc::out_of_range});
}

void encode_host_command_ack(secs::Encoder& e, Hcack ack, std::span<const ParameterError> errors) {
    e.list(2).value<Format::binary>(static_cast<std::uint8_t>(ack)).list(errors.size());
    for (const auto& error : errors) {
        e.list(2).ascii(error.name).value<Format::binary>(static_cast<std::uint8_t>(error.ack));
    }
}

Result<HostCommandAck> decode_host_command_ack(ItemView body) {
    auto list = list_of(body, 2);
    if (!list) {
        return bad();
    }
    auto it = list->begin();
    auto ack = single_byte(*it++);
    auto errors = (*it).list();
    if (!ack || !errors) {
        return bad();
    }
    return HostCommandAck{static_cast<Hcack>(*ack), *errors};
}

// --- S9 ---------------------------------------------------------------------------------------

void encode_mhead(secs::Encoder& e, std::span<const std::uint8_t, 10> header) {
    e.binary(header);
}

} // namespace waferedge::gem
