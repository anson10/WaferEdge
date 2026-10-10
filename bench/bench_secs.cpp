// SECS-II codec: messages/s decoded and encoded, and heap allocations per message (the
// global operator new is counted, tests/support/alloc_counter.cpp; it should read 0).
//
//   build/release/bench/bench-secs
//
// Workloads: the wafer-map event report (gem::encode_wafer_report, docs/secs.md) at the map sizes
// of the datasets, and the S2F41 HOLD the host sends back.
#include "support/alloc_counter.hpp"
#include "support/synth.hpp"
#include "waferedge/gem/messages.hpp"
#include "waferedge/secs/encoder.hpp"
#include "waferedge/secs/item.hpp"

#include <benchmark/benchmark.h>

#include <cstdint>
#include <string>
#include <vector>

namespace {

using namespace waferedge;

std::vector<std::uint8_t> report_body(int n) {
    const WaferMap map = synth::random_map(n, n, 100, 1);
    std::vector<std::uint8_t> body;
    secs::Encoder e(body);
    gem::encode_wafer_report(e, 1, "LOT-0042", 17, map);
    return body;
}

void encode_hold(secs::Encoder& e, std::string_view lot) {
    e.list(2).ascii("HOLD").list(1).list(2).ascii("LOTID").ascii(lot);
}

// Runs `body` per iteration and reports messages/s, bytes/s and allocations per message.
template <typename Body>
void run(benchmark::State& state, std::size_t message_bytes, Body&& body) {
    body(); // warm up: buffers reach their size before counting starts
    const auto before = alloc::count();
    for (auto _ : state) {
        body();
    }
    const auto allocations = alloc::count() - before;
    const auto n = static_cast<std::int64_t>(state.iterations());
    state.SetItemsProcessed(n);
    state.SetBytesProcessed(n * static_cast<std::int64_t>(message_bytes));
    state.counters["allocs/msg"] =
        benchmark::Counter(static_cast<double>(allocations), benchmark::Counter::kAvgIterations);
    state.counters["bytes/msg"] = static_cast<double>(message_bytes);
}

// Validation only: one pass over the headers, the map's bytes are skipped, not read.
void BM_decode_report(benchmark::State& state) {
    const auto body = report_body(static_cast<int>(state.range(0)));
    run(state, body.size(), [&] { benchmark::DoNotOptimize(secs::decode(body)); });
}

// Validation plus every field read into a WaferReport (ids, lot, map view): what the edge
// host does per S6F11 before handing the map on.
void BM_read_report(benchmark::State& state) {
    const auto body = report_body(static_cast<int>(state.range(0)));
    run(state, body.size(), [&] {
        auto report = gem::decode_wafer_report(body);
        benchmark::DoNotOptimize(report);
    });
}

void BM_encode_report(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const WaferMap map = synth::random_map(n, n, 100, 1);
    std::vector<std::uint8_t> buffer;
    std::uint32_t data_id = 0;
    run(state, report_body(n).size(), [&] {
        secs::Encoder e(buffer);
        gem::encode_wafer_report(e, ++data_id, "LOT-0042", 17, map);
        benchmark::DoNotOptimize(e.finish());
    });
}

void BM_read_hold(benchmark::State& state) {
    std::vector<std::uint8_t> body;
    secs::Encoder e(body);
    encode_hold(e, "LOT-0042");
    run(state, body.size(), [&] {
        auto item = secs::decode(body);
        auto params = item->list()->at(1)->list()->at(0)->list();
        benchmark::DoNotOptimize(params->at(1)->text());
    });
}

void BM_encode_hold(benchmark::State& state) {
    std::vector<std::uint8_t> buffer;
    std::size_t size = 0;
    {
        secs::Encoder e(buffer);
        encode_hold(e, "LOT-0042");
        size = e.finish()->size();
    }
    run(state, size, [&] {
        secs::Encoder e(buffer);
        encode_hold(e, "LOT-0042");
        benchmark::DoNotOptimize(e.finish());
    });
}

// The counter's control: SML text allocates, and the column must say so.
void BM_sml_report(benchmark::State& state) {
    const auto body = report_body(static_cast<int>(state.range(0)));
    run(state, body.size(), [&] {
        std::string sml = secs::to_sml(*secs::decode(body));
        benchmark::DoNotOptimize(sml);
    });
}

// Map sides: WaferLens's 24 and 40, the CNN's 64, WM-811K's largest (~200).
void map_sizes(benchmark::internal::Benchmark* b) {
    for (const int n : {24, 40, 64, 200}) {
        b->Arg(n);
    }
}

BENCHMARK(BM_decode_report)->Apply(map_sizes);
BENCHMARK(BM_read_report)->Apply(map_sizes);
BENCHMARK(BM_encode_report)->Apply(map_sizes);
BENCHMARK(BM_read_hold);
BENCHMARK(BM_encode_hold);
BENCHMARK(BM_sml_report)->Arg(40);

} // namespace

BENCHMARK_MAIN();
