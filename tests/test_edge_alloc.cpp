// The edge host's hot path allocates nothing once warm: a wafer report decoded by the GEM
// host, copied into an analyzer slot, classified, its verdict decided, the HOLD sent, the
// tool's answer received. Everything runs on this thread (the analyzer polled by hand), so
// the per-thread allocation counter sees every step.
#include "support/alloc_counter.hpp"
#include "support/protocol_pair.hpp"
#include "support/synth.hpp"
#include "waferedge/gem/equipment.hpp"
#include "waferedge/pipeline/edge_core.hpp"

#include <catch2/catch_test_macros.hpp>

#include <format>
#include <string>
#include <vector>

using namespace waferedge;
using namespace waferedge::pipeline;

TEST_CASE("edge: report, classify, decide, hold allocate nothing once warm", "[edge]") {
    const auto classifier = RuleClassifier::load(WAFEREDGE_SOURCE_DIR "/config/rules.txt");
    REQUIRE(classifier.has_value());
    testing::ProtocolPair pair;
    gem::Equipment equipment{gem::EquipmentConfig{}};
    Analyzer analyzer(*classifier, AnalyzerConfig{});
    const std::vector<Analyzer*> analyzers{&analyzer};
    EdgeConfig config;
    config.decision = {.k = 1, .window = 1, .lots = 16}; // every blob wafer holds its lot
    EdgeCore core(analyzers, config);
    bool ok = true;
    const auto pump = [&] {
        ok = pair.pump(
                 [&](const hsms::Event& e) {
                     equipment.on_hsms(e, pair.tool_link, pair.now);
                     while (auto g = equipment.poll()) {
                         if (const auto* c = std::get_if<gem::LotCommandReceived>(&*g)) {
                             ok = ok &&
                                  equipment.answer(pair.tool_link, c->primary, gem::Hcack::done)
                                      .has_value();
                         }
                     }
                 },
                 [&](const hsms::Event& e) { core.on_hsms(e, pair.host_link, pair.now); }) &&
             ok;
    };
    pair.connect();
    pump();
    equipment.go_online(pair.tool_link);
    pump();

    auto blob = synth::disc(40, 40);
    synth::paint(blob, [](int r, int c) { return synth::rho2(r, c, 40, 40) < 0.03; });
    const WaferMap noisy = synth::random_map(40, 40, 50, 3);
    // Lot names made up front; more lots than the decision table holds, so it evicts too.
    std::vector<std::string> lots;
    lots.reserve(200);
    for (int i = 0; i < 200; ++i) {
        lots.push_back(std::format("LOT-{:06}", i));
    }
    std::size_t next = 0;
    const auto round = [&] {
        const std::string& lot = lots[next++ % lots.size()];
        ok = ok && equipment.report_wafer(pair.tool_link, lot, 1, noisy.view()).has_value();
        ok = ok && equipment.report_wafer(pair.tool_link, lot, 2, blob.view()).has_value();
        pump();
        analyzer.poll();
        core.drain(pair.host_link);
        pump(); // HOLD to the tool, HCACK back
    };
    for (int i = 0; i < 3; ++i) {
        round(); // warm-up: buffers reach their size, the 40 x 40 geometry is cached
    }
    const auto holds_before = core.stats().holds;
    const auto before = alloc::count();
    for (int i = 0; i < 100; ++i) {
        round();
    }
    const auto allocations = alloc::count() - before;
    CHECK(ok);
    CHECK(core.stats().holds - holds_before == 100);
    CHECK(core.holds().back().state == HoldRecord::State::acked);
    CHECK(core.decision().evictions() > 0);
    CHECK(allocations == 0);
}
