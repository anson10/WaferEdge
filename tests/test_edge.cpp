// The edge host's parts: the decision rule, the doorbell (a two-thread stress test with a
// lost-wake-up watchdog), the analyzer (on the caller's thread and on its own), and the edge
// core driven through GEM on a fake clock. The threaded tests run under TSan in CI.
#include "support/protocol_pair.hpp"
#include "support/synth.hpp"
#include "waferedge/gem/equipment.hpp"
#include "waferedge/pipeline/analyzer.hpp"
#include "waferedge/pipeline/doorbell.hpp"
#include "waferedge/pipeline/edge_core.hpp"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <optional>
#include <string>
#include <thread>
#include <vector>

using namespace waferedge;
using namespace waferedge::pipeline;
using namespace std::chrono_literals;
using P = Pattern;

namespace {

LotId lot(std::string_view name) {
    return LotId::from(name).value_or(LotId{});
}

const RuleClassifier& rules() {
    static const RuleClassifier c = *RuleClassifier::load(WAFEREDGE_SOURCE_DIR "/config/rules.txt");
    return c;
}

WaferMap clean() {
    return synth::disc(40, 40);
}
WaferMap center_blob() {
    auto map = synth::disc(40, 40);
    synth::paint(map, [](int r, int c) { return synth::rho2(r, c, 40, 40) < 0.03; });
    return map;
}

void fill(WaferSlot& slot, std::uint64_t seq, std::string_view lot_name, const WaferMap& map) {
    slot.seq = seq;
    slot.lot = lot(lot_name);
    slot.wafer = seq;
    slot.rows = map.rows();
    slot.cols = map.cols();
    std::memcpy(slot.bins.data(), map.view().bins().data(), map.view().bins().size());
}

} // namespace

// --- Decision rule -----------------------------------------------------------------------------

TEST_CASE("decision: k wafers with the same pattern within the window hold the lot", "[decision]") {
    DecisionRule rule({.k = 3, .window = 5, .lots = 8});
    const LotId a = lot("A");
    CHECK_FALSE(rule.observe(a, P::center));
    CHECK_FALSE(rule.observe(a, P::none));
    CHECK_FALSE(rule.observe(a, P::center));
    CHECK_FALSE(rule.held(a));
    CHECK(rule.observe(a, P::center) == P::center); // 3 of the last 4
    CHECK(rule.held(a));
    CHECK_FALSE(rule.observe(a, P::center)); // held once
}

TEST_CASE("decision: none never counts, and different patterns don't add up", "[decision]") {
    DecisionRule rule({.k = 2, .window = 4, .lots = 8});
    const LotId a = lot("A");
    for (int i = 0; i < 10; ++i) {
        CHECK_FALSE(rule.observe(a, P::none));
    }
    CHECK_FALSE(rule.observe(a, P::center));
    CHECK_FALSE(rule.observe(a, P::scratch));
    CHECK_FALSE(rule.observe(a, P::donut));
    CHECK(rule.observe(a, P::scratch) == P::scratch);
}

TEST_CASE("decision: the window slides", "[decision]") {
    DecisionRule rule({.k = 2, .window = 3, .lots = 8});
    const LotId a = lot("A");
    CHECK_FALSE(rule.observe(a, P::edge_loc));
    CHECK_FALSE(rule.observe(a, P::none));
    CHECK_FALSE(rule.observe(a, P::none));
    CHECK_FALSE(rule.observe(a, P::edge_loc)); // the first one has left the window
    CHECK(rule.observe(a, P::edge_loc) == P::edge_loc);
}

TEST_CASE("decision: lots are independent; k = 1 holds on the first pattern", "[decision]") {
    DecisionRule rule({.k = 1, .window = 1, .lots = 8});
    CHECK_FALSE(rule.observe(lot("A"), P::none));
    CHECK(rule.observe(lot("B"), P::random) == P::random);
    CHECK_FALSE(rule.held(lot("A")));
    CHECK(rule.held(lot("B")));
}

TEST_CASE("decision: a full table forgets the least recently seen lot", "[decision]") {
    DecisionRule rule({.k = 1, .window = 1, .lots = 2});
    CHECK(rule.observe(lot("A"), P::center) == P::center);
    CHECK_FALSE(rule.observe(lot("B"), P::none));
    CHECK_FALSE(rule.observe(lot("B"), P::none)); // B is now more recent than A
    CHECK_FALSE(rule.observe(lot("C"), P::none)); // evicts A
    CHECK(rule.evictions() == 1);
    CHECK_FALSE(rule.held(lot("A")));
    CHECK(rule.observe(lot("A"), P::center) == P::center); // starts afresh: held again
}

TEST_CASE("decision: config validation", "[decision]") {
    CHECK(DecisionRule::valid({.k = 1, .window = 1, .lots = 1}));
    CHECK_FALSE(DecisionRule::valid({.k = 0, .window = 1, .lots = 1}));
    CHECK_FALSE(DecisionRule::valid({.k = 3, .window = 2, .lots = 1}));
    CHECK_FALSE(DecisionRule::valid({.k = 1, .window = 33, .lots = 1}));
    CHECK_FALSE(DecisionRule::valid({.k = 1, .window = 1, .lots = 0}));
}

TEST_CASE("lot ids: up to 32 characters, compared by value", "[decision]") {
    CHECK(LotId::from(std::string(32, 'x')).has_value());
    CHECK_FALSE(LotId::from(std::string(33, 'x')).has_value());
    CHECK(lot("LOT-1") == lot("LOT-1"));
    CHECK_FALSE(lot("LOT-1") == lot("LOT-10"));
    CHECK(lot("LOT-1").view() == "LOT-1");
}

// --- Doorbell ----------------------------------------------------------------------------------

TEST_CASE("doorbell: a sleeping consumer never misses a push", "[doorbell]") {
    // The producer pauses now and then so the consumer really goes to sleep. A lost wake-up
    // would leave the consumer asleep with messages waiting: the watchdog notices (no
    // progress for a second while the ring is non-empty), wakes it, and the test fails.
    auto ring = std::make_unique<SpscRing<std::uint64_t, 64>>();
    Doorbell bell;
    constexpr std::uint64_t n = 200'000;
    std::atomic<std::uint64_t> progress{0};
    std::uint64_t bad = 0;
    std::uint64_t sleeps = 0;
    std::thread consumer([&] {
        std::uint64_t expected = 0;
        std::uint64_t out = 0;
        while (expected < n) {
            if (ring->try_pop(out)) {
                bad += out != expected ? 1 : 0;
                progress.store(++expected, std::memory_order_relaxed);
                continue;
            }
            const auto key = bell.prepare();
            if (!ring->empty_approx()) {
                bell.cancel();
                continue;
            }
            ++sleeps;
            bell.wait(key);
        }
    });
    for (std::uint64_t i = 0; i < n;) {
        if (!ring->try_push(i)) {
            std::this_thread::yield();
            continue;
        }
        bell.ring();
        ++i;
        if (i % 2'000 == 0) {
            std::this_thread::sleep_for(50us); // let the consumer drain and fall asleep
        }
    }
    int rescues = 0;
    std::uint64_t last = progress.load();
    auto since = std::chrono::steady_clock::now();
    while (progress.load() < n) {
        std::this_thread::sleep_for(10ms);
        if (progress.load() != last) {
            last = progress.load();
            since = std::chrono::steady_clock::now();
        } else if (std::chrono::steady_clock::now() - since > 1s) {
            ++rescues; // stuck: a lost wake-up
            bell.wake();
            since = std::chrono::steady_clock::now();
        }
    }
    consumer.join();
    CHECK(bad == 0);
    CHECK(rescues == 0);
    CHECK(sleeps > 0); // the sleeping path was exercised
}

// --- Analyzer ----------------------------------------------------------------------------------

TEST_CASE("analyzer: verdicts match the classifier, ids carried through", "[analyzer]") {
    Analyzer analyzer(rules(), {});
    const WaferMap blob = center_blob();
    const WaferMap disc = clean();
    SignalExtractor extractor;
    const Pattern expected = rules().classify(extractor.extract(blob)).pattern;
    REQUIRE(expected == P::center);
    REQUIRE(rules().classify(extractor.extract(disc)).pattern == P::none);

    fill(*analyzer.begin_submit(), 7, "LOT-7", blob);
    analyzer.commit_submit();
    fill(*analyzer.begin_submit(), 8, "LOT-8", disc);
    analyzer.commit_submit();
    CHECK(analyzer.front_verdict() == nullptr); // nothing until analysed
    CHECK(analyzer.poll() == 2);

    const Verdict* v = analyzer.front_verdict();
    REQUIRE(v != nullptr);
    CHECK(v->seq == 7);
    CHECK(v->lot.view() == "LOT-7");
    CHECK(v->pattern == P::center);
    CHECK(v->rule >= 0);
    analyzer.pop_verdict();
    v = analyzer.front_verdict();
    REQUIRE(v != nullptr);
    CHECK(v->lot.view() == "LOT-8");
    CHECK(v->pattern == P::none);
    analyzer.pop_verdict();
    CHECK(analyzer.analysed() == 2);
}

TEST_CASE("analyzer: on its own thread, every wafer once, in order", "[analyzer]") {
    std::atomic<int> signals{0};
    Analyzer analyzer(rules(), {.spin = 20us}, [&] { signals.fetch_add(1); });
    analyzer.start();
    const std::array<WaferMap, 2> maps = {clean(), center_blob()};
    constexpr std::uint64_t n = 2'000;
    std::uint64_t submitted = 0;
    std::uint64_t taken = 0;
    std::uint64_t bad = 0;
    while (taken < n) {
        if (submitted < n) {
            if (WaferSlot* slot = analyzer.begin_submit()) {
                fill(*slot, submitted, "LOT", maps[submitted % 2]);
                analyzer.commit_submit();
                ++submitted;
                if (submitted % 100 == 0) {
                    std::this_thread::sleep_for(200us); // the worker sleeps now and then
                }
            }
        }
        while (const Verdict* v = analyzer.front_verdict()) {
            const P want = taken % 2 == 0 ? P::none : P::center;
            bad += (v->seq != taken || v->pattern != want) ? 1 : 0;
            analyzer.pop_verdict();
            ++taken;
        }
    }
    analyzer.stop();
    CHECK(bad == 0);
    CHECK(analyzer.analysed() == n);
    CHECK(signals.load() == static_cast<int>(n));
}

// --- Edge core through GEM ---------------------------------------------------------------------

namespace {

// A tool (Equipment) and the edge core on a ProtocolPair, the analyzer polled by hand.
struct Rig {
    testing::ProtocolPair pair;
    gem::Equipment equipment{gem::EquipmentConfig{}};
    Analyzer analyzer;
    std::vector<Analyzer*> analyzers{&analyzer};
    EdgeCore core;
    std::vector<std::string> holds_at_tool;

    explicit Rig(DecisionConfig decision)
        : analyzer(rules(), {}), core(analyzers, {.decision = decision}) {
        pair.connect();
        pump();
        equipment.go_online(pair.tool_link);
        pump();
    }

    void pump() {
        REQUIRE(pair.pump(
            [&](const hsms::Event& e) {
                equipment.on_hsms(e, pair.tool_link, pair.now);
                while (auto g = equipment.poll()) {
                    if (const auto* c = std::get_if<gem::LotCommandReceived>(&*g)) {
                        holds_at_tool.emplace_back(c->command.lot);
                        REQUIRE(equipment.answer(pair.tool_link, c->primary, gem::Hcack::done));
                    }
                }
            },
            [&](const hsms::Event& e) { core.on_hsms(e, pair.host_link, pair.now); }));
    }
    // Reports a wafer, analyses it, decides, and lets any HOLD reach the tool and come back.
    void wafer(std::string_view lot_name, std::uint32_t id, const WaferMap& map) {
        REQUIRE(equipment.report_wafer(pair.tool_link, lot_name, id, map.view()));
        pump();
        analyzer.poll();
        core.drain(pair.host_link);
        pump();
    }
};

} // namespace

TEST_CASE("edge core: k of W same-pattern wafers hold the lot at the tool", "[edge]") {
    Rig rig({.k = 2, .window = 3, .lots = 8});
    const WaferMap blob = center_blob();
    const WaferMap disc = clean();
    rig.wafer("LOT-A", 1, disc);
    rig.wafer("LOT-A", 2, blob);
    CHECK(rig.holds_at_tool.empty());
    rig.wafer("LOT-A", 3, blob);
    REQUIRE(rig.holds_at_tool == std::vector<std::string>{"LOT-A"});
    // Lot B: two centers, but never 2 within 3 wafers.
    rig.wafer("LOT-B", 1, blob);
    rig.wafer("LOT-B", 2, disc);
    rig.wafer("LOT-B", 3, disc);
    rig.wafer("LOT-B", 4, blob);
    CHECK(rig.holds_at_tool.size() == 1);

    const auto& s = rig.core.stats();
    CHECK(s.received == 7);
    CHECK(s.submitted == 7);
    CHECK(s.verdicts == 7);
    CHECK(rig.core.idle());
    CHECK(s.by_pattern[static_cast<std::size_t>(P::center)] == 4);
    CHECK(s.by_pattern[static_cast<std::size_t>(P::none)] == 3);
    CHECK(s.holds == 1);
    REQUIRE(rig.core.holds().size() == 1);
    const HoldRecord& h = rig.core.holds()[0];
    CHECK(h.lot.view() == "LOT-A");
    CHECK(h.pattern == P::center);
    CHECK(h.wafer == 3);
    CHECK(h.state == HoldRecord::State::acked);
    CHECK(h.hcack == 0);
}

TEST_CASE("edge core: oversized maps, long lot ids and full rings are counted", "[edge]") {
    Rig rig({.k = 1, .window = 1, .lots = 8});
    rig.wafer("LOT-A", 1, synth::disc(70, 70)); // 4,900 dies > kMaxBins
    rig.wafer(std::string(40, 'L'), 2, clean());
    CHECK(rig.core.stats().oversized == 1);
    CHECK(rig.core.stats().bad_lot == 1);
    CHECK(rig.core.stats().submitted == 0);

    // The analyzer stops taking wafers: its ring fills, then wafers are dropped.
    const WaferMap disc = clean();
    for (std::uint32_t i = 0; i < Analyzer::kSlots + 3; ++i) {
        REQUIRE(rig.equipment.report_wafer(rig.pair.tool_link, "LOT-C", i, disc.view()));
        rig.pump();
    }
    CHECK(rig.core.stats().submitted == Analyzer::kSlots);
    CHECK(rig.core.stats().dropped == 3);
    CHECK(rig.analyzer.poll() == Analyzer::kSlots);
    CHECK(rig.core.drain(rig.pair.host_link) == Analyzer::kSlots);
    CHECK(rig.core.idle());
}
