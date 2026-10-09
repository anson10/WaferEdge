#include "support/synth.hpp"
#include "waferedge/classifier.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <random>
#include <vector>

using namespace waferedge;
using Catch::Approx;

namespace {

Signals signals_with(std::initializer_list<std::pair<Signal, double>> values) {
    Signals s{};
    for (const auto& [which, v] : values) {
        s[static_cast<std::size_t>(which)] = v;
    }
    return s;
}

} // namespace

TEST_CASE("thresholds survive to_text and parse unchanged") {
    const auto a = RuleClassifier::defaults();
    const auto b = RuleClassifier::parse(a.to_text());
    REQUIRE(b.has_value());
    REQUIRE(b->rules().size() == a.rules().size());
    for (std::size_t r = 0; r < a.rules().size(); ++r) {
        for (std::size_t k = 0; k < a.rules()[r].conditions.size(); ++k) {
            CHECK(b->rules()[r].conditions[k].threshold == a.rules()[r].conditions[k].threshold);
        }
    }
}

TEST_CASE("a non-round threshold reads back as the same double") {
    auto c = RuleClassifier::defaults();
    c.rules()[1].conditions[0].threshold = 0.41304347826086957;
    const auto d = RuleClassifier::parse(c.to_text());
    REQUIRE(d.has_value());
    CHECK(d->rules()[1].conditions[0].threshold == 0.41304347826086957);
}

TEST_CASE("broken rules files are rejected with the line or the condition") {
    const auto text = RuleClassifier::defaults().to_text();
    SECTION("missing condition") {
        const auto cut = text.substr(0, text.rfind("loc z"));
        const auto c = RuleClassifier::parse(cut);
        REQUIRE_FALSE(c.has_value());
        CHECK(c.error().find("missing threshold for loc z") != std::string::npos);
    }
    SECTION("given twice") {
        const auto c = RuleClassifier::parse(text + "near_full density >= 0.5\n");
        REQUIRE_FALSE(c.has_value());
        CHECK(c.error().find("twice") != std::string::npos);
    }
    SECTION("unknown rule, wrong direction, bad number") {
        CHECK_FALSE(RuleClassifier::parse(text + "blob density >= 1\n").has_value());
        CHECK_FALSE(RuleClassifier::parse(text + "near_full density <= 1\n").has_value());
        CHECK_FALSE(RuleClassifier::parse("near_full density >= 0.7x\n").has_value());
        CHECK_FALSE(RuleClassifier::parse("near_full density => 0.7\n").has_value());
    }
}

TEST_CASE("the checked-in config/rules.txt loads") {
    const auto c = RuleClassifier::load(WAFEREDGE_SOURCE_DIR "/config/rules.txt");
    REQUIRE(c.has_value());
    CHECK(c->rules().size() == RuleClassifier::defaults().rules().size());
}

TEST_CASE("the first rule that holds wins, and the explanation names it") {
    const auto c = RuleClassifier::defaults();
    // Dense and structured: near_full is first in the list, so it wins over everything.
    auto s = signals_with({{Signal::density, 0.9}, {Signal::edge, 5}, {Signal::z, 50}});
    CHECK(c.classify(s).pattern == Pattern::near_full);
    CHECK(c.classify(s).rule == 0);

    s = signals_with(
        {{Signal::density, 0.2}, {Signal::edge, 3.0}, {Signal::sector, 1.1}, {Signal::z, 12}});
    const auto d = c.classify(s);
    CHECK(d.pattern == Pattern::edge_ring);
    const auto why = c.explain(s);
    CHECK(why.starts_with("edge_ring -> edge_ring:"));
    CHECK(why.find("edge 3 >= 2.5") != std::string::npos);

    CHECK(c.classify(Signals{}).pattern == Pattern::none);
    CHECK(c.classify(Signals{}).rule == -1);
    CHECK(c.explain(Signals{}) == "no rule held -> none");
}

TEST_CASE("metrics on a hand-checked example") {
    using P = Pattern;
    const std::vector truth = {P::none, P::none, P::none, P::center, P::center, P::unknown};
    const std::vector pred = {P::none, P::none, P::center, P::center, P::none, P::scratch};
    const auto m = evaluate(truth, pred);
    CHECK(m.support[0] == 3);
    CHECK(m.support[1] == 2);
    CHECK(m.confusion[0][1] == 1);
    CHECK(m.precision[0] == Approx(2.0 / 3));
    CHECK(m.recall[0] == Approx(2.0 / 3));
    CHECK(m.precision[1] == Approx(0.5));
    CHECK(m.recall[1] == Approx(0.5));
    // Unknown truth is skipped, so the stray scratch prediction counts nowhere.
    CHECK(m.precision[8] == 0.0);
    CHECK(m.accuracy == Approx(3.0 / 5));
    CHECK(m.macro_f1 == Approx((2.0 / 3 + 0.5) / 2)); // only classes in the truth
}

TEST_CASE("weighted macro-F1 equals plain macro-F1 with unit weights") {
    std::mt19937 rng(5);
    std::vector<Pattern> truth;
    std::vector<Pattern> pred;
    for (int i = 0; i < 500; ++i) {
        truth.push_back(static_cast<Pattern>(rng() % kPatternCount));
        pred.push_back(static_cast<Pattern>(rng() % kPatternCount));
    }
    const std::array<double, kPatternCount> ones = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    CHECK(weighted_macro_f1(truth, pred, ones) == Approx(evaluate(truth, pred).macro_f1));
    auto heavy = ones;
    heavy[0] = 10;
    CHECK(weighted_macro_f1(truth, pred, heavy) != Approx(evaluate(truth, pred).macro_f1));
}

TEST_CASE("fitting finds a separating threshold from a bad start") {
    // Two classes that density alone separates at 0.5: none below, near_full above.
    std::vector<Signals> signals;
    std::vector<Pattern> truth;
    for (int i = 0; i < 200; ++i) {
        const double d = i / 200.0;
        signals.push_back(signals_with({{Signal::density, d}}));
        truth.push_back(d >= 0.5 ? Pattern::near_full : Pattern::none);
    }
    auto start = RuleClassifier::defaults();
    start.rules()[0].conditions[0].threshold = 0.95;
    // Candidates are quantiles of the data; with one per map, the exact separator is one.
    FitOptions options;
    options.candidates = 200;
    std::string log;
    const auto fitted = fit(start, signals, truth, options, &log);
    std::vector<Pattern> pred;
    for (const auto& s : signals) {
        pred.push_back(fitted.classify(s).pattern);
    }
    CHECK(evaluate(truth, pred).macro_f1 == 1.0);
    CHECK(log.starts_with("start:"));
}

TEST_CASE("signals of synthetic patterns point the right way") {
    SignalExtractor extractor;
    const int n = 40;
    SECTION("no fails: only density and z, both 0") {
        CHECK(extractor.extract(synth::disc(n, n)) == Signals{});
    }
    SECTION("center blob") {
        auto map = synth::disc(n, n);
        synth::paint(map, [&](int r, int c) { return synth::rho2(r, c, n, n) < 0.03; });
        const auto s = extractor.extract(map);
        CHECK(get(s, Signal::inner) > 3);
        CHECK(get(s, Signal::cluster_rho) < 0.05);
        CHECK(get(s, Signal::cluster_share) == 1.0);
        CHECK(get(s, Signal::z) > 5);
    }
    SECTION("donut: a ring with a hole") {
        auto map = synth::disc(n, n);
        synth::paint(map, [&](int r, int c) {
            const double q = synth::rho2(r, c, n, n);
            return q > 0.09 && q < 0.25; // 0.3 <= rho < 0.5
        });
        const auto s = extractor.extract(map);
        CHECK(get(s, Signal::inner) == 0.0);
        CHECK(get(s, Signal::ring) > 2);
        CHECK(get(s, Signal::cluster_rho) < 0.05); // the ring's centroid is the centre
    }
    SECTION("scratch") {
        auto map = synth::disc(n, n);
        synth::paint(map, [](int r, int c) { return c == 8 + r / 2 && r >= 6 && r < 34; });
        const auto s = extractor.extract(map);
        CHECK(get(s, Signal::elongation) > 5);
        CHECK(get(s, Signal::line_length) > 0.4);
    }
}
