// FabEye's conformal prediction sets and auto-accept rule, mirrored from its
// serving/app.py describe(): class k is in the set when 1 - p_k <= q_k; a wafer is accepted
// without review when its top probability reaches the accept confidence.
#include "waferedge/cnn.hpp"

#include <algorithm>
#include <bit>
#include <format>
#include <fstream>
#include <sstream>

namespace waferedge::cnn {

std::expected<Conformal, std::string> load_conformal(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        return std::unexpected(std::format("cannot open {}", path.string()));
    }
    Conformal c;
    int found = 0;
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line.starts_with('#')) {
            continue;
        }
        std::istringstream f(line);
        std::string key;
        f >> key;
        if (key == "model_sha256") {
            std::string hex;
            f >> hex;
            for (std::size_t i = 0; i < 32 && 2 * i + 1 < hex.size(); ++i) {
                c.model_sha256[i] = static_cast<std::uint8_t>(std::stoi(hex.substr(2 * i, 2), nullptr, 16));
            }
            ++found;
        } else if (key == "thresholds") {
            std::string alpha;
            f >> alpha;
            auto& q = alpha == "0.1" ? c.thresholds_10 : c.thresholds_05;
            for (auto& v : q) {
                f >> v;
            }
            found += f ? 1 : 0;
        } else if (key == "reported_coverage") {
            std::string alpha;
            f >> alpha;
            if (alpha == "0.1") {
                f >> c.reported_coverage_10 >> c.reported_worst_class_10;
            } else {
                f >> c.reported_coverage_05 >> c.reported_worst_class_05;
            }
        } else if (key == "accept_confidence") {
            f >> c.accept_confidence;
            found += f ? 1 : 0;
        } else if (key == "reported_accept") {
            f >> c.reported_accept_rate >> c.reported_accept_error;
        }
    }
    if (found != 4) {
        return std::unexpected(std::format("{}: incomplete conformal calibration", path.string()));
    }
    return c;
}

std::uint16_t prediction_set(std::span<const float> probabilities,
                             const std::array<float, kClasses>& thresholds) noexcept {
    std::uint16_t set = 0;
    for (std::size_t k = 0; k < kClasses; ++k) {
        if (1.0F - probabilities[k] <= thresholds[k]) {
            set = static_cast<std::uint16_t>(set | (1U << k));
        }
    }
    return set;
}

ConformalMetrics evaluate_conformal(std::span<const float> logits, std::span<const Pattern> truth,
                                    const std::array<float, kClasses>& thresholds,
                                    float accept_confidence) {
    std::array<std::size_t, kClasses> covered{};
    std::array<std::size_t, kClasses> total{};
    std::size_t n = 0;
    std::size_t set_sizes = 0;
    std::size_t accepted = 0;
    std::size_t accepted_wrong = 0;
    for (std::size_t i = 0; i < truth.size(); ++i) {
        if (truth[i] == Pattern::unknown) {
            continue;
        }
        const auto t = static_cast<std::size_t>(truth[i]);
        const auto p = softmax(logits.subspan(i * kClasses, kClasses));
        const auto set = prediction_set(p, thresholds);
        ++n;
        ++total[t];
        covered[t] += (set >> t) & 1U;
        set_sizes += static_cast<std::size_t>(std::popcount(set));
        const auto top = static_cast<std::size_t>(std::ranges::max_element(p) - p.begin());
        if (p[top] >= accept_confidence) {
            ++accepted;
            accepted_wrong += top == t ? 0U : 1U;
        }
    }
    ConformalMetrics m;
    if (n == 0) {
        return m;
    }
    std::size_t all_covered = 0;
    m.worst_class_coverage = 1.0;
    for (std::size_t k = 0; k < kClasses; ++k) {
        all_covered += covered[k];
        if (total[k] > 0) {
            m.worst_class_coverage =
                std::min(m.worst_class_coverage, static_cast<double>(covered[k]) / static_cast<double>(total[k]));
        }
    }
    m.coverage = static_cast<double>(all_covered) / static_cast<double>(n);
    m.mean_set_size = static_cast<double>(set_sizes) / static_cast<double>(n);
    m.accept_rate = static_cast<double>(accepted) / static_cast<double>(n);
    m.error_among_accepted = accepted == 0 ? 0.0 : static_cast<double>(accepted_wrong) / static_cast<double>(accepted);
    return m;
}

} // namespace waferedge::cnn
