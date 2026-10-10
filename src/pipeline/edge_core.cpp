#include "waferedge/pipeline/edge_core.hpp"

#include <algorithm>
#include <cstring>
#include <format>
#include <ranges>

namespace waferedge::pipeline {

EdgeCore::EdgeCore(std::span<Analyzer* const> analyzers, EdgeConfig config, Logger log)
    : analyzers_(analyzers.begin(), analyzers.end()), config_(config), host_(config.gem),
      decision_(config.decision), log_(std::move(log)) {
    holds_.reserve(config_.max_holds);
}

void EdgeCore::log(std::string_view line) const {
    if (log_) {
        log_(line);
    }
}

void EdgeCore::on_hsms(const hsms::Event& event, gem::Link& link, TimePoint now) {
    host_.on_hsms(event, link, now);
    while (auto g = host_.poll()) {
        on_gem(*g, link);
    }
}

void EdgeCore::tick(gem::Link& link, TimePoint now) {
    host_.tick(link, now);
    while (auto g = host_.poll()) {
        on_gem(*g, link);
    }
}

void EdgeCore::on_gem(const gem::Event& event, gem::Link& link) {
    if (const auto* w = std::get_if<gem::WaferReported>(&event)) {
        submit(w->report);
    } else if (const auto* r = std::get_if<gem::ReplyReceived>(&event)) {
        if (r->stream == 2 && r->function == 42) {
            if (HoldRecord* h = find_sent(r->system)) {
                h->state = HoldRecord::State::acked;
                h->acked = Clock::now();
                h->hcack = r->code;
                if (log_) {
                    log(std::format("HOLD {}: HCACK {}", h->lot.view(), h->hcack));
                }
            }
        }
        send_waiting(link); // a reply frees a slot in HSMS's window
    } else if (const auto* f = std::get_if<gem::TransactionFailed>(&event)) {
        if (HoldRecord* h = find_sent(f->system)) {
            h->state = HoldRecord::State::failed;
            if (log_) {
                log(std::format("HOLD {}: transaction failed", h->lot.view()));
            }
        }
        send_waiting(link);
    } else if (std::holds_alternative<gem::Communicating>(event)) {
        communicated_ = true;
        log("communicating");
        send_waiting(link);
    } else if (std::holds_alternative<gem::NotCommunicating>(event)) {
        tool_left_ = communicated_;
        log("not communicating");
    } else if (std::holds_alternative<gem::MalformedMessage>(event)) {
        log("malformed message from the tool");
    }
}

void EdgeCore::submit(const gem::WaferReport& report) {
    const std::uint64_t seq = stats_.received++;
    const auto lot = LotId::from(report.lot);
    if (!lot) {
        ++stats_.bad_lot;
        return;
    }
    const auto bins = report.map.bins();
    if (bins.size() > kMaxBins) {
        ++stats_.oversized;
        return;
    }
    // Round robin, skipping workers whose ring is full.
    for (std::size_t i = 0; i < analyzers_.size(); ++i) {
        const std::size_t at = (next_ + i) % analyzers_.size();
        Analyzer& analyzer = *analyzers_[at];
        WaferSlot* slot = analyzer.begin_submit();
        if (slot == nullptr) {
            continue;
        }
        slot->seq = seq;
        slot->lot = *lot;
        slot->wafer = report.wafer;
        slot->received = Clock::now();
        slot->rows = report.map.rows();
        slot->cols = report.map.cols();
        std::memcpy(slot->bins.data(), bins.data(), bins.size()); // the one copy of the map
        analyzer.commit_submit();
        next_ = (at + 1) % analyzers_.size();
        ++stats_.submitted;
        return;
    }
    ++stats_.dropped; // the S6F12 has gone back already: the tool doesn't slow down
}

std::size_t EdgeCore::drain(gem::Link& link) {
    std::size_t n = 0;
    for (Analyzer* analyzer : analyzers_) {
        while (const Verdict* v = analyzer->front_verdict()) {
            decide(*v, link);
            analyzer->pop_verdict();
            ++n;
        }
    }
    return n;
}

void EdgeCore::decide(const Verdict& verdict, gem::Link& link) {
    ++stats_.verdicts;
    ++stats_.by_pattern[static_cast<std::size_t>(verdict.pattern) % kPatternCount];
    const auto pattern = decision_.observe(verdict.lot, verdict.pattern);
    if (!pattern) {
        return;
    }
    ++stats_.holds;
    HoldRecord hold;
    hold.lot = verdict.lot;
    hold.pattern = *pattern;
    hold.wafer = verdict.wafer;
    hold.received = verdict.received;
    hold.analysed = verdict.analysed;
    hold.decided = Clock::now();
    if (holds_.size() == holds_.capacity()) {
        ++stats_.holds_unrecorded;
        send(hold, link); // sent once, not retried
        return;
    }
    send(holds_.emplace_back(hold), link);
}

void EdgeCore::send(HoldRecord& hold, gem::Link& link) {
    const auto system = host_.command(link, {gem::LotAction::hold, hold.lot.view()});
    if (system) {
        hold.state = HoldRecord::State::sent;
        hold.system = *system;
        hold.sent = Clock::now();
        if (log_) {
            log(std::format("HOLD {} ({}, after wafer {})", hold.lot.view(),
                            pattern_name(hold.pattern), hold.wafer));
        }
    } else if (system.error() != gem::GemError::busy &&
               system.error() != gem::GemError::not_communicating) {
        hold.state = HoldRecord::State::failed;
    } // else: stays waiting, retried by send_waiting
}

void EdgeCore::send_waiting(gem::Link& link) {
    // Replies are rare (one per hold), so a scan of the records is cheap enough.
    for (HoldRecord& hold : holds_) {
        if (hold.state == HoldRecord::State::waiting) {
            send(hold, link);
            if (hold.state == HoldRecord::State::waiting) {
                return; // still no room
            }
        }
    }
}

HoldRecord* EdgeCore::find_sent(std::uint32_t system) noexcept {
    // Newest first: the reply is usually to a recent hold.
    for (HoldRecord& h : holds_ | std::views::reverse) {
        if (h.state == HoldRecord::State::sent && h.system == system) {
            return &h;
        }
    }
    return nullptr;
}

} // namespace waferedge::pipeline
