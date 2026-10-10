#include "waferedge/secs/encoder.hpp"

namespace waferedge::secs {

Encoder::Encoder(std::vector<std::uint8_t>& out) noexcept : out_(&out) {
    out.clear(); // keeps the capacity: the point of a reused buffer
}

void Encoder::fail(Errc code) noexcept {
    if (!error_) {
        error_ = code;
    }
}

std::uint8_t* Encoder::begin_item(Format f, std::size_t length_or_count, std::size_t data_bytes) {
    if (error_) {
        return nullptr;
    }
    if (length_or_count > kMaxLength) {
        fail(Errc::too_long);
        return nullptr;
    }
    // The body holds one item: a second top-level item would be unreadable as one message.
    if (done_) {
        fail(Errc::count_mismatch);
        return nullptr;
    }
    if (depth_ > 0) {
        --remaining_[depth_ - 1];
    }
    const auto length = static_cast<std::uint32_t>(length_or_count);
    unsigned n = 3; // length bytes: the fewest that hold the length
    if (length <= 0xFF) {
        n = 1;
    } else if (length <= 0xFFFF) {
        n = 2;
    }
    const std::size_t at = out_->size();
    out_->resize(at + 1 + n + data_bytes);
    std::uint8_t* p = out_->data() + at;
    *p++ = format_byte(f, n);
    for (unsigned i = n; i > 0; --i) {
        *p++ = static_cast<std::uint8_t>(length >> (8U * (i - 1)));
    }
    return p;
}

void Encoder::end_item() noexcept {
    // Pop every list whose last child this was; the outermost completing ends the body.
    while (depth_ > 0 && remaining_[depth_ - 1] == 0) {
        --depth_;
    }
    if (depth_ == 0) {
        done_ = true;
    }
}

Encoder& Encoder::list(std::size_t n) {
    if (begin_item(Format::list, n, 0) == nullptr) {
        return *this;
    }
    if (n == 0) {
        end_item();
        return *this;
    }
    if (depth_ == kMaxDepth) {
        fail(Errc::too_deep);
        return *this;
    }
    remaining_[depth_++] = static_cast<std::uint32_t>(n);
    return *this;
}

Encoder& Encoder::item(ItemView view) {
    // Items are laid out depth first, so the subtree's headers come in exactly the order the
    // encoder wants its calls: walk them, re-emitting each, with no recursion.
    const std::uint8_t* p = view.begin_;
    std::size_t pending = 1;
    while (pending > 0 && ok()) {
        const auto h = detail::read_header(p);
        p += h.header_size;
        --pending;
        if (h.format == Format::list) {
            list(h.length);
            pending += h.length;
        } else {
            std::uint8_t* data = begin_item(h.format, h.length);
            if (data == nullptr) {
                break;
            }
            std::memcpy(data, p, h.length);
            end_item();
            p += h.length;
        }
    }
    return *this;
}

Result<std::span<const std::uint8_t>> Encoder::finish() const noexcept {
    if (error_) {
        return std::unexpected(Error{*error_});
    }
    if (depth_ > 0) {
        return std::unexpected(Error{Errc::count_mismatch});
    }
    return std::span<const std::uint8_t>(*out_);
}

} // namespace waferedge::secs
