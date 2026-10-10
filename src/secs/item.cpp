#include "waferedge/secs/item.hpp"

#include <array>
#include <format>

namespace waferedge::secs {

std::string_view errc_name(Errc code) noexcept {
    switch (code) {
    case Errc::truncated:
        return "truncated";
    case Errc::zero_length_bytes:
        return "zero length bytes";
    case Errc::unknown_format:
        return "unknown format";
    case Errc::bad_array_length:
        return "array length not a multiple of the element size";
    case Errc::too_deep:
        return "lists nested too deep";
    case Errc::trailing_bytes:
        return "trailing bytes after the item";
    case Errc::too_long:
        return "item too long";
    case Errc::count_mismatch:
        return "list child count mismatch";
    case Errc::wrong_format:
        return "wrong format";
    case Errc::out_of_range:
        return "out of range";
    }
    return "?";
}

std::string describe(const Error& error) {
    return std::format("{} at byte {}", errc_name(error.code), error.offset);
}

namespace detail {

const std::uint8_t* skip(const std::uint8_t* p) noexcept {
    std::size_t pending = 1;
    while (pending > 0) {
        const auto h = read_header(p);
        --pending;
        if (h.format == Format::list) {
            pending += h.length;
            p += h.header_size;
        } else {
            p += h.header_size + std::size_t{h.length};
        }
    }
    return p;
}

} // namespace detail

namespace {

// Validates the item at bytes[0] and returns its size in bytes. One pass over the headers,
// iterative: `remaining` holds, for each open list, how many children it still waits for, so
// a hostile depth fails with too_deep instead of overflowing the stack. Array data is never
// read, only skipped.
Result<std::size_t> validate(std::span<const std::uint8_t> bytes) noexcept {
    std::array<std::uint32_t, kMaxDepth> remaining; // NOLINT(*-member-init): written before read
    std::size_t depth = 0;
    std::size_t pos = 0;
    const std::size_t size = bytes.size();
    for (;;) {
        if (pos >= size) {
            return std::unexpected(Error{Errc::truncated, pos});
        }
        const std::uint8_t format_byte = bytes[pos];
        const unsigned n = format_byte & 0x3U;
        if (n == 0) {
            return std::unexpected(Error{Errc::zero_length_bytes, pos});
        }
        const auto format = format_from_code(static_cast<std::uint8_t>(format_byte >> 2U));
        if (!format) {
            return std::unexpected(Error{Errc::unknown_format, pos});
        }
        if (size - pos <= n) {
            return std::unexpected(Error{Errc::truncated, pos});
        }
        std::uint32_t length = 0;
        for (unsigned i = 1; i <= n; ++i) {
            length = (length << 8U) | bytes[pos + i];
        }
        const std::size_t item_pos = pos;
        pos += 1 + n;
        if (*format == Format::list) {
            if (length > 0) {
                if (depth == kMaxDepth) {
                    return std::unexpected(Error{Errc::too_deep, item_pos});
                }
                remaining[depth++] = length;
                continue; // the list completes when its last child does
            }
        } else {
            if (length % element_size(*format) != 0) {
                return std::unexpected(Error{Errc::bad_array_length, item_pos});
            }
            if (size - pos < length) {
                return std::unexpected(Error{Errc::truncated, item_pos});
            }
            pos += length;
        }
        // An item is complete: it counts towards its parent, which may complete in turn.
        for (;;) {
            if (depth == 0) {
                return pos;
            }
            if (--remaining[depth - 1] != 0) {
                break;
            }
            --depth;
        }
    }
}

} // namespace

Result<ItemView> decode_prefix(std::span<const std::uint8_t> bytes) noexcept {
    auto size = validate(bytes);
    if (!size) {
        return std::unexpected(size.error());
    }
    return ItemView(bytes.data());
}

Result<ItemView> decode(std::span<const std::uint8_t> bytes) noexcept {
    auto size = validate(bytes);
    if (!size) {
        return std::unexpected(size.error());
    }
    if (*size != bytes.size()) {
        return std::unexpected(Error{Errc::trailing_bytes, *size});
    }
    return ItemView(bytes.data());
}

Result<ListView> ItemView::list() const noexcept {
    if (!is_list()) {
        return std::unexpected(Error{Errc::wrong_format});
    }
    return ListView(begin_ + header_size_, length_);
}

Result<std::string_view> ItemView::text() const noexcept {
    if (format_ != Format::ascii && format_ != Format::jis8) {
        return std::unexpected(Error{Errc::wrong_format});
    }
    return std::string_view(reinterpret_cast<const char*>(begin_ + header_size_), length_);
}

Result<std::uint64_t> ItemView::unsigned_scalar() const noexcept {
    const auto widen = [](auto r) -> Result<std::uint64_t> {
        if (!r) {
            return std::unexpected(r.error());
        }
        return std::uint64_t{*r};
    };
    switch (format_) {
    case Format::u1:
        return widen(scalar<Format::u1>());
    case Format::u2:
        return widen(scalar<Format::u2>());
    case Format::u4:
        return widen(scalar<Format::u4>());
    case Format::u8:
        return widen(scalar<Format::u8>());
    default:
        return std::unexpected(Error{Errc::wrong_format});
    }
}

Result<ItemView> ListView::at(std::size_t i) const noexcept {
    if (i >= size_) {
        return std::unexpected(Error{Errc::out_of_range});
    }
    auto it = begin();
    for (std::size_t k = 0; k < i; ++k) {
        ++it;
    }
    return *it;
}

namespace {

void append_text(std::string& out, std::string_view s) {
    // Printable runs in quotes, other bytes as hex between them: "AB" 0x0A "C".
    bool in_quotes = false;
    bool first = true;
    for (const char ch : s) {
        const auto byte = static_cast<unsigned char>(ch);
        const bool printable = byte >= 0x20 && byte < 0x7F && ch != '"';
        if (printable) {
            if (!in_quotes) {
                out += first ? "\"" : " \"";
                in_quotes = true;
            }
            out += ch;
        } else {
            if (in_quotes) {
                out += '"';
                in_quotes = false;
            }
            std::format_to(std::back_inserter(out), "{}0x{:02X}", first ? "" : " ", byte);
        }
        first = false;
    }
    if (in_quotes) {
        out += '"';
    }
}

} // namespace

// Recursive, bounded: decode() admits at most kMaxDepth nested lists.
void append_sml(std::string& out, ItemView item, int indent) { // NOLINT(misc-no-recursion)
    out.append(static_cast<std::size_t>(indent), ' ');
    out += '<';
    out += format_name(item.format());
    if (item.is_list()) {
        const auto children = *item.list();
        std::format_to(std::back_inserter(out), " [{}]", children.size());
        if (children.empty()) {
            out += '>';
            return;
        }
        for (const ItemView child : children) {
            out += '\n';
            append_sml(out, child, indent + 2);
        }
        out += '\n';
        out.append(static_cast<std::size_t>(indent), ' ');
        out += '>';
        return;
    }
    if (item.format() == Format::ascii || item.format() == Format::jis8) {
        if (item.size() > 0) {
            out += ' ';
            append_text(out, *item.text());
        }
        out += '>';
        return;
    }
    visit_format(item.format(), [&]<Format F>(std::integral_constant<Format, F>) {
        if constexpr (F != Format::list && F != Format::ascii && F != Format::jis8) {
            // Bound to a name first: `for (v : *item.as<F>())` would iterate a view inside a
            // temporary std::expected that is gone before the loop runs (until C++23's P2718,
            // which GCC 13 lacks). ASan's stack-use-after-scope found it here.
            const auto values = *item.as<F>();
            for (const auto v : values) {
                if constexpr (F == Format::binary) {
                    std::format_to(std::back_inserter(out), " 0x{:02X}", v);
                } else if constexpr (F == Format::boolean) {
                    out += v ? " T" : " F";
                } else if constexpr (sizeof(v) == 1) {
                    std::format_to(std::back_inserter(out), " {}", static_cast<int>(v));
                } else {
                    std::format_to(std::back_inserter(out), " {}", v);
                }
            }
        }
    });
    out += '>';
}

std::string to_sml(ItemView item) {
    std::string out;
    append_sml(out, item);
    return out;
}

} // namespace waferedge::secs
