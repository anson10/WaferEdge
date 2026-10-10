#pragma once

#include "waferedge/secs/item.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

// SECS-II encoding into a caller-owned buffer that is reused from message to message: the
// encoder clears it but keeps its capacity, so once it has grown to the largest message,
// encoding allocates nothing.
//
//   std::vector<std::uint8_t> buffer;          // lives as long as the connection
//   secs::Encoder enc(buffer);
//   enc.list(2).ascii("HOLD").list(0);
//   auto body = enc.finish();                  // Result<span>: the bytes, or the first error
//
// A list's children follow its list(n) call, depth first, as on the wire. Errors are
// sticky: after the first one every call does nothing and finish() reports it, so a builder
// chain needs one check at the end (ADR-0009). Lengths always use the fewest length bytes.
namespace waferedge::secs {

class Encoder {
public:
    // Clears `out` (keeping its capacity) and appends the item to it.
    explicit Encoder(std::vector<std::uint8_t>& out) noexcept;

    // A list of n children; the next n items (each with its own subtree) are its children.
    Encoder& list(std::size_t n);

    template <Format F>
    Encoder& array(std::span<const value_type_t<F>> values) {
        static_assert(F != Format::list, "use list()");
        constexpr std::size_t size = sizeof(value_type_t<F>);
        std::uint8_t* p = begin_item(F, values.size() * size);
        if (p == nullptr) {
            return *this;
        }
        if constexpr (size == 1 && !std::is_same_v<value_type_t<F>, bool>) {
            // One-byte values have no byte order: one copy (the wafer map's bins).
            if (!values.empty()) { // an empty span may hold nullptr, which memcpy forbids
                std::memcpy(p, values.data(), values.size());
            }
        } else {
            for (const auto v : values) {
                store_be(p, v);
                p += size;
            }
        }
        end_item();
        return *this;
    }
    template <Format F>
    Encoder& value(value_type_t<F> v) {
        return array<F>(std::span<const value_type_t<F>>(&v, 1));
    }

    // Shorthands for the formats GEM messages use most.
    Encoder& ascii(std::string_view s) { return array<Format::ascii>(s); }
    Encoder& jis8(std::string_view s) { return array<Format::jis8>(s); }
    Encoder& binary(std::span<const std::uint8_t> bytes) { return array<Format::binary>(bytes); }
    Encoder& boolean(bool b) { return value<Format::boolean>(b); }
    Encoder& u1(std::uint8_t v) { return value<Format::u1>(v); }
    Encoder& u2(std::uint16_t v) { return value<Format::u2>(v); }
    Encoder& u4(std::uint32_t v) { return value<Format::u4>(v); }
    Encoder& u8(std::uint64_t v) { return value<Format::u8>(v); }
    Encoder& i1(std::int8_t v) { return value<Format::i1>(v); }
    Encoder& i2(std::int16_t v) { return value<Format::i2>(v); }
    Encoder& i4(std::int32_t v) { return value<Format::i4>(v); }
    Encoder& i8(std::int64_t v) { return value<Format::i8>(v); }
    Encoder& f4(float v) { return value<Format::f4>(v); }
    Encoder& f8(double v) { return value<Format::f8>(v); }

    // Copies a decoded item (and its subtree), rewriting each header with the fewest length
    // bytes; the data is copied as is. Forwarding a message needs no tree in between.
    Encoder& item(ItemView view);

    // The encoded body, or the first error. Fails if a list is still waiting for children.
    // An encoder that wrote nothing gives an empty body (header-only messages such as S1F1).
    [[nodiscard]] Result<std::span<const std::uint8_t>> finish() const noexcept;
    [[nodiscard]] bool ok() const noexcept { return !error_.has_value(); }

private:
    // Writes the header, reserves `bytes` of data and returns where to write them, or
    // nullptr after an error.
    std::uint8_t* begin_item(Format f, std::size_t length_or_count, std::size_t data_bytes);
    std::uint8_t* begin_item(Format f, std::size_t bytes) { return begin_item(f, bytes, bytes); }
    // Closes every list the item just written completes.
    void end_item() noexcept;
    void fail(Errc code) noexcept;

    template <typename T>
    static void store_be(std::uint8_t* p, T v) noexcept {
        if constexpr (std::is_same_v<T, bool>) {
            *p = v ? 1 : 0;
        } else if constexpr (sizeof(T) == 1) {
            *p = std::bit_cast<std::uint8_t>(v);
        } else {
            using U = std::conditional_t<
                sizeof(T) == 2, std::uint16_t,
                std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>>;
            auto raw = std::bit_cast<U>(v);
            if constexpr (std::endian::native == std::endian::little) {
                raw = std::byteswap(raw);
            }
            std::memcpy(p, &raw, sizeof raw);
        }
    }

    std::vector<std::uint8_t>* out_;
    std::array<std::uint32_t, kMaxDepth> remaining_{}; // children still due, per open list
    std::size_t depth_ = 0;
    bool done_ = false; // the body's one top-level item is complete
    std::optional<Errc> error_;
};

} // namespace waferedge::secs
