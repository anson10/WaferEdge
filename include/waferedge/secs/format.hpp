#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>

// SECS-II (SEMI E5) item formats. Every item starts with a format byte:
//
//   bit 7 ........ 2 | 1 .. 0
//   format code (6)  | number of length bytes (1, 2 or 3)
//
// followed by 1-3 big-endian length bytes (data bytes for arrays, number of children for a
// list) and the data, big-endian. The codes are octal in the standard; they are written in
// octal here so they can be checked against any public table. docs/secs.md has the layout.
namespace waferedge::secs {

enum class Format : std::uint8_t {
    list = 000,    // L: a count of child items, no data of its own
    binary = 010,  // B: raw bytes
    boolean = 011, // BOOLEAN: one byte per value, non-zero is true
    ascii = 020,   // A: one byte per character
    jis8 = 021,    // J: JIS-8, one byte per character
    i8 = 030,
    i1 = 031,
    i2 = 032,
    i4 = 034,
    f8 = 040,
    f4 = 044,
    u8 = 050,
    u1 = 051,
    u2 = 052,
    u4 = 054,
};

// Every format, in the order docs/secs.md lists them.
inline constexpr std::array kAllFormats = {
    Format::list, Format::binary, Format::boolean, Format::ascii, Format::jis8,
    Format::i1,   Format::i2,     Format::i4,      Format::i8,    Format::u1,
    Format::u2,   Format::u4,     Format::u8,      Format::f4,    Format::f8,
};

// The value type an array item of format F holds once decoded (host byte order).
template <Format F>
struct FormatTraits;
template <>
struct FormatTraits<Format::binary> {
    using type = std::uint8_t;
};
template <>
struct FormatTraits<Format::boolean> {
    using type = bool;
};
template <>
struct FormatTraits<Format::ascii> {
    using type = char;
};
template <>
struct FormatTraits<Format::jis8> {
    using type = char;
};
template <>
struct FormatTraits<Format::i1> {
    using type = std::int8_t;
};
template <>
struct FormatTraits<Format::i2> {
    using type = std::int16_t;
};
template <>
struct FormatTraits<Format::i4> {
    using type = std::int32_t;
};
template <>
struct FormatTraits<Format::i8> {
    using type = std::int64_t;
};
template <>
struct FormatTraits<Format::u1> {
    using type = std::uint8_t;
};
template <>
struct FormatTraits<Format::u2> {
    using type = std::uint16_t;
};
template <>
struct FormatTraits<Format::u4> {
    using type = std::uint32_t;
};
template <>
struct FormatTraits<Format::u8> {
    using type = std::uint64_t;
};
template <>
struct FormatTraits<Format::f4> {
    using type = float;
};
template <>
struct FormatTraits<Format::f8> {
    using type = double;
};

template <Format F>
using value_type_t = typename FormatTraits<F>::type;

struct FormatInfo {
    bool valid = false;
    std::uint8_t element_size = 0; // bytes per value; 0 for a list
    std::string_view name;         // as SML writes it: L, B, BOOLEAN, A, J, I1, ..., F8
};

namespace detail {

constexpr std::array<FormatInfo, 64> make_format_table() {
    std::array<FormatInfo, 64> t{};
    const auto set = [&t](Format f, std::uint8_t size, std::string_view name) {
        t[static_cast<std::size_t>(f)] = FormatInfo{true, size, name};
    };
    set(Format::list, 0, "L");
    set(Format::binary, 1, "B");
    set(Format::boolean, 1, "BOOLEAN");
    set(Format::ascii, 1, "A");
    set(Format::jis8, 1, "J");
    set(Format::i1, 1, "I1");
    set(Format::i2, 2, "I2");
    set(Format::i4, 4, "I4");
    set(Format::i8, 8, "I8");
    set(Format::u1, 1, "U1");
    set(Format::u2, 2, "U2");
    set(Format::u4, 4, "U4");
    set(Format::u8, 8, "U8");
    set(Format::f4, 4, "F4");
    set(Format::f8, 8, "F8");
    return t;
}

} // namespace detail

// Indexed by the 6-bit format code: the decoder's only lookup per item.
inline constexpr std::array<FormatInfo, 64> kFormatTable = detail::make_format_table();

[[nodiscard]] constexpr const FormatInfo& info(Format f) noexcept {
    return kFormatTable[static_cast<std::size_t>(f)];
}
[[nodiscard]] constexpr std::size_t element_size(Format f) noexcept {
    return info(f).element_size;
}
[[nodiscard]] constexpr std::string_view format_name(Format f) noexcept {
    return info(f).name;
}
// The format for a 6-bit code, or nothing if E5 defines none (or this codec doesn't: C2,
// the two-byte localized string, code 022, is out of scope; docs/secs.md).
[[nodiscard]] constexpr std::optional<Format> format_from_code(std::uint8_t code) noexcept {
    if (code >= kFormatTable.size() || !kFormatTable[code].valid) {
        return std::nullopt;
    }
    return static_cast<Format>(code);
}
// The format byte with n length bytes (1..3).
[[nodiscard]] constexpr std::uint8_t format_byte(Format f, unsigned n_length_bytes) noexcept {
    return static_cast<std::uint8_t>((static_cast<unsigned>(f) << 2U) | n_length_bytes);
}

// Calls fn(std::integral_constant<Format, F>{}) for the run-time format f, so generic code can
// use value_type_t<F> for every array format. fn must also accept the list's constant.
template <typename Fn>
constexpr decltype(auto) visit_format(Format f, Fn&& fn) {
    using enum Format;
    // clang-format off
    switch (f) {
    case list:    return std::forward<Fn>(fn)(std::integral_constant<Format, list>{});
    case binary:  return std::forward<Fn>(fn)(std::integral_constant<Format, binary>{});
    case boolean: return std::forward<Fn>(fn)(std::integral_constant<Format, boolean>{});
    case ascii:   return std::forward<Fn>(fn)(std::integral_constant<Format, ascii>{});
    case jis8:    return std::forward<Fn>(fn)(std::integral_constant<Format, jis8>{});
    case i1:      return std::forward<Fn>(fn)(std::integral_constant<Format, i1>{});
    case i2:      return std::forward<Fn>(fn)(std::integral_constant<Format, i2>{});
    case i4:      return std::forward<Fn>(fn)(std::integral_constant<Format, i4>{});
    case i8:      return std::forward<Fn>(fn)(std::integral_constant<Format, i8>{});
    case u1:      return std::forward<Fn>(fn)(std::integral_constant<Format, u1>{});
    case u2:      return std::forward<Fn>(fn)(std::integral_constant<Format, u2>{});
    case u4:      return std::forward<Fn>(fn)(std::integral_constant<Format, u4>{});
    case u8:      return std::forward<Fn>(fn)(std::integral_constant<Format, u8>{});
    case f4:      return std::forward<Fn>(fn)(std::integral_constant<Format, f4>{});
    case f8:      return std::forward<Fn>(fn)(std::integral_constant<Format, f8>{});
    }
    // clang-format on
    std::unreachable(); // formats only come from format_from_code or the enumerators
}

// The table and the C++ types agree, checked at compile time for every array format.
namespace detail {
template <Format F>
constexpr bool traits_match_table() {
    return sizeof(value_type_t<F>) == element_size(F) &&
           std::is_trivially_copyable_v<value_type_t<F>>;
}
} // namespace detail
static_assert(
    detail::traits_match_table<Format::binary>() && detail::traits_match_table<Format::boolean>() &&
    detail::traits_match_table<Format::ascii>() && detail::traits_match_table<Format::jis8>() &&
    detail::traits_match_table<Format::i1>() && detail::traits_match_table<Format::i2>() &&
    detail::traits_match_table<Format::i4>() && detail::traits_match_table<Format::i8>() &&
    detail::traits_match_table<Format::u1>() && detail::traits_match_table<Format::u2>() &&
    detail::traits_match_table<Format::u4>() && detail::traits_match_table<Format::u8>() &&
    detail::traits_match_table<Format::f4>() && detail::traits_match_table<Format::f8>());
static_assert(format_byte(Format::list, 1) == 0x01 && format_byte(Format::u4, 1) == 0xB1 &&
              format_byte(Format::ascii, 2) == 0x42);
static_assert(!format_from_code(022).has_value() && format_from_code(054) == Format::u4);

} // namespace waferedge::secs
