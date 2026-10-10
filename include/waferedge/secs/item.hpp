#pragma once

#include "waferedge/secs/format.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <iterator>
#include <span>
#include <string>
#include <string_view>

// Zero-copy SECS-II decoding (ADR-0009). decode() validates a message body once, without
// allocating, and returns an ItemView: a pointer into the caller's buffer plus the item's
// header fields. Children and values are read from that buffer on demand, and integers and
// floats are byte-swapped as they are read; nothing is copied. The buffer must outlive every
// view into it.
namespace waferedge::secs {

enum class Errc : std::uint8_t {
    truncated,         // the input ends inside an item's header or data
    zero_length_bytes, // the format byte says 0 length bytes (E5 allows 1-3)
    unknown_format,    // the 6-bit code is not a format this codec knows
    bad_array_length,  // data bytes are not a multiple of the element size
    too_deep,          // lists nested deeper than kMaxDepth
    trailing_bytes,    // bytes left after the body's single item
    too_long,          // encode: more than 0xFFFFFF data bytes or children
    count_mismatch,    // encode: a list got more or fewer children than it declared
    wrong_format,      // access: the item is not of the requested format
    out_of_range,      // access: no such child or element
};

struct Error {
    Errc code;
    std::size_t offset = 0; // byte offset in the decoded buffer (decode); 0 otherwise
    friend bool operator==(const Error&, const Error&) = default;
};

[[nodiscard]] std::string_view errc_name(Errc code) noexcept;
[[nodiscard]] std::string describe(const Error& error);

template <typename T>
using Result = std::expected<T, Error>;

// Lists may nest this deep. Real GEM messages use 4-6 levels; the limit bounds the work and
// the recursion of anything that walks a decoded tree (to_sml), whatever a peer sends.
inline constexpr std::size_t kMaxDepth = 64;
// Three length bytes: the most data bytes (or children) one item can have.
inline constexpr std::size_t kMaxLength = 0xFF'FFFF;

namespace detail {

template <typename T>
[[nodiscard]] inline T load_be(const std::uint8_t* p) noexcept {
    if constexpr (std::is_same_v<T, bool>) {
        return *p != 0;
    } else if constexpr (sizeof(T) == 1) {
        return std::bit_cast<T>(*p);
    } else {
        using U =
            std::conditional_t<sizeof(T) == 2, std::uint16_t,
                               std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>>;
        U raw;
        std::memcpy(&raw, p, sizeof raw);
        if constexpr (std::endian::native == std::endian::little) {
            raw = std::byteswap(raw);
        }
        return std::bit_cast<T>(raw);
    }
}

struct Header {
    Format format;
    std::uint8_t header_size; // format byte + length bytes
    std::uint32_t length;     // data bytes, or children for a list
};

// Reads a header that decode() has already validated.
[[nodiscard]] inline Header read_header(const std::uint8_t* p) noexcept {
    const unsigned n = *p & 0x3U;
    std::uint32_t length = 0;
    for (unsigned i = 1; i <= n; ++i) {
        length = (length << 8U) | p[i];
    }
    return {static_cast<Format>(*p >> 2U), static_cast<std::uint8_t>(1 + n), length};
}

// Past the end of the validated item at p, walking headers only: a list adds its children
// to the count still to skip, so no stack is needed.
[[nodiscard]] const std::uint8_t* skip(const std::uint8_t* p) noexcept;

} // namespace detail

// The values of an array item, read big-endian from the buffer on access.
template <typename T>
class ArrayView {
public:
    class iterator {
    public:
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        iterator() = default;
        explicit iterator(const std::uint8_t* p) noexcept : p_(p) {}
        T operator*() const noexcept { return detail::load_be<T>(p_); }
        iterator& operator++() noexcept {
            p_ += sizeof(T);
            return *this;
        }
        iterator operator++(int) noexcept {
            auto old = *this;
            ++*this;
            return old;
        }
        friend bool operator==(iterator, iterator) = default;

    private:
        const std::uint8_t* p_ = nullptr;
    };

    ArrayView() = default;
    ArrayView(const std::uint8_t* data, std::size_t size) noexcept : data_(data), size_(size) {}

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    // Precondition: i < size().
    [[nodiscard]] T operator[](std::size_t i) const noexcept {
        return detail::load_be<T>(data_ + i * sizeof(T));
    }
    [[nodiscard]] iterator begin() const noexcept { return iterator(data_); }
    [[nodiscard]] iterator end() const noexcept { return iterator(data_ + size_ * sizeof(T)); }
    // The encoded (big-endian) bytes: for one-byte types, the values themselves.
    [[nodiscard]] std::span<const std::uint8_t> bytes() const noexcept {
        return {data_, size_ * sizeof(T)};
    }

private:
    const std::uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
};
static_assert(std::forward_iterator<ArrayView<std::uint32_t>::iterator>);

class ListView;
class Encoder;

// One validated item. Only decode() and ListView make them, so every ItemView points at
// bytes that have passed validation; reading it needs no further checks.
class ItemView {
public:
    ItemView() = default;

    [[nodiscard]] Format format() const noexcept { return format_; }
    [[nodiscard]] bool is_list() const noexcept { return format_ == Format::list; }
    // Children for a list, values for an array (characters for A / J).
    [[nodiscard]] std::size_t size() const noexcept {
        return is_list() ? length_ : length_ / element_size(format_);
    }
    // The whole item as encoded, header included. O(1) for an array; a list walks its
    // subtree's headers to find its end.
    [[nodiscard]] std::span<const std::uint8_t> encoded() const noexcept {
        const std::size_t extent = is_list()
                                       ? static_cast<std::size_t>(detail::skip(begin_) - begin_)
                                       : std::size_t{header_size_} + length_;
        return {begin_, extent};
    }
    // The item's data: the values of an array, the children's encodings for a list.
    [[nodiscard]] std::span<const std::uint8_t> payload() const noexcept {
        return encoded().subspan(header_size_);
    }

    [[nodiscard]] Result<ListView> list() const noexcept;

    template <Format F>
    [[nodiscard]] Result<ArrayView<value_type_t<F>>> as() const noexcept {
        static_assert(F != Format::list, "use list()");
        if (format_ != F) {
            return std::unexpected(Error{Errc::wrong_format});
        }
        return ArrayView<value_type_t<F>>(begin_ + header_size_, length_ / sizeof(value_type_t<F>));
    }
    // An array of exactly one value, as GEM's scalar fields (DATAID, CEID, ...) are sent.
    template <Format F>
    [[nodiscard]] Result<value_type_t<F>> scalar() const noexcept {
        auto values = as<F>();
        if (!values) {
            return std::unexpected(values.error());
        }
        if (values->size() != 1) {
            return std::unexpected(Error{Errc::out_of_range});
        }
        return (*values)[0];
    }
    // A or J text, unvalidated (tools send 8-bit bytes in A items too).
    [[nodiscard]] Result<std::string_view> text() const noexcept;
    // A single U1/U2/U4/U8 value widened: GEM lets a tool pick the width of its ids.
    [[nodiscard]] Result<std::uint64_t> unsigned_scalar() const noexcept;

private:
    friend Result<ItemView> decode(std::span<const std::uint8_t> bytes) noexcept;
    friend Result<ItemView> decode_prefix(std::span<const std::uint8_t> bytes) noexcept;
    friend class ListView;
    friend class Encoder;

    // p points at a validated item.
    explicit ItemView(const std::uint8_t* p) noexcept : ItemView(p, detail::read_header(p)) {}
    ItemView(const std::uint8_t* p, detail::Header h) noexcept
        : begin_(p), length_(h.length), header_size_(h.header_size), format_(h.format) {}

    const std::uint8_t* begin_ = nullptr;
    std::uint32_t length_ = 0;
    std::uint8_t header_size_ = 0;
    Format format_ = Format::list;
};

// The children of a list item. Moving to the next child skips the current one's headers
// (O(size of the child's subtree)); index with at() only for short lists.
class ListView {
public:
    class iterator {
    public:
        using value_type = ItemView;
        using difference_type = std::ptrdiff_t;
        iterator() = default;
        iterator(const std::uint8_t* p, std::size_t remaining) noexcept
            : p_(p), remaining_(remaining) {}
        ItemView operator*() const noexcept { return ItemView(p_); }
        iterator& operator++() noexcept {
            p_ = detail::skip(p_);
            --remaining_;
            return *this;
        }
        iterator operator++(int) noexcept {
            auto old = *this;
            ++*this;
            return old;
        }
        // Iterators compare by children left: end() needs no walk to find the end pointer.
        friend bool operator==(const iterator& a, const iterator& b) noexcept {
            return a.remaining_ == b.remaining_;
        }

    private:
        const std::uint8_t* p_ = nullptr;
        std::size_t remaining_ = 0;
    };

    ListView() = default;
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] iterator begin() const noexcept { return {first_, size_}; }
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static): the range interface
    [[nodiscard]] iterator end() const noexcept { return {nullptr, 0}; }
    [[nodiscard]] Result<ItemView> at(std::size_t i) const noexcept;

private:
    friend class ItemView;
    ListView(const std::uint8_t* first, std::size_t size) noexcept : first_(first), size_(size) {}
    const std::uint8_t* first_ = nullptr;
    std::size_t size_ = 0;
};
static_assert(std::forward_iterator<ListView::iterator>);

// The body of a data message is one item: decode() wants exactly one, filling all of
// `bytes`; decode_prefix() takes the first item and ignores what follows it.
[[nodiscard]] Result<ItemView> decode(std::span<const std::uint8_t> bytes) noexcept;
[[nodiscard]] Result<ItemView> decode_prefix(std::span<const std::uint8_t> bytes) noexcept;

// SML (SECS Message Language), the text form logs and tool manuals use:
//   <L [2]
//     <U4 17>
//     <A "LOT-1">
//   >
// For people and tests, not for the hot path: it allocates.
void append_sml(std::string& out, ItemView item, int indent = 0);
[[nodiscard]] std::string to_sml(ItemView item);

} // namespace waferedge::secs
