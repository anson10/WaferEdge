// Round-trip properties of the SECS-II codec on random item trees:
//   tree -> encode -> decode -> tree is the identity (values compared bit for bit, so NaN
//   payloads and -0.0 must survive), and re-encoding a decoded body gives the same bytes.
#include "waferedge/secs/encoder.hpp"
#include "waferedge/secs/item.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

using namespace waferedge::secs;

namespace {

// An owning item tree, the test's model. Each array value is kept as its bit pattern
// zero-extended to 64 bits, so one vector holds every format and == compares bits.
struct Node {
    Format format = Format::list;
    std::vector<Node> children;
    std::vector<std::uint64_t> bits;
    friend bool operator==(const Node&, const Node&) = default;
};

template <typename T>
std::uint64_t to_bits(T v) {
    if constexpr (std::is_same_v<T, bool>) {
        return v ? 1 : 0;
    } else {
        using U = std::conditional_t<
            sizeof(T) == 1, std::uint8_t,
            std::conditional_t<sizeof(T) == 2, std::uint16_t,
                               std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>>>;
        return std::bit_cast<U>(v);
    }
}

template <typename T>
T from_bits(std::uint64_t bits) {
    if constexpr (std::is_same_v<T, bool>) {
        return bits != 0;
    } else {
        using U = std::conditional_t<
            sizeof(T) == 1, std::uint8_t,
            std::conditional_t<sizeof(T) == 2, std::uint16_t,
                               std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>>>;
        return std::bit_cast<T>(static_cast<U>(bits));
    }
}

// Random trees: lists of 0..5 children up to depth 6, arrays of 0..40 values (sometimes
// 300, to cross into two length bytes), every format equally likely. Uses only mt19937's
// raw output so the trees are the same on every platform.
class Generator {
public:
    explicit Generator(std::uint32_t seed) : rng_(seed) {}

    Node tree(int depth = 0) {
        Node n;
        n.format = kAllFormats[below(kAllFormats.size())];
        if (n.format == Format::list) {
            const std::size_t count = depth >= 6 ? 0 : below(6);
            for (std::size_t i = 0; i < count; ++i) {
                n.children.push_back(tree(depth + 1));
            }
            return n;
        }
        const std::size_t count = below(20) == 0 ? 300 : below(41);
        const std::size_t width = element_size(n.format) * 8;
        for (std::size_t i = 0; i < count; ++i) {
            std::uint64_t v = (std::uint64_t{rng_()} << 32U) | rng_();
            if (width < 64) {
                v &= (std::uint64_t{1} << width) - 1;
            }
            if (n.format == Format::boolean) {
                v &= 1U; // the decoder reads any non-zero byte as true
            }
            n.bits.push_back(v);
        }
        return n;
    }

private:
    std::size_t below(std::size_t n) { return rng_() % n; }
    std::mt19937 rng_;
};

void encode(Encoder& e, const Node& n) {
    if (n.format == Format::list) {
        e.list(n.children.size());
        for (const auto& c : n.children) {
            encode(e, c);
        }
        return;
    }
    visit_format(n.format, [&]<Format F>(std::integral_constant<Format, F>) {
        if constexpr (F != Format::list) {
            using T = value_type_t<F>;
            std::vector<T> values;
            for (const auto b : n.bits) {
                values.push_back(from_bits<T>(b));
            }
            // vector<bool> has no data(): copy bools through a plain array.
            if constexpr (std::is_same_v<T, bool>) {
                const auto plain = std::make_unique<bool[]>(values.size());
                std::ranges::copy(values, plain.get());
                e.array<F>(std::span<const bool>(plain.get(), values.size()));
            } else {
                e.array<F>(values);
            }
        }
    });
}

Node to_node(ItemView item) {
    Node n;
    n.format = item.format();
    if (item.is_list()) {
        const ListView children = *item.list(); // named: see append_sml
        for (const ItemView child : children) {
            n.children.push_back(to_node(child));
        }
        return n;
    }
    visit_format(item.format(), [&]<Format F>(std::integral_constant<Format, F>) {
        if constexpr (F != Format::list) {
            const auto values = *item.as<F>();
            for (const auto v : values) {
                n.bits.push_back(to_bits(v));
            }
        }
    });
    return n;
}

} // namespace

TEST_CASE("random trees: encode then decode is the identity") {
    std::vector<std::uint8_t> buffer; // one buffer for every tree, as on a connection
    std::vector<std::uint8_t> again;
    for (std::uint32_t seed = 0; seed < 3000; ++seed) {
        const Node tree = Generator(seed).tree();
        Encoder e(buffer);
        encode(e, tree);
        auto body = e.finish();
        REQUIRE(body.has_value());

        auto item = decode(*body);
        INFO("seed " << seed);
        REQUIRE(item.has_value());
        CHECK(item->encoded().size() == body->size());
        CHECK(to_node(*item) == tree);

        // Re-encoding the decoded view (no tree in between) gives the same bytes.
        Encoder copy(again);
        copy.item(*item);
        auto copied = copy.finish();
        REQUIRE(copied.has_value());
        CHECK(std::ranges::equal(*copied, *body));
    }
}

TEST_CASE("random trees: a decoded child list re-encodes on its own") {
    std::vector<std::uint8_t> buffer;
    std::vector<std::uint8_t> sub;
    for (std::uint32_t seed = 0; seed < 500; ++seed) {
        const Node tree = Generator(seed).tree();
        if (tree.format != Format::list || tree.children.empty()) {
            continue;
        }
        Encoder e(buffer);
        encode(e, tree);
        auto item = decode(*e.finish());
        REQUIRE(item.has_value());
        std::size_t i = 0;
        const ListView children = *item->list();
        for (const ItemView child : children) {
            INFO("seed " << seed << " child " << i);
            Encoder c(sub);
            c.item(child);
            auto bytes = c.finish();
            REQUIRE(bytes.has_value());
            CHECK(std::ranges::equal(*bytes, child.encoded()));
            CHECK(to_node(*decode(*bytes)) == tree.children[i]);
            ++i;
        }
        CHECK(i == tree.children.size());
    }
}
