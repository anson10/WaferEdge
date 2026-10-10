#include "secs_item_target.hpp"

#include "waferedge/secs/encoder.hpp"
#include "waferedge/secs/item.hpp"

#include <algorithm>
#include <cstdlib>
#include <span>
#include <string>
#include <vector>

namespace waferedge::fuzz {

namespace {

// The oracle: a failed check is a bug even when nothing crashed.
void check(bool ok) {
    if (!ok) {
        std::abort();
    }
}

} // namespace

int secs_item(const std::uint8_t* data, std::size_t size) {
    using namespace secs;
    const std::span<const std::uint8_t> bytes(data, size);
    const auto prefix = decode_prefix(bytes);
    const auto whole = decode(bytes);

    // decode() and decode_prefix() agree: same error, or decode() fails only on the bytes
    // after a valid first item.
    if (!prefix) {
        check(!whole && whole.error() == prefix.error());
        return 0;
    }
    check(prefix->encoded().size() <= size);
    if (!whole) {
        check(whole.error().code == Errc::trailing_bytes &&
              whole.error().offset == prefix->encoded().size());
        return 0;
    }
    check(whole->encoded().size() == size);

    // Walk every value (to_sml reads them all) and copy the item through the encoder.
    const std::string sml = to_sml(*whole);
    std::vector<std::uint8_t> copy;
    Encoder encoder(copy);
    encoder.item(*whole);
    const auto encoded = encoder.finish();
    check(encoded.has_value() && encoded->size() <= size); // the fewest length bytes

    // The copy decodes to the same tree, and copying it again changes nothing.
    const auto again = decode(*encoded);
    check(again.has_value() && to_sml(*again) == sml);
    std::vector<std::uint8_t> copy2;
    Encoder encoder2(copy2);
    encoder2.item(*again);
    const auto encoded2 = encoder2.finish();
    check(encoded2.has_value() && std::ranges::equal(*encoded2, *encoded));
    return 0;
}

} // namespace waferedge::fuzz
