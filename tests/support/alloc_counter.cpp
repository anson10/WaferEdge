#include "support/alloc_counter.hpp"

#include <cstdlib>
#include <new>

namespace {

thread_local std::uint64_t t_count = 0;

void* allocate(std::size_t size) {
    ++t_count;
    if (void* p = std::malloc(size == 0 ? 1 : size)) {
        return p;
    }
    throw std::bad_alloc();
}

void* allocate_aligned(std::size_t size, std::align_val_t align) {
    ++t_count;
    const auto a = static_cast<std::size_t>(align);
    // aligned_alloc wants a size that is a multiple of the alignment.
    if (void* p = std::aligned_alloc(a, (size + a - 1) / a * a)) {
        return p;
    }
    throw std::bad_alloc();
}

} // namespace

std::uint64_t waferedge::alloc::count() noexcept {
    return t_count;
}

// The replaceable forms the program can reach; nothrow and sized forms forward to these.
void* operator new(std::size_t size) {
    return allocate(size);
}
void* operator new[](std::size_t size) {
    return allocate(size);
}
void* operator new(std::size_t size, std::align_val_t align) {
    return allocate_aligned(size, align);
}
void* operator new[](std::size_t size, std::align_val_t align) {
    return allocate_aligned(size, align);
}
void operator delete(void* p) noexcept {
    std::free(p);
}
void operator delete[](void* p) noexcept {
    std::free(p);
}
void operator delete(void* p, std::size_t) noexcept {
    std::free(p);
}
void operator delete[](void* p, std::size_t) noexcept {
    std::free(p);
}
void operator delete(void* p, std::align_val_t) noexcept {
    std::free(p);
}
void operator delete[](void* p, std::align_val_t) noexcept {
    std::free(p);
}
void operator delete(void* p, std::size_t, std::align_val_t) noexcept {
    std::free(p);
}
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept {
    std::free(p);
}
