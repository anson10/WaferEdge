#include "gpu/iota_kernel.hpp"
#include "waferedge/machine.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vector>

using namespace waferedge;

TEST_CASE("the GPU is found and described") {
    const auto gpu = gpu_info();
    if (!gpu) {
        SKIP("no usable GPU: " << gpu.error());
    }
    CHECK_FALSE(gpu->name.empty());
    CHECK(gpu->multiprocessors > 0);
    CHECK(gpu->runtime_version >= 12000);
    // The binary is built for sm_86; it runs on 8.6 and newer.
    CHECK(gpu->compute_major * 10 + gpu->compute_minor >= 86);
}

TEST_CASE("a kernel built by this toolchain gives the host's answer") {
    if (!gpu_info()) {
        SKIP("no usable GPU");
    }
    // Not a multiple of the block size, so the bounds check in the kernel matters.
    std::vector<int> out(1'000'003, -1);
    REQUIRE(test::iota_on_device(out).empty());
    for (std::size_t i = 0; i < out.size(); ++i) {
        if (out[i] != static_cast<int>(i)) {
            FAIL("out[" << i << "] = " << out[i]);
        }
    }
}
