// The GEMM ladder against a CPU reference that accumulates in double. Labelled gpu: runs
// locally (ctest --preset cuda), not in CI.
#include "waferedge/backend.hpp"
#include "waferedge/gemm.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cuda_runtime_api.h>

#include <cmath>
#include <random>
#include <vector>

using namespace waferedge;
using gpu::GemmKernel;

namespace {

void require_gpu() {
    if (!backend::Cuda::available()) {
        SKIP("no CUDA device with compute capability 8.6 or newer");
    }
}

// A device buffer for the test's lifetime.
template <typename T>
struct Device {
    T* ptr = nullptr;
    explicit Device(const std::vector<T>& host) {
        REQUIRE(cudaMalloc(reinterpret_cast<void**>(&ptr), host.size() * sizeof(T)) == cudaSuccess);
        REQUIRE(cudaMemcpy(ptr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice) ==
                cudaSuccess);
    }
    ~Device() { cudaFree(ptr); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    std::vector<T> download(std::size_t count) const {
        std::vector<T> host(count);
        REQUIRE(cudaMemcpy(host.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost) ==
                cudaSuccess);
        return host;
    }
};

std::vector<float> random_matrix(int rows, int cols, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<float> v(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
    for (auto& x : v) {
        x = static_cast<float>(rng() % 2001) / 1000.0F - 1.0F; // [-1, 1], same on every platform
    }
    return v;
}

// C = A * B in double; also sum |a||b| per element, the scale of the rounding error a float
// sum of K products can carry.
void reference(int m, int n, int k, const std::vector<float>& a, const std::vector<float>& b,
               std::vector<double>& c, std::vector<double>& scale) {
    c.assign(static_cast<std::size_t>(m) * static_cast<std::size_t>(n), 0.0);
    scale.assign(c.size(), 0.0);
    for (int i = 0; i < m; ++i) {
        for (int p = 0; p < k; ++p) {
            const double av = a[static_cast<std::size_t>(i) * static_cast<std::size_t>(k) +
                                static_cast<std::size_t>(p)];
            for (int j = 0; j < n; ++j) {
                const double bv = b[static_cast<std::size_t>(p) * static_cast<std::size_t>(n) +
                                    static_cast<std::size_t>(j)];
                const auto at = static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                                static_cast<std::size_t>(j);
                c[at] += av * bv;
                scale[at] += std::abs(av * bv);
            }
        }
    }
}

// Elements outside |got - want| <= tolerance * scale + 1e-6.
std::size_t count_wrong(const std::vector<float>& got, const std::vector<double>& want,
                        const std::vector<double>& scale, double tolerance) {
    std::size_t wrong = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        wrong += std::abs(got[i] - want[i]) <= tolerance * scale[i] + 1e-6 ? 0U : 1U;
    }
    return wrong;
}

} // namespace

TEST_CASE("every fp32 GEMM rung equals the double-precision reference, edges included") {
    require_gpu();
    const auto kernel = GENERATE(GemmKernel::naive, GemmKernel::tiled, GemmKernel::register_blocked,
                                 GemmKernel::vectorized);
    // Multiples of every tile size, and sizes that leave partial tiles on every side.
    // The vectorized rung runs itself when K and N are multiples of 4 (64, 128x256x72,
    // 260x132x300, 1024) and falls back to rung 3 otherwise.
    for (const auto& [m, n, k] :
         {std::array{64, 64, 64}, std::array{128, 256, 72}, std::array{1, 1, 1},
          std::array{100, 70, 33}, std::array{257, 129, 300}, std::array{31, 513, 7},
          std::array{260, 132, 300}, std::array{129, 4, 4}, std::array{1024, 1024, 1024}}) {
        const auto a = random_matrix(m, k, 1);
        const auto b = random_matrix(k, n, 2);
        std::vector<double> want;
        std::vector<double> scale;
        reference(m, n, k, a, b, want, scale);
        const Device<float> da(a);
        const Device<float> db(b);
        const Device<float> dc(
            std::vector<float>(static_cast<std::size_t>(m) * static_cast<std::size_t>(n), -7.0F));
        REQUIRE(gpu::gemm(kernel, m, n, k, da.ptr, db.ptr, dc.ptr) == 0);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        const auto got = dc.download(want.size());
        INFO(gpu::gemm_name(kernel) << " " << m << "x" << n << "x" << k);
        // fp32 rounding grows with the number of terms; 1e-5 relative to sum |a||b| leaves
        // room for any summation order and still catches a missing or doubled term.
        CHECK(count_wrong(got, want, scale, 1e-5 * std::sqrt(static_cast<double>(k))) == 0);
    }
}

TEST_CASE("the tensor-core GEMM equals the reference on fp16-rounded inputs") {
    require_gpu();
    for (const auto& [m, n, k] :
         {std::array{16, 16, 16}, std::array{64, 64, 64}, std::array{48, 80, 32},
          std::array{144, 208, 272}, std::array{1024, 512, 2304}}) {
        auto a = random_matrix(m, k, 3);
        auto b = random_matrix(k, n, 4);
        std::vector<std::uint16_t> ah(a.size());
        std::vector<std::uint16_t> bh(b.size());
        gpu::to_half(a, ah);
        gpu::to_half(b, bh);
        gpu::from_half(ah, a); // the reference sees exactly the fp16 values the GPU does
        gpu::from_half(bh, b);
        std::vector<double> want;
        std::vector<double> scale;
        reference(m, n, k, a, b, want, scale);
        const Device<std::uint16_t> da(ah);
        const Device<std::uint16_t> db(bh);
        const Device<float> dc(std::vector<float>(want.size(), -7.0F));
        REQUIRE(gpu::gemm_tensor_core(m, n, k, da.ptr, db.ptr, dc.ptr) == 0);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        INFO(m << "x" << n << "x" << k);
        // Products of fp16 values are exact in fp32; only the fp32 accumulation rounds.
        CHECK(count_wrong(dc.download(want.size()), want, scale,
                          1e-5 * std::sqrt(static_cast<double>(k))) == 0);
    }
    // Not multiples of 16: refused, nothing launched.
    const Device<std::uint16_t> one(std::vector<std::uint16_t>(16 * 16));
    const Device<float> out(std::vector<float>(16 * 16));
    CHECK(gpu::gemm_tensor_core(15, 16, 16, one.ptr, one.ptr, out.ptr) ==
          static_cast<int>(cudaErrorInvalidValue));
}
