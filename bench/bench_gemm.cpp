// The GEMM ladder against cuBLAS (the yardstick, linked into this benchmark only, never into
// the library: ADR-0008).
//
//   build/cuda/bench/bench-gemm --benchmark_repetitions=3 --benchmark_report_aggregates_only=true
//
// Square C = A * B, fp32 (and fp16 in / fp32 out for the tensor-core rung). Kernel time only,
// from CUDA events around a run of launches (inputs stay on the GPU); GFLOPS = 2 n^3 / time.
#include "waferedge/backend.hpp"
#include "waferedge/gemm.hpp"
#include "waferedge/machine.hpp"

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <format>
#include <random>
#include <vector>

namespace {

using namespace waferedge;

struct Buffers {
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    std::uint16_t* ah = nullptr;
    std::uint16_t* bh = nullptr;
    explicit Buffers(int n) {
        const auto count = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
        std::vector<float> host(count);
        std::mt19937 rng(1);
        for (auto& x : host) {
            x = static_cast<float>(rng() % 2001) / 1000.0F - 1.0F;
        }
        std::vector<std::uint16_t> half(count);
        gpu::to_half(host, half);
        cudaMalloc(reinterpret_cast<void**>(&a), count * sizeof(float));
        cudaMalloc(reinterpret_cast<void**>(&b), count * sizeof(float));
        cudaMalloc(reinterpret_cast<void**>(&c), count * sizeof(float));
        cudaMalloc(reinterpret_cast<void**>(&ah), count * 2);
        cudaMalloc(reinterpret_cast<void**>(&bh), count * 2);
        cudaMemcpy(a, host.data(), count * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(b, host.data(), count * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(ah, half.data(), count * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(bh, half.data(), count * 2, cudaMemcpyHostToDevice);
    }
    ~Buffers() {
        for (void* p : {static_cast<void*>(a), static_cast<void*>(b), static_cast<void*>(c),
                        static_cast<void*>(ah), static_cast<void*>(bh)}) {
            cudaFree(p);
        }
    }
    Buffers(const Buffers&) = delete;
    Buffers& operator=(const Buffers&) = delete;
};

// Times `launch` on the GPU: a few launches per iteration between two events.
template <typename Launch>
void time_gpu(benchmark::State& state, int n, Launch&& launch) {
    if (!backend::Cuda::available()) {
        state.SkipWithError("no CUDA device");
        return;
    }
    const int reps = n <= 1024 ? 20 : (n <= 2048 ? 5 : 2);
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    if (launch() != 0) { // warm-up, and fail fast
        state.SkipWithError("launch failed");
        return;
    }
    cudaDeviceSynchronize();
    for (auto _ : state) {
        cudaEventRecord(start);
        for (int r = 0; r < reps; ++r) {
            launch();
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(static_cast<double>(ms) / 1e3 / reps);
    }
    const double flops = 2.0 * n * static_cast<double>(n) * n;
    state.counters["GFLOPS"] =
        benchmark::Counter(flops / 1e9, benchmark::Counter::kIsIterationInvariantRate);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void BM_ladder(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const auto kernel = static_cast<gpu::GemmKernel>(state.range(1));
    state.SetLabel(std::string(gpu::gemm_name(kernel)));
    const Buffers buf(n);
    time_gpu(state, n, [&] { return gpu::gemm(kernel, n, n, n, buf.a, buf.b, buf.c); });
}

void BM_tensor_core(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const Buffers buf(n);
    time_gpu(state, n, [&] { return gpu::gemm_tensor_core(n, n, n, buf.ah, buf.bh, buf.c); });
}

// Row-major C = A * B is column-major C^T = B^T * A^T: pass B first.
void BM_cublas_sgemm(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const Buffers buf(n);
    cublasHandle_t handle = nullptr;
    cublasCreate(&handle);
    const float one = 1.0F;
    const float zero = 0.0F;
    time_gpu(state, n, [&] {
        return static_cast<int>(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, buf.b,
                                            n, buf.a, n, &zero, buf.c, n));
    });
    cublasDestroy(handle);
}

void BM_cublas_hgemm_tensor(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const Buffers buf(n);
    cublasHandle_t handle = nullptr;
    cublasCreate(&handle);
    const float one = 1.0F;
    const float zero = 0.0F;
    time_gpu(state, n, [&] {
        return static_cast<int>(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one,
                                             buf.bh, CUDA_R_16F, n, buf.ah, CUDA_R_16F, n, &zero,
                                             buf.c, CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
                                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    });
    cublasDestroy(handle);
}

// The fair yardstick for the tensor-core rung: cuBLASLt's heuristic proposes up to 16
// algorithms for this exact problem (fp16 in, fp32 accumulate and out, 32 MB of workspace);
// each is timed and the fastest is benchmarked. cublasGemmEx above uses the default choice.
void BM_cublaslt_best_tensor(benchmark::State& state) {
    if (!backend::Cuda::available()) {
        state.SkipWithError("no CUDA device");
        return;
    }
    const int n = static_cast<int>(state.range(0));
    const auto un = static_cast<std::uint64_t>(n);
    const Buffers buf(n);
    cublasLtHandle_t lt = nullptr;
    cublasLtCreate(&lt);
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatrixLayout_t in_layout = nullptr;
    cublasLtMatrixLayout_t out_layout = nullptr;
    cublasLtMatrixLayoutCreate(&in_layout, CUDA_R_16F, un, un, n); // square: one layout for A, B
    cublasLtMatrixLayoutCreate(&out_layout, CUDA_R_32F, un, un, n);
    std::size_t workspace_size = std::size_t{32} << 20;
    void* workspace = nullptr;
    cudaMalloc(&workspace, workspace_size);
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &workspace_size, sizeof workspace_size);
    std::array<cublasLtMatmulHeuristicResult_t, 16> found{};
    int returned = 0;
    cublasLtMatmulAlgoGetHeuristic(lt, op, in_layout, in_layout, out_layout, out_layout, pref,
                                   static_cast<int>(found.size()), found.data(), &returned);
    const float one = 1.0F;
    const float zero = 0.0F;
    // Row-major C = A * B as column-major C^T = B^T * A^T: B first.
    auto run = [&](const cublasLtMatmulAlgo_t& algo) {
        return static_cast<int>(cublasLtMatmul(lt, op, &one, buf.bh, in_layout, buf.ah, in_layout,
                                               &zero, buf.c, out_layout, buf.c, out_layout, &algo,
                                               workspace, workspace_size, nullptr));
    };
    int best = -1;
    float best_ms = 1e30F;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < returned; ++i) {
        if (run(found[static_cast<std::size_t>(i)].algo) != 0) {
            continue;
        }
        cudaEventRecord(start);
        for (int r = 0; r < 3; ++r) {
            run(found[static_cast<std::size_t>(i)].algo);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        if (ms < best_ms) {
            best_ms = ms;
            best = i;
        }
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (best < 0) {
        state.SkipWithError("no cuBLASLt algorithm ran");
    } else {
        state.SetLabel(std::format("best of {} algorithms", returned));
        time_gpu(state, n, [&] { return run(found[static_cast<std::size_t>(best)].algo); });
    }
    cudaFree(workspace);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(in_layout);
    cublasLtMatrixLayoutDestroy(out_layout);
    cublasLtMatmulDescDestroy(op);
    cublasLtDestroy(lt);
}

void sizes_and_rungs(benchmark::internal::Benchmark* b) {
    for (const std::int64_t n : {1024, 2048, 4096}) {
        for (const auto k : {gpu::GemmKernel::naive, gpu::GemmKernel::tiled,
                             gpu::GemmKernel::register_blocked, gpu::GemmKernel::vectorized}) {
            b->Args({n, static_cast<std::int64_t>(k)});
        }
    }
}

BENCHMARK(BM_ladder)->Apply(sizes_and_rungs)->ArgNames({"n", "rung"})->UseManualTime();
BENCHMARK(BM_cublas_sgemm)->Arg(1024)->Arg(2048)->Arg(4096)->UseManualTime();
BENCHMARK(BM_tensor_core)->Arg(1024)->Arg(2048)->Arg(4096)->UseManualTime();
BENCHMARK(BM_cublas_hgemm_tensor)->Arg(1024)->Arg(2048)->Arg(4096)->UseManualTime();
BENCHMARK(BM_cublaslt_best_tensor)->Arg(1024)->Arg(2048)->Arg(4096)->UseManualTime();

} // namespace

int main(int argc, char** argv) {
    benchmark::AddCustomContext("machine", describe_machine());
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
