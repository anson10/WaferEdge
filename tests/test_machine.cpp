#include "waferedge/machine.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace waferedge;

TEST_CASE("build info names the version, compiler and standard") {
    const auto build = build_info();
    CHECK(build.version == "0.1.0");
    CHECK_FALSE(build.compiler.empty());
    CHECK_FALSE(build.build_type.empty());
    // GCC 13 reports 202100L for -std=c++23, later compilers 202302L.
    CHECK(build.cxx_standard > 202002L);
}

TEST_CASE("cpu info reports at least one hardware thread") {
    CHECK(cpu_info().hardware_threads >= 1);
}

TEST_CASE("gpu info fails with a reason when there is no GPU path") {
    const auto gpu = gpu_info();
    if (!build_info().cuda_enabled) {
        REQUIRE_FALSE(gpu.has_value());
        CHECK(gpu.error().find("without CUDA") != std::string::npos);
    }
    CHECK_FALSE(gpu_info(-1).has_value());
}

TEST_CASE("describe_machine has a line for the build, the CPU and the GPU") {
    const auto text = describe_machine();
    CHECK(text.starts_with("waferedge: "));
    CHECK(text.find("\ncpu: ") != std::string::npos);
    CHECK(text.find("\ngpu: ") != std::string::npos);
    CHECK(text.ends_with('\n'));
}
