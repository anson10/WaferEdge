# The Python module (phase 5): waferedge._core, built by scikit-build-core from pyproject.toml.
pybind11_add_module(_core MODULE bindings.cpp)
# Not waferedge_options: pybind11's headers don't build warning-free under -Wconversion.
target_link_libraries(_core PRIVATE waferedge::waferedge)
target_compile_definitions(_core PRIVATE WAFEREDGE_PY_VERSION="${PROJECT_VERSION}")
install(TARGETS _core LIBRARY DESTINATION waferedge)
