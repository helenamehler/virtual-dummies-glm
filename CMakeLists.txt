# UPSTREAM, EXTENDED: taulantkoka/virtual-dummies (Koka et al., 2026), modified for GLM/Cox score-based selection (Mehler, Koka, Muma, 2026). GPLv3, see NOTICE.
cmake_minimum_required(VERSION 3.20)
project(virtual_dummies LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ==============================================================================
# Options
# ==============================================================================
option(BUILD_BENCHMARKS "Build the memory/runtime benchmarks (requires Armadillo)" OFF)

# ==============================================================================
# Python + pybind11
# ==============================================================================
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.5
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# ==============================================================================
# BLAS vendor
# ==============================================================================
add_library(cblas_vendor INTERFACE)
set(VD_HAVE_BLAS OFF)
if(APPLE)
  find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
  target_link_libraries(cblas_vendor INTERFACE ${ACCELERATE_FRAMEWORK})
  target_compile_definitions(cblas_vendor INTERFACE HAS_ACCELERATE=1)
  set(VD_HAVE_BLAS ON)
else()
  find_package(BLAS QUIET)
  if(BLAS_FOUND)
    target_link_libraries(cblas_vendor INTERFACE BLAS::BLAS)
    set(VD_HAVE_BLAS ON)
    message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
  else()
    message(WARNING
      "No BLAS found -- building WITHOUT EIGEN_USE_BLAS (slower GEMM).\n"
      "Install a BLAS (e.g. conda install -c conda-forge openblas) or set\n"
      "-DBLA_VENDOR=OpenBLAS / -DCMAKE_PREFIX_PATH=<blas-root> to enable it.")
  endif()
endif()

# ==============================================================================
# OpenMP helper
# ==============================================================================
function(enable_openmp tgt)
  if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(LIBOMP_ROOT "/opt/homebrew/opt/libomp")
    find_library(OMP_LIB omp HINTS "${LIBOMP_ROOT}/lib")
    find_path(OMP_INC omp.h HINTS "${LIBOMP_ROOT}/include")
    if(OMP_LIB AND OMP_INC)
      target_compile_options(${tgt} PRIVATE -Xpreprocessor -fopenmp)
      target_include_directories(${tgt} PRIVATE "${OMP_INC}")
      target_link_libraries(${tgt} PRIVATE "${OMP_LIB}")
    endif()
  else()
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
      target_link_libraries(${tgt} PRIVATE OpenMP::OpenMP_CXX)
    endif()
  endif()
endfunction()

# ==============================================================================
# Eigen headers (vendored or fetched)
# ==============================================================================
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/extern/eigen-5.0.0/Eigen/Dense")
  set(EIGEN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/eigen-5.0.0")
else()
  include(FetchContent)
  FetchContent_Declare(eigen_fetch
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    SOURCE_SUBDIR  ""
  )
  FetchContent_Populate(eigen_fetch)
  set(EIGEN_INCLUDE_DIR "${eigen_fetch_SOURCE_DIR}")
endif()

# ==============================================================================
# vd_selectors  (always built — the main Python extension)
# ==============================================================================
pybind11_add_module(vd_selectors
  src/bindings.cpp
  src/vd_base.cpp
  src/vd_lars.cpp
  src/vd_omp.cpp
  src/vd_afs.cpp
  src/vd_afs_logistic.cpp
  src/trex.cpp
  src/vd_afs_poisson.cpp
  src/vd_afs_multinomial.cpp
  src/vd_afs_cox.cpp
)

target_compile_features(vd_selectors PRIVATE cxx_std_17)
target_include_directories(vd_selectors PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${EIGEN_INCLUDE_DIR}
)
target_link_libraries(vd_selectors PRIVATE cblas_vendor)

if(MSVC)
  target_compile_options(vd_selectors PRIVATE /O2)
else()
  target_compile_options(vd_selectors PRIVATE -O3)
endif()

target_compile_definitions(vd_selectors PRIVATE
  EIGEN_NO_DEBUG
  NDEBUG
)
if(VD_HAVE_BLAS)
  target_compile_definitions(vd_selectors PRIVATE EIGEN_USE_BLAS)
endif()

if(APPLE)
  set_target_properties(vd_selectors PROPERTIES
    LINK_FLAGS "-undefined dynamic_lookup"
  )
endif()

enable_openmp(vd_selectors)

install(TARGETS vd_selectors LIBRARY DESTINATION vd_selectors)

# ==============================================================================
# Memory and runtime benchmarks  (opt-in, Sec. 4.2 of the paper)
#
# One executable per response family, each comparing the virtual dummy solver
# against its explicitly augmented counterpart at n = 10^4, p = 10^5, L = 5p.
# Sources live in experiments/sec4_2_memory_runtime/. The AD baselines are
# ad_afs_*.hpp, our C++ ports of the matching AD_AFS_*.py in
# experiments/sec4_1_equivalence/. Every driver shares bench_harness.hpp for
# the fork/pin/RSS measurement, and needs Eigen plus Armadillo. Run the
# executables from the repository root so results/ lands in the right place.
#
#   cmake -B build -DBUILD_BENCHMARKS=ON
#   cmake --build build --target bench_memory_poisson -j
#   ./build/bench_memory_poisson        # writes results/sec4_2_memory_runtime/poisson/
#
# Set VD_MEM_CAP_GB and the family's AD cap (ADPOIS_MEM_CAP_GB, ADLOGIT_MEM_CAP_GB,
# ADMULTI_MEM_CAP_GB, ADCOX_MEM_CAP_GB) to skip runs that would exceed the node.
# The AD side needs about 48 GB at this point.
# ==============================================================================
if(BUILD_BENCHMARKS)
  find_package(Armadillo QUIET)
  if(NOT Armadillo_FOUND)
    message(FATAL_ERROR
      "Armadillo not found (required for BUILD_BENCHMARKS).\n"
      "Install it or set -DCMAKE_PREFIX_PATH=/path/to/armadillo")
  endif()

  function(link_armadillo tgt)
    if(TARGET Armadillo::Armadillo)
      target_link_libraries(${tgt} PRIVATE Armadillo::Armadillo)
    elseif(TARGET armadillo)
      target_link_libraries(${tgt} PRIVATE armadillo)
    else()
      target_link_libraries(${tgt} PRIVATE ${ARMADILLO_LIBRARIES})
    endif()
    if(ARMADILLO_INCLUDE_DIRS)
      target_include_directories(${tgt} PRIVATE ${ARMADILLO_INCLUDE_DIRS})
    endif()
  endfunction()

  find_package(Threads REQUIRED)

  foreach(vd_family logistic poisson multinomial cox)
    set(vd_tgt "bench_memory_${vd_family}")

    add_executable(${vd_tgt}
      experiments/sec4_2_memory_runtime/bench_memory_${vd_family}.cpp
      src/vd_base.cpp
      src/vd_afs_${vd_family}.cpp
    )

    target_compile_features(${vd_tgt} PRIVATE cxx_std_17)
    target_include_directories(${vd_tgt} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR}/experiments/sec4_2_memory_runtime
      ${EIGEN_INCLUDE_DIR}
    )

    if(MSVC)
      target_compile_options(${vd_tgt} PRIVATE /O2)
    else()
      target_compile_options(${vd_tgt} PRIVATE -O3 -fstrict-aliasing)
    endif()

    target_compile_definitions(${vd_tgt} PRIVATE EIGEN_NO_DEBUG NDEBUG)
    if(VD_HAVE_BLAS)
      target_compile_definitions(${vd_tgt} PRIVATE EIGEN_USE_BLAS)
    endif()

    target_link_libraries(${vd_tgt} PRIVATE cblas_vendor Threads::Threads)
    link_armadillo(${vd_tgt})
  endforeach()
endif()
