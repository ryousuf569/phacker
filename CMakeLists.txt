cmake_minimum_required(VERSION 3.18...3.31)
project(${SKBUILD_PROJECT_NAME} LANGUAGES NONE)

# The kernel is optional: with no compiler the wheel ships the NumPy reference alone.
include(CheckLanguage)
check_language(CXX)
if(NOT CMAKE_CXX_COMPILER)
  message(WARNING "No C++ compiler found, building phacker without the C++ kernel")
  return()
endif()
enable_language(CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core NB_STATIC
  src/cpp/bindings.cpp
  src/cpp/block_stats.cpp
  src/cpp/cscv_kernel.cpp)

if(MSVC)
  # plain /openmp is OpenMP 2.0 and rejects `omp simd`
  target_compile_options(_core PRIVATE /O2 /openmp:experimental)
else()
  find_package(OpenMP COMPONENTS CXX)  # absent on Apple clang; the kernel then runs serially
  if(OpenMP_CXX_FOUND)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
  endif()
  target_compile_options(_core PRIVATE -O3 -fno-math-errno -Wno-unknown-pragmas)
  # never -march=native in a wheel (SIGILL on older CPUs); x86-64-v2 is the portable baseline
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(_core PRIVATE -march=x86-64-v2)
  endif()
endif()

install(TARGETS _core LIBRARY DESTINATION phacker)
