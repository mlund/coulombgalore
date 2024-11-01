cmake_minimum_required(VERSION 3.11.4 FATAL_ERROR)

project(coulombgalore)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
enable_testing()

include(cmake/CPM.cmake)

## GCC
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-Wall -Wextra -Wpedantic -Wunreachable-code -Wstrict-aliasing)
## Clang
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wunreachable-code -Wstrict-aliasing)
endif()

# dependencies 

CPMAddPackage("gh:doctest/doctest#v2.4.11")
CPMAddPackage("gh:nlohmann/json@3.11.3")

# Eigen
CPMAddPackage(NAME Eigen VERSION 3.4.0 DOWNLOAD_ONLY YES
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz)
if(Eigen_ADDED)
    add_library(Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen INTERFACE ${Eigen_SOURCE_DIR})
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(example test/example.cpp)
target_link_libraries(example Eigen nlohmann_json)

add_executable(tests test/unittests.cpp)
target_link_libraries(tests doctest Eigen nlohmann_json)

add_test(NAME tests COMMAND tests)

