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

CPMAddPackage("gh:onqtam/doctest#2.4.6")

# Eigen
CPMAddPackage(NAME Eigen VERSION 3.4.0 DOWNLOAD_ONLY YES
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz)
if(Eigen_ADDED)
    add_library(Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen INTERFACE ${Eigen_SOURCE_DIR})
endif()

# modern json
CPMAddPackage(
    NAME nlohmann_json VERSION 3.10.5 OPTIONS "JSON_BuildTests OFF"
    URL https://github.com/nlohmann/json/releases/download/v3.10.5/include.zip)
if (nlohmann_json_ADDED)
    add_library(nlohmann_json INTERFACE IMPORTED)
    target_include_directories(nlohmann_json INTERFACE ${nlohmann_json_SOURCE_DIR}/include)
endif()

include_directories(${CMAKE_SOURCE_DIR})

add_executable(example test/example.cpp)
target_link_libraries(example Eigen nlohmann_json)

add_executable(unittests test/unittests.cpp)
target_link_libraries(unittests doctest Eigen nlohmann_json)

add_test(NAME unittests COMMAND unittests)
