cmake_minimum_required(VERSION 3.11.4 FATAL_ERROR)

project(coulombgalore)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
enable_testing()

include(FetchContent)

## GCC
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-Wall -Wextra -Wpedantic -Wunreachable-code -Wstrict-aliasing)
## Clang
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wunreachable-code -Wstrict-aliasing)
endif()

# download modern json
FetchContent_Declare(
    modernjson
    URL "https://github.com/nlohmann/json/releases/download/v3.5.0/include.zip"
    URL_HASH SHA256=3564da9c5b0cf2e032f97c69baedf10ddbc98030c337d0327a215ea72259ea21)
FetchContent_GetProperties(modernjson)
if(NOT modernjson_POPULATED)
    FetchContent_Populate(modernjson)
endif()

# download Eigen
FetchContent_Declare(
    eigen
    URL "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz"
    URL_HASH MD5=9e30f67e8531477de4117506fe44669b)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
endif()
 
# download doctest
FetchContent_Declare(
    doctest
    URL "https://github.com/onqtam/doctest/archive/2.2.0.tar.gz"
    URL_MD5 3f91262877b792e545f971e54afb7602)
FetchContent_GetProperties(doctest)
if(NOT doctest_POPULATED)
    FetchContent_Populate(doctest)
    add_definitions(-DDOCTEST_CONFIG_DISABLE)
endif()

include_directories(SYSTEM ${eigen_SOURCE_DIR} ${doctest_SOURCE_DIR} ${modernjson_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR})

add_executable(example test/example.cpp)
add_executable(unittests test/unittests.cpp)
add_test(NAME unittests COMMAND unittests)
