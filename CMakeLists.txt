# Based in https://github.com/TheLartians/ModernCppStarter/blob/master/CMakeLists.txt

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(coulombgalore VERSION 0.1.2 LANGUAGES CXX)

# ---- Include guards ----

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
    message(
        FATAL_ERROR
        "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
    )
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
enable_testing()

include(cmake/CPM.cmake)

# dependencies 
CPMAddPackage("gh:doctest/doctest#v2.4.12")
CPMAddPackage("gh:nlohmann/json@3.12.0")
CPMAddPackage("gh:TheLartians/PackageProject.cmake@1.12.0")

CPMAddPackage(NAME Eigen VERSION 3.4.0 DOWNLOAD_ONLY YES
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz)
if(Eigen_ADDED)
    add_library(Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen INTERFACE ${Eigen_SOURCE_DIR})
endif()

file(GLOB_RECURSE headers CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")

add_library(${PROJECT_NAME} INTERFACE ${headers})
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)
target_link_libraries(${PROJECT_NAME} INTERFACE Eigen)

target_include_directories(
    ${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}-${PROJECT_VERSION}>
)

# Compile flags

# being a cross-platform target, we enforce standards conformance on MSVC
target_compile_options(${PROJECT_NAME} INTERFACE "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive->")

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-Wall -Wextra -Wpedantic -Wunreachable-code -Wstrict-aliasing)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wunreachable-code -Wstrict-aliasing)
endif()

# the location where the project's version header will be placed should match the project's regular
# header paths
string(TOLOWER ${PROJECT_NAME}/version.h VERSION_HEADER_LOCATION)

packageProject(
    NAME ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    NAMESPACE ${PROJECT_NAME}
    BINARY_DIR ${PROJECT_BINARY_DIR}
    INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include
    INCLUDE_DESTINATION include/${PROJECT_NAME}-${PROJECT_VERSION}
    VERSION_HEADER "${VERSION_HEADER_LOCATION}"
    DEPENDENCIES "eigen 3"
)

# example
add_executable(example EXCLUDE_FROM_ALL test/example.cpp)
target_link_libraries(example PRIVATE ${PROJECT_NAME} nlohmann_json)
target_include_directories(example INTERFACE ${Eigen_SOURCE_DIR})

# unittests
add_executable(tests test/unittests.cpp)
target_link_libraries(tests PRIVATE ${PROJECT_NAME} doctest nlohmann_json)
target_include_directories(tests INTERFACE ${Eigen_SOURCE_DIR})
add_test(NAME tests COMMAND tests)

