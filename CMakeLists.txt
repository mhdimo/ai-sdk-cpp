cmake_minimum_required(VERSION 3.20)
project(ai-sdk-cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# GCC 11 and 12 cannot build a working library from this source, and the way
# they fail is worth stating precisely, because it is not the way one expects.
#
# They first refuse outright: the assembler rejects the operator() of a
# coroutine lambda declared inside a function with C language linkage as a
# duplicate symbol. That failure is loud and is what bindings/c/ai_sdk.cpp was
# rearranged to avoid. Past it, they miscompile the coroutines badly enough
# that 30 of the 158 unit tests crash -- streaming, sessions, tool execution,
# the C binding's permission gates. Measured on ubuntu:22.04 and ubuntu:24.04,
# at both -O0 and -O3, against Boost 1.83: GCC 11.4 and GCC 12.3 each fail the
# same 30, and only the compiler differs between a passing and a failing run.
# GCC 13 and Clang pass all 158.
#
# So the honest outcome on GCC 11/12 is a library that builds cleanly and then
# segfaults, which is a much worse thing to hand someone than a build error.
# Refuse by name instead. These versions are still the default on Ubuntu 22.04,
# RHEL 9 and Debian 12, so this is not a rare configuration to arrive in --
# Clang is available on all three and builds this fine.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13
        AND NOT AI_SDK_ALLOW_UNSUPPORTED_COMPILER)
    message(FATAL_ERROR
        "GCC ${CMAKE_CXX_COMPILER_VERSION} cannot build this project. Its coroutine "
        "support is broken for the pattern the library relies on: the tests fail by "
        "crashing, and a build that succeeds would still be a library that segfaults "
        "at runtime.\n"
        "Use GCC 13 or newer, or Clang.\n"
        "To build anyway and accept the breakage -- for example to reproduce it -- "
        "configure with -DAI_SDK_ALLOW_UNSUPPORTED_COMPILER=ON.")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Build all targets position-independent so the static core/providers can be
# linked into the shared C binding (libai_sdk.so). Required on Linux; harmless
# elsewhere. Without this, TLS relocations (e.g. boost::asio) fail when linking
# a non-PIC static archive into a shared object.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(AI_SDK_BUILD_SHARED "Build shared library" ON)
option(AI_SDK_BUILD_STATIC "Build static library" ON)
option(AI_SDK_BUILD_TESTS "Build tests" ON)
option(AI_SDK_BUILD_EXAMPLES "Build examples" ON)
option(AI_SDK_BUILD_CLI "Build CLI agent tool" ON)
option(AI_SDK_BUILD_BINDINGS "Build C/Python/Node bindings" ON)
option(AI_SDK_BUILD_PYTHON "Build Python pybind11 bindings" OFF)
option(AI_SDK_PROVIDER_OPENAI "Build OpenAI provider" ON)
option(AI_SDK_PROVIDER_ANTHROPIC "Build Anthropic provider" ON)
option(AI_SDK_PROVIDER_GOOGLE "Build Google provider" ON)
option(AI_SDK_PROVIDER_DEEPSEEK "Build DeepSeek provider" ON)
option(AI_SDK_PROVIDER_MOONSHOTAI "Build MoonshotAI provider" ON)
option(AI_SDK_PROVIDER_ZAI "Build z.ai (GLM) provider" ON)
option(AI_SDK_PROVIDER_BEDROCK "Build Amazon Bedrock provider" ON)
option(AI_SDK_PROVIDER_OPENAI_COMPATIBLE "Build OpenAI-Compatible provider" ON)
option(AI_SDK_COVERAGE "Instrument for Clang source-based coverage (see scripts/coverage.sh)" OFF)
# Escape hatch for the compiler gate near the top of this file. Off by default
# because the thing it permits is a library that crashes; see the comment there.
option(AI_SDK_ALLOW_UNSUPPORTED_COMPILER "Build on a compiler measured to produce a broken library (GCC < 13)" OFF)

# Whether the shared C binding carries its own copy of the C++ runtime.
#
# This is the artifact that ships, and on Linux it has to start on distributions
# older than the one that built it. libstdc++ versions its symbols, so the usual
# shape -- a build image years ahead of the target -- produces a library that
# refuses to load, on the user's machine, which is the one place the check
# cannot run. Linking the runtime in removes the dependency: the result needs
# only libc, libm and the ELF loader.
#
# Default on for GNU, and off otherwise for a reason rather than by omission.
# This project configures Clang on Linux with -stdlib=libc++, and there is no
# -static-libc++ to match -static-libstdc++; defaulting it on there would give
# an option that reports itself enabled while doing nothing at all, which is the
# exact class of defect it exists to prevent. Turn it on with Clang and the
# build says so instead.
if(UNIX AND NOT APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    option(AI_SDK_STATIC_CXX_RUNTIME "Link the C++ runtime into the shared C binding" ON)
else()
    option(AI_SDK_STATIC_CXX_RUNTIME "Link the C++ runtime into the shared C binding" OFF)
endif()

# The version the runtime reports (ai_sdk_version(), the MCP clientInfo block).
# Sourced from the project version so it cannot drift from the release number
# the manifests carry. See include/ai/version.hpp.
add_compile_definitions(AI_SDK_VERSION="${PROJECT_VERSION}")

# Coverage instrumentation. These flags only make the build *emit* .profraw
# files; turning those into a report needs llvm-profdata + llvm-cov, which is
# what scripts/coverage.sh drives. Clang-specific: the profile format is read
# back by llvm-cov, which will not read GCC's gcov output.
if(AI_SDK_COVERAGE)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR
            "AI_SDK_COVERAGE requires Clang (found ${CMAKE_CXX_COMPILER_ID}). "
            "llvm-cov reads the profile Clang emits; GCC would need gcov instead.")
    endif()
    add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
    # The profiling runtime has to be linked in too, or nothing writes a profile.
    add_link_options(-fprofile-instr-generate)
endif()

# Expose provider enablement to the preprocessor so the CLI / C binding can
# conditionally compile provider code (they include all providers by default).
foreach(pv IN ITEMS
    ANTHROPIC OPENAI GOOGLE DEEPSEEK MOONSHOTAI ZAI BEDROCK OPENAI_COMPATIBLE
    XAI MISTRAL GROQ FIREWORKS TOGETHERAI PERPLEXITY COHERE)
    if(AI_SDK_PROVIDER_${pv})
        add_compile_definitions(AI_SDK_PROVIDER_${pv})
    endif()
endforeach()

find_package(Boost 1.82 REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)

if(AI_SDK_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.0
    )
    FetchContent_MakeAvailable(catch2)
    enable_testing()
endif()

set(AI_SDK_CORE_SOURCES
    src/error/ai_error.cpp
    src/error/api_call_error.cpp
    src/util/json.cpp
    src/schema/json_schema.cpp
    src/schema/validator.cpp
    src/stream/sse_parser.cpp
    src/http/client.cpp
    src/http/connection_pool.cpp
    src/http/multipart.cpp
    src/http/websocket.cpp
    src/core/generate_text.cpp
    src/core/stream_text.cpp
    src/core/generate_object.cpp
    src/core/embed.cpp
    src/core/batch.cpp
    src/agent/tool_loop_agent.cpp
    src/agent/agent.cpp
    src/session/session.cpp
    src/session/context_strategy.cpp
    src/session/store.cpp
    src/permission/permission.cpp
    src/tool/tool_set.cpp
    src/tools/standard/standard_toolkit.cpp
    src/memory/memory.cpp
    src/coding_agent.cpp
    src/model/middleware.cpp
    src/mcp/mcp_client.cpp
    src/mcp/transport.cpp
    src/telemetry/telemetry.cpp
    src/util/token_count.cpp
    src/util/rate_limiter.cpp
    src/stream/data_stream.cpp
    src/core/generate_image.cpp
    src/core/generate_speech.cpp
    src/core/transcribe.cpp
    src/core/rerank.cpp
    src/core/boost_json_src.cpp
)

if(AI_SDK_BUILD_STATIC)
    add_library(ai-sdk-core-static STATIC ${AI_SDK_CORE_SOURCES})
    set_target_properties(ai-sdk-core-static PROPERTIES OUTPUT_NAME ai-sdk-core)
    target_include_directories(ai-sdk-core-static PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(ai-sdk-core-static PUBLIC
        Boost::headers
        OpenSSL::SSL OpenSSL::Crypto
        Threads::Threads
    )
    target_compile_options(ai-sdk-core-static PUBLIC
        $<$<CXX_COMPILER_ID:GNU>:-fcoroutines>
        $<$<CXX_COMPILER_ID:Clang>:-stdlib=libc++>
    )
endif()

if(AI_SDK_BUILD_SHARED)
    add_library(ai-sdk-core-shared SHARED ${AI_SDK_CORE_SOURCES})
    set_target_properties(ai-sdk-core-shared PROPERTIES OUTPUT_NAME ai-sdk-core)
    target_include_directories(ai-sdk-core-shared PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(ai-sdk-core-shared PUBLIC
        Boost::headers
        OpenSSL::SSL OpenSSL::Crypto
        Threads::Threads
    )
    target_compile_options(ai-sdk-core-shared PUBLIC
        $<$<CXX_COMPILER_ID:GNU>:-fcoroutines>
        $<$<CXX_COMPILER_ID:Clang>:-stdlib=libc++>
    )
    target_compile_definitions(ai-sdk-core-shared PRIVATE AI_SDK_EXPORTS)
endif()

if(AI_SDK_BUILD_STATIC)
    add_library(ai-sdk-core ALIAS ai-sdk-core-static)
elseif(AI_SDK_BUILD_SHARED)
    add_library(ai-sdk-core ALIAS ai-sdk-core-shared)
else()
    message(FATAL_ERROR "At least one of AI_SDK_BUILD_STATIC or AI_SDK_BUILD_SHARED must be enabled")
endif()

if(AI_SDK_PROVIDER_ANTHROPIC)
    add_subdirectory(providers/anthropic)
endif()
if(AI_SDK_PROVIDER_OPENAI)
    add_subdirectory(providers/openai)
endif()
if(AI_SDK_PROVIDER_GOOGLE)
    add_subdirectory(providers/google)
endif()
if(AI_SDK_PROVIDER_DEEPSEEK)
    add_subdirectory(providers/deepseek)
endif()
if(AI_SDK_PROVIDER_MOONSHOTAI)
    add_subdirectory(providers/moonshotai)
endif()
if(AI_SDK_PROVIDER_ZAI)
    add_subdirectory(providers/zai)
endif()
if(AI_SDK_PROVIDER_BEDROCK)
    add_subdirectory(providers/bedrock)
endif()
if(AI_SDK_PROVIDER_OPENAI_COMPATIBLE)
    add_subdirectory(providers/openai-compatible)
endif()

if(AI_SDK_BUILD_CLI)
    add_subdirectory(cli)
endif()

if(AI_SDK_BUILD_BINDINGS)
    add_subdirectory(bindings/c)
endif()

if(AI_SDK_BUILD_PYTHON)
    add_subdirectory(bindings/python)
endif()

if(AI_SDK_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(AI_SDK_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

include(GNUInstallDirs)
if(AI_SDK_BUILD_STATIC)
    install(TARGETS ai-sdk-core-static EXPORT AiSdkTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()
if(AI_SDK_BUILD_SHARED)
    install(TARGETS ai-sdk-core-shared EXPORT AiSdkTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()
install(DIRECTORY include/ai DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# --- find_package(ai-sdk-cpp) support -------------------------------------
# Export every provider that was actually built (into the AiSdkTargets export
# set, namespace ai-sdk-cpp::) and install its headers into the merged
# include/ai tree so consumers can use the full C++ API after find_package.
include(CMakePackageConfigHelpers)
set(_ai_sdk_provider_libs
    ai-sdk-openai ai-sdk-anthropic ai-sdk-google ai-sdk-deepseek
    ai-sdk-moonshotai ai-sdk-bedrock ai-sdk-zai ai-sdk-openai-compatible)
foreach(_lib IN LISTS _ai_sdk_provider_libs)
    if(TARGET ${_lib})
        install(TARGETS ${_lib}
            EXPORT AiSdkTargets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
        string(REPLACE "ai-sdk-" "" _prov ${_lib})
        install(DIRECTORY providers/${_prov}/include/ai
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    endif()
endforeach()

install(EXPORT AiSdkTargets
    NAMESPACE ai-sdk-cpp::
    FILE ai-sdk-cppTargets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ai-sdk-cpp)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ai-sdk-cppConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ai-sdk-cppConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ai-sdk-cpp)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ai-sdk-cppConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ai-sdk-cppConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ai-sdk-cppConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ai-sdk-cpp)
