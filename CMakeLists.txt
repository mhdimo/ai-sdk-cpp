cmake_minimum_required(VERSION 3.20)
project(ai-sdk-cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

# Expose provider enablement to the preprocessor so the CLI / C binding can
# conditionally compile provider code (they include all providers by default).
foreach(pv IN ITEMS
    ANTHROPIC OPENAI GOOGLE DEEPSEEK MOONSHOTAI ZAI BEDROCK OPENAI_COMPATIBLE
    XAI MISTRAL GROQ FIREWORKS TOGETHERAI PERPLEXITY COHERE)
    if(AI_SDK_PROVIDER_${pv})
        add_compile_definitions(AI_SDK_PROVIDER_${pv})
    endif()
endforeach()

find_package(Boost 1.82 REQUIRED COMPONENTS json)
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
)

if(AI_SDK_BUILD_STATIC)
    add_library(ai-sdk-core-static STATIC ${AI_SDK_CORE_SOURCES})
    set_target_properties(ai-sdk-core-static PROPERTIES OUTPUT_NAME ai-sdk-core)
    target_include_directories(ai-sdk-core-static PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(ai-sdk-core-static PUBLIC
        Boost::json
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
        Boost::json
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

add_library(ai-sdk-core ALIAS ai-sdk-core-static)

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
