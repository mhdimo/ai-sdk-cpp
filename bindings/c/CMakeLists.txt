add_library(ai-sdk-c SHARED
    ai_sdk.cpp
)

target_include_directories(ai-sdk-c PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ai-sdk-c PRIVATE ai-sdk-core)
# Only link providers that are actually built (some may be disabled).
foreach(pv
    ai-sdk-openai ai-sdk-anthropic ai-sdk-google ai-sdk-deepseek ai-sdk-zai)
    if(TARGET ${pv})
        target_link_libraries(ai-sdk-c PRIVATE ${pv})
    endif()
endforeach()

set_target_properties(ai-sdk-c PROPERTIES
    OUTPUT_NAME ai_sdk
    PUBLIC_HEADER ai_sdk.h
)

# See AI_SDK_STATIC_CXX_RUNTIME in the root CMakeLists for why. Kept here rather
# than as a global linker flag so it applies to this one target: the tests and
# the CLI link the same code and have no reason to carry a second copy of the
# runtime.
if(AI_SDK_STATIC_CXX_RUNTIME)
    # CMAKE_CXX_COMPILER_ID, spelled in full. The bare name is not a CMake
    # variable at all, so it is the empty string, the comparison is false on
    # every compiler, and this silently takes the branch below -- leaving the
    # runtime dynamically linked while the build reports the option as ON.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_link_options(ai-sdk-c PRIVATE -static-libstdc++ -static-libgcc)
    else()
        message(WARNING
            "AI_SDK_STATIC_CXX_RUNTIME is ON but the compiler is "
            "${CMAKE_CXX_COMPILER_ID}, not GNU, so -static-libstdc++ has nothing "
            "to act on and will be skipped. The shared library will depend on "
            "the runtime installed on the machine that loads it.")
    endif()
endif()

install(TARGETS ai-sdk-c
    EXPORT AiSdkTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
