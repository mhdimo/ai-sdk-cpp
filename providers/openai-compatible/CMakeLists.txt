add_library(ai-sdk-openai-compatible
    src/openai_compatible_provider.cpp
)

target_include_directories(ai-sdk-openai-compatible PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ai-sdk-openai-compatible PUBLIC ai-sdk-core ai-sdk-openai)
