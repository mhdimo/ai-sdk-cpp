add_library(ai-sdk-zai
    src/zai_provider.cpp
)

target_include_directories(ai-sdk-zai PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ai-sdk-zai PUBLIC ai-sdk-core ai-sdk-anthropic ai-sdk-openai)
