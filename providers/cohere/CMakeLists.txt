add_library(ai-sdk-cohere
    src/cohere_provider.cpp
)

target_include_directories(ai-sdk-cohere PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ai-sdk-cohere PUBLIC ai-sdk-core ai-sdk-openai)
