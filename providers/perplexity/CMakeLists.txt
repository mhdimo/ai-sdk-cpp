add_library(ai-sdk-perplexity
    src/perplexity_provider.cpp
)

target_include_directories(ai-sdk-perplexity PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ai-sdk-perplexity PUBLIC ai-sdk-core ai-sdk-openai)
