add_library(ai-sdk-mistral
    src/mistral_provider.cpp
)

target_include_directories(ai-sdk-mistral PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ai-sdk-mistral PUBLIC ai-sdk-core ai-sdk-openai)
