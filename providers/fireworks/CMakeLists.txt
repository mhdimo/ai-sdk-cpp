add_library(ai-sdk-fireworks
    src/fireworks_provider.cpp
)

target_include_directories(ai-sdk-fireworks PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ai-sdk-fireworks PUBLIC ai-sdk-core ai-sdk-openai)
