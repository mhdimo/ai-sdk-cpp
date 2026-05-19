add_library(ai-sdk-xai
    src/xai_provider.cpp
)

target_include_directories(ai-sdk-xai PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ai-sdk-xai PUBLIC ai-sdk-core ai-sdk-openai)
