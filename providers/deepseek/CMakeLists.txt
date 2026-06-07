add_library(ai-sdk-deepseek
    src/deepseek_provider.cpp
)

target_include_directories(ai-sdk-deepseek PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ai-sdk-deepseek PUBLIC ai-sdk-core ai-sdk-openai)
