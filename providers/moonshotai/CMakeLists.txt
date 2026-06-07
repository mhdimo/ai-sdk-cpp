add_library(ai-sdk-moonshotai
    src/moonshotai_provider.cpp
)

target_include_directories(ai-sdk-moonshotai PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ai-sdk-moonshotai PUBLIC ai-sdk-core ai-sdk-openai)
