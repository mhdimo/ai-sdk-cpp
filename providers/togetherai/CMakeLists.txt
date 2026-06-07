add_library(ai-sdk-togetherai
    src/togetherai_provider.cpp
)

target_include_directories(ai-sdk-togetherai PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ai-sdk-togetherai PUBLIC ai-sdk-core ai-sdk-openai)
