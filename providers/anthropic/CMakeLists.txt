add_library(ai-sdk-anthropic
    src/anthropic_provider.cpp
    src/anthropic_model.cpp
    src/anthropic_file_storage.cpp
    src/anthropic_batch.cpp
    src/convert_prompt.cpp
    src/parse_response.cpp
)

target_include_directories(ai-sdk-anthropic PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ai-sdk-anthropic PUBLIC ai-sdk-core)
