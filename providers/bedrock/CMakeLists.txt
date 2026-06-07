add_library(ai-sdk-bedrock
    src/bedrock_provider.cpp
    src/bedrock_model.cpp
    src/sigv4.cpp
)

target_include_directories(ai-sdk-bedrock PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ai-sdk-bedrock PUBLIC ai-sdk-core)
