add_library(ai-sdk-google
    src/google_provider.cpp
    src/google_model.cpp
)

target_include_directories(ai-sdk-google PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ai-sdk-google PUBLIC ai-sdk-core)
