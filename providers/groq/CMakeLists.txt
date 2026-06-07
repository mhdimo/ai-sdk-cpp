add_library(ai-sdk-groq
    src/groq_provider.cpp
)

target_include_directories(ai-sdk-groq PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ai-sdk-groq PUBLIC ai-sdk-core ai-sdk-openai)
