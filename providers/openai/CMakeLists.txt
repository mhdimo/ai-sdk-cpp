add_library(ai-sdk-openai
    src/openai_model.cpp
    src/openai_embedding.cpp
    src/openai_responses_model.cpp
    src/openai_batch.cpp
    src/openai_realtime.cpp
    src/convert_prompt.cpp
    src/parse_response.cpp
)

target_include_directories(ai-sdk-openai PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ai-sdk-openai PUBLIC ai-sdk-core)
