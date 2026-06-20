add_library(ai-sdk-c SHARED
    ai_sdk.cpp
)

target_include_directories(ai-sdk-c PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ai-sdk-c PRIVATE ai-sdk-core)
# Only link providers that are actually built (some may be disabled).
foreach(pv
    ai-sdk-openai ai-sdk-anthropic ai-sdk-google ai-sdk-deepseek ai-sdk-groq
    ai-sdk-xai ai-sdk-mistral ai-sdk-fireworks ai-sdk-togetherai
    ai-sdk-perplexity ai-sdk-cohere)
    if(TARGET ${pv})
        target_link_libraries(ai-sdk-c PRIVATE ${pv})
    endif()
endforeach()

set_target_properties(ai-sdk-c PROPERTIES
    OUTPUT_NAME ai_sdk
    PUBLIC_HEADER ai_sdk.h
)

install(TARGETS ai-sdk-c
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)
