add_executable(ai-sdk-tests
    test_json_schema.cpp
    test_sse_parser.cpp
    test_validator.cpp
    test_messages.cpp
    test_cancellation.cpp
    test_streaming.cpp
    test_batch.cpp
)

target_link_libraries(ai-sdk-tests PRIVATE
    ai-sdk-core
    Catch2::Catch2WithMain
)

# Provider-specific tests are only compiled when their provider is enabled.
if(AI_SDK_PROVIDER_ANTHROPIC)
    target_sources(ai-sdk-tests PRIVATE test_anthropic_structured.cpp)
    target_link_libraries(ai-sdk-tests PRIVATE ai-sdk-anthropic)
endif()

include(CTest)
include(Catch)
catch_discover_tests(ai-sdk-tests)
