add_executable(ai-sdk-tests
    test_json_schema.cpp
    test_sse_parser.cpp
    test_validator.cpp
    test_messages.cpp
    test_cancellation.cpp
    test_streaming.cpp
    test_batch.cpp
    test_session.cpp
    test_permission.cpp
    test_standard_toolkit.cpp
    test_session_store.cpp
    test_memory.cpp
    test_coding_agent.cpp
    test_retry.cpp
    test_rate_limiter.cpp
    test_mcp.cpp
    test_embeddings.cpp
    test_rerank.cpp
    test_middleware.cpp
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

if(AI_SDK_PROVIDER_OPENAI AND AI_SDK_PROVIDER_GOOGLE)
    target_sources(ai-sdk-tests PRIVATE test_provider_http.cpp)
    target_link_libraries(ai-sdk-tests PRIVATE ai-sdk-openai ai-sdk-google)
endif()

include(CTest)
include(Catch)
catch_discover_tests(ai-sdk-tests)

# Live (network) smoke tests: a separate executable that is NOT registered with
# ctest, so the default suite stays fully offline. Run manually:
#   AI_SDK_RUN_LIVE=1 ANTHROPIC_API_KEY=... ./scripts/run_smoke.sh
if(AI_SDK_PROVIDER_ANTHROPIC)
    add_executable(ai-sdk-live-tests test_live_smoke.cpp)
    target_link_libraries(ai-sdk-live-tests PRIVATE
        ai-sdk-core ai-sdk-anthropic Catch2::Catch2WithMain)
endif()
