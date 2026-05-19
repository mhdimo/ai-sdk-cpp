add_executable(ai-sdk-tests
    test_json_schema.cpp
    test_sse_parser.cpp
    test_validator.cpp
    test_messages.cpp
    test_cancellation.cpp
)

target_link_libraries(ai-sdk-tests PRIVATE
    ai-sdk-core
    Catch2::Catch2WithMain
)

include(CTest)
include(Catch)
catch_discover_tests(ai-sdk-tests)
