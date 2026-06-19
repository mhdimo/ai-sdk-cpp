add_executable(example-basic-generate basic_generate/main.cpp)
target_link_libraries(example-basic-generate PRIVATE ai-sdk-core ai-sdk-deepseek)

add_executable(example-streaming-chat streaming_chat/main.cpp)
target_link_libraries(example-streaming-chat PRIVATE ai-sdk-core ai-sdk-deepseek)

add_executable(example-tool-agent tool_agent/main.cpp)
target_link_libraries(example-tool-agent PRIVATE ai-sdk-core ai-sdk-anthropic)

add_executable(example-claude-code-clone claude_code_clone/main.cpp)
target_link_libraries(example-claude-code-clone PRIVATE ai-sdk-core ai-sdk-anthropic ai-sdk-openai ai-sdk-deepseek)
