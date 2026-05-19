add_executable(ai-agent main.cpp)

target_link_libraries(ai-agent PRIVATE
    ai-sdk-core
    ai-sdk-anthropic
    ai-sdk-openai
    ai-sdk-bedrock
)

install(TARGETS ai-agent RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
