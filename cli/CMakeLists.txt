add_executable(ai-agent main.cpp)

target_link_libraries(ai-agent PRIVATE ai-sdk-core)
# Only link providers that are actually built (some may be disabled).
foreach(pv ai-sdk-anthropic ai-sdk-openai ai-sdk-bedrock)
    if(TARGET ${pv})
        target_link_libraries(ai-agent PRIVATE ${pv})
    endif()
endforeach()

install(TARGETS ai-agent RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
