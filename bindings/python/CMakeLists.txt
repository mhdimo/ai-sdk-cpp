find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.12.0
    )
    FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(_native src/module.cpp)

target_include_directories(_native PRIVATE
    ${CMAKE_SOURCE_DIR}/bindings/c
)

target_link_libraries(_native PRIVATE ai-sdk-c)

install(TARGETS _native DESTINATION ai_sdk)
