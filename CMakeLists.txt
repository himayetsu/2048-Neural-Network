cmake_minimum_required(VERSION 3.14)
project(2048_AI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Torch REQUIRED)
find_package(SDL2 REQUIRED)
find_package(SDL2_ttf REQUIRED)

# The entry point includes the guarded component .cpp files. Compile it once
# rather than also compiling those components as separate translation units.
add_executable(2048_ai 2048_ai.cpp)

# Link libraries
target_link_libraries(2048_ai PRIVATE ${TORCH_LIBRARIES} SDL2::SDL2 SDL2_ttf::SDL2_ttf)

# Set compile options
target_compile_options(2048_ai PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>
)

include(CTest)
if(BUILD_TESTING)
    add_executable(game_engine_tests tests/game_engine_tests.cpp)
    target_include_directories(game_engine_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME game_engine_tests COMMAND game_engine_tests)

    add_executable(training_tests tests/training_tests.cpp)
    target_include_directories(training_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(training_tests PRIVATE ${TORCH_LIBRARIES})
    add_test(NAME training_tests COMMAND training_tests)
endif()

# Copy DLLs on Windows
if(WIN32)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET 2048_ai POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:2048_ai>)
endif()
