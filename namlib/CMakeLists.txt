# Component name
set(component_name namlib)


include(FetchContent)
cmake_policy(SET CMP0135 NEW)

# ENGINE API
FetchContent_Declare(
    engine_api
    URL https://packages.braingines.com/artifactory/partners/common/engine_api_unstable/engine_api-v2.2.0_62.zip
    HTTP_HEADER "Authorization: Bearer $ENV{GPUAUDIO_ART_KEY}"
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/engine_api
)
FetchContent_MakeAvailable(engine_api)

# GPU AUDIO CLIENT
FetchContent_Declare(
    gpu_audio_client
    URL https://packages.braingines.com/artifactory/partners/common/gpu_audio_client_unstable/gpu_audio_client-v0.1.1_12.zip
    HTTP_HEADER "Authorization: Bearer $ENV{GPUAUDIO_ART_KEY}"
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu_audio_client
)
FetchContent_MakeAvailable(gpu_audio_client)

# JSON
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

# Google Tests
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

# Component project
add_library(${component_name} STATIC)

set(target_src
    src/GPUCreate.cpp
    src/GPUNeuralAmpModeler.cpp
)

set(target_headers
    src/GPUNeuralAmpModeler.h
)

set(nam_src
    src/NAM/dsp.cpp
    src/NAM/get_dsp.cpp
    src/NAM/lstm.cpp
    src/NAM/wavenet.cpp
)

set(nam_headers
    include/NAM/dsp.h
    include/NAM/get_dsp.h
    include/NAM/lstm.h
    include/NAM/tensor_data.h
    include/NAM/version.h
    include/NAM/wavenet.h
)

# Source files
target_sources(${component_name} PRIVATE
    ${target_src}
    ${nam_src}
    ${target_headers}
    ${nam_headers}
)
source_group(NAM ${CMAKE_CURRENT_SOURCE_DIR} FILES ${nam_src} ${nam_headers})

# Include directories
target_include_directories(${component_name} PUBLIC
    ../include
    include
)

# Header-only dependencies if they were fetched from the top-level project
if (DEFINED FETCH_LOCATION)
    target_include_directories(${component_name} PUBLIC
        ${FETCH_LOCATION}/include
    )
endif ()

# Compile definitions
if (WIN32)
    set(win_common_private_compile_definitions
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )
endif ()

target_compile_definitions(${component_name} PRIVATE
    ${win_common_private_compile_definitions}
)

# Link libraries
target_link_libraries(${component_name} PUBLIC nlohmann_json::nlohmann_json)

if (NOT DEFINED NAMLIB_DISABLE_TESTS)
    # Unit tests
    set(tests_name ${component_name}_tests)

    add_executable(${tests_name})

    # Source files
    target_sources(${tests_name} PRIVATE
        tests/AudioFile.h
        tests/TestCommon.h
        tests/GPUNeuralAmpModelerTests.cpp
    )

    # Include directories
    target_include_directories(${tests_name} PRIVATE
        ../include
        include
    )
endif ()

# Compile definitions
if (WIN32)
    set(win_common_private_compile_definitions
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )
elseif(APPLE)
    set(apple_common_private_compile_definitions
        GPU_AUDIO_MAC
)
endif ()

if (NOT DEFINED NAMLIB_DISABLE_TESTS)
    target_compile_definitions(${tests_name} PRIVATE
        ${win_common_private_compile_definitions}
        ${apple_common_private_compile_definitions}
        BUILD_TYPE="$<CONFIG>"
    )

    # Link libraries
    target_link_libraries(${tests_name} PRIVATE
        ${component_name}
        gtest_main
        nlohmann_json::nlohmann_json
    )

    gtest_add_tests(TARGET ${tests_name})
endif ()

# process file executable
add_executable(nam_process_file)

# Source files
target_sources(nam_process_file PRIVATE
    tests/AudioFile.h
    tests/nam_process_file.cpp
)

# Include directories
target_include_directories(nam_process_file PRIVATE
    ../include
    include
)

target_compile_definitions(nam_process_file PRIVATE
    ${win_common_private_compile_definitions}
    BUILD_TYPE="$<CONFIG>"
)

# Link libraries
target_link_libraries(nam_process_file PRIVATE
    ${component_name}
    nlohmann_json::nlohmann_json
)

set_property(TARGET ${component_name} PROPERTY COMPILE_WARNING_AS_ERROR OFF)
set_property(TARGET ${tests_name} PROPERTY COMPILE_WARNING_AS_ERROR OFF)
set_property(TARGET nam_process_file PROPERTY COMPILE_WARNING_AS_ERROR OFF)
