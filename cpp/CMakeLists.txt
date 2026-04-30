cmake_minimum_required(VERSION 3.18)
project(echo-tts LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Options ──────────────────────────────────────────────────────────
option(ECHO_CUDA   "Enable CUDA backend for GGML"          ON)
option(ECHO_ONNX   "Enable ONNX Runtime for S1-DAC"        ON)

# ── GGML (via FetchContent) ──────────────────────────────────────────
include(FetchContent)

set(GGML_CUDA ${ECHO_CUDA} CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    ggml
    GIT_REPOSITORY https://github.com/ggml-org/ggml.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)

FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.18.0
    GIT_SHALLOW    TRUE
)

FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(ggml httplib json)

# ── ONNX Runtime ─────────────────────────────────────────────────────
if(ECHO_ONNX)
    # User must set ONNXRUNTIME_ROOT to the extracted ONNX Runtime package
    # e.g. cmake -DONNXRUNTIME_ROOT=C:/onnxruntime-win-x64-gpu-1.18.0 ..
    if(NOT DEFINED ONNXRUNTIME_ROOT)
        message(STATUS "ONNXRUNTIME_ROOT not set, looking in default paths...")
        # Try common locations
        find_path(ONNXRUNTIME_INCLUDE_DIR
            NAMES onnxruntime_cxx_api.h
            PATH_SUFFIXES include
        )
        find_library(ONNXRUNTIME_LIB
            NAMES onnxruntime
        )
    else()
        set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
        find_library(ONNXRUNTIME_LIB
            NAMES onnxruntime
            PATHS "${ONNXRUNTIME_ROOT}/lib"
            NO_DEFAULT_PATH
        )
    endif()

    if(ONNXRUNTIME_LIB)
        message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_LIB}")
        add_library(onnxruntime SHARED IMPORTED)
        set_target_properties(onnxruntime PROPERTIES
            IMPORTED_LOCATION "${ONNXRUNTIME_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
        )
        if(WIN32)
            # On Windows, also need the import lib
            find_file(ONNXRUNTIME_DLL
                NAMES onnxruntime.dll
                PATHS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/bin"
                NO_DEFAULT_PATH
            )
            find_library(ONNXRUNTIME_IMPLIB
                NAMES onnxruntime
                PATHS "${ONNXRUNTIME_ROOT}/lib"
                NO_DEFAULT_PATH
            )
            if(ONNXRUNTIME_IMPLIB)
                set_target_properties(onnxruntime PROPERTIES
                    IMPORTED_IMPLIB "${ONNXRUNTIME_IMPLIB}"
                )
            endif()
        endif()
    else()
        message(WARNING "ONNX Runtime not found. Set ONNXRUNTIME_ROOT. S1-DAC support disabled.")
        set(ECHO_ONNX OFF)
    endif()
endif()

# ── Source files ─────────────────────────────────────────────────────
set(ECHO_SOURCES
    src/main.cpp
    src/echo_model.cpp
    src/echo_sampler.cpp
    src/echo_pca.cpp
    src/echo_tokenizer.cpp
    src/echo_audio.cpp
    src/echo_postprocess.cpp
    src/echo_pipeline.cpp
    src/echo_server.cpp
)

if(ECHO_ONNX)
    list(APPEND ECHO_SOURCES src/echo_dac_onnx.cpp)
endif()

add_executable(echo-tts ${ECHO_SOURCES})

target_include_directories(echo-tts PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/extern
)

target_link_libraries(echo-tts PRIVATE ggml httplib::httplib nlohmann_json::nlohmann_json)

if(ECHO_ONNX)
    target_link_libraries(echo-tts PRIVATE onnxruntime)
    target_compile_definitions(echo-tts PRIVATE ECHO_HAS_ONNX=1)
endif()

if(ECHO_CUDA)
    target_compile_definitions(echo-tts PRIVATE ECHO_HAS_CUDA=1)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(echo-tts PRIVATE CUDA::cudart)
endif()

# ── Post-build: copy required DLLs beside the executable ──────────────
if(WIN32)
    # ONNX Runtime DLLs
    if(ECHO_ONNX AND ONNXRUNTIME_ROOT)
        file(GLOB ORT_DLLS "${ONNXRUNTIME_ROOT}/lib/*.dll")
        foreach(ORT_DLL ${ORT_DLLS})
            add_custom_command(TARGET echo-tts POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ORT_DLL}"
                    "$<TARGET_FILE_DIR:echo-tts>"
                COMMENT "Copying ONNX Runtime DLL: ${ORT_DLL}"
            )
        endforeach()
    endif()

    # GGML DLLs (built by FetchContent into bin/<config>)
    add_custom_command(TARGET echo-tts POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_BINARY_DIR}/bin/$<CONFIG>/ggml.dll"
            "${CMAKE_BINARY_DIR}/bin/$<CONFIG>/ggml-base.dll"
            "${CMAKE_BINARY_DIR}/bin/$<CONFIG>/ggml-cpu.dll"
            "$<TARGET_FILE_DIR:echo-tts>"
        COMMENT "Copying GGML DLLs"
    )
    if(ECHO_CUDA)
        add_custom_command(TARGET echo-tts POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_BINARY_DIR}/bin/$<CONFIG>/ggml-cuda.dll"
                "$<TARGET_FILE_DIR:echo-tts>"
            COMMENT "Copying GGML CUDA DLL"
        )
    endif()
endif()

# ── Install ──────────────────────────────────────────────────────────
install(TARGETS echo-tts RUNTIME DESTINATION bin)

# ── Portable packaging ────────────────────────────────────────────────
add_custom_target(package_portable
    COMMAND powershell -ExecutionPolicy Bypass -File
        "${CMAKE_SOURCE_DIR}/../scripts/package_portable.ps1"
        -BuildDir "$<TARGET_FILE_DIR:echo-tts>"
        -CudaVersion 12
        -WithFfmpeg:$<IF:$<BOOL:${ECHO_CUDA}>,true,false>
    COMMENT "Creating portable release ZIP..."
    DEPENDS echo-tts
)
