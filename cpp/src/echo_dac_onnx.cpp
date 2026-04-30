// Echo-TTS C++ — S1-DAC ONNX Runtime Wrapper Implementation

#ifdef ECHO_HAS_ONNX

#include "echo_dac_onnx.h"
#include <cstdio>
#include <cassert>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

EchoDACSession::EchoDACSession()
    : env_(ORT_LOGGING_LEVEL_WARNING, "echo-tts-dac")
    , memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    session_opts_.SetIntraOpNumThreads(4);
    session_opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

EchoDACSession::~EchoDACSession() = default;

bool EchoDACSession::load(const std::string & encoder_path, const std::string & decoder_path) {
    try {
        // Try to add CUDA execution provider if available
#ifdef ECHO_HAS_CUDA
        OrtCUDAProviderOptions cuda_opts{};
        cuda_opts.device_id = 0;
        cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
        session_opts_.AppendExecutionProvider_CUDA(cuda_opts);
#endif

        // Convert paths to wide strings on Windows
#ifdef _WIN32
        auto to_wstring = [](const std::string & s) -> std::wstring {
            int len = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
            std::wstring ws(len, 0);
            MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &ws[0], len);
            ws.resize(len - 1);  // remove null terminator
            return ws;
        };
        std::wstring enc_w = to_wstring(encoder_path);
        std::wstring dec_w = to_wstring(decoder_path);
        encoder_session_ = std::make_unique<Ort::Session>(env_, enc_w.c_str(), session_opts_);
        decoder_session_ = std::make_unique<Ort::Session>(env_, dec_w.c_str(), session_opts_);
#else
        encoder_session_ = std::make_unique<Ort::Session>(env_, encoder_path.c_str(), session_opts_);
        decoder_session_ = std::make_unique<Ort::Session>(env_, decoder_path.c_str(), session_opts_);
#endif

        printf("[echo_dac] Loaded encoder: %s\n", encoder_path.c_str());
        printf("[echo_dac] Loaded decoder: %s\n", decoder_path.c_str());
        return true;

    } catch (const Ort::Exception & e) {
        fprintf(stderr, "[echo_dac] ONNX Runtime error: %s\n", e.what());
        encoder_session_.reset();
        decoder_session_.reset();
        return false;
    }
}

std::vector<float> EchoDACSession::encode(
    const float * audio,
    int batch_size,
    int audio_length,
    int & out_time_steps
) {
    assert(encoder_session_ && "Encoder not loaded");

    // Input: (batch, 1, length)
    std::array<int64_t, 3> input_shape = { batch_size, 1, audio_length };
    size_t input_size = batch_size * 1 * audio_length;

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(audio), input_size,
        input_shape.data(), input_shape.size()
    );

    const char * input_names[]  = { "audio" };
    const char * output_names[] = { "z_q" };

    auto outputs = encoder_session_->Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1
    );

    // Output: (batch, 1024, time)
    auto & output_tensor = outputs[0];
    auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto output_shape = output_info.GetShape();

    out_time_steps = static_cast<int>(output_shape[2]);

    const float * output_data = output_tensor.GetTensorData<float>();
    size_t output_size = batch_size * 1024 * out_time_steps;

    return std::vector<float>(output_data, output_data + output_size);
}

std::vector<float> EchoDACSession::decode(
    const float * z_q,
    int batch_size,
    int time_steps,
    int & out_audio_length
) {
    assert(decoder_session_ && "Decoder not loaded");

    // Input: (batch, 1024, time)
    std::array<int64_t, 3> input_shape = { batch_size, 1024, time_steps };
    size_t input_size = batch_size * 1024 * time_steps;

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(z_q), input_size,
        input_shape.data(), input_shape.size()
    );

    const char * input_names[]  = { "z_q" };
    const char * output_names[] = { "audio" };

    auto outputs = decoder_session_->Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1
    );

    // Output: (batch, 1, length)
    auto & output_tensor = outputs[0];
    auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto output_shape = output_info.GetShape();

    out_audio_length = static_cast<int>(output_shape[2]);

    const float * output_data = output_tensor.GetTensorData<float>();
    size_t output_size = batch_size * 1 * out_audio_length;

    return std::vector<float>(output_data, output_data + output_size);
}

void EchoDACSession::release_encoder() {
    encoder_session_.reset();
}

#endif // ECHO_HAS_ONNX
