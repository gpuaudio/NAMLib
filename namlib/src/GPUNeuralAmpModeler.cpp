#include "GPUNeuralAmpModeler.h"

#include <gpu_audio_client/GpuAudioManager.h>

#include <engine_api/DeviceInfoProvider.h>
#include <engine_api/LauncherSpecification.h>
#include <engine_api/ModuleInfo.h>

#define _USE_MATH_DEFINES
#include <algorithm>
#include <array>
#include <cassert>
#include <math.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>

template <ExecutionMode EXEC_MODE>
GPUNeuralAmpModeler<EXEC_MODE>::GPUNeuralAmpModeler(uint32_t buffer_samples_per_channel, uint32_t threads_per_block) {
    // buffer settings and double buffering configuration (see `gpu_audio_client` for details)
    m_executor_config = {
        .retain_threshold = 0.625,
        .launch_threshold = 0.7275,
        .nchannels_in = ChannelCount,
        .nchannels_out = ChannelCount,
        .max_samples_per_channel = buffer_samples_per_channel};

    // initialize GPU processor specification
    m_processor_spec.max_buffer_samples = buffer_samples_per_channel;
    m_processor_spec.threads_per_block = threads_per_block;

    // create gpu_audio engine and make sure a supported GPU is installed/selected
    const auto& gpu_audio = GpuAudioManager::GetGpuAudio();
    const auto& device_info_provider = gpu_audio->GetDeviceInfoProvider();
    const auto dev_idx = GpuAudioManager::GetDeviceIndex();
    if (dev_idx >= device_info_provider.GetDeviceCount()) {
        throw std::runtime_error("No supported device found");
    }
    // get all the information about the GPU device required to create a launcher
    GPUA::engine::v2::LauncherSpecification launcher_spec = {};
    if ((device_info_provider.GetDeviceInfo(dev_idx, launcher_spec.device_info) != GPUA::engine::v2::ErrorCode::eSuccess) || !launcher_spec.device_info) {
        throw std::runtime_error("Failed to get device info");
    }

    // create a launcher for the specified GPU device
    if ((gpu_audio->CreateLauncher(launcher_spec, m_launcher) != GPUA::engine::v2::ErrorCode::eSuccess) || !m_launcher) {
        throw std::runtime_error("Failed to create launcher");
    }

    // create a processing graph to create the processor(s) in
    if ((m_launcher->CreateProcessingGraph(m_graph) != GPUA::engine::v2::ErrorCode::eSuccess) || !m_graph) {
        gpu_audio->DeleteLauncher(m_launcher);
        throw std::runtime_error("Failed to create processing graph");
    }

    // get the module provider from the launcher to access all available modules (read as processors here)
    auto& module_provider = m_launcher->GetModuleProvider();
    const auto module_count = module_provider.GetModulesCount();
    GPUA::engine::v2::ModuleInfo info {};
    bool processor_module_found = false;
    // iterate the module infos and try to find the nam processor by matching the id
    for (uint32_t i = 0; i < module_count; ++i) {
        if ((module_provider.GetModuleInfo(i, info) == GPUA::engine::v2::ErrorCode::eSuccess) && info.id && (std::wcscmp(info.id, L"nam") == 0)) {
            processor_module_found = true;
            break;
        }
    }
    // we could not find the nam processor
    if (!processor_module_found) {
        m_launcher->DeleteProcessingGraph(m_graph);
        gpu_audio->DeleteLauncher(m_launcher);
        throw std::runtime_error("Failed to find required processor module");
    }

    // get the processor's module; we need this to create and destroy the processor instance
    if ((module_provider.GetModule(info, m_module) != GPUA::engine::v2::ErrorCode::eSuccess) || !m_module) {
        m_launcher->DeleteProcessingGraph(m_graph);
        gpu_audio->DeleteLauncher(m_launcher);
        throw std::runtime_error("Failed to load required processor module");
    }
};

template <ExecutionMode EXEC_MODE>
GPUNeuralAmpModeler<EXEC_MODE>::~GPUNeuralAmpModeler() {
    // delete executor and processor
    common_disarm();
    // delete the processing graph
    m_launcher->DeleteProcessingGraph(m_graph);
    // delete the launcher
    GpuAudioManager::GetGpuAudio()->DeleteLauncher(m_launcher);
}

template <ExecutionMode EXEC_MODE>
void GPUNeuralAmpModeler<EXEC_MODE>::common_arm() {
    std::lock_guard<std::mutex> lock(m_armed_mutex);

    if (!m_armed) {
        // use the processor module and the processor specification to create a processor instance in the graph
        if (m_module->CreateProcessor(m_graph, &m_processor_spec, sizeof(m_processor_spec), m_processor) != GPUA::engine::v2::ErrorCode::eSuccess || !m_processor) {
            throw std::runtime_error("Failed to create processor");
        }
        // create an executor that manages input and output buffers and performs the actual launches
        m_process_executor = new ProcessExecutor<EXEC_MODE>(m_launcher, m_graph, 1u, &m_processor, m_executor_config);
        m_armed = true;
    }
}

template <ExecutionMode EXEC_MODE>
void GPUNeuralAmpModeler<EXEC_MODE>::common_disarm() {
    std::lock_guard<std::mutex> lock(m_armed_mutex);

    if (m_armed) {
        // delete the executor - ensures that all launches have finished before destroying itself.
        if (m_process_executor) {
            delete m_process_executor;
            m_process_executor = nullptr;
        }

        // as no more launches are active, it's safe to destroy the processor
        if (m_processor) {
            m_module->DeleteProcessor(m_processor);
            m_processor = nullptr;
        }
        m_armed = false;
    }
}

template <ExecutionMode EXEC_MODE>
void GPUNeuralAmpModeler<EXEC_MODE>::common_process(float const* const* in_buffer, float* const* out_buffer, int nsamples) {
    // If the processor was not armed ahead of time, arm it on the first process call.
    if (!m_armed) {
        common_arm();
    }

    // If we get more samples to process than the buffers can currently hold, we increase their size if buffer growth is enabled.
    // The current content of the buffers is lost - only relevant if double buffering is enabled, as we then get a full buffer of
    // zeros after the buffer realloc.
    if (m_buffer_growth_enabled && nsamples > m_executor_config.max_samples_per_channel && m_executor_config.max_samples_per_channel < MaxSampleCount) {
        do {
            m_executor_config.max_samples_per_channel = std::min(m_executor_config.max_samples_per_channel * 2, MaxSampleCount);
        } while (nsamples > m_executor_config.max_samples_per_channel && m_executor_config.max_samples_per_channel < MaxSampleCount);
        renewExecutor();
    }

    thread_local std::vector<float const*> input_ptrs;
    thread_local std::vector<float*> output_ptrs;
    if (nsamples > m_executor_config.max_samples_per_channel) {
        input_ptrs.assign(in_buffer, in_buffer + ChannelCount);
        output_ptrs.assign(out_buffer, out_buffer + ChannelCount);
    }

    uint32_t remainingSamples = nsamples;
    while (remainingSamples != 0) {
        uint32_t thisLaunchSamples = std::min(m_executor_config.max_samples_per_channel, remainingSamples);

        if constexpr (EXEC_MODE == ExecutionMode::eAsync) {
            m_process_executor->template ExecuteAsync<AudioDataLayout::eChannelsIndividual>(thisLaunchSamples, in_buffer);
            m_process_executor->template RetrieveOutput<AudioDataLayout::eChannelsIndividual>(thisLaunchSamples, out_buffer);
        }
        else {
            m_process_executor->template Execute<AudioDataLayout::eChannelsIndividual>(thisLaunchSamples, in_buffer, out_buffer);
        }

        remainingSamples -= thisLaunchSamples;
        if (remainingSamples != 0) {
            for (auto& ptr : input_ptrs)
                ptr += thisLaunchSamples;
            in_buffer = input_ptrs.data();

            for (auto& ptr : output_ptrs)
                ptr += thisLaunchSamples;
            out_buffer = output_ptrs.data();
        }
    }
}

template <ExecutionMode EXEC_MODE>
void GPUNeuralAmpModeler<EXEC_MODE>::common_prewarm(uint32_t nprewarm_samples) {
    if (nprewarm_samples == 0)
        return;

    // stick to the internal buffer size to not trigger buffer growth
    uint32_t const nsamples_per_launch = m_executor_config.max_samples_per_channel;
    // setup dummy input & output buffers
    std::vector<float> input(nsamples_per_launch, 0.0f);
    float* input_ptr = input.data();
    std::vector<float> output(nsamples_per_launch, 0.0f);
    float* output_ptr = output.data();

    // process dummy data to initialize the model state
    uint32_t nsamples_processed = 0;
    while (nsamples_processed < nprewarm_samples) {
        this->common_process(&input_ptr, &output_ptr, nsamples_per_launch);
        nsamples_processed += nsamples_per_launch;
    }
}

template <ExecutionMode EXEC_MODE>
void GPUNeuralAmpModeler<EXEC_MODE>::common_enable_buffer_growth(bool enable) {
    m_buffer_growth_enabled = enable;
}

template <ExecutionMode EXEC_MODE>
uint32_t GPUNeuralAmpModeler<EXEC_MODE>::common_get_latency() {
    // distance between first sample to process and first processed sample retrieved in a single process call
    return m_executor_config.max_samples_per_channel;
}

template <ExecutionMode EXEC_MODE>
void GPUNeuralAmpModeler<EXEC_MODE>::renewExecutor() {
    std::lock_guard<std::mutex> lock(m_armed_mutex);

    // delete executor; ensures that all launches finished
    if (m_process_executor) {
        delete m_process_executor;
        m_process_executor = nullptr;
    }

    // create new executor to apply changes in the executor config or in the processor
    m_process_executor = new ProcessExecutor<EXEC_MODE>(m_launcher, m_graph, 1u, &m_processor, m_executor_config);
}

template <ExecutionMode EXEC_MODE>
GPUNeuralAmpModelerWavenet<EXEC_MODE>::GPUNeuralAmpModelerWavenet(uint32_t buffer_samples_per_channel, uint32_t threads_per_block,
    const std::vector<nam::wavenet::LayerArrayParams>& layer_array_params, const float head_scale, const bool with_head, std::vector<float> weights, const double expected_sample_rate) :
    GPUNeuralAmpModeler<EXEC_MODE>(buffer_samples_per_channel, threads_per_block),
    nam::wavenet::GPUWaveNet(layer_array_params, head_scale, with_head, weights, expected_sample_rate) {
    nam::wavenet::GPUWaveNet::SetupProcessorData(GPUNeuralAmpModeler<EXEC_MODE>::m_processor_spec);
}

template <ExecutionMode EXEC_MODE>
void GPUNeuralAmpModelerWavenet<EXEC_MODE>::arm() {
    GPUNeuralAmpModeler<EXEC_MODE>::common_arm();
}

template <ExecutionMode EXEC_MODE>
void GPUNeuralAmpModelerWavenet<EXEC_MODE>::disarm() {
    GPUNeuralAmpModeler<EXEC_MODE>::common_disarm();
}

template <ExecutionMode EXEC_MODE>
void GPUNeuralAmpModelerWavenet<EXEC_MODE>::process(float* input, float* output, const int nsamples) {
    GPUNeuralAmpModeler<EXEC_MODE>::common_process(&input, &output, nsamples);
}

template <ExecutionMode EXEC_MODE>
void GPUNeuralAmpModelerWavenet<EXEC_MODE>::prewarm() {
    GPUNeuralAmpModeler<EXEC_MODE>::common_prewarm(PrewarmSamples());
}

template <ExecutionMode EXEC_MODE>
void GPUNeuralAmpModelerWavenet<EXEC_MODE>::enable_buffer_growth(bool enable) {
    GPUNeuralAmpModeler<EXEC_MODE>::common_enable_buffer_growth(enable);
}

template <ExecutionMode EXEC_MODE>
uint32_t GPUNeuralAmpModelerWavenet<EXEC_MODE>::get_latency() {
    return GPUNeuralAmpModeler<EXEC_MODE>::common_get_latency();
}

template class GPUNeuralAmpModelerWavenet<ExecutionMode::eSync>;
template class GPUNeuralAmpModelerWavenet<ExecutionMode::eAsync>;

template <ExecutionMode EXEC_MODE>
GPUNeuralAmpModelerLSTM<EXEC_MODE>::GPUNeuralAmpModelerLSTM(uint32_t buffer_samples_per_channel, uint32_t threads_per_block,
    const int num_layers, const int input_size, const int hidden_size, std::vector<float>& weights, const double expected_sample_rate) :
    GPUNeuralAmpModeler<EXEC_MODE>(buffer_samples_per_channel, threads_per_block),
    nam::lstm::GPULongShortTermMemory(num_layers, input_size, hidden_size, weights, expected_sample_rate) {
    nam::lstm::GPULongShortTermMemory::SetupProcessorData(GPUNeuralAmpModeler<EXEC_MODE>::m_processor_spec);
}

template <ExecutionMode EXEC_MODE>
void GPUNeuralAmpModelerLSTM<EXEC_MODE>::arm() {
    GPUNeuralAmpModeler<EXEC_MODE>::common_arm();
}

template <ExecutionMode EXEC_MODE>
void GPUNeuralAmpModelerLSTM<EXEC_MODE>::disarm() {
    GPUNeuralAmpModeler<EXEC_MODE>::common_disarm();
}

template <ExecutionMode EXEC_MODE>
void GPUNeuralAmpModelerLSTM<EXEC_MODE>::process(float* input, float* output, const int nsamples) {
    GPUNeuralAmpModeler<EXEC_MODE>::common_process(&input, &output, nsamples);
}

template <ExecutionMode EXEC_MODE>
void GPUNeuralAmpModelerLSTM<EXEC_MODE>::prewarm() {
    GPUNeuralAmpModeler<EXEC_MODE>::common_prewarm(PrewarmSamples());
}

template <ExecutionMode EXEC_MODE>
void GPUNeuralAmpModelerLSTM<EXEC_MODE>::enable_buffer_growth(bool enable) {
    GPUNeuralAmpModeler<EXEC_MODE>::common_enable_buffer_growth(enable);
}

template <ExecutionMode EXEC_MODE>
uint32_t GPUNeuralAmpModelerLSTM<EXEC_MODE>::get_latency() {
    return GPUNeuralAmpModeler<EXEC_MODE>::common_get_latency();
}

template class GPUNeuralAmpModelerLSTM<ExecutionMode::eSync>;
template class GPUNeuralAmpModelerLSTM<ExecutionMode::eAsync>;
