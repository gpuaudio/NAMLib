#ifndef GPUA_GPU_NEURAL_AMP_MODELER_H
#define GPUA_GPU_NEURAL_AMP_MODELER_H

#include <NeuralAmpModelerInterface.h>

#include <engine_api/GraphLauncher.h>
#include <engine_api/Module.h>
#include <engine_api/ProcessingGraph.h>
#include <engine_api/Processor.h>

#include <gpu_audio_client/ProcessExecutorSync.h>
#include <gpu_audio_client/ProcessExecutorAsync.h>

#include "NAM/wavenet.h"
#include "NAM/lstm.h"

#include <array>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

/**
 * Common base for GPUNeuralAmpModelerWavenet and GPUNeuralAmpModelerLSTM
 */
template <ExecutionMode EXEC_MODE>
class GPUNeuralAmpModeler {
protected:
    NamConfig::Specification m_processor_spec;

    /**
     * @brief Constructor
     * @param buffer_samples_per_channel [in] maximum number of samples per channel in the processing-buffer
     * @param threads_per_block [in] number of threads per block used in the GPU process function
     */
    GPUNeuralAmpModeler(uint32_t buffer_samples_per_channel, uint32_t threads_per_block);

    /**
     * @brief Destructor
     */
    virtual ~GPUNeuralAmpModeler();

    /**
     * @brief Get the client library ready for processing with the current configuration
     */
    void common_arm();

    /**
     * @brief Clean up and get ready for destruction or re-configuration
     */
    void common_disarm();

    /**
     * @brief Process samples provided in input and write them to output buffers.
     * @param input [in] pointer to pointers to the channels of the input audio data
     * @param output [in/out] pointer to pointers to the channels of the output audio data
     */
    void common_process(float const* const* in_buffer, float* const* out_buffer, int nsamples);

    /**
     * @brief Process dummy data to get the network into an initial state
     * @param nprewarm_samples [in] number of samples to process to reach initial state
     */
    void common_prewarm(uint32_t nprewarm_samples);

    /**
     * @brief Turn automatic buffer growth on or off.
     * @param enable [in] true to enable, false to disable
     */
    void common_enable_buffer_growth(bool enable);

    /**
     * @brief Get the current latency introduced by double buffering
     * @return latency in number of samples
     */
    uint32_t common_get_latency();

private:
    std::mutex m_armed_mutex;
    bool m_armed {false};

    static constexpr uint32_t ChannelCount {1u};
    static constexpr uint32_t MaxSampleCount {4096u};

    GPUA::engine::v2::GraphLauncher* m_launcher {nullptr};
    GPUA::engine::v2::ProcessingGraph* m_graph {nullptr};
    GPUA::engine::v2::Module* m_module {nullptr};

    GPUA::engine::v2::Processor* m_processor {nullptr};

    bool m_buffer_growth_enabled {true};

    ProcessExecutorConfig m_executor_config;
    ProcessExecutor<EXEC_MODE>* m_process_executor {nullptr};

    void renewExecutor();
};

/**
 * GPU Neural Amp Modeler: Wavenet
 */
template <ExecutionMode EXEC_MODE>
class GPUNeuralAmpModelerWavenet : public NeuralAmpModelerInterface, public GPUNeuralAmpModeler<EXEC_MODE>, public nam::wavenet::GPUWaveNet {
public:
    GPUNeuralAmpModelerWavenet(uint32_t buffer_samples_per_channel, uint32_t threads_per_block,
        const std::vector<nam::wavenet::LayerArrayParams>& layer_array_params, const float head_scale, const bool with_head, std::vector<float> weights, const double expected_sample_rate = -1.0);

    ////////////////////////////////
    // NeuralAmpModelerInterface methods
    virtual void arm() override;
    virtual void disarm() override;

    virtual void enable_buffer_growth(bool enable) override;
    virtual uint32_t get_latency() override;
    // NeuralAmpModelerInterface methods
    ////////////////////////////////

    ////////////////////////////////
    // DSP interface methods
    virtual void prewarm() override;
    // DSP interface methods
    ////////////////////////////////

    ////////////////////////////////
    // NeuralAmpModelerInterface & DSP methods
    virtual void process(float* input, float* output, const int nsamples) override;
    // NeuralAmpModelerInterface & DSP methods
    ////////////////////////////////
};

/**
 * GPU Neural Amp Modeler: LSTM
 */
template <ExecutionMode EXEC_MODE>
class GPUNeuralAmpModelerLSTM : public NeuralAmpModelerInterface, public GPUNeuralAmpModeler<EXEC_MODE>, public nam::lstm::GPULongShortTermMemory {
public:
    GPUNeuralAmpModelerLSTM(uint32_t buffer_samples_per_channel, uint32_t threads_per_block,
        const int num_layers, const int input_size, const int hidden_size, std::vector<float>& weights, const double expected_sample_rate = -1.0);

    ////////////////////////////////
    // NeuralAmpModelerInterface methods
    virtual void arm() override;
    virtual void disarm() override;

    virtual void enable_buffer_growth(bool enable) override;
    virtual uint32_t get_latency() override;
    // NeuralAmpModelerInterface methods
    ////////////////////////////////

    ////////////////////////////////
    // DSP interface methods
    virtual void prewarm() override;
    // DSP interface methods
    ////////////////////////////////

    ////////////////////////////////
    // NeuralAmpModelerInterface & DSP methods
    virtual void process(float* input, float* output, const int nsamples) override;
    // NeuralAmpModelerInterface & DSP methods
    ////////////////////////////////
};

#endif // GPUA_GPU_NEURAL_AMP_MODELER_H
