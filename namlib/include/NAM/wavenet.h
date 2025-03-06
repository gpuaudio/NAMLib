#pragma once

#include <string>
#include <vector>

#include <nam_processor/NamSpecification.h>

#include "dsp.h"
#include "tensor_data.h"

namespace nam {
namespace wavenet {

class DilatedConvData {
public:
    DilatedConvData(const int in_channels, const int out_channels, const int kernel_size, const int bias,
        const int dilation);
    void set_weights(std::vector<float>::const_iterator& weights);
    void set_size(const int in_channels, const int out_channels, const int kernel_size, const bool do_bias,
        const int m_dilation);

    long get_in_channels() const { return this->m_weight.size() > 0 ? this->m_weight[0].cols() : 0; };
    long get_kernel_size() const { return this->m_weight.size(); };
    long get_out_channels() const { return this->m_weight.size() > 0 ? this->m_weight[0].rows() : 0; };

    std::vector<MatrixData> const& get_weights() const { return this->m_weight; };
    VectorData const& get_bias() const { return this->m_bias; };
    int get_dilation() const { return this->m_dilation; };

private:
    std::vector<MatrixData> m_weight;
    VectorData m_bias;
    int m_dilation;
};

class Conv1x1Data {
public:
    Conv1x1Data(const int in_channels, const int out_channels, const bool m_bias);
    void set_weights(std::vector<float>::const_iterator& weights);

    MatrixData const& get_weights() const { return this->m_weight; };
    VectorData const& get_bias() const { return this->m_bias; };
    bool get_do_bias() const { return this->m_do_bias; };

private:
    MatrixData m_weight;
    VectorData m_bias;
    bool m_do_bias;
};

class LayerData {
public:
    LayerData(const int condition_size, const int channels, const int kernel_size, const int dilation,
        const std::string activation, const bool gated) :
        m_conv(channels, gated ? 2 * channels : channels, kernel_size, true, dilation),
        m_input_mixin(condition_size, gated ? 2 * channels : channels, false),
        m_1x1(channels, channels, true) {};
    void set_weights(std::vector<float>::const_iterator& weights);
    long get_channels() const { return this->m_conv.get_in_channels(); };
    int get_dilation() const { return this->m_conv.get_dilation(); };
    long get_kernel_size() const { return this->m_conv.get_kernel_size(); };

    DilatedConvData const& get_conv() const { return this->m_conv; };
    Conv1x1Data const& get_input_mixin() const { return this->m_input_mixin; };
    Conv1x1Data const& get_1x1() const { return this->m_1x1; };

private:
    // The dilated convolution at the front of the block
    DilatedConvData m_conv;
    // Input mixin
    Conv1x1Data m_input_mixin;
    // The post-activation 1x1 convolution
    Conv1x1Data m_1x1;
};

class LayerArrayParams {
public:
    LayerArrayParams(const int input_size_, const int condition_size_, const int head_size_, const int channels_,
        const int kernel_size_, const std::vector<int>&& dilations_, const std::string activation_,
        const bool gated_, const bool head_bias_) :
        input_size(input_size_),
        condition_size(condition_size_),
        head_size(head_size_),
        channels(channels_),
        kernel_size(kernel_size_),
        dilations(std::move(dilations_)),
        activation(activation_),
        gated(gated_),
        head_bias(head_bias_) {
    }

    const int input_size;
    const int condition_size;
    const int head_size;
    const int channels;
    const int kernel_size;
    std::vector<int> dilations;
    const std::string activation;
    const bool gated;
    const bool head_bias;
};

// An array of layers with the same channels, kernel sizes, activations.
class LayerArrayData {
public:
    LayerArrayData(const int input_size, const int condition_size, const int head_size, const int channels,
        const int kernel_size, const std::vector<int>& dilations, const std::string activation, const bool gated,
        const bool head_bias);

    void set_weights(std::vector<float>::const_iterator& it);

    // "Zero-indexed" receptive field.
    // E.g. a 1x1 convolution has a z.i.r.f. of zero.
    long get_receptive_field() const;

    Conv1x1Data const& get_rechannel() const { return this->m_rechannel; };
    std::vector<LayerData> const& get_layers() const { return this->_layers; };
    Conv1x1Data const& get_head_rechannel() const { return this->_head_rechannel; };

private:
    // The rechannel before the layers
    Conv1x1Data m_rechannel;

    // The layer objects
    std::vector<LayerData> _layers;

    // Rechannel for the head
    Conv1x1Data _head_rechannel;
};

// The main WaveNet model
class GPUWaveNet : public DSP {
    std::vector<uint32_t> m_dilations;
    std::vector<float const*> m_weights;

    uint32_t _max_num_channels {0u};
    std::vector<LayerArrayData> _layer_arrays;

    float m_head_scale;
    int m_nprewarm_samples {0};

protected:
    int PrewarmSamples() override { return m_nprewarm_samples; };

public:
    /**
     * @brief Constructor; set up the CPU data structures
     */
    GPUWaveNet(const std::vector<LayerArrayParams>& layer_array_params, const float head_scale, const bool with_head,
        std::vector<float> weights, const double expected_sample_rate = -1.0);

    /**
     * @brief Set weights in the network
     * @param weights [in] all weights required in the network
     */
    void set_weights(std::vector<float> const& weights);

    /**
     * @brief Converts the CPU data structures into a processor specification
     * @param spec [out] specification to create the GPU processor from
     */
    void SetupProcessorData(NamConfig::Specification& spec);
};

}; // namespace wavenet
}; // namespace nam
