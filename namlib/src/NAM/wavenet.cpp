#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <sstream>

#include <NAM/wavenet.h>

nam::wavenet::DilatedConvData::DilatedConvData(const int in_channels, const int out_channels, const int kernel_size,
    const int bias, const int dilation) {
    this->set_size(in_channels, out_channels, kernel_size, bias, dilation);
}
void nam::wavenet::DilatedConvData::set_weights(std::vector<float>::const_iterator& weights) {
    if (this->m_weight.size() > 0) {
        const long out_channels = this->m_weight[0].rows();
        const long in_channels = this->m_weight[0].cols();
        // Crazy ordering because that's how it gets flattened.
        for (auto i = 0; i < out_channels; i++)
            for (auto j = 0; j < in_channels; j++)
                for (size_t k = 0; k < this->m_weight.size(); k++)
                    this->m_weight[k](i, j) = *(weights++);
    }
    for (long i = 0; i < this->m_bias.size(); i++)
        this->m_bias(i) = *(weights++);
}

void nam::wavenet::DilatedConvData::set_size(const int in_channels, const int out_channels, const int kernel_size, const bool do_bias,
    const int m_dilation) {
    this->m_weight.resize(kernel_size);
    for (size_t i = 0; i < this->m_weight.size(); i++)
        this->m_weight[i].resize(out_channels,
            in_channels); // y = Ax, input array (C,L)
    if (do_bias)
        this->m_bias.resize(out_channels);
    else
        this->m_bias.resize(0);
    this->m_dilation = m_dilation;
}

// Conv1x1Data =================================================================

nam::wavenet::Conv1x1Data::Conv1x1Data(const int in_channels, const int out_channels, const bool m_bias) {
    this->m_weight.resize(out_channels, in_channels);
    this->m_do_bias = m_bias;
    if (m_bias)
        this->m_bias.resize(out_channels);
}

void nam::wavenet::Conv1x1Data::set_weights(std::vector<float>::const_iterator& weights) {
    for (int i = 0; i < this->m_weight.rows(); i++)
        for (int j = 0; j < this->m_weight.cols(); j++)
            this->m_weight(i, j) = *(weights++);
    if (this->m_do_bias)
        for (int i = 0; i < this->m_bias.size(); i++)
            this->m_bias(i) = *(weights++);
}

// Layers =================================================================

void nam::wavenet::LayerData::set_weights(std::vector<float>::const_iterator& weights) {
    this->m_conv.set_weights(weights);
    this->m_input_mixin.set_weights(weights);
    this->m_1x1.set_weights(weights);
}

// LayerArray =================================================================

nam::wavenet::LayerArrayData::LayerArrayData(const int input_size, const int condition_size, const int head_size,
    const int channels, const int kernel_size, const std::vector<int>& dilations,
    const std::string activation, const bool gated, const bool head_bias) :
    m_rechannel(input_size, channels, false),
    _head_rechannel(channels, head_size, head_bias) {
    for (size_t i = 0; i < dilations.size(); i++)
        this->_layers.push_back(LayerData(condition_size, channels, kernel_size, dilations[i], activation, gated));
}

long nam::wavenet::LayerArrayData::get_receptive_field() const {
    long result = 0;
    for (size_t i = 0; i < this->_layers.size(); i++)
        result += this->_layers[i].get_dilation() * (this->_layers[i].get_kernel_size() - 1);
    return result;
}

void nam::wavenet::LayerArrayData::set_weights(std::vector<float>::const_iterator& weights) {
    this->m_rechannel.set_weights(weights);
    for (size_t i = 0; i < this->_layers.size(); i++)
        this->_layers[i].set_weights(weights);
    this->_head_rechannel.set_weights(weights);
}

// WaveNet ====================================================================
nam::wavenet::GPUWaveNet::GPUWaveNet(const std::vector<nam::wavenet::LayerArrayParams>& layer_array_params,
    const float head_scale, const bool with_head, std::vector<float> weights,
    const double expected_sample_rate) :
    DSP(expected_sample_rate),
    m_head_scale(head_scale) {
    if (with_head)
        throw std::runtime_error("Head not implemented!");
    for (size_t i = 0; i < layer_array_params.size(); i++) {
        _max_num_channels = std::max(_max_num_channels, static_cast<uint32_t>(layer_array_params[i].channels));
        this->_layer_arrays.push_back(nam::wavenet::LayerArrayData(
            layer_array_params[i].input_size, layer_array_params[i].condition_size, layer_array_params[i].head_size,
            layer_array_params[i].channels, layer_array_params[i].kernel_size, layer_array_params[i].dilations,
            layer_array_params[i].activation, layer_array_params[i].gated, layer_array_params[i].head_bias));
        if (i > 0)
            if (layer_array_params[i].channels != layer_array_params[i - 1].head_size) {
                std::stringstream ss;
                ss << "channels of layer " << i << " (" << layer_array_params[i].channels
                   << ") doesn't match head_size of preceding layer (" << layer_array_params[i - 1].head_size << "!\n";
                throw std::runtime_error(ss.str().c_str());
            }
    }
    this->set_weights(weights);

    m_nprewarm_samples = 1;
    for (size_t i = 0; i < this->_layer_arrays.size(); i++)
        m_nprewarm_samples += this->_layer_arrays[i].get_receptive_field();
}

void nam::wavenet::GPUWaveNet::set_weights(std::vector<float> const& weights) {
    std::vector<float>::const_iterator it = weights.cbegin();
    for (size_t i = 0; i < this->_layer_arrays.size(); i++)
        this->_layer_arrays[i].set_weights(it);
    // this->_head.set_params_(it);
    this->m_head_scale = *(it++);
    if (it != weights.end()) {
        std::stringstream ss;
        for (size_t i = 0; i < weights.size(); i++)
            if (weights[i] == *it) {
                ss << "Weight mismatch: assigned " << i + 1 << " weights, but " << weights.size() << " were provided.";
                throw std::runtime_error(ss.str().c_str());
            }
        ss << "Weight mismatch: provided " << weights.size() << " weights, but the model expects more.";
        throw std::runtime_error(ss.str().c_str());
    }
}

void nam::wavenet::GPUWaveNet::SetupProcessorData(NamConfig::Specification& spec) {
    // reset
    m_dilations.clear();
    m_weights.clear();

    // loop over all layer arrays to convert them to the data structs for the GPU processor
    for (uint32_t la_id {0u}; la_id < _layer_arrays.size(); ++la_id) {
        auto const& layer_array = _layer_arrays[la_id];

        // add re-channel weights and bias weights of the re-channel 1x1 convolution
        m_weights.push_back(layer_array.get_rechannel().get_weights().data());
        m_weights.push_back(layer_array.get_rechannel().get_bias().data());

        // loop over the layers of the layer array
        for (uint32_t l_id {0u}; l_id < layer_array.get_layers().size(); ++l_id) {
            auto const& layer = layer_array.get_layers()[l_id];

            // add the k weights for the dilated convolution
            for (uint32_t k {0u}; k < layer.get_conv().get_kernel_size(); ++k) {
                m_weights.push_back(layer.get_conv().get_weights()[k].data());
            }
            // add bias weights and dilation for the dilated convolution
            m_weights.push_back(layer.get_conv().get_bias().data());
            m_dilations.push_back(layer.get_conv().get_dilation());

            // add weights and bias weights for the input mix-in 1x1 convolution
            m_weights.push_back(layer.get_input_mixin().get_weights().data());
            m_weights.push_back(layer.get_input_mixin().get_bias().data());

            // add weights and bias weights for the post-activation 1x1 convolution
            m_weights.push_back(layer.get_1x1().get_weights().data());
            m_weights.push_back(layer.get_1x1().get_bias().data());
        }

        // add weights and bias weights for the head re-channel 1x1 convolution
        m_weights.push_back(layer_array.get_head_rechannel().get_weights().data());
        m_weights.push_back(layer_array.get_head_rechannel().get_bias().data());
    }

    // set remaining members of the processor spec
    spec.type = NamConfig::NetType::eWaveNet;
    spec.head_scale = &m_head_scale;
    spec.dilations = m_dilations.data();
    spec.weights = reinterpret_cast<void const**>(m_weights.data());
}
