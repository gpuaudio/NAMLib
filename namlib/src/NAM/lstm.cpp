#include <algorithm>
#include <string>
#include <vector>

#include <NAM/lstm.h>

nam::lstm::LSTMCellData::LSTMCellData(const int input_size, const int hidden_size, std::vector<float>::iterator& weights) {
    // Resize arrays
    this->m_w.resize(4 * hidden_size, input_size + hidden_size);
    this->m_b.resize(4 * hidden_size);
    this->m_xh.resize(input_size + hidden_size);
    this->m_ifgo.resize(4 * hidden_size);
    this->m_c.resize(hidden_size);

    // Assign in row-major because that's how PyTorch goes.
    for (int i = 0; i < this->m_w.rows(); i++)
        for (int j = 0; j < this->m_w.cols(); j++)
            this->m_w(i, j) = *(weights++);
    for (int i = 0; i < this->m_b.size(); i++)
        this->m_b(i) = *(weights++);
    const int h_offset = input_size;
    for (int i = 0; i < hidden_size; i++)
        this->m_xh(i + h_offset) = *(weights++);
    for (int i = 0; i < hidden_size; i++)
        this->m_c(i) = *(weights++);
}

nam::lstm::GPULongShortTermMemory::GPULongShortTermMemory(const int num_layers, const int input_size, const int hidden_size, std::vector<float>& weights,
    const double expected_sample_rate) :
    DSP(expected_sample_rate) {
    std::vector<float>::iterator it = weights.begin();
    for (int i = 0; i < num_layers; i++)
        this->m_layers.push_back(LSTMCellData(i == 0 ? input_size : hidden_size, hidden_size, it));
    this->m_head_weight.resize(hidden_size);
    for (int i = 0; i < hidden_size; i++)
        this->m_head_weight(i) = *(it++);
    this->m_head_bias = *(it++);
    assert(it == weights.end());
}

int nam::lstm::GPULongShortTermMemory::PrewarmSamples() {
    int result = (int)(0.5 * mExpectedSampleRate);
    // If the expected sample rate wasn't provided, it'll be -1.
    // Make sure something still happens.
    return result <= 0 ? 1 : result;
}

void nam::lstm::GPULongShortTermMemory::SetupProcessorData(NamConfig::Specification& spec) {
    // reset
    m_weights.clear();

    // add the head weights
    m_weights.push_back(m_head_weight.data());
    uint32_t num_layers = static_cast<uint32_t>(m_layers.size());
    // add the w, b, xh and c weights for each layer
    for (uint32_t lid {0}; lid < num_layers; ++lid) {
        m_weights.push_back(m_layers[lid].get_w().data());
        m_weights.push_back(m_layers[lid].get_b().data());
        m_weights.push_back(m_layers[lid].get_xh().data());
        m_weights.push_back(m_layers[lid].get_c().data());
    }

    // set the remaining members of the processor specification
    spec.type = NamConfig::NetType::eLSTM;
    spec.nlayers = num_layers;
    spec.hidden_size = static_cast<uint32_t>(m_head_weight.size());
    spec.nlocal_samples = 32u;
    spec.head_scale = &m_head_bias;
    spec.weights = reinterpret_cast<void const**>(m_weights.data());
}
