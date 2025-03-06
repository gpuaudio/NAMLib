#pragma once
// LSTM implementation

#include <map>
#include <vector>

#include <nam_processor/NamSpecification.h>

#include "dsp.h"
#include "tensor_data.h"

namespace nam {
namespace lstm {
// A Single LSTM cell
// i input
// f forget
// g cell
// o output
// c cell state
// h hidden state
class LSTMCellData {
    // Parameters
    // xh -> ifgo
    // (dx+dh) -> (4*dh)
    MatrixData m_w;
    VectorData m_b;

    // State
    // Concatenated input and hidden state
    VectorData m_xh;
    // Input, Forget, Cell, Output gates
    VectorData m_ifgo;

    // Cell state
    VectorData m_c;

public:
    LSTMCellData(const int input_size, const int hidden_size, std::vector<float>::iterator& weights);

    MatrixData const& get_w() { return m_w; };
    VectorData const& get_b() { return m_b; };
    VectorData const& get_xh() { return m_xh; };
    VectorData const& get_c() { return m_c; };
};

/**
 * GPU implementation of the LSTM model
 */
class GPULongShortTermMemory : public DSP {
    std::vector<float const*> m_weights;

    VectorData m_head_weight;
    float m_head_bias;
    std::vector<LSTMCellData> m_layers;

protected:
    // Hacky, but a half-second seems to work for most models.
    int PrewarmSamples() override;

public:
    /**
     * @brief Constructor; set up the CPU data structures
     */
    GPULongShortTermMemory(const int num_layers, const int input_size, const int hidden_size, std::vector<float>& weights,
        const double expected_sample_rate = -1.0);

    /**
     * @brief Converts the CPU data structures into a processor specification
     * @param spec [out] specification to create the GPU processor from
     */
    void SetupProcessorData(NamConfig::Specification& spec);
};
}; // namespace lstm
}; // namespace nam
