#pragma once

#include <string>
#include <vector>

#include <nam_processor/NamSpecification.h>

#include "dsp.h"

class VectorData {
    std::vector<float> m_data;

public:
    void resize(uint64_t new_size) {
        m_data.resize(new_size);
    }

    uint64_t size() const {
        return m_data.size();
    }

    float& operator()(uint64_t pos) {
        return m_data.at(pos);
    }
    float const& operator()(uint64_t pos) const {
        return m_data.at(pos);
    }

    float* data() {
        return m_data.data();
    }
    float const* data() const {
        return m_data.data();
    }
};

class MatrixData {
    uint64_t nrows {0u};
    uint64_t ncols {0u};
    std::vector<float> m_data;

public:
    void resize(uint64_t new_num_rows, uint64_t new_num_cols) {
        nrows = new_num_rows;
        ncols = new_num_cols;
        m_data.resize(new_num_rows * new_num_cols);
    }

    uint64_t rows() const {
        return nrows;
    }

    uint64_t cols() const {
        return ncols;
    }

    float& operator()(uint64_t row, uint64_t col) {
        return m_data.at(col * nrows + row);
    }
    float const& operator()(uint64_t row, uint64_t col) const {
        return m_data.at(col * nrows + row);
    }

    float* data() {
        return m_data.data();
    }
    float const* data() const {
        return m_data.data();
    }
};
