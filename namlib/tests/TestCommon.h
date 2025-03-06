#ifndef GPUA_NAM_LIB_TEST_COMMON_H
#define GPUA_NAM_LIB_TEST_COMMON_H

#include <GPUCreate.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>

struct TestData {
    uint64_t m_nchannels;
    uint64_t m_nsamples;

    float** m_data;

    enum class DataMode {
        Constant,
        Sin,
        Random,
        Saw
    };

    TestData() :
        m_nchannels {0u},
        m_nsamples {0u},
        m_data {nullptr} {}

    TestData(uint64_t nchannels, uint64_t nsamples, float val, DataMode mode = DataMode::Constant) :
        m_nchannels {nchannels},
        m_nsamples {nsamples} {
        m_data = new float*[m_nchannels];

        std::random_device rd;
        std::mt19937 e(rd());
        std::uniform_real_distribution<float> dist(-1.f, 1.0f);

        for (int ch = 0; ch < m_nchannels; ++ch) {
            m_data[ch] = new float[m_nsamples];
            switch (mode) {
            case DataMode::Constant:
                std::fill_n(m_data[ch], nsamples, val);
                break;
            case DataMode::Sin:
                for (uint32_t i = 0; i < nsamples; ++i) {
                    m_data[ch][i] = val * sin(ch * 0.1f + i * 0.02f);
                }
                break;
            case DataMode::Random:
                for (uint32_t i = 0; i < nsamples; ++i) {
                    m_data[ch][i] = val * dist(e);
                }
                break;
            case DataMode::Saw:
                for (uint32_t i = 0; i < nsamples; ++i) {
                    if (i % int(val + 1) == 0) {
                        m_data[ch][i] = 0.0f;
                        continue;
                    }
                    m_data[ch][i] = 1.0f / val + m_data[ch][i - 1];
                }
                break;
            }
        }
    }

    TestData(TestData const& other) :
        m_nchannels {other.m_nchannels},
        m_nsamples {other.m_nsamples} {
        m_data = new float*[m_nchannels];

        for (int ch = 0; ch < m_nchannels; ++ch) {
            m_data[ch] = new float[m_nsamples];
            std::memcpy(m_data[ch], other.m_data[ch], m_nsamples * sizeof(float));
        }
    }

    ~TestData() {
        reset();
    }

    void reset() {
        if (m_data != nullptr) {
            for (int ch = 0; ch < m_nchannels; ++ch) {
                delete[] m_data[ch];
                m_data[ch] = nullptr;
            }

            delete[] m_data;
            m_data = nullptr;
        }
        m_nchannels = 0u;
        m_nsamples = 0u;
    }

    void read(std::ifstream& in) {
        reset();

        in.read(reinterpret_cast<char*>(&m_nchannels), sizeof(m_nchannels));
        m_data = new float*[m_nchannels];

        in.read(reinterpret_cast<char*>(&m_nsamples), sizeof(m_nsamples));

        for (int ch = 0; ch < m_nchannels; ++ch) {
            m_data[ch] = new float[m_nsamples];
            in.read(reinterpret_cast<char*>(m_data[ch]), m_nsamples * sizeof(float));
        }
    }
    void read(char const* path) {
        std::ifstream in(path, std::ios::in | std::ios::binary);
        if (!in) {
            throw std::runtime_error("Could not open " + std::string(path) + " for reading");
        }
        read(in);
        in.close();
    }

    void write(std::ofstream& out) {
        out.write(reinterpret_cast<char const*>(&m_nchannels), sizeof(m_nchannels));

        out.write(reinterpret_cast<char const*>(&m_nsamples), sizeof(m_nsamples));

        for (int ch = 0; ch < m_nchannels; ++ch) {
            out.write(reinterpret_cast<char const*>(m_data[ch]), m_nsamples * sizeof(float));
        }
    }
    void write(char const* path) {
        std::ofstream out(path, std::ios::out | std::ios::binary);
        if (!out) {
            throw std::runtime_error("Could not open " + std::string(path) + " for writing");
        }
        write(out);
        out.close();
    }

    TestData& operator=(TestData const& other) {
        if (this != &other) {
            if (m_data != nullptr) {
                for (int ch = 0; ch < m_nchannels; ++ch) {
                    delete[] m_data[ch];
                    m_data[ch] = nullptr;
                }

                delete[] m_data;
                m_data = nullptr;
            }

            m_nchannels = other.m_nchannels;
            m_nsamples = other.m_nsamples;
            m_data = new float*[m_nchannels];

            for (int ch = 0; ch < m_nchannels; ++ch) {
                m_data[ch] = new float[m_nsamples];
                std::memcpy(m_data[ch], other.m_data[ch], m_nsamples * sizeof(float));
            }
        }
        return *this;
    }

    float* const* operator()() {
        return m_data;
    }

    float& at(uint64_t channel, uint64_t sample) {
        if (channel >= m_nchannels || sample >= m_nsamples)
            throw std::runtime_error("TestData out-of-bounds access");

        return m_data[channel][sample];
    }

    float const& at(uint64_t channel, uint64_t sample) const {
        if (channel >= m_nchannels || sample >= m_nsamples)
            throw std::runtime_error("TestData out-of-bounds access");

        return m_data[channel][sample];
    }

    bool operator==(TestData const& other) const {
        if (this->m_nchannels != other.m_nchannels || this->m_nsamples != other.m_nsamples)
            return false;
        bool match {true};
        for (int ch = 0; ch < m_nchannels; ++ch) {
            for (int s = 0; s < m_nsamples; ++s) {
                if (std::abs(this->m_data[ch][s] - other.m_data[ch][s]) > 0.0001) {
                    match = false;
                    // printf("Mismatch at channel %d, sample %d: %f != %f\n", ch, s, this->m_data[ch][s], other.m_data[ch][s]);
                }
            }
        }
        return match;
    }

    bool operator!=(TestData const& other) const {
        return !(*this == other);
    }

    float* getChannel(uint64_t channel) {
        if (channel >= m_nchannels)
            throw std::runtime_error("TestData out-of-bounds access");

        return m_data[channel];
    }

    float const* getChannel(uint64_t channel) const {
        if (channel >= m_nchannels)
            throw std::runtime_error("TestData out-of-bounds access");

        return m_data[channel];
    }

    void setChannel(uint64_t channel, float* data) {
        if (channel >= m_nchannels)
            throw std::runtime_error("TestData out-of-bounds access");

        std::memcpy(m_data[channel], data, m_nsamples * sizeof(float));
    }

    void shiftChannel(uint64_t channel, uint64_t delta) {
        if (channel >= m_nchannels)
            throw std::runtime_error("TestData out-of-bounds access");
        if (delta >= m_nsamples)
            throw std::runtime_error("Shift-delta too large\n");

        std::memcpy(m_data[channel] + delta, m_data[channel], (m_nsamples - delta) * sizeof(float));
        std::fill_n(m_data[channel], delta, 0.0f);
    }

    void printNonZeros(std::ostream& out) const {
        for (uint32_t ch = 0; ch < m_nchannels; ++ch) {
            for (int s = 0; s < m_nsamples; ++s) {
                if (std::fabs(m_data[ch][s]) > 0.0001) {
                    out << "(" << ch << ", " << s << ") = " << m_data[ch][s] << "\n";
                }
            }
        }
        out << std::flush;
    }
};

bool CompareBuffers(TestData const& lhs, uint32_t lhs_off, TestData const& rhs, uint32_t rhs_off, float const tol = 1e-6f) {
    if (lhs.m_nchannels != rhs.m_nchannels || lhs.m_nsamples != rhs.m_nsamples)
        return false;

    bool match {true};
    for (uint64_t ch = 0u; ch < lhs.m_nchannels; ++ch) {
        float const* lhs_data = lhs.getChannel(ch);
        float const* rhs_data = rhs.getChannel(ch);
        for (uint32_t s = 0u; s < lhs.m_nsamples - std::max(lhs_off, rhs_off); ++s) {
            // printf("%u \t%f \t%f \n", s, lhs_data[s + lhs_off], rhs_data[s + rhs_off]);
            if (std::abs(lhs_data[s + lhs_off] - rhs_data[s + rhs_off]) > tol) {
                match = false;
                // printf("%u \t%f \t%f \n", s, lhs_data[s + lhs_off], rhs_data[s + rhs_off]);
                break;
            }
        }
        if (!match) {
            break;
        }
    }
    return match;
}

inline std::ostream& operator<<(std::ostream& os, const TestData& data) {
    data.printNonZeros(os);
    return os;
}

struct ChannelAccess : public TestData {
    TestData* m_buffer {nullptr};

    ChannelAccess(TestData* buffer, uint32_t channel = 0) :
        m_buffer {buffer} {
        m_nchannels = 1u;
        m_nsamples = m_buffer->m_nsamples;
        setChannel(channel);
    }

    ~ChannelAccess() {
        m_data = nullptr;
    }

    void setChannel(uint32_t channel) {
        m_data = &m_buffer->m_data[channel % m_buffer->m_nchannels];
    }

    float& at(uint64_t sample) {
        if (sample >= m_nsamples)
            throw std::runtime_error("ChannelAccess::at: out-of-bounds access");

        return m_data[0][sample];
    }

    float const& at(uint64_t sample) const {
        if (sample >= m_nsamples)
            throw std::runtime_error("ChannelAccess::at: out-of-bounds access");

        return m_data[0][sample];
    }
};

struct ChannelReader : public TestData {
    TestData const* m_buffer {nullptr};

    ChannelReader(TestData const* buffer, uint32_t channel = 0) :
        m_buffer {buffer} {
        m_nchannels = 1u;
        m_nsamples = m_buffer->m_nsamples;
        setChannel(channel);
    }

    ~ChannelReader() {
        m_data = nullptr;
    }

    void setChannel(uint32_t channel) {
        m_data = &m_buffer->m_data[channel % m_buffer->m_nchannels];
    }

    float const& at(uint64_t sample) const {
        if (sample >= m_nsamples)
            throw std::runtime_error("ChannelReader::at: out-of-bounds access");

        return m_data[0][sample];
    }
};

#endif // GPUA_NAM_LIB_TEST_COMMON_H
