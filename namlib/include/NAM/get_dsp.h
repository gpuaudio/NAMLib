#include <fstream>

#include "dsp.h"

namespace nam {
// Get NAM from a .nam file at the provided location
std::unique_ptr<DSP> get_dsp(const std::filesystem::path config_filename, uint32_t nframes, uint32_t nthreads);

// Get NAM from a provided configuration struct
std::unique_ptr<DSP> get_dsp(dspData& conf, uint32_t nframes, uint32_t nthreads);

// Get NAM from a provided .nam file path and store its configuration in the provided conf
std::unique_ptr<DSP> get_dsp(const std::filesystem::path config_filename, dspData& returnedConfig, uint32_t nframe, uint32_t nthreads);
}; // namespace nam
