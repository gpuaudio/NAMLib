#pragma once

#include <NAM/dsp.h>
#include "NeuralAmpModelerInterface.h"

#include <cstdint>

/**
 * @brief Create an instance of the GPU Neural Amp Modeler
 * @param config [in] path to a NAM configuration file
 * @param nframes_max [in] capacity of the processing-buffer per channel
 * @param nthreads_per_block [in] Number of threads per block used in the GPU process function
 * @return nam::DSP pointer to the created GPU Neural Amp Modeler instance
 */
std::unique_ptr<nam::DSP> createGpuProcessor(char const* config, uint32_t nframes_max = 64u, uint32_t nthreads_per_block = 128u);
