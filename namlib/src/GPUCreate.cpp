#include <GPUCreate.h>

#include "GPUNeuralAmpModeler.h"

std::unique_ptr<nam::DSP> createGpuProcessor(char const* config, uint32_t nframes_max, uint32_t nthreads_per_block) {
    return std::move(nam::get_dsp(config, nframes_max, nthreads_per_block));
}
