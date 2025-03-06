#ifndef GPU_NEURAL_AMP_MODELER_INTERFACE_H
#define GPU_NEURAL_AMP_MODELER_INTERFACE_H

#include <nam_processor/NamSpecification.h>
#include <cstdint>

/**
 * Public interface for the Neural Amp Modeler
 */
class NeuralAmpModelerInterface {
public:
    /**
     * @brief Default destructor
     */
    virtual ~NeuralAmpModelerInterface() {};

    /**
     * @brief Process samples provided in input and write them output buffers.
     * @param input [in] pointer to the samples of the input audio data
     * @param output [in/out] pointer to the memory to write the output audio data to
     * @param num_frames [in] number of samples to process
     */
    virtual void process(float* input, float* output, const int num_frames) = 0;

    /**
     * @brief Get the client library ready for processing with the current configuration
     */
    virtual void arm() = 0;

    /**
     * @brief Clean up and get ready for destruction or re-configuration
     */
    virtual void disarm() = 0;

    /**
     * @brief Turn automatic buffer growth on or off.
     * @param enable [in] true to enable, false to disable
     */
    virtual void enable_buffer_growth(bool enable) = 0;

    /**
     * @brief Get the current latency introduced by double buffering
     * @return latency in number of samples
     */
    virtual uint32_t get_latency() = 0;
};

#endif // GPU_NEURAL_AMP_MODELER_INTERFACE_H
