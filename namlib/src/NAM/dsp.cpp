#include <algorithm> // std::max_element
#include <algorithm>
#include <cmath> // pow, tanh, expf
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <NAM/dsp.h>

nam::DSP::DSP(const double expected_sample_rate) :
    mExpectedSampleRate(expected_sample_rate) {
}

void nam::DSP::prewarm() {
    const int prewarmSamples = PrewarmSamples();
    if (prewarmSamples == 0)
        return;

    const size_t bufferSize = std::max(mMaxBufferSize, 1);
    std::vector<NAM_SAMPLE> inputBuffer, outputBuffer;
    inputBuffer.resize(bufferSize);
    outputBuffer.resize(bufferSize);
    for (auto it = inputBuffer.begin(); it != inputBuffer.end(); ++it) {
        (*it) = (NAM_SAMPLE)0.0;
    }

    NAM_SAMPLE* inputPtr = inputBuffer.data();
    NAM_SAMPLE* outputPtr = outputBuffer.data();
    int samplesProcessed = 0;
    while (samplesProcessed < prewarmSamples) {
        this->process(inputPtr, outputPtr, bufferSize);
        samplesProcessed += bufferSize;
    }
}

void nam::DSP::process(NAM_SAMPLE* input, NAM_SAMPLE* output, const int num_frames) {
    // Default implementation is the null operation
    for (int i = 0; i < num_frames; i++)
        output[i] = input[i];
}

double nam::DSP::GetLoudness() const {
    if (!HasLoudness()) {
        throw std::runtime_error("Asked for loudness of a model that doesn't know how loud it is!");
    }
    return mLoudness;
}

void nam::DSP::Reset(const double sampleRate, const int maxBufferSize) {
    // Some subclasses might want to throw an exception if the sample rate is "wrong".
    // This could be under a debugging flag potentially.
    mExternalSampleRate = sampleRate;
    mHaveExternalSampleRate = true;
    mMaxBufferSize = maxBufferSize;

    // Subclasses might also want to pre-warm, but let them call that themselves in case
    // they want to e.g. do some allocations first.
}
void nam::DSP::SetLoudness(const double loudness) {
    mLoudness = loudness;
    mHasLoudness = true;
}
