#include "AudioFile.h"

#include <GPUCreate.h>

#include <cstdint>
#include <cstdio>

bool processWav(std::string const& config, std::string const& input_wav, std::string const& output_wav) {
    // load input wav from `input_wav` to an `AudioFile`
    AudioFile<float> input;
    if (!input.load(input_wav)) {
        return false;
    }

    // create the processor
    constexpr uint32_t nframes_max {64u};
    constexpr uint32_t nthreads_per_block {128u};
    auto namlib = createGpuProcessor(config.c_str(), nframes_max, nthreads_per_block);
    if (namlib == nullptr) {
        return false;
    }

    int nframes_total = input.getNumSamplesPerChannel();

    std::vector<float> out_data(nframes_total, 0);
    // call process with `nframes_max`-sized chunks of input (one internal process call at a time)
    for (int cursor = 0; cursor < nframes_total; cursor += nframes_max) {
        // process channel 0 of the input; the processor can only do mono
        float* in_pointer = input.samples[0].data() + cursor;
        float* out_pointer = out_data.data() + cursor;

        uint32_t nframes = std::min<uint32_t>(nframes_max, nframes_total - cursor);
        namlib->process(in_pointer, out_pointer, nframes);
    }

    // write the output into an `AudioFile`
    AudioFile<float> output {input};
    output.setNumChannels(1);

    std::fill_n(std::begin(output.samples[0]), output.getNumSamplesPerChannel(), 0.f);
    output.samples[0] = std::move(out_data);

    // save `AudiFile` to `output_wav`
    if (!output.save(output_wav)) {
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Error: usage nam_process_file.exe [config.nam] [input.wav]\n");
        return 1;
    }
    std::string configpath(argv[1]);
    std::string infilepath(argv[2]);

    // create filename and path of the output wav
    std::string configfile = std::filesystem::path(configpath).replace_extension().filename().string();
    auto illegal_char = [](auto const& c) {
        return ::isspace(c) || !::isalnum(c);
    };
    configfile.erase(std::remove_if(configfile.begin(), configfile.end(), illegal_char), configfile.end());
    std::string outputfile = std::filesystem::path(infilepath).filename().replace_extension().string() + "_" + configfile + ".wav";
    std::string outfilepath = (std::filesystem::path(infilepath).parent_path() / outputfile).string();

    // process the audio file
    bool success = processWav(configpath, infilepath, outfilepath);

    printf("%s\n", success ? "Success" : "Something went wrong");

    return 0;
}
