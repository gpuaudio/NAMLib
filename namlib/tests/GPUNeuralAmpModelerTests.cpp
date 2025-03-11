#include <gtest/gtest.h>

#include <AudioFile/AudioFile.h>
#include "TestCommon.h"

#include <GPUCreate.h>
#include "../include/nam_processor/NamSpecification.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <numeric>
#include <random>
#include <string>
#include <sstream>
#include <vector>

namespace {

static constexpr char const* g_configs_path {"../deps/NAM_models/"};
static constexpr char const* g_lstm_configs_path {"../deps/LSTM_NAM_models/"};
static constexpr char const* g_audio_path {"../deps/audio/"};
static constexpr char const* g_default_config {"Tim R Fender TwinVerb Norm Bright.nam"};
static constexpr char const* g_default_lstm_config {"LSTM-2-020.nam"};
static constexpr char const* g_validation_path {"../deps/validation/"};

static constexpr uint32_t g_buffer_length {128u};
static constexpr uint32_t g_threads_per_block {256u};

static constexpr uint32_t g_lstm_buffer_length {128u};
static constexpr uint32_t g_lstm_threads_per_block {256u};

std::filesystem::path get_config_path(std::string filename) {
    return (std::filesystem::path(g_configs_path) / filename);
}

std::filesystem::path get_lstm_config_path(std::string filename) {
    return (std::filesystem::path(g_lstm_configs_path) / filename);
}

std::string get_default_config_path() {
    return (std::filesystem::path(g_configs_path) / std::filesystem::path(g_default_config)).string();
}

std::string get_default_lstm_config_path() {
    return (std::filesystem::path(g_lstm_configs_path) / std::filesystem::path(g_default_lstm_config)).string();
}

void get_all_wavs(std::vector<std::string>& wavs) {
    for (auto const& it : std::filesystem::directory_iterator(std::filesystem::path(g_audio_path))) {
        if (it.is_regular_file() && it.path().extension() == std::string(".wav")) {
            wavs.push_back(it.path().string());
        }
    }
}

void validateWithFiles(std::string to_validate_config, nam::DSP* to_validate_model, uint32_t const buffer_size, uint32_t const double_buffering_latency) {
    to_validate_model->Reset(to_validate_model->GetExpectedSampleRate(), buffer_size);

    for (uint32_t fid {0u};; ++fid) {
        auto validation_file = std::filesystem::path(g_validation_path) / (std::to_string(fid) + ".val");
        if (!std::filesystem::exists(validation_file))
            return;

        std::ifstream in(validation_file, std::ios::in | std::ios::binary);
        if (!in) {
            printf("Could not open \"%s\" for reading\n", validation_file.string().c_str());
            continue;
        }
        uint64_t reference_config_size;
        in.read(reinterpret_cast<char*>(&reference_config_size), sizeof(reference_config_size));
        std::string reference_config;
        reference_config.resize(reference_config_size);
        in.read(&reference_config[0], reference_config_size);

        if (to_validate_config != reference_config)
            continue;

        TestData input_buffer;
        input_buffer.read(in);
        TestData reference_output_buffer;
        reference_output_buffer.read(in);

        in.close();

        uint32_t nframes_total = static_cast<uint32_t>(input_buffer.m_nsamples);

        float* input_data = input_buffer.getChannel(0);

        TestData to_validate_output_buffer(1u, nframes_total, 0.0f, TestData::DataMode::Constant);
        float* to_validate_output_data = to_validate_output_buffer.getChannel(0);

        float tolerance {1e-5f};

        // ASSERT_FALSE(CompareBuffers(reference_output_buffer, 0u, to_validate_output_buffer, 0u, tolerance));

        printf("Validating file %s...", validation_file.string().c_str());

        uint32_t frame_offset {0u};
        uint32_t nlaunches = (nframes_total + buffer_size - 1) / buffer_size;
        for (uint32_t lid {0u}; lid < nlaunches; lid++) {
            uint32_t nframes = std::min(buffer_size, nframes_total - frame_offset);
            to_validate_model->process(input_data + frame_offset, to_validate_output_data + frame_offset, nframes);
            frame_offset += nframes;
        }

        bool success {true};
        if (nframes_total > double_buffering_latency) {
            success = CompareBuffers(reference_output_buffer, 0u, to_validate_output_buffer, double_buffering_latency, tolerance);
        }
        printf("%s [tolerance %f]\n", success ? " success!" : " failed!", tolerance);
        ASSERT_TRUE(success);
    }
}
} // namespace

TEST(NamLib, CreateDestroy) {
    std::unique_ptr<nam::DSP> namlib = createGpuProcessor(get_default_config_path().c_str(), g_buffer_length, g_threads_per_block);
    ASSERT_NE(namlib, nullptr);
}

TEST(NamLibLSTM, CreateDestroy) {
    std::unique_ptr<nam::DSP> namlib = createGpuProcessor(get_default_lstm_config_path().c_str(), g_lstm_buffer_length, g_lstm_threads_per_block);
    ASSERT_NE(namlib, nullptr);
}

TEST(NamLib, CreateArmDisarmDestroy) {
    constexpr uint32_t processor_nframes {g_buffer_length};

    std::unique_ptr<nam::DSP> namlib = createGpuProcessor(get_default_config_path().c_str(), g_buffer_length, g_threads_per_block);
    ASSERT_NE(namlib, nullptr);

    dynamic_cast<NeuralAmpModelerInterface*>(namlib.get())->arm();
    dynamic_cast<NeuralAmpModelerInterface*>(namlib.get())->disarm();
}

TEST(NamLibLSTM, CreateArmDisarmDestroy) {
    constexpr uint32_t processor_nframes {g_buffer_length};

    std::unique_ptr<nam::DSP> namlib = createGpuProcessor(get_default_lstm_config_path().c_str(), g_lstm_buffer_length, g_lstm_threads_per_block);
    ASSERT_NE(namlib, nullptr);

    dynamic_cast<NeuralAmpModelerInterface*>(namlib.get())->arm();
    dynamic_cast<NeuralAmpModelerInterface*>(namlib.get())->disarm();
}

TEST(NamLib, CreateArmProcessDisarmDestroy) {
    constexpr uint32_t nlaunches {150u};
    constexpr uint32_t nframes {g_buffer_length};
    constexpr uint32_t nchannels {1u};
    constexpr uint32_t nsamples {nlaunches * nframes};

    TestData in_buffer(nchannels, nsamples, 1.0f, TestData::DataMode::Sin);
    TestData out_buffer(nchannels, nsamples, 1.0f, TestData::DataMode::Sin);

    std::unique_ptr<nam::DSP> namlib = createGpuProcessor(get_default_config_path().c_str(), g_buffer_length, g_threads_per_block);
    ASSERT_NE(namlib, nullptr);

    dynamic_cast<NeuralAmpModelerInterface*>(namlib.get())->arm();
    for (uint32_t lid = 0u; lid < nlaunches; ++lid) {
        dynamic_cast<NeuralAmpModelerInterface*>(namlib.get())->process(in_buffer.getChannel(0) + lid * nframes, out_buffer.getChannel(0) + lid * nframes, nframes);
    }
    dynamic_cast<NeuralAmpModelerInterface*>(namlib.get())->disarm();
}

TEST(NamLibLSTM, CreateArmProcessDisarmDestroy) {
    constexpr uint32_t nlaunches {1u};
    constexpr uint32_t nframes {g_buffer_length};
    constexpr uint32_t nchannels {1u};
    constexpr uint32_t nsamples {nlaunches * nframes};

    TestData in_buffer(nchannels, nsamples, 1.0f, TestData::DataMode::Sin);
    TestData out_buffer(nchannels, nsamples, 0.0f, TestData::DataMode::Constant);

    std::unique_ptr<nam::DSP> namlib = createGpuProcessor(get_default_lstm_config_path().c_str(), g_lstm_buffer_length, g_lstm_threads_per_block);
    ASSERT_NE(namlib, nullptr);

    dynamic_cast<NeuralAmpModelerInterface*>(namlib.get())->arm();
    for (uint32_t lid = 0u; lid < nlaunches; ++lid) {
        dynamic_cast<NeuralAmpModelerInterface*>(namlib.get())->process(in_buffer.getChannel(0) + lid * nframes, out_buffer.getChannel(0) + lid * nframes, nframes);
    }
    // out_buffer.printNonZeros(std::cout);
    dynamic_cast<NeuralAmpModelerInterface*>(namlib.get())->disarm();
}

TEST(NamLib, CreateArmProcessPartialBuffersDisarmDestroy) {
    constexpr uint32_t nlaunches {150u};
    constexpr uint32_t nframes_max {g_buffer_length};
    constexpr uint32_t nchannels {1u};

    std::random_device dev;
    auto seed = dev();
    std::cout << "Using seed " << seed << std::endl;
    std::mt19937 rne(seed);

    std::uniform_int_distribution<std::mt19937::result_type> nframes_dist(1, nframes_max);

    std::vector<uint32_t> nframes_per_launch(nlaunches);
    uint32_t nframes_total {0u};
    for (uint32_t lid {0u}; lid < nlaunches; ++lid) {
        nframes_per_launch[lid] = nframes_dist(rne);
        nframes_total += nframes_per_launch[lid];
    }

    TestData in_buffer(nchannels, nframes_total, 1.0f, TestData::DataMode::Sin);
    TestData out_buffer(nchannels, nframes_total, 1.0f, TestData::DataMode::Sin);

    std::unique_ptr<nam::DSP> namlib = createGpuProcessor(get_default_config_path().c_str(), g_buffer_length, g_threads_per_block);
    ASSERT_NE(namlib, nullptr);

    dynamic_cast<NeuralAmpModelerInterface*>(namlib.get())->arm();
    uint32_t frame_offset {0u};
    for (uint32_t lid = 0u; lid < nlaunches; ++lid) {
        dynamic_cast<NeuralAmpModelerInterface*>(namlib.get())->process(in_buffer.getChannel(0) + frame_offset, out_buffer.getChannel(0) + frame_offset, nframes_per_launch[lid]);
        frame_offset += nframes_per_launch[lid];
    }
    dynamic_cast<NeuralAmpModelerInterface*>(namlib.get())->disarm();
}

TEST(NamLib, ValidateResultFromFile) {
    std::vector<std::string> configs = {
        g_default_config,
        "George B V4 Countess 300eps.nam",
        "Helga B 5150 BlockLetter - NoBoost.nam",
        "Helga B 6505+ Green ch - MXR Drive.nam",
        "Helga B 6505+ Green ch - NoBoost.nam",
        "Helga B 6505+ Red ch - MXR Drive V2.nam",
        "Helga B 6534+ MXR M77 - Helga Behrens.nam",
        "Phillipe P Bug333-Clean-Cab-ESR0,007.nam",
        "Phillipe P Bug6262-Crunch-NoDrive-Cab-ESR0,004.nam",
        "Phillipe P JVM-OD2-RD-NoDrive-Cab-ESR0,006.nam",
        "Phillipe P VOXAC15-TopBoost.nam",
        "Tim R JCM2000 Crunch 805'd.nam"};

    constexpr bool test_all_supported {true};
    constexpr bool test_buffer_growth_and_device_iteration {true};
    constexpr bool test_buffer_partially_filled {true};
#if defined(GPU_AUDIO_MAC)
    std::pair<uint32_t, uint32_t> buflen_nthreads[6] = {
        {64u, 64u},
        {64u, 128u},
        {64u, 256u},
        {128u, 128u},
        {128u, 256u},
        {256u, 256u}};
#else
    std::pair<uint32_t, uint32_t> buflen_nthreads[10] = {
        {64u, 64u},
        {64u, 128u},
        {64u, 256u},
        {64u, 512u},
        {128u, 128u},
        {128u, 256u},
        {128u, 512u},
        {256u, 256u},
        {256u, 512u},
        {512u, 512u}};
#endif

    for (auto const& config : configs) {
        printf("Testing config %s\n", config.c_str());
        if (!test_all_supported) {
            std::unique_ptr<nam::DSP> namlib = createGpuProcessor(get_config_path(config).string().c_str(), g_buffer_length, g_threads_per_block);
            validateWithFiles(config, namlib.get(), g_buffer_length, g_buffer_length);
            if (test_buffer_growth_and_device_iteration) {
                namlib = createGpuProcessor(get_config_path(config).string().c_str(), g_buffer_length, g_threads_per_block);
                validateWithFiles(config, namlib.get(), 2 * g_buffer_length, 2 * g_buffer_length);
            }
            continue;
        }
        for (auto const& [buflen, nthreads] : buflen_nthreads) {
            printf("Testing buflen %u; nthreads %u\n", buflen, nthreads);
            std::unique_ptr<nam::DSP> namlib = createGpuProcessor(get_config_path(config).string().c_str(), buflen, nthreads);
            printf("Testing full buffers\n");
            validateWithFiles(config, namlib.get(), buflen, buflen);
            if (test_buffer_growth_and_device_iteration) {
                // we have to create a new one to clean the state of prev. execution
                // in order to compare with the validation files from the CPU version
                namlib = createGpuProcessor(get_config_path(config).string().c_str(), buflen, nthreads);
                // call with 2*buflen, to trigger buffer growth and device iteration
                printf("Testing device buffer iteration\n");
                validateWithFiles(config, namlib.get(), 2u * buflen, 2u * buflen);
            }
            if (test_buffer_partially_filled) {
                // we have to create a new one to clean the state of prev. execution
                // in order to compare with the validation files from the CPU version
                namlib = createGpuProcessor(get_config_path(config).string().c_str(), buflen, nthreads);
                // call with buflen / 2 to check partially filled buffers
                printf("Testing partially filled buffers\n");
                validateWithFiles(config, namlib.get(), buflen >> 1, buflen);
            }
        }
    }
}

TEST(NamLibLSTM, ValidateResultFromFile) {
    std::vector<std::string> configs = {
        g_default_lstm_config,
        "LSTM-1-016.nam",
        "LSTM-2-008.nam",
        "LSTM-3-028.nam",
        "LSTM-4-012.nam"};

    constexpr bool test_all_supported {true};
    constexpr bool test_buffer_growth_and_device_iteration {true};
    constexpr bool test_buffer_partially_filled {true};
    std::pair<uint32_t, uint32_t> buflen_nthreads[1] = {
        {128u, 256u}};

    for (auto const& config : configs) {
        printf("Testing config %s\n", config.c_str());
        if (!test_all_supported) {
            std::unique_ptr<nam::DSP> namlib = createGpuProcessor(get_lstm_config_path(config).string().c_str(), g_lstm_buffer_length, g_lstm_threads_per_block);
            validateWithFiles(config, namlib.get(), g_lstm_buffer_length, g_lstm_buffer_length);
            if (test_buffer_growth_and_device_iteration) {
                namlib = createGpuProcessor(get_lstm_config_path(config).string().c_str(), g_lstm_buffer_length, g_lstm_threads_per_block);
                validateWithFiles(config, namlib.get(), 2 * g_lstm_buffer_length, 2 * g_lstm_buffer_length);
            }
            continue;
        }
        for (auto const& [buflen, nthreads] : buflen_nthreads) {
            printf("Testing buflen %u; nthreads %u\n", buflen, nthreads);
            std::unique_ptr<nam::DSP> namlib = createGpuProcessor(get_lstm_config_path(config).string().c_str(), buflen, nthreads);
            printf("Testing full buffers\n");
            validateWithFiles(config, namlib.get(), buflen, buflen);
            if (test_buffer_growth_and_device_iteration) {
                // we have to create a new one to clean the state of prev. execution
                // in order to compare with the validation files from the CPU version
                namlib = createGpuProcessor(get_lstm_config_path(config).string().c_str(), buflen, nthreads);
                // call with 2*buflen, to trigger buffer growth and device iteration
                printf("Testing device buffer iteration\n");
                validateWithFiles(config, namlib.get(), 2 * buflen, 2 * buflen);
            }
            if (test_buffer_partially_filled) {
                // we have to create a new one to clean the state of prev. execution
                // in order to compare with the validation files from the CPU version
                namlib = createGpuProcessor(get_lstm_config_path(config).string().c_str(), buflen, nthreads);
                // call with buflen / 2 to check partially filled buffers
                printf("Testing partially filled buffers\n");
                validateWithFiles(config, namlib.get(), buflen >> 1, buflen);
            }
        }
    }
}

bool processWav(std::string const& config, std::string const& input_wav, std::string const& output_wav, bool async) {
    // load input wav
    AudioFile<float> input;
    if (!input.load(input_wav)) {
        return false;
    }

    // create processor with nframes_max samples per buffer and sync execution
    constexpr uint32_t nframes_max {g_buffer_length};
    auto namlib = createGpuProcessor(config.c_str(), g_buffer_length, g_threads_per_block);
    if (namlib == nullptr) {
        return false;
    }

    int nframes_total = input.getNumSamplesPerChannel();

    std::vector<float> out_data(nframes_total, 0);
    // call process with junks of input (one internal process call at a time)
    for (int cursor = 0; cursor < nframes_total; cursor += nframes_max) {
        // process channel 0 of the input - processor can only do mono
        float* in_pointer = input.samples[0].data() + cursor;
        float* out_pointer = out_data.data() + cursor;

        uint32_t nframes = std::min<uint32_t>(nframes_max, nframes_total - cursor);
        namlib->process(in_pointer, out_pointer, nframes);
    }
    // call process with full input (nframes_total + nframes - 1 / nframes internal process calls)
    // namlib->process(input.samples[0].data(), out_data.data(), nframes_total);

    // write the output
    AudioFile<float> output {input};
    output.setNumChannels(1);

    std::fill_n(std::begin(output.samples[0]), output.getNumSamplesPerChannel(), 0.f);
    output.samples[0] = std::move(out_data);

    if (!output.save(output_wav)) {
        return false;
    }
    return true;
}

TEST(NamLib, ProcessAudioFileWithSupportedConfigs) {
    std::vector<std::string> configs = {
        "George B V4 Countess 300eps.nam",
        "Helga B 5150 BlockLetter - NoBoost.nam",
        "Helga B 6505+ Green ch - MXR Drive.nam",
        "Helga B 6505+ Green ch - NoBoost.nam",
        "Helga B 6505+ Red ch - MXR Drive V2.nam",
        "Helga B 6534+ MXR M77 - Helga Behrens.nam",
        "Phillipe P Bug333-Clean-Cab-ESR0,007.nam",
        "Phillipe P Bug6262-Crunch-NoDrive-Cab-ESR0,004.nam",
        "Phillipe P JVM-OD2-RD-NoDrive-Cab-ESR0,006.nam",
        "Phillipe P VOXAC15-TopBoost.nam",
        "Tim R JCM2000 Crunch 805'd.nam"};

    std::vector<std::string> wavs;
    get_all_wavs(wavs);

    std::filesystem::path output_path;
    if (!wavs.empty()) {
        output_path = std::filesystem::path(g_audio_path) / "out";
        std::filesystem::create_directory(output_path);
    }
    for (auto& config : configs) {
        for (auto const& wav : wavs) {
            std::string configname = std::filesystem::path(config).replace_extension().filename().string();
            auto illegal_char = [](auto const& c) {
                return ::isspace(c) || !::isalnum(c);
            };
            configname.erase(std::remove_if(configname.begin(), configname.end(), illegal_char), configname.end());

            std::string output_wav = std::filesystem::path(wav).filename().replace_extension().string() + "_" + configname + "_out.wav";

            ASSERT_TRUE(processWav(get_config_path(config).string(), wav, (output_path / output_wav).string(), false));
        }
    }
}

TEST(NamLib, GetLatencyWithBufferGrowth) {
    constexpr uint32_t nchannels {1u};
    constexpr uint32_t nsamples_max {8192u};

    TestData in_out_buffer(nchannels, nsamples_max, 1.0f, TestData::DataMode::Sin);
    float* buffer_ptr = in_out_buffer.getChannel(0);

    uint32_t constexpr initial_buffer_size {128u};
    std::unique_ptr<nam::DSP> namlib = createGpuProcessor(get_default_config_path().c_str(), initial_buffer_size, 256u);
    ASSERT_NE(namlib, nullptr);

    auto NAMLib_interface = dynamic_cast<NeuralAmpModelerInterface*>(namlib.get());
    uint32_t nsamples = NAMLib_interface->get_latency();
    ASSERT_EQ(nsamples, initial_buffer_size);

    // process half a buffer - no latency change
    namlib->process(buffer_ptr, buffer_ptr, nsamples >> 1u);
    ASSERT_EQ(nsamples, NAMLib_interface->get_latency());

    // process a full buffer - no latency change
    namlib->process(buffer_ptr, buffer_ptr, nsamples);
    ASSERT_EQ(nsamples, NAMLib_interface->get_latency());

    // process a full buffer plus one sample - latency doubles as buffer size doubles
    namlib->process(buffer_ptr, buffer_ptr, nsamples + 1);
    ASSERT_EQ(nsamples <<= 1u, NAMLib_interface->get_latency());

    // process a full buffer - no latency change
    namlib->process(buffer_ptr, buffer_ptr, nsamples);
    ASSERT_EQ(nsamples, NAMLib_interface->get_latency());

    // turn off buffer growth
    NAMLib_interface->enable_buffer_growth(false);

    // process a full buffer plus one sample - no latency change
    namlib->process(buffer_ptr, buffer_ptr, nsamples + 1);
    ASSERT_EQ(nsamples, NAMLib_interface->get_latency());

    // turn on buffer growth
    NAMLib_interface->enable_buffer_growth(true);

    // process a full buffer plus one sample - latency doubles as buffer size doubles
    namlib->process(buffer_ptr, buffer_ptr, nsamples + 1);
    ASSERT_EQ(nsamples <<= 1, NAMLib_interface->get_latency());

    // process three full buffers plus one sample - latency quadruples as buffer size doubles
    namlib->process(buffer_ptr, buffer_ptr, 3u * nsamples + 1);
    ASSERT_EQ(nsamples <<= 2, NAMLib_interface->get_latency());

    // process a full buffer plus one sample - latency doubles as buffer size doubles
    namlib->process(buffer_ptr, buffer_ptr, nsamples + 1);
    ASSERT_EQ(nsamples <<= 1, NAMLib_interface->get_latency());

    // process a full buffer plus one sample - no change as we reached the max buffer size / latency of 4096
    namlib->process(buffer_ptr, buffer_ptr, nsamples + 1);
    ASSERT_EQ(nsamples, NAMLib_interface->get_latency());
}
