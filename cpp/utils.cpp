#include <sndfile.h>
#include "utils.h"

std::vector<float> loadAudio(const std::string& path, int& sampleRate){
    SF_INFO sfinfo;
    sfinfo.format = 0;

    
    SNDFILE* file = sf_open(path.c_str(), SFM_READ, &sfinfo);
    if (!file) {
        throw std::runtime_error("Failde to open file: " + std::string(sf_strerror(nullptr)));
    }

    sampleRate = sfinfo.samplerate;

    std::vector<float> buffer(sfinfo.frames * sfinfo.channels); // creates buffer to store data from file
    sf_read_float(file, buffer.data(), buffer.size()); // read file
    sf_close(file);

    if (sfinfo.channels > 1) { // converts stereo audio to mono. (just mean from channels)
        std::vector<float> monoBuffer(sfinfo.frames);
        for (int i = 0; i < sfinfo.frames; i++) {
            float sum = 0.0f;
            for (int c = 0; c < sfinfo.channels; c++) {
                sum += buffer[i * sfinfo.channels + c];
            }
            monoBuffer[i] = sum / sfinfo.channels;
        }
        return monoBuffer;
    }
    
    return buffer;
}

std::string getModelPath(const std::string& metadataPath, Backend_Type backend_type){
    std::filesystem::path modelPath;
    std::filesystem::path metadata(metadataPath);
    std::filesystem::path directory = metadata.parent_path();
    if (backend_type == ONNXRuntime){
        modelPath = directory / "model.onnx";
    }
    else {
        throw std::runtime_error("Unsupported backend type for model path generation");
    }
    return modelPath.string();
}
