#include <sndfile.h>
#include <json/json.h>
#include <fstream>

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

MetaDataConfig loadMetaData(const std::string& path){
    if (!std::filesystem::exists(path)){
        throw std::filesystem::filesystem_error(
            "file not found", 
            path, 
            std::make_error_code(std::errc::no_such_file_or_directory)
        );
    }

    Json::Value configuration;
    std::ifstream ifs;
    ifs.open(path);

    Json::CharReaderBuilder builder;
    builder["collectComments"] = false;
    JSONCPP_STRING errs;

    if (!parseFromStream(builder, ifs, &configuration, &errs)) {
        std::cout << errs << std::endl;
        throw std::runtime_error("Failed to parse JSON configuration");
    }

    int sampleRate = configuration["sample_rate"].asInt();
    int maxLength = configuration["max_length"].asInt();
    std::string framework = configuration["framework"].asString();

    std::unordered_map<int, std::string> id2label;
    const Json::Value& labels = configuration["id2label"];

    for (Json::ValueConstIterator it = labels.begin(); it != labels.end(); ++it) {
        int key = std::stoi(it.key().asString());
        std::string value = (*it).asString();
        id2label[key] = value;
    }

    return MetaDataConfig(sampleRate, maxLength, framework, id2label);
}

void printMetaData(MetaDataConfig metadata){
    std::cout << "Framework: " << metadata.Framework << std::endl;
    std::cout << "Sample rate: " << metadata.SampleRate << std::endl;
    std::cout << "Max intput length (s): " << metadata.MaxLenght / metadata.SampleRate << std::endl;
    std::cout << "Labels mapping: ";
    printMap(metadata.Id2Label);

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
