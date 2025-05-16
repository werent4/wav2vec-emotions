#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <string>

#include "utils.h"
#include "preprocessor.h"
#include "inference.h"

// TODO: this should be parsed from metadata.json
#define MD_SAMPLE_RATE 16000
#define MD_MAX_LENGHT 80000
#define FRAMEWORK "onnx"

std::unordered_map<int, std::string> ID2LABEL = {
    {0, "angry"},
    {1, "sad"},
    {2, "neutral"},
    {3, "happy"},
    {4, "excited"},
    {5, "frustrated"},
    {6, "fear"},
    {7, "surprise"},
    {8, "disgust"},
    {9, "unknown"}
};

int main(int argc, char* argv[]) {
    if (argc != 3){
        std::cerr << "Usage: /path/to/file.wav /path/to/exported_model/metadata.json" << std::endl;
        std::wcerr << "If you don't have exported_model folder, feel free to use our convertor (python/src/export.py)";
        return 1;
    }
    std::string audio_path = argv[1];
    std::string model_path = argv[2];
    int sampleRate;

    std::vector<float> audio_data = loadAudio(audio_path, sampleRate);
    std::cout << audio_data.size() << std::endl;
    std::cout << sampleRate << std::endl;

    FeaturesExtractor featuresextractor = FeaturesExtractor(sampleRate);
    std::vector<float> processed_audio = featuresextractor.process(audio_data);

    MetaDataConfig config = MetaDataConfig(MD_SAMPLE_RATE, MD_MAX_LENGHT, FRAMEWORK, ID2LABEL);
    Model model = Model(model_path, config, Device_Type(1));
    std::string class_name = model.predict(processed_audio);
    std::cout << class_name << std::endl;

    return 0;
}