#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <string>

#include "utils.h"
#include "preprocessor.h"
#include "inference.h"

int main(int argc, char* argv[]) {
    if (argc != 4){
        std::cerr << "Usage: /path/to/file.wav /path/to/exported_model/metadata.json, device_type(0/1)" << std::endl;
        std::wcerr << "If you don't have exported_model folder, feel free to use our convertor (python/src/export.py)";
        return 1;
    }
    std::string audio_path = argv[1];
    std::string metadata_path = argv[2];
    Device_Type device_type = Device_Type(atoi(argv[3]));
    int sampleRate;

    std::vector<float> audio_data = loadAudio(audio_path, sampleRate);
    MetaDataConfig model_metadata = loadMetaData(metadata_path);

    FeaturesExtractor featuresextractor = FeaturesExtractor(sampleRate, true, model_metadata.MaxLenght);
    std::vector<float> processed_audio = featuresextractor.process(audio_data);


    Model model = Model(metadata_path, model_metadata, device_type);
    std::string class_name = model.predict(processed_audio);
    std::cout << "Predicted class: " <<class_name << std::endl;

    return 0;
}