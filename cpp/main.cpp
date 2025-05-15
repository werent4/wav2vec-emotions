#include <iostream>
#include <vector>
#include <algorithm>
#include <preprocessor.h>
#include <utils.h>


int main(int argc, char* argv[]) {
    if (argc != 3){
        std::cerr << "Usage: /path/to/file.wav /path/to/onnx_model/metadata.json" << std::endl;
        std::wcerr << "If you don't have onnx_model folder, feel free to use our convertor (python/src/export.py)";
        return 1;
    }
    std::string path = argv[1];
    int sampleRate;

    std::vector<float> audio_data = loadAudio(path, sampleRate);
    print1DVector(audio_data);
    std::cout << audio_data.size() << std::endl;
    std::cout << sampleRate << std::endl;

    return 0;
}