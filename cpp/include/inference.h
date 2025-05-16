#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>

#include "backends.h"

struct MetaDataConfig {
    int SampleRate;
    int MaxLenght;
    std::string Framework;
    std::unordered_map<int, std::string> Id2Label;

    MetaDataConfig(int SampleRate_, int MaxLenght_, std::string Framework_, std::unordered_map<int, std::string> Id2Label_){
        SampleRate = SampleRate_;
        MaxLenght = MaxLenght_;
        Framework = Framework_;
        Id2Label = Id2Label_;
    };
};

class Model{
    private:
        Backend_Type backend_type;
        std::unordered_map<int, std::string> id2label;

        Backend* backend;
    
    public:
        std::string metadata_path;    
        Device_Type device_type;

    public:
        Model(const std::string& metadata_path_, const MetaDataConfig& config, Device_Type device_type_): 
            device_type(device_type_), metadata_path(metadata_path_), id2label(config.Id2Label){

            if (StringToBackendType.find(config.Framework) != StringToBackendType.end()) {
                backend_type = StringToBackendType[config.Framework];
            } else {
                std::cerr << "Unknown framework type: " << config.Framework << std::endl;
                std::cout << "Using onnx as default backend" << std::endl;

                backend_type = ONNXRuntime; 
            }

            init_backend(backend_type);
        }
        ~Model(){
            release_backend();
        };
        std::string predict(std::vector<float> m_inputs);

    private:
        void init_backend(Backend_Type backend_type);
        std::vector<float> cpuSoftmax1D(const std::vector<float> logits);

        void release_backend();

        std::string getClass(int prediction){
            auto it = id2label.find(prediction);
            return it != id2label.end() ? it->second : "no class found";
        }
};
