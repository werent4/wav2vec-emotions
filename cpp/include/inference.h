#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>

#include "utils.h"
#include "backends.h"


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
                std::string errorMsg = "Unknown framework type: " + config.Framework;
                std::cerr << errorMsg << std::endl;
                throw std::runtime_error(errorMsg);
            }

            init_backend(backend_type);
            
            std::cout << "Successfully created model with following configuration:" << std::endl;
            printMetaData(config);

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
