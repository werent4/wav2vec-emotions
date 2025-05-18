#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>

#include "utils.h"
#include "backends.h"

/**
 * @description: Main inference class for emotion recognition
 */	
class Model{
    private:
        /**
         * @description: Type of backend to use for inference
         */
        Backend_Type backend_type;

        /**
         * @description: Mapping from numerical class ID to emotion label
         */
        std::unordered_map<int, std::string> id2label;

        /**
         * @description: Backend instance for model inference
         */
        Backend* backend;
    
    public:
        /**
         * @description: Path to model metadata file
         */
        std::string metadata_path;    

        /**
         * @description: Device (CPU/GPU) on which to run inference
         */
        Device_Type device_type;

    public:
        /**
         * @description:                        Constructor for Model class
         * @param {string&} metadata_path_      Path to model metadata file
         * @param {MetaDataConfig&} config      Configuration loaded from metadata
         * @param {Device_Type} device_type_    Device on which to run inference
         * @return {Model} Model instance
         */
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

        /**
         * @description:                        Main inference method to predict emotion from audio input
         * @param {vector<float>} m_inputs      Preprocessed audio input as float vector
         * @return {string} predicted_class     Predicted emotion class name
         */
        std::string predict(std::vector<float> m_inputs);

    private:
        /**
         * @description:                        Initialize backend with specified type
         * @param {Backend_Type} backend_type   Type of backend to initialize
         */
        void init_backend(Backend_Type backend_type);

        /**
         * @description: Apply softmax function to logits on CPU
         * @param {vector<float>} logits   Raw logits from model
         * @return {vector<float>}         Probability distribution over classes
         */
        std::vector<float> cpuSoftmax1D(const std::vector<float> logits);

        /**
         * @description: Release backend resources
         */
        void release_backend();

        /**
         * @description:                Convert numerical class ID to emotion label
         * @param {int} prediction      Predicted class ID
         * @return {string}             Corresponding emotion label
         */
        std::string getClass(int prediction){
            auto it = id2label.find(prediction);
            return it != id2label.end() ? it->second : "no class found";
        }
};
