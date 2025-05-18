#pragma once

#include <iostream>
#include <string>
#include <unordered_map>

/**
 * @description: Device Type
 */
enum Device_Type
{
	CPU,
	GPU,
};

/**
 * @description: Backend Type
 */
enum Backend_Type
{
	ONNXRuntime,
};

/**
 * @description: Mapping of strings to instances of Backend_Type
 */
inline std::unordered_map<std::string, Backend_Type> StringToBackendType = {
    {"onnx", ONNXRuntime} 
};


/**
 * @description: Configuration structure for model metadata
 */
struct MetaDataConfig {
    /**
     * @description: Audio sampling rate in Hz (typically 16000 for wav2vec models)
     */
    int SampleRate;

    /**
     * @description: Maximum length of input audio in samples
     */
    int MaxLenght;

    /**
     * @description: Framework type string identifier (e.g., "onnx")
     */
    std::string Framework;

    /**
     * @description: Mapping from numerical emotion class IDs to emotion labels
     */
    std::unordered_map<int, std::string> Id2Label;

    /**
     * @description:                    Constructor for MetaDataConfig
     * @param {int} SampleRate_         Audio sampling rate in Hz
     * @param {int} MaxLenght_          Maximum input length in samples
     * @param {string} Framework_       Framework identifier
     * @param {unordered_map} Id2Label_ Mapping from emotion IDs to labels
     */
    MetaDataConfig(int SampleRate_, int MaxLenght_, std::string Framework_, std::unordered_map<int, std::string> Id2Label_){
        SampleRate = SampleRate_;
        MaxLenght = MaxLenght_;
        Framework = Framework_;
        Id2Label = Id2Label_;
    };
};