#pragma once

#include <iostream>
#include <string>
#include <unordered_map>

/**
 * @description: device type
 */
enum Device_Type
{
	CPU,
	GPU,
};

enum Backend_Type
{
	ONNXRuntime,
};

inline std::unordered_map<std::string, Backend_Type> StringToBackendType = {
    {"onnx", ONNXRuntime} 
};

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