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
