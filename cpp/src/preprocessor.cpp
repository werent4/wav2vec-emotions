#include <bits/stdc++.h>
#include "preprocessor.h"
#include "utils.h"
#ifdef _CUDA_PREPROCESS
    #include "cuda_preprocess.cuh"
#endif

std::vector<float> FeaturesExtractor::process(const std::vector<float> m_inputs){
    std::vector<float> m_inputs_ = m_inputs;

    m_inputs_ = pad(m_inputs_);

    float mean = accumulate(m_inputs.begin(), m_inputs.end(), 0.0f) / m_inputs.size();
#ifdef _CUDA_PREPROCESS
    cudaDeviceProp deviceProp;
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&deviceProp, deviceId);

    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;

    int threadsPerBlock = 256;
    if (maxThreadsPerBlock < 256) {
        threadsPerBlock = maxThreadsPerBlock;
        threadsPerBlock = (threadsPerBlock / 32) * 32;
    }

    int dataSize = m_inputs_.size();

    int blockSize = 1;
    if (dataSize > 10000) blockSize = 8;
    else if (dataSize > 1000) blockSize = 4;
    else blockSize = 2;

    return gpuNormalizationLauncher(m_inputs_, mean, blockSize, threadsPerBlock);
    // float variance_ = gpuVarianceLauncher(m_inputs_, mean, blockSize, threadsPerBlock);
    // float divider = std::sqrt(variance_ + eps);
    // for (auto& element: m_inputs_){
    //     element = (element - mean) / divider;
    // }
#else
    if (do_normalize) normalize(m_inputs_, mean);
    return m_inputs_;
#endif
}

std::vector<float> FeaturesExtractor::pad(std::vector<float>& m_inputs){
    if (m_inputs.size() > max_length) {
        // Crop
        return std::vector<float>(m_inputs.begin(), m_inputs.begin() + max_length);
    } else if (m_inputs.size() < max_length) {
        // Pad
        std::vector<float> padded = m_inputs;
        padded.resize(max_length, paddingValue);
        return padded;
    }
    return m_inputs;
}

void FeaturesExtractor::normalize(std::vector<float>& m_inputs, float mean){
    const double eps = 1e-7; 
    float variance_ = variance(m_inputs, mean);

    float divider = std::sqrt(variance_ + eps);
    for (auto& element: m_inputs){
        element = (element - mean) / divider;
    }
}

float FeaturesExtractor::variance(const std::vector<float>& m_inputs, float mean) {
    if (m_inputs.size() <= 1) {
        return 0.0;
    }

    double sum = 0.0;

    for (const auto& element : m_inputs) {
        double sub_mean = static_cast<double>(element) - mean;
        sum += sub_mean * sub_mean; 
    }
    
    return sum / (m_inputs.size() - 1);
}