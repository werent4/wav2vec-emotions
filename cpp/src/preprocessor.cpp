#include <bits/stdc++.h>
#include "preprocessor.h"

std::vector<float> FeaturesExtractor::process(const std::vector<float> m_inputs){
    std::vector<float> m_inputs_ = m_inputs;
    if (do_normalize) normalize(m_inputs);
    
    return m_inputs_;
}

void FeaturesExtractor::normalize(std::vector<float> m_inputs){
    double eps = 1e-7; 
    float mean = accumulate(m_inputs.begin(), m_inputs.end(), 0) / m_inputs.size();
    float variance_ = variance(m_inputs, mean);

    float divider = std::sqrt(variance_ + eps);
    for (auto& element: m_inputs){
        element = (element - mean) / divider;
    }
}

float FeaturesExtractor::variance(const std::vector<float> m_inputs, float mean){
    std::vector<float> m_inputs_ = m_inputs;
    float sum = 0.0f;

    for (auto& element: m_inputs_){
        float sub_mean = element - mean;
        sum += sub_mean * sub_mean; 
    }
    return sum / m_inputs_.size();
}