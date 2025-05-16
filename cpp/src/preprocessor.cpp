#include <bits/stdc++.h>
#include "preprocessor.h"
#include "utils.h"

std::vector<float> FeaturesExtractor::process(const std::vector<float> m_inputs){
    std::vector<float> m_inputs_ = m_inputs;

    m_inputs_ = pad(m_inputs_);

    if (do_normalize) normalize(m_inputs_);
    // print1DVector(m_inputs_);
    return m_inputs_;
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

void FeaturesExtractor::normalize(std::vector<float>& m_inputs){
    const double eps = 1e-7; 
    float mean = accumulate(m_inputs.begin(), m_inputs.end(), 0.0f) / m_inputs.size();
    float variance_ = variance(m_inputs, mean);

    float divider = std::sqrt(variance_ + eps);
    for (auto& element: m_inputs){
        element = (element - mean) / divider;
    }
}

float FeaturesExtractor::variance(const std::vector<float>& m_inputs, float mean) {
    double sum = 0.0;

    for (const auto& element : m_inputs) {
        double sub_mean = static_cast<double>(element) - mean;
        sum += sub_mean * sub_mean; 
    }
    
    if (m_inputs.size() <= 1) {
        return 0.0;
    }
    
    return sum / (m_inputs.size() - 1);
}