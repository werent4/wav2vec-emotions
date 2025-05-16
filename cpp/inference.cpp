#include "inference.h"
#include "utils.h"

void Model::init_backend(Backend_Type backend_type){
    std::string model_path = getModelPath(metadata_path, backend_type);
    try {
        backend = createBackend(backend_type, model_path, device_type);
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize backend: " << e.what() << std::endl;
        throw;
    }
};

std::string Model::predict(std::vector<float> m_inputs){
    std::vector<float> logits = backend->predict(m_inputs);
    std::vector<float> probabilities = cpuSoftmax1D(logits);
    std::string class_name = getClass(*std::max_element(probabilities.begin(), probabilities.end()));
    return class_name;
}

std::vector<float> Model::cpuSoftmax1D(const std::vector<float> logits){
    if (logits.empty()) return {};
    
    std::vector<float> probabilities = logits;
    float max_val = *std::max_element(logits.begin(), logits.end());
    
    float sum = 0.0f;
    for (size_t i = 0; i < probabilities.size(); i++) {
        probabilities[i] = std::exp(probabilities[i] - max_val);
        sum += probabilities[i];
    }
    
    if (sum > 0) {
        for (size_t i = 0; i < probabilities.size(); i++) {
            probabilities[i] /= sum;
        }
    }

    return probabilities;
}

void Model::release_backend(){
    if (backend) delete backend;
};