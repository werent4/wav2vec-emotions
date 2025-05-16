#pragma once

#include <iostream>
#include <thread>
#include <onnxruntime_cxx_api.h>

#include "types.h"

class Backend{
    public:
    virtual ~Backend() = default;
    virtual std::vector<float> predict(const std::vector<float>& m_inputs) = 0;
};

class OnnxBackend : public Backend {
    private:
        Ort::Env m_env;
        Ort::Session* m_session;
        Device_Type device_type;

    public:
        OnnxBackend(const std::string& model_path, Device_Type device_type);
        ~OnnxBackend() override {
            m_session->release();
            m_env.release();
        }
        
        std::vector<float> predict(const std::vector<float>& m_inputs) override;
};


Backend* createBackend(Backend_Type type, const std::string& model_path, Device_Type device);