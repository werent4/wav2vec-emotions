#pragma once

#include <iostream>
#include <thread>
#include <onnxruntime_cxx_api.h>

#include "types.h"

/**
 * @description: Base backend class
*/
class Backend{
    public:
    virtual ~Backend() = default;
	/**
	 * @description: 				     Main metgod to get predictions via specific backend
	 * @param {vector<float>&} m_inputs  Prepared model inputs	    
	 * @return {vector<float>} logits    Raw logits
	 */	
    virtual std::vector<float> predict(const std::vector<float>& m_inputs) = 0;
};

class OnnxBackend : public Backend {
    private:
        /**
        * @description: ONNX env
        */
        Ort::Env m_env;

        /**
        * @description: ONNX session
        */
        Ort::Session* m_session;
        
        /**
        * @description: device on which run backend
        */
        Device_Type device_type;

    public:
        /**
         * @description: 				     Constructor
         * @param {string&} model_path       Path to model file
         * @param {Device_Type} device	 	 Device (CPU/GPU) on which run backend
         * @return {OnnxBackend} backend
         */	
        OnnxBackend(const std::string& model_path, Device_Type device_type);
        ~OnnxBackend() override {
            m_session->release();
            m_env.release();
        }
        
        std::vector<float> predict(const std::vector<float>& m_inputs) override;
};

/**
 * @description: 				     Factory fuction to create backends
 * @param {Backend_Type} type        Type of backend to be created
 * @param {string&} model_path       Path to model file
 * @param {Device_Type} device	 	 Device (CPU/GPU) on which run backend
 * @return {Backend*} backend
 */	
Backend* createBackend(Backend_Type type, const std::string& model_path, Device_Type device);