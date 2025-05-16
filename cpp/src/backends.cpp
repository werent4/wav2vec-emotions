#include "backends.h"

Backend* createBackend(Backend_Type type, const std::string& model_path, Device_Type device) {
    switch (type) {
        case ONNXRuntime:
            return new OnnxBackend(model_path, device);
        // case LibTorch:
        //     return new LibTorchBackend(model_path, device);
        default:
            throw std::runtime_error("Unsupported backend type!");
    }
}

OnnxBackend::OnnxBackend(const std::string& model_path, Device_Type device_type){
	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(std::thread::hardware_concurrency()/2);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (device_type == GPU)
	{
		OrtCUDAProviderOptions cuda_option;
		cuda_option.device_id = 0;
		cuda_option.arena_extend_strategy = 0;
		cuda_option.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
		cuda_option.gpu_mem_limit = SIZE_MAX;
		cuda_option.do_copy_in_default_stream = 1;
		session_options.AppendExecutionProvider_CUDA(cuda_option);
	}

#ifdef _WIN32
    throw std::runtime_error("Windows is not supported yet");
#endif

#ifdef __linux__
	m_session = new Ort::Session(m_env, model_path.c_str(), session_options);
#endif
}

std::vector<float> OnnxBackend::predict(const std::vector<float>& m_inputs){
    if (!m_session) {
        throw std::runtime_error("ONNX session not initialized");
    }
    std::vector<const char*> input_names = {"input"};
    std::vector<const char*> output_names = {"logits"};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(m_inputs.size())};

    // CreateTensor (const OrtMemoryInfo *info, T *p_data, size_t p_data_element_count, const int64_t *shape, size_t shape_len)
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        const_cast<float*>(m_inputs.data()), 
        m_inputs.size(), //
        input_shape.data(), 
        input_shape.size()
    );

    std::vector<Ort::Value> outputs = m_session->Run(
        Ort::RunOptions{nullptr}, 
        input_names.data(), 
        &input_tensor, 
        1, 
        output_names.data(), 
        1
    );

    if (outputs.size() == 0 || !outputs[0].IsTensor()) {
        throw std::runtime_error("Invalid output tensor");
    }
    
    float* output_data = const_cast<float*>(outputs[0].GetTensorData<float>());
    size_t output_size = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> logits(output_data, output_data + output_size);
    return logits;
}


