#include "cuda_preprocess.cuh"

__global__
void gpuVariance(float *x, float *out, float mean, const int BLOCK_SIZE, int n_cols){
    extern __shared__ float sdata[]; // shared memory for each block

    int tid = blockIdx.x*blockDim.x + threadIdx.x; // global thread id
    int local_tid = threadIdx.x;  // local thread id

    float sum = 0.0f; // local sum 

    int start_idx = tid * BLOCK_SIZE;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        if (start_idx + i < n_cols) {
            float diff = x[start_idx + i] - mean;
            sum += diff * diff; 
        }
    }

    sdata[local_tid] = sum;
    __syncthreads(); // synchronize threads in the block

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) { // reduce within the block
        if (local_tid < stride) {
            sdata[local_tid] += sdata[local_tid + stride];
        }
        __syncthreads();
    }

    if (local_tid == 0) { // only 1st thread writes the result
        out[blockIdx.x] = sdata[0];
    }
}

__global__
void gpuNormalization(float *x, float *out, float mean, float devider, const int BLOCK_SIZE, int n_cols){
    int tid = blockIdx.x*blockDim.x + threadIdx.x; // global thread id
    int start_idx = tid * BLOCK_SIZE;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        if (start_idx + i < n_cols) {
            out[start_idx + i] = (x[start_idx + i] - mean) / devider;
        }
    }

}

std::vector<float> gpuNormalizationLauncher(std::vector<float> m_inputs, float mean, const int blockSize, const int threadsPerBlock) {
    const double eps = 1e-7;

    int blocksPerGrid = (m_inputs.size() + threadsPerBlock * blockSize - 1) / (threadsPerBlock * blockSize);

    float h_inputs[m_inputs.size()], h_variance_out[blocksPerGrid], h_outputs[m_inputs.size()];
    float *d_inputs, *d_variance_outputs, *d_outputs;

    for (int i = 0; i < m_inputs.size(); i++){
        h_inputs[i] = m_inputs[i];
    }

    cudaMalloc(&d_inputs, 1 * m_inputs.size() * sizeof(float)); // allocate memory for 1d matrix
    cudaMalloc(&d_variance_outputs, 1 * blocksPerGrid * sizeof(float)); // allocate memory for variance intermideate outputs 
    cudaMalloc(&d_outputs, 1 * m_inputs.size() * sizeof(float)); // allocate memory for outputs

    cudaMemcpy(d_inputs, h_inputs, 1 * m_inputs.size() * sizeof(float), cudaMemcpyHostToDevice); // copy inputs to device

    gpuVariance<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_inputs, d_variance_outputs, mean, blockSize, m_inputs.size());
    cudaMemcpy(h_variance_out, d_variance_outputs, blocksPerGrid  * sizeof(float), cudaMemcpyDeviceToHost); // copy variance intermideate outputs to host

    float total_sum = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        total_sum += h_variance_out[i];
    }

    float gpu_variance = total_sum / (m_inputs.size());

    float divider = std::sqrt(gpu_variance + eps);
    gpuNormalization<<<blocksPerGrid, threadsPerBlock>>>(d_inputs, d_outputs, mean, divider, blockSize, m_inputs.size());
    cudaMemcpy(h_outputs, d_outputs, 1 * m_inputs.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_inputs);
    cudaFree(d_variance_outputs);
    cudaFree(d_outputs);

    std::vector<float> normalized_m_inputs(m_inputs.size());
    for (int i = 0; i < m_inputs.size(); i++){
        normalized_m_inputs[i] = h_outputs[i];
    }
    return normalized_m_inputs;
}