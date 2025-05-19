#pragma once
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

std::vector<float> gpuNormalizationLauncher(std::vector<float> m_inputs, float mean, const int blockSize, const int threadsPerBlock);