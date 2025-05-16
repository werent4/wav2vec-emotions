#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>
#include <filesystem>

#include "inference.h"

/**
 * @description: loads audio file
 * @param {string&} path	path to audio file
 * @param {int} sampleRate	sample rate
 * @return {vector<float>}
 */
std::vector<float> loadAudio(const std::string& path, int& sampleRate);

/**
 * @description: Returns model path based on path to model's metadata_path and backend type 
 * @param {string&} metadata_path	    path model's metadata
 * @param {Backend_Type} backend_type	type of backend to run model
 * @return {string}                     path to model in Backend_Type format
 */
std::string getModelPath(const std::string& metadata_path, Backend_Type backend_type);

/**
 * @description: prints vector
 * @param {std::vector<Number>} data  vector data
 * @return {*}
 */
template <typename Number>
void print1DVector(std::vector<Number> data) {
    std::cout << "[";
    for (size_t i = 0; i < data.size(); i++) {
        std::cout << data[i];
        if (i < data.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

#endif // UTILS_H