#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>

/**
 * @description: loads audio file
 * @param {string&} path	path to audio file
 * @param {int} sampleRate	sample rate
 * @return {vector<float>}
 */
std::vector<float> loadAudio(const std::string& path, int& sampleRate);

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