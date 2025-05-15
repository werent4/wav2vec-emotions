#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>

template <typename Number>
void print1DVector(std::vector<Number> data) {
    std::cout << "[";
    for (size_t i = 0; i < data.size(); i++) {
        std::cout << data[i];
        if (i < data.size() - 1) {
            std::cout << " ";
        }
    }
    std::cout << "]" << std::endl;
}

#endif // UTILS_H