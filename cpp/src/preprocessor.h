#include <iostream>
#include <vector>

/**
 * @description: loads audio file
 * @param {string&} path	path to audio file
 * @param {int} sampleRate	sample rate
 * @return {vector<float>}
 */
std::vector<float> loadAudio(const std::string& path, int& sampleRate);