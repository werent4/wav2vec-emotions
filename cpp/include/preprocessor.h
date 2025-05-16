#include <iostream>
#include <vector>

using m_inputs_batch = std::vector<std::vector<float>>;

/*
 * @description: Preprocessing class for audio. This class is adaptation of Wav2Vec2FeatureExtractor from transformers library
 *                  see more https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/feature_extraction_wav2vec2.py.
 */
class FeaturesExtractor{
    private: 
        int SamplingRate = 16000;
    public:
        bool do_normalize= true;
        float paddingValue = 0.0f;
        int max_length = 80000; // SamplingRate * duration (seconds)

    public:
        FeaturesExtractor(int SamplingRate_, bool do_normalize_ = true ){
            if (SamplingRate_ != SamplingRate){
                std::cerr << "This class was developed specifically to support wav2vec models that expect audio with a sample rate of 16000" << std::endl;
            }
            
            do_normalize = do_normalize_;

        };
        // ~FeaturesExtractor(){};

        /* @description: main extractor method.
         * @param  {vector<float>} m_inputs   array of sound amplitudes in time domain
         * @return {vector<float>}
         */
        std::vector<float> process(const std::vector<float> m_inputs);

    private:
        /* @description: Every array in the list is normalized to have zero mean and unit variance
         * @param  {vector<float>} m_inputs  array of sound amplitudes in time domain
         * @return {*}
         */
        void normalize(std::vector<float> m_inputs);

        /* @description: Variance 
         * @param  {vector<float>} m_inputs  array of sound amplitudes in time domain
         * @param  {float} mean              mean of m_inputs 
         * @return {vector<float>}
         */        
        float variance(const std::vector<float> m_inputs, float mean);
        std::vector<float> pad(std::vector<float> m_inputs);
};

