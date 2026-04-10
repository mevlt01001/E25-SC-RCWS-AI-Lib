#ifndef AUDIO_MODEL_HPP
#define AUDIO_MODEL_HPP

#include <vector>
#include <torch/torch.h>
#include "dr_wav.h"
#include "miniaudio.h"

//TODO: DOCs each entity

/*
AudioModel class works to access mic, to record audio data and provide inference funcitons it on torch::jit.
Example:
``` 
    AudioModel* audio_model = new AudioModel("path/to/model.pt", 16000, 8);
    audio_model->start_recording(); // Starts the voice recording.
    audio_model->stop_recording();  // Stops the voice recording.

    // Access to the recorded data
    std::vector<float>* audio_data = audio_model->get_audio_buffer();

    // Access info if voice recording
    bool audio_recording = audio_model->get_is_recording();

    // Convert to torch tensor from vector
    torch::Tensor audio_tensor = audio_model->toTorchTensor();

    // Inference on torch::jit
    torch::Tensor result = audio_model->inference(audio_tensor);

    // Access to targets 
    std::vector<bool>& targets = audio_model->get_targets();
```
*/
class AudioModel {
    private:
        bool is_recording;
        std::vector<float> audio_buffer;    
        torch::jit::script::Module model;   
        ma_device device;                   
        ma_device_config deviceConfig;      
        std::mutex buffer_mutex;                                                
        std::string ds_config_file_path;              
        std::string audio_model_file_path;             
        int target_sr;                                      
        int max_seconds;                                     
        std::vector<bool> targets;          
        AudioModel* audio_model;                            
        std::atomic<bool> is_audio_busy{false};

        static void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount);

    public:
        AudioModel(const std::string& model_path, int sample_rate = 16000, int max_sec = 8);
        ~AudioModel();

        void start_recording();
        void stop_recording();
        torch::Tensor toTorchTensor();
        torch::Tensor resample_audio(torch::Tensor audio_data, int orig_sr, int target_sr);
        torch::Tensor preprocess_audio(torch::Tensor audio_data, int max_seconds, int target_sr);
        torch::Tensor inference(torch::Tensor final_audio);

        std::vector<bool> get_targets() const;
        bool get_is_recording() const;
        std::vector<float>* get_audio_buffer();


};
#endif  // AUDIO_MODEL_HPP