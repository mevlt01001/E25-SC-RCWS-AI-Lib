#ifndef AUDIO_MODEL_HPP
#define AUDIO_MODEL_HPP

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include <torch/torch.h>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>

class AudioModel {
    private:
        bool is_recording;          
        torch::jit::script::Module model;   
        ma_device device;                   
        ma_device_config deviceConfig;      
        std::vector<float> audio_buffer;    
        std::mutex buffer_mutex;                                                
        std::string ds_config_file_path;              
        std::string audio_model_file_path;             
        int target_sr;                                      
        int max_seconds;                                     
        std::vector<bool> current_targets; 
        AudioModel* audio_model;                            
        std::atomic<bool> is_audio_busy{false};
        std::vector<bool> targets;          

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
};
#endif  // AUDIO_MODEL_HPP