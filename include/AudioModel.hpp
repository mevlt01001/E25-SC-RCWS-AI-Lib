#ifndef AUDIO_MODEL_HPP
#define AUDIO_MODEL_HPP

#include <vector>
#include <torch/torch.h>
#include "dr_wav.h"
#include "miniaudio.h"

/**
 * @brief AudioModel is a thread-safe class designed to interface with a microphone, 
 * record audio data, and perform inference using a pre-trained PyTorch (`torch::jit`) model.
 * * Example:
 * ``` 
 * AudioModel* audio_model = new AudioModel("path/to/model.pt", 16000, 8);
 * audio_model->start_recording(); // Starts the voice recording.
 * audio_model->stop_recording();  // Stops the voice recording.
 * * // Access to the recorded data
 * std::vector<float>* audio_data = audio_model->get_audio_buffer();
 * * // Access info if voice recording
 * bool audio_recording = audio_model->get_is_recording();
 * * // Convert to torch tensor from vector
 * torch::Tensor audio_tensor = audio_model->toTorchTensor();
 * * // Inference on torch::jit
 * torch::Tensor result = audio_model->inference(audio_tensor);
 * * // Access to targets 
 * std::vector<bool>& targets = audio_model->get_targets();
 * ```
 */
class AudioModel {
    private:
        // Holds a boolean if the model recording.
        bool is_recording;
        // Holds recorded data
        std::vector<float> audio_buffer;
        // Pretrained Audio recognition model
        torch::jit::script::Module model;
        ma_device device;       
        ma_device_config deviceConfig;      
        // Mutex for audio buffer while recording
        std::mutex buffer_mutex;           
        // Target sample rate for audio capture
        int target_sr;                     
        // Maximum duration for audio capture in seconds
        int max_seconds;                   
        // Output from sound recognition model [R, G, B] respectively. 
        std::vector<bool> targets; 

        /**
         * @brief Internal callback function triggered by miniaudio to process incoming audio data.
         * Safely appends the captured audio frames into the `audio_buffer` using a mutex lock.
         * * @param pDevice Pointer to the miniaudio device triggering the callback.
         * * @param pOutput Pointer to the output buffer (unused in recording).
         * * @param pInput Pointer to the input buffer containing captured audio frames.
         * * @param frameCount The number of frames available in the input buffer.
         */
        static void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount);

    public:
        /**
         * @brief Initializes the AudioModel with the specified PyTorch model and audio parameters.
         * Loads the JIT model, configures the miniaudio device for recording, and sets up buffers.
         * If the recorded audio is shorter than `max_sec`, it will be padded with zeros during inference.
         * * @param model_path Path to the pre-trained PyTorch JIT model (`.pt`).
         * * @param sample_rate The sample rate for audio recording (default is 16000 Hz).
         * * @param max_sec The maximum duration of the audio recording in seconds (default is 8).
         */
        AudioModel(const std::string& model_path, int sample_rate = 16000, int max_sec = 8);

        /**
         * @brief Destroys the AudioModel instance and safely uninitializes the audio device.
         * Releases the miniaudio context (`ma_device_uninit`) to free microphone resources.
         */
        ~AudioModel();

        /**
         * @brief Starts the audio recording process.
         * Safely clears the existing `audio_buffer` using `buffer_mutex` and begins capturing 
         * microphone input via the miniaudio device.
         */
        void start_recording();

        /**
         * @brief Stops the ongoing audio recording process.
         * Pauses the miniaudio device to stop capturing microphone input.
         */
        void stop_recording();

        /**
         * @brief Converts the recorded floating-point audio data into a PyTorch tensor.
         * Transmutes the raw `audio_buffer` vector into a 2D tensor `[1, num_samples]`.
         * If the `audio_buffer` is empty, it returns an empty torch tensor.
         * * @return `torch::Tensor` The constructed PyTorch tensor containing the audio data.
         */
        torch::Tensor toTorchTensor();

        /**
         * @brief Preprocesses the raw audio tensor to match the model's expected input length.
         * Truncates the tensor if it exceeds the maximum allowed frames (`max_seconds * target_sr`),
         * or pads it with zeros if it is shorter than the required length.
         * * @param audio_data The raw audio tensor to be preprocessed.
         * * @param max_seconds The target duration in seconds.
         * * @param target_sr The target sample rate.
         * * @return `torch::Tensor` The preprocessed audio tensor ready for model inference.
         */
        torch::Tensor preprocess_audio(torch::Tensor audio_data, int max_seconds, int target_sr);

        /**
         * @brief Executes model inference on the provided audio tensor to recognize voice commands.
         * Takes the input tensor, runs it through the loaded PyTorch model, and updates the internal 
         * `targets` boolean vector based on the predictions. If the input is empty or an error occurs, 
         * it returns an empty tensor.
         * * @param recorded_data The preprocessed audio tensor to perform inference on.
         * * @return `torch::Tensor` The raw output tensor from the model.
         */
        torch::Tensor inference(torch::Tensor recorded_data);

        /**
         * @brief Retrieves the boolean flags indicating which commands/colors were recognized.
         * * @return `std::vector<bool>` A vector of booleans representing the recognized targets.
         */
        std::vector<bool> get_targets() const;

        /**
         * @brief Checks the current recording status of the audio model.
         * * @return `bool` True if the microphone is currently recording, false otherwise.
         */
        bool get_is_recording() const;

        /**
         * @brief Retrieves a pointer to the raw audio buffer data.
         * This method is thread-safe.
         * * @return `const std::vector<float>*` A constant pointer to the internal audio buffer.
         */
        const std::vector<float>* get_audio_buffer();
};
#endif // AUDIO_MODEL_HPP