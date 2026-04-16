# Sound Controlled Object Selection and Tracking Module

*This module was developed for an AI-controlled robotics project, specifically for the __EngineerHub'25: Voice Controlled, Remote Controlled Weapon Station Software__, rather than for general-purpose use.*

*All instructions assume the NVIDIA Jetson AGX Orin (or Orin NX) utilizing this library has **DeepStream 7.1** installed. If not, please follow [these installation instructions](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_Installation.html#jetson-setup).*

*Enjoy with your laser :)*

---

This module consists of two main classes: [`AI`](include/AIClass.hpp) and [`AudioModel`](include/AudioModel.hpp). 

The `AI` class acts as an orchestrator and wraps the `AudioModel` class. **Therefore, you should interact only with the `AI` class and avoid using `AudioModel` directly.**

## Compiling

The compiling process requires Torch libraries since `libAI_class.so` depends on them.
The best practice for using the `torch` library in C/C++ is fetching it from a Python package, as the libraries come pre-compiled. 
We highly recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to prevent system-wide conflicts.

*Setting up the conda environment:*
```shell
miniconda create -n libtorch python=3.10
conda activate libtorch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

After installing `torch`, you can compile your code with a C++ compiler by adding the following flags:
```shell
g++ example_main_usage.cpp \
-I./include/ \
-L./lib \
-lAI \
-Wl,-rpath,./lib \
-Wl,-rpath-link,$(conda info --base)/envs/libtorch/lib/python3.10/site-packages/torch/lib \
-Wl,-rpath-link,$(conda info --base)/envs/libtorch/lib/python3.10/site-packages/torch.libs
```

---

## `class AI`
This is the orchestrator class. It works in conjunction with the [`AudioModel`](include/AudioModel.hpp), which handles multi-threaded audio recording and target recognition functions seamlessly alongside the DeepStream pipeline.

### Example Usage:
```cpp
// 1. Initialize the `AI` Class.
AI* ai = new AI(
      "path/to/deepstream_app.txt", // DeepStream configuration file path
      "path/to/audio_model.pt",     // Pre-trained Audio Recognition model file path
      16000,                        // Target sample rate for the audio model
      8                             // Maximum recording duration in seconds
); 

// 2. Start pipelines
ai->run_deepstream();               // Starts DeepStream pipeline in a detached thread
ai->start_recording();              // Starts voice recording
ai->stop_recording();               // Stops recording and triggers inference

// 3. Fetch Data (Thread-Safe)
// Returns three boolean data representing whether the selected color is Red, Green, or Blue.
const std::vector<bool>* targets_color = ai->get_current_targets_color();

// Returns the location of the locked bounding box based on voice command.
// Format: [cx, cy, w, h, class_id, tracking_id]. Returns {} if no target is found.
std::vector<float> target_loc = ai->get_current_target_loc();

// Returns the pointer to the raw sound amplitude data.
const std::vector<float>* stored_data_ptr = ai->get_audio_data();

// 4. Reset tracking if needed
ai->reset_tracking();               // Clears current target lock until a new voice command is given.

delete ai;
```

### Class Members

#### Variables
- **`ds_config_file_path`**: DeepStream configuration file path.
- **`audio_model_file_path`**: Audio recognition model `.pt` file path.
- **`target_sr`**: Sample rate expected by the model.
- **`max_seconds`**: Maximum audio length expected by the model.
- **`current_targets_color`**: Holds model-recognized colors as a boolean array (Red, Green, Blue). Default initialized to false.
- **`current_targets_loc`**: Holds current target locations dynamically in `[cx, cy, w, h, class_id, tracking_id]` format.
- **`selected_target_id`**: Holds the tracked target's DeepStream Object ID.
- **`audio_model`**: Pointer to the `AudioModel` class instance.
- **`is_audio_busy`**: Atomic boolean indicating whether audio is currently recording. Used for `AI::start_recording()` and `AI::stop_recording()`.
- **`get_target_loc_mutex`**: Recursive mutex ensuring thread-safe data transfer between DeepStream callbacks and C++ logic.

#### Functions
- **`AI(...)`**: Constructor configuring both DeepStream and Audio parameters.
- **`run_deepstream()`**: Starts the DeepStream pipeline. The `process_ds_data()` method will be triggered by each frame.
- **`start_recording()`**: Starts voice recording securely.
- **`stop_recording()`**: Stops voice recording and starts the PyTorch inference thread.
- **`get_current_targets_color()`**: Returns the active colors identified by the audio model.
- **`get_current_target_loc()`**: Recursively locks onto and returns the 6-element location data of the valid target.
- **`get_audio_data()`**: Returns a pointer to the stored voice buffer.
- **`get_is_recording()`**: Returns a boolean indicating if the microphone is active.
- **`reset_tracking()`**: Clears the active target colors and tracking IDs.
- **`get_class_info()`**: Prints debug information to the standard output.
- **`process_ds_data(...)`**: Processes bounding box data from DeepStream. Updates `AI::current_targets_loc`. It is private.
- **`audio_inference(...)`**: Triggers inference for voice data. It is private.
- **`class_info()`**: Prints out all the internal states held by the class. It is private.

---

## `class AudioModel` (Internal Use)
`AudioModel` is a thread-safe class designed to access the microphone, record audio data via `miniaudio`, and provide inference functions using `torch::jit`.

### Example Usage:
```cpp
AudioModel* audio_model = new AudioModel("path/to/model.pt", 16000, 8);
audio_model->start_recording(); // Starts the voice recording.
audio_model->stop_recording();  // Stops the voice recording.

// Access to the recorded data
const std::vector<float>* audio_data = audio_model->get_audio_buffer();

// Access info if voice recording
bool audio_recording = audio_model->get_is_recording();

// Convert to torch tensor from vector
torch::Tensor audio_tensor = audio_model->toTorchTensor();

// Inference on torch::jit
torch::Tensor result = audio_model->inference(audio_tensor);

// Access to targets 
std::vector<bool> targets = audio_model->get_targets();
```

### Class Members

#### Variables
- **`is_recording`**: Holds a boolean if the model is recording.
- **`audio_buffer`**: Holds recorded raw floating-point audio data.
- **`model`**: Pretrained PyTorch JIT model.
- **`buffer_mutex`**: Mutex protecting the audio buffer during active recording.
- **`targets`**: Recognized targets boolean values for Red, Green, and Blue respectively.

#### Functions
- **`AudioModel(const std::string& model_path, int sample_rate = 16000, int max_sec = 8)`**: Constructor.
  - `model_path`: Absolute path to the audio model.
  - `sample_rate`: Expected sample rate. Default is 16000 (16Khz).
  - `max_sec`: Maximum seconds the model can process. If recorded data is longer, it will be cropped. If shorter, it will be padded with zeros. Default is 8.
- **`data_callback(...)`**: Microphone data record function in each frame. It is private.
- **`start_recording()`**: Clears the `audio_buffer` securely and starts reading from the microphone.
- **`stop_recording()`**: Stops the microphone stream.
- **`toTorchTensor()`**: Converts the vector buffer into a 2D `torch::Tensor`.
- **`preprocess_audio(...)`**: Truncates or pads the raw audio tensor to match the `max_seconds` exactly.
- **`inference(torch::Tensor recorded_data)`**: Feeds the preprocessed data into the model, updates the `targets` boolean array, and returns the raw output tensor.
- **`get_targets()`**: Returns attached `targets`.
- **`get_is_recording()`**: Returns `is_recording` flag.
- **`get_audio_buffer()`**: Returns a pointer to the `audio_buffer`. This is thread-safe.