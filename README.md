# `Sound Controlled Object Selection and Tracking Module`
*This module developed to an AI controlled robotics project, not for general purpose.*

Module occurs with to class: [`AI`](include/AIClass.hpp) and [`AudioModel`](include/AudioModel.hpp). 

`AI` class wraps `AudioModel` class, thats why you must not use `AudioModel` class directly.


## `class AI`
This is an orchestrator class. This class works in conjunction with the [`AudioModel`](include/AudioModel.hpp), which handles audio recording and target recognition functions.

### Example Usage:
```cpp
// Init `AI` Class.
AI* ai = new AI(
      "path/to/deepstream_app.txt", // Deepstream configuration file path
      "path/to/audio_model.pt",     // Pre trained Audio Recognition model file path
      16000,                        // Pre trained Audio Recognition model sample rate
      8                             // Pre trained Audio Recognition model maximum sound duration
); 

ai->run_deepstream();               // Starts Deepstream pipeline
ai->start_recording();              // Starts voice recording.
ai->stop_recording();               // Stops voice recording.

// Retruns three boolean data that represents wheter selected color red green blue respectively.
std::vector<bool> targets_color = ai->get_current_targets_color();

// Returns the location one of bounding boxes that pairs `AI::selected_target_id` `AI::current_targets_color`.
std::vector<int> target_loc = ai->get_current_targets_loc();

// Returns the pointer of the stored sound amplitude data. Normalized 0-1.
std::vector<float>* stored_data_ptr = ai->get_audio_data();

delete ai;
```

### Class Members
#### Variables
- **`ds_config_file_path`**: Deepstream configuration file path.
- **`audio_model_file_path`**: Audio recognition model pack file path.
- **`target_sr`**: Model trained sample rate.
- **`max_seconds`**: Model trained maximum audio lenght.
- **`current_targets_color`**: Holds model recognized colors as boolean three value, red green blue respectively. Default initialized to false.
- **`current_targets_loc`**: Holds current targets locations in xywh format.
- **`selected_target_id`**: Holds the tracked target ID.
- **`audio_model`**: Pointer of the AudioModel class which provides audio process tools.
- **`is_audio_busy`**: Atomic boolean indicating whether audio is recording. Used for `AI::start_recording()` and `AI::stop_recording()`.

#### Functions
- **`AI(std::string ds_config_path, std::string audio_model_path, int sample_rate, int max_sec)`**: Constructor.
  - `ds_config_path`: Deepstream configuration file path.
  - `audio_model_path`: Pre-trained Audio Recognition model file path.
  - `sample_rate`: Pre-trained Audio Recognition model sample rate.
  - `max_sec`: Pre-trained Audio Recognition model maximum sound duration.
- **`run_deepstream()`**: Starts Deepstream pipeline. The `AI::process_ds_data()` method will be triggered by each frame processed on Deepstream.
- **`start_recording()`**: Starts voice recording.
- **`stop_recording()`**: Stops voice recording.
- **`get_current_targets_color()`**: Returns three boolean data that represents whether the selected color is red, green, or blue respectively.
- **`get_current_targets_loc()`**: Returns the location of one of the bounding boxes that equal to `AI::selected_target_id` and `AI::current_targets_color` which was recognized with the sound model. The selected bbox ID will be attached to `AI::selected_target_id`.
- **`get_audio_data()`**: Returns a pointer to the stored voice data.
- **`get_is_recording()`**: Returns a boolean value that indicates whether voice is currently recording.
- **`get_class_info()`**: Designed to debug all data. Prints out all the information that the class holds by calling `AI::class_info()`.
- **`process_ds_data(DsObjectData* obj_list, int num_objects, int frame_num)`**: Processes the `DsObjectData`s obtained from Deepstream. Updates `AI::current_targets_loc`. It will be triggered by each frame. It is private.
- **`audio_inference(std::vector<bool>& targets)`**: Triggers inference for voice data. It will be triggered at the end of the `AI::start_recording()` method. It is private.
- **`class_info()`**: Prints out all the internal information and states held by the class. It is private.

## `class AudioModel`
`AudioModel` is a thread-safe class works to access microphone, to record audio data and provide inference funcitons it on `torch::jit`.

### Example Usage:
```cpp
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

### Class Members
#### Variables
- `is_recording`: Holds a boolean if the model recording.\
- `audio_buffer`: Holds recorded data.\
- `model`: Pretrained Audio recognition model.\
- `buffer_mutex`: Mutex for audio buffer while data saving.\
- `ds_config_file_path`: Deepstream configuration file path.\
- `audio_model_file_path`: Audio recognition model pack file path.\
- `target_sr`: Model trained sample rate.\
- `max_seconds`: Madel trained maximum audio lenght.\
- `targets`: Recognized Targest boolean val for Red, Green, Blue respectively.

#### Functions
- **`AudioModel(const std::string& model_path, int sample_rate = 16000, int max_sec = 8)`**: **Constructor**. Class has just one constructor.\
  - `model_path` is audio model absolute path. Used for `torch::jit::load(model_path)`.\
  - `sample_rate` is microphone sound access count per time. It effect sound quality directly. And model must have trained dedicated sample rate. Default is 16000 (16Khz).\
  - `max_sec` is maximum second of model can process. if recorded data longer than `max_sec` then it will cropped till 'max_sec', but if recorded data shorter than `max_sec` then it will be padded with zeros till data riched as data as much amound of 'max_sec'. This default is 8 (8 seconds).

- **`data_callback`**: Microphone data record funciton in each frame (sound package). It is private.

- **`start_recording()`**: Firstly cleans to the `audio_buffer` using `buffer_mutex` and then starts to read microphone.

- **`stop_recording()`**: Stops recording.

- **`toTorchTensor()`**: Convert to torch tensor from vector data. If `audio_buffer` is empty, it returns an empty torch tensor.

- **`preprocess_audio()`**: Data preprocessor according to the 'max_sec'

- **`inference(torch::Tensor recorded_data)`**: Takes input data (`recorded_data`), runs model, attach `targets` returns out data. Returned data is empty when `audio_buffer` is empty or inference had met an error. In the other hand, returns raw color data which model recognized.

- **`get_targets()`**: Returns attached `targets`.

- **`get_is_recording()`**: Returns `is_recordings` to figure out data whether recorfing.

- **`get_audio_buffer()`**: Returns 'audio_buffer', this is thread-safe.

