#ifndef AICLASS_HPP
#define AICLASS_HPP

#include <vector>
#include <atomic>

// TODO: implelente `AI::get_current_targets_color()` `AI::get_current_target_loc()`


class AudioModel;                         // Audio Class Declaration `include/AudioModel.hpp`
typedef struct DsObjectData DsObjectData; // A type declaration for struct DsObjectData to DsObjectData
struct DsObjectData;                      // A struct to store `NvDsObjectMeta` member such as `class_id` `object_id` `rect_params`.


/**
 * This is an orchestrator class. This class works in conjunction with the
 * [`AudioModel`](./AudioModel.hpp), which handles audio recording and 
 * target recognition functions.
 * 
 * Usage Example:
```
// Inıt `AI` Class.
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
std::vector<int> target_loc = ai->get_current_target_loc();

// Returns the pointer of the stored sound amplitude data. Normalized 0-1.
std::vector<float>* stored_data_ptr = ai->get_audio_data();

delete ai;
```
*/
class AI {
    private:
        // Deepstream configuration file path
        std::string ds_config_file_path;     
        // Audio recognition model pack file path         
        std::string audio_model_file_path;       
        // Model trained sample rate         
        int target_sr;                 
        // Madel trained maximum audio lenght                     
        int max_seconds;                     
        // Holds model recognized colors as boolean three value, red green blue respectively.
        std::vector<bool> current_targets_color = std::vector<bool>(3, false); 
        // holds current targets locations xywh format.
        std::vector<std::vector<int>> current_targets_loc = std::vector<std::vector<int>>(1, std::vector<int>(4, 0));
        // Holds the tracked target ID.
        int selected_target_id = -1;
        // Pointer of audio model class which provides audio process tools.
        AudioModel* audio_model = nullptr;                            
        // Whether audio is recording information for `AI::start_recording()` and `AI::stop_recording()`.
        std::atomic<bool> is_audio_busy{false}; 

    public:
        AI(std::string ds_config_path, std::string audio_model_path, int sample_rate, int max_sec);

        // Starts Deepstream pipeline. The `AI::process_ds_data()` method will be triggered by each frame processed on Deepstream.
        void run_deepstream();

        // Starts voice recording.
        void start_recording();

        // Stops voice recording.
        void stop_recording();

        // Retruns three boolean data that represents wheter selected color red green blue respectively.
        std::vector<bool> get_current_targets_color();

        // Returns the location one of bounding boxes that pairs `AI::selected_target_id` `AI::current_targets_color` which recognized with sound model. 
        // Selected bbox ID will be attached to `AI::selected_target_id`.
        std::vector<int> get_current_target_loc();

        // Returns pointer of stored voice data. 
        std::vector<float>* get_audio_data();

        // Returns a boolean value that tell whether voice is recording. 
        bool get_is_recording();
        
        // Designed to debug all data. Prints out all the information that class holded by `AI::class_info()`
        void get_class_info();

    private:
        // Process the `DsObjectData`s got from Deepstream. Updates`AI::current_targets_loc`. It will be triggered by each frame.
        void process_ds_data(DsObjectData* obj_list, int num_objects, int frame_num);

        // Triggers inference for voice data. It will be trigered by end of the `AI::start_recording()` method.
        void audio_inference(std::vector<bool>& targets);

        // Prints out all the information that class holded.
        void class_info();
};
#endif // AICLASS_HPP