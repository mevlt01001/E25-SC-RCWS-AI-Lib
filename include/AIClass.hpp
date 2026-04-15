#ifndef AICLASS_HPP
#define AICLASS_HPP

#include <vector>
#include <atomic>
#include <mutex>

// Last Commit: DsObjectData updated, Mutex added, Test (CPP file) added. 
//              DsObjectData updated to CXCYWH from XYWH. Mutex (AI::get_target_loc_mutex) added to AI::get_current_target_loc() and AI::process_ds_data. Example_main.cpp added to test. 
// TODO: redesing Tracking ID and target loc function. Update DOC and readme.


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
        std::vector<bool> current_targets_color;
        // holds current targets locations xywh-classId-trackID format.
        std::vector<std::vector<float>> current_targets_loc;
        // Holds the target tracked ID.
        int selected_target_id = -1;
        // Pointer of audio model class which provides audio process tools.
        AudioModel* audio_model = nullptr;                            
        // Whether audio is recording information for `AI::start_recording()` and `AI::stop_recording()`.
        std::atomic<bool> is_audio_busy{false}; 
        // A mutex for detecten objects list to prevent rece condition between deepstrteam_app_main.c and this class.
        std::mutex get_target_loc_mutex;

    public:
        /**
        * Class has just one constructor. 
        * `ds_config_path`: Deepstream configuration file path.
        * `audio_model_path`: Pre-trained Audio Recognition model file path.
        * `sample_rate`: Pre-trained Audio Recognition model sample rate.
        * `max_sec`: Pre-trained Audio Recognition model maximum sound duration.
        */
        AI(std::string ds_config_path, std::string audio_model_path, int sample_rate, int max_sec);

        // Starts Deepstream pipeline. The `AI::process_ds_data()` method will be triggered by each frame processed on Deepstream.
        void run_deepstream();

        // Starts voice recording.
        void start_recording();

        // Stops voice recording.
        void stop_recording();

        // Retruns three boolean data that represents wheter selected color red green blue respectively.
        const std::vector<bool>* get_current_targets_color();

        /**
         * Retruns an 6 elements array pointer `AI::current_targets_loc[i]`.
         * Firstly check if not setted `AI::selected_target_id` to -1. if not, set to `AI::current_targets_loc[0][5]` and returns `AI::current_targets_loc[0]` pointer.
         * If set, retuns `AI::current_targets_loc[i]` which its classid pairs `AI::selected_target_id`. If not found, `AI::selected_target_id` get setted to `-1` return `nullptr`.
         */
        std::vector<float> get_current_target_loc();

        // Returns pointer of stored voice data. 
        const std::vector<float>* get_audio_data();

        // Returns a boolean value that tell whether voice is recording. 
        bool get_is_recording();
        
        // Designed to debug all data. Prints out all the information that class holded by `AI::class_info()`
        void get_class_info();

        // Process the `DsObjectData`s got from Deepstream. Updates`AI::current_targets_loc`. It will be triggered by each frame.
        void process_ds_data(DsObjectData* obj_list, int num_objects, int frame_num);

        void reset_tracking_color();

    private:
        // Triggers inference for voice data. It will be trigered by end of the `AI::start_recording()` method.
        void audio_inference(std::vector<bool>& targets);

        // Prints out all the information that class holded.
        void class_info();
};
#endif // AICLASS_HPP