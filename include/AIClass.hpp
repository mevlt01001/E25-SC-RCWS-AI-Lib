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
 * @brief This is an orchestrator class. This class works in conjunction with the
 * `AudioModel` class, which handles audio recording and target recognition functions.
 * * Usage Example:
 * ```
 * // Init `AI` Class.
 * AI* ai = new AI(
 * "path/to/deepstream_app.txt", // Deepstream configuration file path
 * "path/to/audio_model.pt",     // Pre trained Audio Recognition model file path
 * 16000,                        // Pre trained Audio Recognition model sample rate
 * 8                             // Pre trained Audio Recognition model maximum sound duration
 * ); 
 * * ai->run_deepstream();               // Starts Deepstream pipeline
 * ai->start_recording();              // Starts voice recording.
 * ai->stop_recording();               // Stops voice recording.
 * * // Returns three boolean data that represents whether selected color is red, green, or blue respectively.
 * std::vector<bool> targets_color = ai->get_current_targets_color();
 * * // Returns the location of one of the bounding boxes that pairs `AI::selected_target_id` and `AI::current_targets_color`.
 * std::vector<int> target_loc = ai->get_current_target_loc();
 * * // Returns the pointer of the stored sound amplitude data. Normalized 0-1.
 * std::vector<float>* stored_data_ptr = ai->get_audio_data();
 * * delete ai;
 * ```
 */
class AI {
    private:
        // Deepstream configuration file path
        std::string ds_config_file_path;     
        // Audio recognition model pack file path         
        std::string audio_model_file_path;       
        // Model trained sample rate         
        int target_sr;                 
        // Model trained maximum audio length                     
        int max_seconds;                     
        // Holds model recognized colors as boolean three value, red green blue respectively.
        std::vector<bool> current_targets_color;
        // Holds current targets locations xywh-classId-trackID format.
        std::vector<std::vector<float>> current_targets_loc;
        // Holds the target tracked ID.
        int selected_target_id = -1;
        // Pointer of audio model class which provides audio process tools.
        AudioModel* audio_model = nullptr;                            
        // Whether audio is recording information for `AI::start_recording()` and `AI::stop_recording()`.
        std::atomic<bool> is_audio_busy{false}; 
        // A recursive mutex for detected objects list to prevent race condition between deepstream_app_main.c and this class.
        // Used `std::recursive_mutex` because `AI::get_current_target_loc` can work recursively.
        std::recursive_mutex get_target_loc_mutex;

    public:
        /**
         * @brief Initializes the AI class with DeepStream and Audio model configurations.
         * Sets up the file paths, audio parameters, initializes the AudioModel instance,
         * and registers the external bounding box callback for DeepStream.
         * * @param ds_config_path Path to the DeepStream configuration text file.
         * * @param audio_model_path Path to the PyTorch audio recognition model file.
         * * @param sample_rate Target sample rate for the audio model.
         * * @param max_sec Maximum duration in seconds for audio recording.
         */
        AI(std::string ds_config_path, std::string audio_model_path, int sample_rate, int max_sec);

        /**
         * @brief Starts the DeepStream pipeline in a detached background thread.
         * Creates simulated command-line arguments to pass to `deepstream_app_main` (deepstream-app -C config.txt)
         * and isolates the pipeline execution to prevent blocking the main application thread.
         */
        void run_deepstream();

        /**
         * @brief Initiates audio recording for voice command recognition.
         * Checks if the audio model is already busy. If not, it sets the busy flag
         * and triggers the recording process in the underlying `AudioModel`.
         */
        void start_recording();

        /**
         * @brief Stops the ongoing audio recording and triggers model inference.
         * If a recording is active, it stops the audio buffer and spawns a detached thread
         * to run inference without blocking the main loop. Unlocks the busy flag upon completion.
         */
        void stop_recording();

        /**
         * @brief Retrieves the currently active target colors identified by the audio model.
         * * @return `const std::vector<bool>*` A pointer to a 3-element boolean vector representing [Red, Green, Blue].
         */
        const std::vector<bool>* get_current_targets_color();

        /**
         * @brief Retrieves the location data of the currently tracked target.
         * This method is thread-safe and safely reads from `current_targets_loc`. 
         * The logic flow is as follows:
         * - If no target is currently selected (`selected_target_id == -1`), it locks onto the first available target in the list.
         * - If a target is already being tracked, it returns its updated location.
         * - If the tracked target leaves the camera frame (is lost), it automatically resets the tracking ID 
         * and recursively calls itself to lock onto a new target.
         * * @return std::vector<float> A 6-element bounding box array `[cx, cy, w, h, class_id, tracking_id]`. 
         * Returns an empty vector `{}` if there are no targets on the screen.
         */
        std::vector<float> get_current_target_loc();

        /**
         * @brief Retrieves the raw audio buffer data from the audio model.
         * * @return const std::vector<float>* A pointer to the floating-point audio data buffer.
         */
        const std::vector<float>* get_audio_data();

        /**
         * @brief Checks the current recording status of the audio model.
         * * @return bool True if the system is currently recording audio, false otherwise.
         */
        bool get_is_recording();
        
        /**
         * @brief Public interface to output the current state and configuration of the AI class.
         * Calls the private `class_info()` method to print variables to standard output.
         */
        void get_class_info();

        /**
         * @brief Processes bounding box data received from the DeepStream pipeline.
         * This method is thread-safe. It locks `get_target_loc_mutex` to safely clear
         * and update `current_targets_loc` with objects that match the actively selected colors.
         * * @param obj_list Pointer to the array of object data from the current frame.
         * * @param num_objects Number of objects detected.
         * * @param frame_num The current frame number.
         */
        void process_ds_data(DsObjectData* obj_list, int num_objects, int frame_num);

        /**
         * @brief Resets the actively tracked colors and the currently selected target ID.
         * Clears the boolean color flags (sets all to false) and resets `selected_target_id` to -1,
         * effectively halting the tracking of the current object until a new voice command is processed.
         */
        void reset_tracking();

    private:
        /**
         * @brief Performs inference on the recorded audio buffer using the PyTorch model.
         * Converts the audio buffer to a LibTorch tensor, runs it through the model,
         * and updates the provided targets boolean array based on recognized vocal commands.
         * * @param targets Reference to the boolean vector where the recognized color flags will be stored.
         */
        void audio_inference(std::vector<bool>& targets);

        /**
         * @brief Prints detailed internal state information of the AI object to standard output.
         * Used for debugging purposes to inspect paths, configuration parameters,
         * and pointer statuses.
         */
        void class_info();
};
#endif // AICLASS_HPP