#include <iostream>
#include <thread>
#include <string>
#include <chrono>
#include <atomic>
#include "AIClass.hpp"

int main(int argc, char* argv[]) {

    AI* ai = new AI(
        "/home/orin2/Downloads/customDeepstreamSample/deepstream_app.txt", 
        "/home/orin2/workspace/deepstream-app-stuff/audio_model.pt",           
        16000,                      
        8                           
    ); 

    ai->run_deepstream();

    std::atomic<bool> is_running{true};

    std::thread input_thread([&]() {
        std::string input;
        while (is_running) {
            std::getline(std::cin, input); 
            
            if (input == "q") {
                is_running = false; 
            } 
            else if (input == "r") { 
                ai->reset_tracking();
            } 
            else if (input.empty()) {
                if (ai->get_is_recording()) {
                    ai->stop_recording();
                } else {
                    ai->start_recording();
                }
            }
        }
    });

    while (is_running) {
        std::string durum = ai->get_is_recording() ? "Recording..." : "Not Recording!";
        
        std::string renk_str = "-,-,-";
        const std::vector<bool>* colors = ai->get_current_targets_color();
        if (colors && colors->size() >= 3) {
            renk_str = std::to_string((*colors)[0]) + "," + 
                       std::to_string((*colors)[1]) + "," + 
                       std::to_string((*colors)[2]);
        }

        std::string konum_str = "Yok";
        std::vector<float> loc = ai->get_current_target_loc();
        if (!loc.empty()) {
            konum_str = "x:" + std::to_string((int)(loc)[0]) + 
                        " y:" + std::to_string((int)(loc)[1]);
        }

        std::string tracking_id_str = "No Target";
        int selected_id = ai->get_selected_target_id();
        if (selected_id != -1) {
            tracking_id_str = std::to_string(selected_id);
        }

        int ses_boyut = 0;
        const std::vector<float>* audio_data = ai->get_audio_data();
        if (audio_data) {
            ses_boyut = audio_data->size();
        }

        std::cout << "\r\033[K"
                  << "[" << durum << "] "
                  << "Color(R,G,B):" << renk_str << " | "
                  << "Location:" << konum_str << " | "
                  << "Tracking ID:" << tracking_id_str << " | "
                  << "Sound Buff Lenght:" << ses_boyut << " | "
                  << "[r/q/ENTER]: " << std::flush;

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    input_thread.join(); 
    delete ai;
    return 0;
}