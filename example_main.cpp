#include <iostream>
#include <thread>
#include <string>
#include <chrono>
#include <atomic>
#include "AIClass.hpp"

int main(int argc, char* argv[]) {
    std::cout << "--- AI Sistemi Canli Test Ortami ---\n";
    std::cout << "Bilgi: Ekran yenileme hizi maksimumda. Degerler anlik guncelleniyor.\n\n";

    AI* ai = new AI(
        "/home/orin2/Downloads/customDeepstreamSample/deepstream_app.txt", 
        "/home/orin2/workspace/deepstream-app/audio_model.pt",           
        16000,                      
        8                           
    ); 

    ai->run_deepstream();

    std::atomic<bool> is_running{true};

    // KULLANICI GIRDILERINI DINLEYEN THREAD
    std::thread input_thread([&]() {
        std::string input;
        while (is_running) {
            std::getline(std::cin, input); 
            
            if (input == "q") {
                is_running = false; 
            } 
            else if (input == "r") { 
                ai->reset_tracking_color();
            } 
            else if (input.empty()) { // Sadece Enter'a basildiysa
                if (ai->get_is_recording()) {
                    ai->stop_recording();
                } else {
                    ai->start_recording();
                }
            }
        }
    });

    // ANA THREAD: Ekrana durmaksizin (kaydirmadan) veri basar
    while (is_running) {
        // 1. Durum Bilgisi
        std::string durum = ai->get_is_recording() ? "KAYITTA" : "BEKLEMEDE";
        
        // 2. Renk Bilgisi
        std::string renk_str = "-,-,-";
        const std::vector<bool>* colors = ai->get_current_targets_color();
        if (colors && colors->size() >= 3) {
            renk_str = std::to_string((*colors)[0]) + "," + 
                       std::to_string((*colors)[1]) + "," + 
                       std::to_string((*colors)[2]);
        }

        // 3. Konum Bilgisi
        std::string konum_str = "Yok";
        std::vector<float> loc = ai->get_current_target_loc();
        if (!loc.empty()) {
            // Ekrana rahat sigmasi icin ondaliklari atip int olarak basiyoruz
            konum_str = "x:" + std::to_string((int)(loc)[0]) + 
                        " y:" + std::to_string((int)(loc)[1]);
        }

        // 4. Ses Bilgisi
        int ses_boyut = 0;
        const std::vector<float>* audio_data = ai->get_audio_data();
        if (audio_data) {
            ses_boyut = audio_data->size();
        }

        // SİHİRLİ SATIR: 
        // \r       -> İmleci satırın en başına al.
        // \033[K   -> İmleçten itibaren satırın sağında kalan her şeyi sil (eski uzun yazılar kalmasın).
        // std::flush -> \n kullanmadığımız için yazıyı hemen terminale itmeye zorlar.
        std::cout << "\r\033[K"
                  << "[" << durum << "] "
                  << "Renk(R,G,B):" << renk_str << " | "
                  << "Konum:" << konum_str << " | "
                  << "SesBuf:" << ses_boyut << " | "
                  << "Komut [r/q/ENTER]: " << std::flush;

        // KÜÇÜK BİR YAZILIM TESTİ SIRRI:
        // Döngüden 500ms'yi kaldırmak harika ama araya en azından 1 milisaniye (veya mikro saniye) 
        // koymazsan while(true) döngüsü CPU'nun 1 çekirdeğini %100 kilitleyebilir. 
        // 1ms gecikme, ekranda fark edilmez bile ama CPU'ya inanılmaz bir nefes aldırır.
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Kapanis
    input_thread.join(); 
    delete ai;
    
    std::cout << "\n\nSistem basariyla kapatildi. Eline saglik!\n";
    return 0;
}