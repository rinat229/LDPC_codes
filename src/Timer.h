#include <chrono>
#include <iostream>

class Timer
{
    public: 
        Timer(){
            start_point = std::chrono::high_resolution_clock::now();
        }

        void Stop(){
            auto end_point = std::chrono::high_resolution_clock::now();

            auto start = std::chrono::time_point_cast<std::chrono::microseconds> (start_point).time_since_epoch().count();
            auto end = std::chrono::time_point_cast<std::chrono::microseconds> (end_point).time_since_epoch().count();

            auto duration = end - start;
            std::cout << "duration - " << duration << "mcs" <<  std::endl;
        }
    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> start_point;

};