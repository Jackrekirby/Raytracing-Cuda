#pragma once
#include <chrono>
#include <vector>
#include <map>
#include <string>

typedef std::chrono::time_point<std::chrono::system_clock> Time;

class StopWatch {
public:
    StopWatch(Time start_time) : start_time(start_time) {

    }

    StopWatch() : start_time(std::chrono::system_clock::now()) {

    }

    void stop() {
        stop_time = std::chrono::system_clock::now();
    }

    int delta() {
        std::chrono::duration<float> difference = stop_time - start_time;

        return static_cast<int>(difference.count() * 1000);

    }

    Time start_time;
    Time stop_time;
};

class TimeIt {
public:
    TimeIt() { }

    void start() {
        std::string name = "timer_" + std::to_string(stack.size());
        start(name);
    }

    void start(const std::string& name) {
        stack.push_back(name);
        start_times[name] = StopWatch();
    }

    void stop() {
        const std::string& name = stack.back();
        stop(name);
        stack.pop_back();
    }

    void stop(const std::string& name) {
        StopWatch& stopwatch = start_times[name];
        stopwatch.stop();

        VAR(name + " time", stopwatch.delta());
    }

    std::chrono::time_point<std::chrono::system_clock> start_time;
    std::map<std::string, StopWatch> start_times;
    std::vector<std::string> stack;
};