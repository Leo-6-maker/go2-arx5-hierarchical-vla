// Short command handoff only. Never hold this lock during policy inference,
// physics integration, PD computation or ROS calls. PD always uses try_lock.
#pragma once
#include <mutex>
namespace wm02_command_pair {
inline std::mutex& mutex() { static std::mutex value; return value; }
template<class Function> void publish(Function function) {
    std::lock_guard<std::mutex> lock(mutex());
    function();
}
template<class Function> bool consume(Function function) {
    std::unique_lock<std::mutex> lock(mutex(),std::try_to_lock);
    return lock.owns_lock() && function();
}
}
