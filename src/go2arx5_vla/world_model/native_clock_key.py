"""Serialize the ROS key event with its existing native control consumer.

The old callback can inject ENTER after RobotControl tests the key but before
ClearInput, losing the event. This changes event delivery only. It neither
retries the toggle nor changes any native controller or integration formula.
"""

CALLBACK = '[this](const std_msgs::msg::String::SharedPtr msg) { this->InjectKey(msg->data); }'
SERIALIZED = '''[this](const std_msgs::msg::String::SharedPtr msg) {
            const std::lock_guard<std::recursive_mutex> key_guard(sim->mtx);
            this->InjectKey(msg->data);
            double key_flags[]={double(msg->data=="ENTER"),double(simulation_running),double(task_clock_paused_)};
            WM02InitNumeric(wm02_init::KEY_RECEIVED,mj_data->time,0,key_flags,3);
        }'''
TOGGLE = '        simulation_running = !simulation_running;'
APPLIED = '''
        double key_flags[]={double(simulation_running),double(task_clock_paused_)};
        WM02InitNumeric(wm02_init::KEY_APPLIED,mj_data->time,0,key_flags,2);'''


def patch(cpp):
    if (cpp.count(CALLBACK) != 1 or cpp.count(TOGGLE) != 1
            or cpp.count('const std::lock_guard<std::recursive_mutex> lock(sim->mtx);') != 1
            or cpp.count('    this->control.ClearInput();') != 1):
        raise ValueError('EXACT_SINGLE_NATIVE_KEY_CALLBACK_AND_CONTROL_CONSUMER_REQUIRED')
    changed = cpp.replace(CALLBACK, SERIALIZED).replace(TOGGLE, TOGGLE+APPLIED)
    if changed.replace(SERIALIZED, CALLBACK).replace(TOGGLE+APPLIED, TOGGLE) != cpp:
        raise ValueError('KEY_EVENT_PATCH_CHANGED_UNRELATED_NATIVE_TEXT')
    return changed
