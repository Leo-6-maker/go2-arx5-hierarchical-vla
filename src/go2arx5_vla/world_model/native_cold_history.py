"""Use the original ObservationBuffer reset API at a cold real-observation boundary."""

def patch(cpp,hpp,utils):
    insert='        this->history_obs_buf.insert(clamped_obs);'
    member='    std::recursive_mutex task_model_pause_mutex_;'
    reset='            PI0289ResetStartup(this->mj_model, this->mj_data);'
    for text,needle in((cpp,insert),(hpp,member),(cpp,reset)):
        if text.count(needle)!=1:raise ValueError('Exact frozen cold-history anchor required:'+needle)
    extra='''        // This is explicit cold padding from the first real observation,
        // not thirty historical physical steps or a changed policy weight.
        if (!wm02_cold_history_ready_) {
            this->history_obs_buf.reset({0}, clamped_obs);
            wm02_cold_history_ready_ = true;
        }
'''
    new_cpp=cpp.replace(insert,extra+insert).replace(reset,reset+'\n            wm02_cold_history_ready_ = false;')
    new_hpp=hpp.replace(member,member+'\n    bool wm02_cold_history_ready_ = false;')
    # No gain, command, inference-period or ongoing-history change is allowed.
    restored=new_cpp.replace(extra,'').replace(reset+'\n            wm02_cold_history_ready_ = false;',reset)
    if restored!=cpp or new_hpp.replace('\n    bool wm02_cold_history_ready_ = false;','')!=hpp:raise ValueError('Unexpected non-cold-history source change')
    return new_cpp,new_hpp,utils
