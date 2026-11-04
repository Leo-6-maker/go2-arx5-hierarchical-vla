"""Bind unchanged native periods to actual physical time without blocking PD."""
INCLUDE='#include "control_clock.hpp"\n'
BEGIN='''    if (this->rl_init_done && simulation_running)
    {
'''
GATE='''        const int64_t wm02_control_period_ns=std::llround(double(this->params.Get<float>("dt"))*1e9);
        const int64_t wm02_policy_period_ns=wm02_control_period_ns*this->params.Get<int>("decimation");
        const double wm02_policy_time_s=PI0294SimNs.load()*1e-9;
        if(!wm02_control_clock::clock().policy_due(wm02_policy_time_s,wm02_policy_period_ns))return;
'''


def patch(cpp,utils):
    anchors=[(cpp,BEGIN),(cpp,'        // this->TorqueProtect(this->output_dof_tau);'),(cpp,'    this->SetCommand(&this->robot_command);'),(cpp,'            wm02_cold_history_ready_ = false;'),(utils,'            // run single step, let next iteration deal with timing\n            mj_step(m, d);'),(utils,'              // call mj_step\n              mj_step(m, d);')]
    if any(text.count(anchor)!=1 for text,anchor in anchors):raise ValueError('EXACT_NATIVE_CALLBACK_CLOCK_ANCHORS_REQUIRED')
    changed=INCLUDE+cpp.replace(BEGIN,BEGIN+GATE)
    complete='        wm02_control_clock::clock().policy_complete(wm02_policy_time_s,wm02_policy_period_ns);\n'
    changed=changed.replace(anchors[1][1],complete+anchors[1][1])
    mark='\n    wm02_control_clock::clock().control(this->mj_data->time,std::llround(double(this->params.Get<float>("dt"))*1e9));'
    changed=changed.replace(anchors[2][1],anchors[2][1]+mark)
    reset='\n            wm02_control_clock::clock().reset();'
    changed=changed.replace(anchors[3][1],anchors[3][1]+reset)
    a='            if(!wm02_control_clock::clock().ready(d->time))continue;\n'
    b='              if(!wm02_control_clock::clock().ready(d->time))break;\n'
    new_utils=INCLUDE+utils.replace(anchors[4][1],anchors[4][1].replace('            mj_step',a+'            mj_step')).replace(anchors[5][1],anchors[5][1].replace('              mj_step',b+'              mj_step'))
    inverse=changed[len(INCLUDE):].replace(BEGIN+GATE,BEGIN).replace(complete,'').replace(mark,'').replace(reset,'')
    if inverse!=cpp or new_utils[len(INCLUDE):].replace(a,'').replace(b,'')!=utils:raise ValueError('CLOCK_PATCH_CHANGED_NATIVE_MATH_OR_PERIOD_TEXT')
    return changed,new_utils
