import struct,subprocess,shutil
from pathlib import Path
import pytest
from go2arx5_vla.world_model.native_callback_replay_v61 import RECORD,prefix_schedule,patch


def trace(path,events):
    with path.open('wb')as f:
        f.write(b'WM02IT01'+struct.pack('<II',RECORD.size,65536))
        for ticket,(kind,step,policy,n)in enumerate(events):
            token=step if kind==12 else policy
            f.write(RECORD.pack(ticket,100+ticket,step*.002,kind,n,0,policy,policy,policy,token,*([0.]*96)))


def test_extracts_actual_callbacks_and_zero_callback_tick(tmp_path):
    p=tmp_path/'trace';trace(p,[(2,0,1,0),(5,0,1,12),(12,1,1,86),
        (9,1,1,90),(10,1,1,86),(9,1,1,90),(10,1,1,86),(12,2,1,86),
        (12,3,1,86),(9,3,1,90),(10,3,1,86),(12,4,1,86)])
    v=prefix_schedule(p,stop_tick=4)
    assert v['callbacks']==3
    assert v['rows']==[dict(native_tick=1,phases=[dict(policy=1,count=2)]),dict(native_tick=2,phases=[]),dict(native_tick=3,phases=[dict(policy=1,count=1)])]
    assert not v['state_restore_proven'] and v['physics_integrations']==0


def test_missing_trace_and_non_native_clock_are_rejected(tmp_path):
    p=tmp_path/'trace';trace(p,[(12,1,1,86),(12,3,1,86)])
    with pytest.raises(ValueError,match='CONTIGUOUS_TWO_MS'):prefix_schedule(p,stop_tick=3)


def test_patch_preserves_all_original_control_math_and_gates():
    cpp='void RL_Sim::RobotControl()\n{\n    const std::lock_guard<std::recursive_mutex> lock(sim->mtx);\nthis->GetState(&this->robot_state);\nORIGINAL_PAUSE_ACK\n    this->StateController(&this->robot_state, &this->robot_command);\nORIGINAL_CONTROL_MATH\n}\nvoid RL_Sim::StartRosInterface()\n{\n}\n        if(!wm02_applied_control_clock::clock().policy_due(wm02_policy_time_s,wm02_policy_period_ns))return;\n    wm02_applied_control_clock::clock().control(this->mj_data->time,std::llround(double(this->params.Get<float>("dt"))*1e9));'
    utils='if(!wm02_applied_control_clock::clock().ready(d->time))continue;\nmj_step(m,d);\nif(!wm02_applied_control_clock::clock().ready(d->time))break;\nmj_step(m,d);'
    a,b=patch(cpp,utils)
    assert 'ORIGINAL_CONTROL_MATH'in a and b.count('mj_step(m,d);')==2
    assert a.index('this->GetState')<a.index('ORIGINAL_PAUSE_ACK')<a.index('control_allowed(')<a.index('this->StateController')
    with pytest.raises(ValueError):patch(a,b)
    with pytest.raises(ValueError):patch(cpp,utils.replace('break','continue')+utils)


@pytest.mark.skipif(not shutil.which('g++'),reason='C++ compiler required')
def test_real_cpp_gate_reproduces_missing_tick11_callback_and_preserves_old_gate(tmp_path):
    schedule=tmp_path/'s';lines=['WM02_CALLBACK_REPLAY_V61 13']
    for i in range(1,13):lines.append(str(i)+' '+('2 1 3 2 1'if i==10 else'1 2 2'if i==11 else'0'))
    schedule.write_text('\n'.join(lines)+'\n')
    source=tmp_path/'t.cpp';source.write_text(r'''
#include "callback_replay_v61.hpp"
#include "applied_control_clock.hpp"
int main(int argc,char**argv){
 wm02_callback_replay_v61::Clock c(argv[1]),off;
 if(!off.control_allowed(11,2)||!off.integration_ready(11))return 1;
 if(c.integration_ready(10)||c.policy_allowed(10,2))return 2;
 for(int i=0;i<3;++i){if(!c.control_allowed(10,1))return 3;c.control_complete(10);}
 if(c.control_allowed(10,1)||!c.policy_allowed(10,2)||c.integration_ready(10))return 4;
 if(!c.control_allowed(10,2))return 5;c.control_complete(10);
 if(!c.integration_ready(10)||c.control_allowed(10,2))return 6;
 if(c.integration_ready(11)||!c.control_allowed(11,2))return 7;
 c.control_complete(11);if(c.integration_ready(11))return 8;
 c.control_complete(11);if(!c.integration_ready(11)||c.control_allowed(11,2))return 9;
 // The replay gate is an additional conjunct;it never authorizes the old clock.
 wm02_applied_control_clock::Clock old;
 if(old.ready(.022)&&c.integration_ready(11))return 10;
 if(!c.integration_ready(12)||c.control_allowed(12,2)||!c.integration_ready(13))return 11;
 return 0;
}
''')
    include=Path(__file__).resolve().parents[2]/'src/go2arx5_vla/target_mujoco';binary=tmp_path/'test'
    subprocess.run(['g++','-std=c++17','-O2','-pthread','-I'+str(include),str(source),'-o',str(binary)],check=True,capture_output=True)
    subprocess.run([str(binary),str(schedule)],check=True,timeout=10)
