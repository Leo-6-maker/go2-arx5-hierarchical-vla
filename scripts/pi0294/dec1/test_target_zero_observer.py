"""Compiled C ABI mocks only, never MuJoCo/models/ROS or native qualification.

The fake library supplies a fixed CPU-fixture realtime clock so historical
cutoffs stay tested without ever passing a fake now to an actual owner.
No fake library or fixture clock is an allowed native dispatch input.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest
from scripts.pi0294.task_state import parse_state

HEADER = r'''
#pragma once
#define mjVERSION_HEADER 327
typedef double mjtNum;
struct mjContact { int geom1,geom2; double dist,frame[9],pos[3]; };
struct mjModel { int nq,nv,nu; };
struct mjData { double time; double *qpos,*qvel,*ctrl; int ncon; mjContact* contact; };
extern "C" {
void mj_forward(const mjModel*,mjData*);
void mj_deleteData(mjData*);void mj_deleteModel(mjModel*);
void mj_contactForce(const mjModel*,const mjData*,int,double*);
int mj_version();
void mj_step(const mjModel*,mjData*);void mj_step1(const mjModel*,mjData*);
void mj_step2(const mjModel*,mjData*);void mj_Euler(const mjModel*,mjData*);
void mj_RungeKutta(const mjModel*,mjData*,int);void mj_implicit(const mjModel*,mjData*);
}
'''
MOCK = r'''
#include <mujoco/mujoco.h>
#include <fstream>
#include <cstdlib>
#include <dlfcn.h>
#include <time.h>
#include <unistd.h>
extern "C" int mock_lock_held();
extern "C" int mj_version(){return std::getenv("FAKE_WRONG_VERSION")?328:327;}
extern "C" int clock_gettime(clockid_t c,timespec* p) {
 if(c==CLOCK_REALTIME){p->tv_sec=1792790000;p->tv_nsec=0;return 0;}
 using F=int(*)(clockid_t,timespec*);static auto f=(F)dlsym(RTLD_NEXT,"clock_gettime");return f(c,p);
}
extern "C" void mj_forward(const mjModel*,mjData*) {}
extern "C" void mj_deleteData(mjData*) {}
extern "C" void mj_deleteModel(mjModel*) {}
extern "C" void mj_contactForce(const mjModel*,const mjData*,int,double* v) {
 if(!mock_lock_held())_exit(99);
 for(int i=0;i<6;i++)v[i]=i+.5;
}
void integration(mjData* d) {
 std::ofstream f(std::getenv("FAKE_INTEGRATION_EFFECT"));f<<"ORIGINAL_WAS_CALLED";d->time+=.002;
}
extern "C" void mj_step(const mjModel*,mjData* d){integration(d);}
extern "C" void mj_step1(const mjModel*,mjData* d){integration(d);}
extern "C" void mj_step2(const mjModel*,mjData* d){integration(d);}
extern "C" void mj_Euler(const mjModel*,mjData* d){integration(d);}
extern "C" void mj_RungeKutta(const mjModel*,mjData* d,int){integration(d);}
extern "C" void mj_implicit(const mjModel*,mjData* d){integration(d);}
'''
DRIVER = r'''
#include <mujoco/mujoco.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <thread>
#include <unistd.h>
thread_local bool held=false;
extern "C" int mock_lock_held(){return held;}
struct RL_Sim {
 std::recursive_mutex mutex;
 __attribute__((noinline)) void PauseTaskClock(unsigned long step);
} original;
void RL_Sim::PauseTaskClock(unsigned long step) {
 std::lock_guard<std::recursive_mutex> lock(mutex);held=true;
 std::string path=std::getenv("GR00T_TASK_PAUSE_ACK"),tmp=path+".pending";
 std::ofstream f(tmp);f<<"{\"step\":"<<step<<",\"CPU_fixture_only\":true}\n";f.close();
 if(std::getenv("FAKE_NO_ACK_RENAME")){held=false;return;}
 std::rename(tmp.c_str(),path.c_str());held=false;
}
extern "C" void GR00TTaskPause(unsigned long step){
 if(!std::getenv("FAKE_INLINE_ORIGINAL_PAUSE")){original.PauseTaskClock(step);return;}
 std::lock_guard<std::recursive_mutex> lock(original.mutex);held=true;
 std::string p=std::getenv("GR00T_TASK_PAUSE_ACK"),t=p+".pending";
 std::ofstream f(t);f<<"{\"step\":"<<step<<"}\n";f.close();std::rename(t.c_str(),p.c_str());held=false;
}
int main(int argc,char**argv) {
 mjModel m{34,32,20};double q[34]{},v[32]{},u[20]{};q[3]=1.;u[4]=.25;
 mjContact contact{};contact.geom1=5;contact.geom2=6;contact.dist=-.001;
 contact.frame[0]=contact.frame[4]=contact.frame[8]=1;contact.pos[0]=.2;
 mjData d{0,q,v,u,1,&contact};const char* mode=argc>1?argv[1]:"normal";
 if(!std::strcmp(mode,"mj_step")){mj_step(&m,&d);return 91;}
 if(!std::strcmp(mode,"mj_step1")){mj_step1(&m,&d);return 91;}
 if(!std::strcmp(mode,"mj_step2")){mj_step2(&m,&d);return 91;}
 if(!std::strcmp(mode,"mj_Euler")){mj_Euler(&m,&d);return 91;}
 if(!std::strcmp(mode,"mj_RungeKutta")){mj_RungeKutta(&m,&d,4);return 91;}
 if(!std::strcmp(mode,"mj_implicit")){mj_implicit(&m,&d);return 91;}
 if(!std::strcmp(mode,"nonzero_time"))d.time=.002;
 if(!std::strcmp(mode,"nonfinite"))q[0]=NAN;
 mj_forward(&m,&d);
 if(!std::strcmp(mode,"second_data")){mjData other=d;mj_forward(&m,&other);}
 std::ofstream request(std::getenv("GR00T_ZERO_IO_PAUSE_REQUEST"));request<<"0\n";request.close();
 for(int i=0;i<400 && access(std::getenv("GR00T_TASK_STATE"),F_OK);++i)
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
 if(access(std::getenv("GR00T_TASK_STATE"),F_OK))return 88;
 if(!std::strcmp(mode,"early_delete")){mj_deleteData(&d);return 87;}
 if(!std::strcmp(mode,"bad_ack_caller")){
  std::string p=std::getenv("GR00T_TASK_PAUSE_ACK"),t=p+".rogue";
  std::ofstream f(t);f<<"{}";f.close();std::rename(t.c_str(),p.c_str());return 86;
 }
 // Repeat the same ORIGINAL mutex-held call to provide fresh captures.
 original.PauseTaskClock(0);original.PauseTaskClock(0);
 std::ofstream stop(std::getenv("GR00T_ZERO_IO_STOP"));stop<<"stop\n";stop.close();
 for(int i=0;i<400 && access(std::getenv("GR00T_ZERO_IO_STOPPED"),F_OK);++i)
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
 if(access(std::getenv("GR00T_ZERO_IO_STOPPED"),F_OK))return 85;
 mj_deleteData(&d);mj_deleteModel(&m);return 0;
}
'''


@pytest.fixture(scope='module')
def compiled(tmp_path_factory):
    if not shutil.which('g++'): pytest.skip('C++ compiler unavailable; native build remains unqualified')
    p=tmp_path_factory.mktemp('zero-observer-CPU-ABI-mock');inc=p/'mujoco';inc.mkdir()
    (inc/'mujoco.h').write_text(HEADER);(p/'mock.cpp').write_text(MOCK);(p/'driver.cpp').write_text(DRIVER)
    cpp=Path(__file__).with_name('target_zero_observer.cpp')
    commands=[['g++','-O2','-std=c++17','-fPIC','-shared','-I'+str(p),str(p/'mock.cpp'),'-ldl','-o',str(p/'libmock.so')],
        ['g++','-O2','-std=c++17','-fPIC','-shared','-I'+str(p),str(cpp),'-pthread','-ldl','-o',str(p/'observer.so')],
        ['g++','-O2','-std=c++17','-rdynamic','-I'+str(p),str(p/'driver.cpp'),'-L'+str(p),'-lmock',
         '-Wl,-rpath,'+str(p),'-pthread','-o',str(p/'driver')]]
    for cmd in commands: subprocess.run(cmd,check=True,capture_output=True,timeout=30)
    return p


def run(compiled,tmp_path,mode='normal',change=None):
    paths=dict(GR00T_TASK_STEP_QUOTA='quota',GR00T_TASK_STATE='state.bin',GR00T_TASK_PAUSE_ACK='ack.json',
        GR00T_ZERO_IO_PAUSE_REQUEST='request',GR00T_ZERO_IO_STOP='stop',GR00T_ZERO_IO_STOPPED='stopped.json',
        GR00T_TASK_ERROR='errors',FAKE_INTEGRATION_EFFECT='forbidden-effect')
    env=dict(os.environ,LD_PRELOAD=str(compiled/'observer.so'),GR00T_ZERO_IO_ROLE='DEC1_FROZEN_TARGET_NATIVE_IO_QUALIFICATION_V1',
        GR00T_ZERO_IO_DEADLINE_EPOCH='1792790120',**{k:str(tmp_path/v) for k,v in paths.items()})
    (tmp_path/'quota').write_text('0\n')
    if change:change(env)
    result=subprocess.run([str(compiled/'driver'),mode],env=env,capture_output=True,timeout=5)
    assert not (tmp_path/'forbidden-effect').exists()
    return result


def test_exact_actual_packet_and_contacts_are_copied_under_original_pause_mutex(compiled,tmp_path):
    assert run(compiled,tmp_path).returncode==0
    packet=parse_state((tmp_path/'state.bin').read_bytes())
    assert packet['step']==packet['time_s']==0 and packet['qpos'][3]==1 and packet['ctrl'][4]==.25
    assert len(packet['contacts'])==1 and packet['contacts'][0]['geom0']==5
    assert packet['contacts'][0]['wrench'].tolist()==[i+.5 for i in range(6)]
    proof=json.loads((tmp_path/'stopped.json').read_bytes())
    assert proof['observer_stopped'] and proof['original_pause_called'] and proof['snapshots']>=3
    assert proof['physics_integrations']==0


def test_original_mutex_held_C_ABI_inlining_still_has_exact_main_executable_origin(compiled,tmp_path):
    assert run(compiled,tmp_path,change=lambda env:env.update(FAKE_INLINE_ORIGINAL_PAUSE='1')).returncode==0
    assert parse_state((tmp_path/'state.bin').read_bytes())['step']==0


@pytest.mark.parametrize('name',['mj_step','mj_step1','mj_step2','mj_Euler','mj_RungeKutta','mj_implicit'])
def test_every_integration_entry_denies_before_original_function(compiled,tmp_path,name):
    assert run(compiled,tmp_path,name).returncode==94
    assert (tmp_path/'errors').read_text()=='ZERO_IO_INTEGRATION_FORBIDDEN_'+name+'\n'


@pytest.mark.parametrize('mode,reason',[('nonzero_time','ACTUAL_STATE_OR_ZERO_QUOTA_CHANGED'),
    ('nonfinite','NONFINITE_ACTUAL_STATE'),('second_data','AMBIGUOUS_NATIVE_DATA_POINTER'),
    ('early_delete','DELETE_BEFORE_OBSERVER_STOPPED'),('bad_ack_caller','ACK_NOT_FROM_FROZEN_PAUSE_METHOD')])
def test_wrong_state_lifecycle_or_ack_caller_never_qualifies(compiled,tmp_path,mode,reason):
    assert run(compiled,tmp_path,mode).returncode==94
    assert reason in (tmp_path/'errors').read_text()
    assert not (tmp_path/'stopped.json').exists()


@pytest.mark.parametrize('fault',['role','quota','deadline','old_state','no_native_rename','runtime_version'])
def test_current_role_zero_new_output_deadline_and_actual_hook_required(compiled,tmp_path,fault):
    def change(env):
        if fault=='role':env['GR00T_ZERO_IO_ROLE']='OLD_R4'
        if fault=='quota':(tmp_path/'quota').write_text('25\n')
        if fault=='deadline':env['GR00T_ZERO_IO_DEADLINE_EPOCH']='1792789999'
        if fault=='old_state':(tmp_path/'state.bin').write_bytes(b'CPU_OLD_STATE')
        if fault=='no_native_rename':env['FAKE_NO_ACK_RENAME']='1'
        if fault=='runtime_version':env['FAKE_WRONG_VERSION']='1'
    assert run(compiled,tmp_path,change=change).returncode==94
    assert not (tmp_path/'stopped.json').exists()


def test_source_does_not_replace_native_pause_or_control_fields():
    text=Path(__file__).with_name('target_zero_observer.cpp').read_text()
    assert 'dlsym(RTLD_DEFAULT, "GR00TTaskPause")' in text
    assert 'pause(0);' in text and '_ZN6RL_Sim14PauseTaskClockEm' in text
    assert 'qpos[' not in text and 'qvel[' not in text and 'ctrl[' not in text
    assert 'RL_Sim*' not in text and 'mj_stepSkip' not in text
