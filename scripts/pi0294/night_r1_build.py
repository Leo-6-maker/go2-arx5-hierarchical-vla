"""CPU-only resource adapter build and exact unchanged-control comparison."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

from .night_r1_runtime import read, sha
from scripts.pi0292.durable import write


def run(root, native):
    root, native = Path(root), Path(native)
    out=root/'runtime/TASK_ADAPTER_R1_01';out.mkdir()
    source=Path(__file__).parent
    old=(source/'task_actuation.cpp').read_text()
    new=(source/'task_actuation_r1.cpp').read_text()
    start=old.index('\nvoid read_command');new_start=new.index('\nvoid read_command')
    assert old[start:]==new[new_start:]
    control_start=old.index(' for(int i=0;i<2;++i) {',old.index('extern "C"'))
    control_end=old.index(' double before=',control_start)
    loop=old[control_start:control_end]
    new_loop=new[new.index(' for(int i=0;i<2;++i) {',new.index('extern "C"')):new.index(' double before=',new.index('extern "C"'))]
    assert loop==new_loop
    include=native/'src/ctrl/rl_sar/library/mujoco'
    commands=[]
    def execute(argv,label,**kwargs):
        p=subprocess.run(argv,capture_output=True,text=True,timeout=180,**kwargs)
        (out/(label+'.log')).write_text(p.stdout+p.stderr)
        commands.append(dict(label=label,exit=p.returncode))
        write(out/'BUILD_COMMANDS.json',commands)
        p.check_returncode();return p.stdout
    execute(['g++','-std=c++17','-O2','-shared','-fPIC','-I'+str(include/'include'),'-I'+str(source),
             '-L'+str(include/'lib'),'-Wl,-rpath,'+str(include/'lib'),str(source/'task_actuation_r1.cpp'),
             '-lmujoco','-ldl','-o',str(out/'task_actuation.so')],'adapter')
    old_build=read(root/'runtime/TASK_ADAPTER_BUILD_03/BUILD_RECEIPT.json')
    assert sha(root/'runtime/TASK_ADAPTER_BUILD_03/task_observer.so')==old_build['binaries']['task_observer']
    (out/'task_observer.so').symlink_to(root/'runtime/TASK_ADAPTER_BUILD_03/task_observer.so')
    quota_probe=out/'quota_probe.cpp'
    quota_probe.write_text('#include "task_quota_r1.hpp"\n#include <iostream>\nint main(int argc,char**argv){uint64_t v=0;bool ok=gr00t_quota_r1::read_quota(argc>1?argv[1]:nullptr,v);std::cout<<ok<<" "<<v;return 0;}\n')
    execute(['g++','-std=c++17','-O2','-I'+str(source),str(quota_probe),'-o',str(out/'quota_probe')],'quota-probe')
    cases=[];case_root=out/'cases';case_root.mkdir()
    def case(label, quota, bound, warmup=5, task=128, run='run', card='card', reservation='reservation', env_updates=None, missing=False):
        q=case_root/(label+'.quota');q.write_text(str(quota)+'\n')
        binding=case_root/(label+'.binding')
        if not missing:binding.write_text('\n'.join(['GRTQUOTA2',run,card,reservation,str(bound),str(warmup),str(task)])+'\n')
        env=dict(os.environ,GR00T_TASK_QUOTA_BINDING=str(binding),GR00T_TASK_RUN_UUID='run',GR00T_TASK_CARD_SHA256='card',GR00T_TASK_RESERVATION_ID='reservation')
        env.update(env_updates or {})
        answer=execute([str(out/'quota_probe'),str(q)],'quota-'+label,env=env)
        passed=answer.split()[0]=='1';cases.append(dict(label=label,accepted=passed))
        return passed
    for duration in [32,64,96,128]:
        limit=(5+duration)*500+3
        assert case('valid_'+str(duration),limit,limit,task=duration)
        assert not case('over_card_'+str(duration),limit+1,limit,task=duration)
    for label,value in [('zero','0'),('negative','-1'),('decimal','2500.0'),('trailing','2500 x'),('nan','nan'),('overflow','18446744073709551616')]:
        assert not case(label,value,66503)
    assert not case('missing_binding',2500,66503,missing=True)
    assert not case('cross_run',2500,66503,run='other')
    assert not case('cross_card',2500,66503,card='other')
    assert not case('cross_reservation',2500,66503,reservation='other')
    assert not case('over_absolute',66504,66504)
    assert not case('bound_over_duration',2500,66503,task=32)
    assert not case('fractional_bound',2500,'66503.0')
    assert not case('fractional_duration',2500,66503,task='128.0')
    assert case('startup_2500',2500,2503,task=0)
    # Compile the two exact original/new finite-gripper blocks in isolated CPU
    # functions. No mjModel is loaded and mj_step is never invoked.
    prefix='''#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>
struct Model{int jnt_qposadr[2]={0,1};int jnt_dofadr[2]={0,1};struct{double timestep=.002;}opt;};
struct Data{double qpos[2],qvel[2],ctrl[20],time;};
int joint[2]={0,1},actuator[2]={18,19};double target[2],opening,expiry;bool has_command;
void fail(const char* s){throw std::runtime_error(s);}
void probe(Model* m,Data* d){
'''
    main='''}
int main(){Model m;Data d{};std::cout<<std::setprecision(17);
while(std::cin>>d.qpos[0]>>d.qpos[1]>>d.qvel[0]>>d.qvel[1]>>target[0]>>target[1]>>opening>>has_command>>d.time>>expiry){
for(int j=0;j<20;j++)d.ctrl[j]=j*.0125;probe(&m,&d);
std::cout<<target[0]<<" "<<target[1];for(double x:d.ctrl)std::cout<<" "<<x;std::cout<<"\\n";} }
'''
    for name,block in [('old',loop),('new',new_loop)]:
        path=out/('control_'+name+'.cpp');path.write_text(prefix+block+main)
        execute(['g++','-std=c++17','-O2',str(path),'-o',str(out/('control_'+name))],'control-compile-'+name)
    states=[]
    # Legal short-state probes include every saved night startup point.
    oldnight=Path(read(root/'RUN_MANIFEST.json')['dependencies']['prior_night_run'])
    raw=oldnight/'factory/episodes/TFN_V2_DEV_T2_00_E1/physics.jsonl'
    for line in raw.read_text().splitlines():
        row=json.loads(line)
        if row.get('kind')=='sample':
            # Record qpos uses scalar MuJoCo joint layout, fingers q25/q26.
            q=row['qpos'];dq=row['qvel'];t=row['sim_time_s']
            states.append([q[25],q[26],dq[24],dq[25],.002,.002,.002/.044,0,t,-1])
    for q in [.002,.022,.042]:
        for dq in [-.1,0.,.1]:
            for flag in [0,1]:states.append([q,q,dq,-dq,q,q,.042/.044,flag,1.,1.4])
    text=''.join(' '.join(map(str,row))+'\n' for row in states)
    output_old=execute([str(out/'control_old')],'control-old-output',input=text)
    output_new=execute([str(out/'control_new')],'control-new-output',input=text)
    assert output_old==output_new
    tail=old[old.index(' if(steps==limit)'):]
    assert tail==new[new.index(' if(steps==limit)'):]
    # The exact existing tail hook is exercised with a CPU mock callback.
    tail_block=tail[:tail.rfind('\n}')]
    tail_code='''#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <cassert>
#define RTLD_DEFAULT nullptr
uint64_t seen=0,calls=0;
void mock_pause(uint64_t n){seen=n;++calls;}
void* dlsym(void*,const char* name){return std::strcmp(name,"GR00TTaskPause")==0?reinterpret_cast<void*>(&mock_pause):nullptr;}
void fail(const char* s){throw std::runtime_error(s);}
void test_tail(uint64_t steps,uint64_t limit){
'''+tail_block+'''}
int main(){for(uint64_t duration:{32,64,96,128}){uint64_t limit=(5+duration)*500+3;calls=0;seen=0;test_tail(limit-1,limit);assert(!calls);test_tail(limit,limit);assert(calls==1&&seen==limit);test_tail(limit+1,limit);assert(calls==1);} }
'''
    tail_path=out/'tail_probe.cpp';tail_path.write_text('#include <initializer_list>\n'+tail_code)
    execute(['g++','-std=c++17','-O2',str(tail_path),'-o',str(out/'tail_probe')],'tail-compile')
    execute([str(out/'tail_probe')],'tail-exact-mock')
    receipt=dict(status='PASS_CPU_QUOTA_CONTROL_AND_TAIL_HOOK_CONTRACTS_NOT_LONG_PHYSICS',
        sources=dict(task_actuation=sha(source/'task_actuation_r1.cpp'),quota_header=sha(source/'task_quota_r1.hpp')),
        binaries=dict(task_actuation=sha(out/'task_actuation.so'),task_observer=sha(out/'task_observer.so')),
        quota_cases=cases,control_states=len(states),control_output_byte_equal=True,original_gripper_control_block_byte_equal=True,
        original_pause_tail_block_byte_equal=True,long_domain_tail_CPU_mock_PASS=True,
        native_WBC_rebuild=False,native_pause_binary_changed=False,safety_limits_changed=False,new_physics=0)
    write(out/'BUILD_RECEIPT.json',receipt);write(root/'evidence/TASK_QUOTA_COMPATIBILITY.json',receipt)
    return receipt
