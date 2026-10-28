"""Reuse the source-pinned incremental PI0288 build in a new PI0289 workspace."""
import base64
import argparse
import sys
from pathlib import Path
WORKTREE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(WORKTREE/'scripts/pi0287_target_sim'))
from owner import remote
ROOT='/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0289'

def body(variant=1):
    build=(WORKTREE/'scripts/pi0288_mujoco/initialization_build_remote.py').read_text()
    build=build.replace("/go2-arx5/PI0288'","/go2-arx5/PI0289'").replace("repair=root/'initialization-repair'","repair=root/'startup-v1'\nrepair.mkdir(exist_ok=True)")
    build=build.replace('PI0288ResetHome(mnew, dnew)','PI0289ResetStartup(mnew, dnew)').replace('PI0288ResetHome(m, d)','PI0289ResetStartup(m, d)')
    build=build.replace('PI0288ResetHome(this->mj_model, this->mj_data)','PI0289ResetStartup(this->mj_model, this->mj_data)').replace('PI0288ResetHome(m_, d_)','PI0289ResetStartup(m_, d_)')
    build=build.replace('#include "native_home.hpp"','#include "native_startup.hpp"')
    build=build.replace("(overlay/'native_home.hpp').write_bytes((repair/'native_home.hpp').read_bytes())", "for name in ['native_home.hpp','native_startup.hpp']:(overlay/name).write_bytes((repair/name).read_bytes())")
    build=build.replace("    (overlay/name).write_text(after)",'''    if name=='rl_sim_mujoco.cpp':
        after=after.replace('    // start physics thread','    sim->run = 0;\\n    simulation_running = false;\\n    // start physics thread')
        token='    this->SetCommand(&this->robot_command);'
        assert after.count(token)==1
        after=after.replace(token,token+'\\n    { std::lock_guard<std::mutex> lock(ros_arm_mutex_); PI0289ControlTrace(this->mj_data, this->fsm.current_state_->GetStateName(), ros_arm_mode_, simulation_running, this->robot_command.motor_command); }')
    (overlay/name).write_text(after)''')
    a=build.index("execute(['g++','-O2','-std=c++17'",build.index("scene=private/"))
    b=build.index("a=subprocess.run([str(repair/'audit_home')",a)
    build=build[:a]+'''cache=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0288/initialization-repair/private-scene.mjb')
assert sha(cache)=='168b1232df2a50a163709c1c1e57d13d39aa3efb92a4c98005e797924142376d'
parity=json.loads((cache.parent/'COMPILED_MODEL_PARITY.json').read_text())
(repair/'COMPILED_MODEL_PARITY.json').write_text(json.dumps(dict(reused_parent_receipt=parity,new_compiler_steps=0)))
''' + build[b:]
    build=build.replace("compiler_self_test_steps=parity['compiler_self_test_steps']", "compiler_self_test_steps=0")
    build=build.replace('USER_DELEGATED_HOME_QUALIFICATION','PI0289_OCS2_MUJOCO_AUTONOMOUS_BRINGUP_AND_DATA_GO_V1').replace('DELIVERED_HOME_ONLY','DELIVERED_HOME_WITH_MPC_INITIAL_ARM_AND_PAUSED_NATIVE_HOLD_V1')
    build=build.replace("repair/'native_home.hpp',repair/'audit_home.cpp'","repair/'native_home.hpp',repair/'native_startup.hpp',repair/'audit_home.cpp'")
    build=build.replace("'x5_joint2/3 have zero lower margin'","'x5_joint2/3 .9 rad from delivered MPC initialState; RL normalizer unchanged'")
    build=build.replace("config['initialization']=dict(","config['startup_variant']='V1_MPC_INITIAL_ARM_PAUSED_HOLD'\nconfig['initialization']=dict(")
    if variant==2:
        build=build.replace("repair=root/'startup-v1'","repair=root/'startup-v2'")
        build=build.replace("    (overlay/name).write_text(after)","    if name=='rl_sim_mujoco.cpp':\n        assert after.count('    // loop\\n')==1\n        after=after.replace('    // loop\\n','    this->fsm.RequestStateChange(\"RLFSMStateRLLocomotion\");\\n\\n    // loop\\n')\n    (overlay/name).write_text(after)")
        build=build.replace('V1_MPC_INITIAL_ARM_PAUSED_HOLD','V2_MPC_INITIAL_ARM_PAUSED_NATIVE_LOCOMOTION_HOLD')
    return build

def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',type=int,choices=[1,2],default=1);args=p.parse_args()
    audit=(WORKTREE/'src/go2arx5_vla/target_mujoco/audit_home.cpp').read_text().replace('#include "native_home.hpp"','#include "native_startup.hpp"').replace('PI0288ResetHome(m,d)','PI0289ResetStartup(m,d)').replace('\\\"key\\\":\\\"home\\\"','\\\"key\\\":\\\"home_plus_MPC_initial_arm_V1\\\"')
    paths=['src/go2arx5_vla/target_mujoco/native_home.hpp','src/go2arx5_vla/target_mujoco/native_startup.hpp','scripts/pi0289_mujoco/run.py','scripts/pi0289_mujoco/force_audit.cpp']
    payload={Path(p).name:base64.b64encode((WORKTREE/p).read_bytes()).decode() for p in paths}
    payload['audit_home.cpp']=base64.b64encode(audit.encode()).decode()
    # Preserve old observer; new read-only channels and stricter warning/first-contact termination.
    observer=(WORKTREE/'src/go2arx5_vla/target_mujoco/observer.cpp').read_text()
    observer=observer.replace('  real(m,d);','''  if(trace.is_open() && !count) {
    trace<<"{\\\"kind\\\":\\\"first_step_input\\\",\\\"data_pointer\\\":"<<reinterpret_cast<uintptr_t>(d)<<",\\\"time\\\":"<<d->time<<",\\\"qpos\\\":";array(trace,d->qpos,m->nq);trace<<"}\\n";trace.flush();
  }
  real(m,d);''')
    observer=observer.replace('  bool illegal_contact=false;', '  for(int i=0;i<mjNWARNING;++i)stop|=d->warning[i].number>0;\n  bool illegal_contact=false;')
    observer=observer.replace('if(d->time+1e-12>=next_sample || positive', 'if(d->time<=.2 || d->time+1e-12>=next_sample || positive')
    observer=observer.replace('    trace<<",\\\"tcp_xyz\\\":";', '''    trace<<",\\\"dynamics_state_time_s\\\":"<<d->time-m->opt.timestep;
    trace<<",\\\"qfrc_actuator\\\":";array(trace,d->qfrc_actuator,m->nv);
    trace<<",\\\"qfrc_passive\\\":";array(trace,d->qfrc_passive,m->nv);
    trace<<",\\\"qfrc_constraint\\\":";array(trace,d->qfrc_constraint,m->nv);
    trace<<",\\\"qfrc_bias\\\":";array(trace,d->qfrc_bias,m->nv);
    trace<<",\\\"qacc\\\":";array(trace,d->qacc,m->nv);
    trace<<",\\\"tcp_xyz\\\":";''')
    payload['observer.cpp']=base64.b64encode(observer.encode()).decode()
    setup=f'''from pathlib import Path
import json,shutil
r=Path({ROOT!r});r.mkdir(exist_ok=True)
for n in ['runtime','runtime/tmp','runtime/ros-logs','private','evidence']:(r/n).mkdir(exist_ok=True)
parent=r.parent/'PI0288'
assert not (r/'startup-v{args.variant}').exists(),'Do not restage an existing startup variant'
if (r/'evidence/BUDGET.json').exists():
    ledger=json.loads((r/'evidence/BUDGET.json').read_text());assert all(x['status']!='RUNNING' for x in ledger['episodes'])
config=json.loads((parent/'private/config.json').read_text());config.update(release='1b9990c780e652a81cc585ef345af250de9bdfea',canonical=6092294476)
(r/'private/config.json').write_text(json.dumps(config,indent=2))
'''
    suffix='''
execute(['g++','-shared','-fPIC','-O2','-std=c++17','-I'+str(lib/'include'),str(repair/'observer.cpp'),'-L'+str(lib/'lib'),'-Wl,-rpath,'+str(lib/'lib'),'-lmujoco','-ldl','-o',str(root/'runtime/observer.so')],'compile-observer')
execute(['g++','-O2','-std=c++17','-I'+str(lib/'include'),str(repair/'force_audit.cpp'),'-L'+str(lib/'lib'),'-Wl,-rpath,'+str(lib/'lib'),'-lmujoco','-o',str(repair/'force_audit')],'compile-force-audit')
config['initialization']['pinned_files'].update({str(root/'runtime/observer.so'):sha(root/'runtime/observer.so'),str(repair/'observer.cpp'):sha(repair/'observer.cpp')})
(repair/'config.json').write_text(json.dumps(config,indent=2)+'\\n')
'''
    print(remote(setup+'\npayload='+repr(payload)+'\n'+body(args.variant)+suffix))

if __name__=='__main__':main()
