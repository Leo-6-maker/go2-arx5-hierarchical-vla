import shutil
import subprocess
from pathlib import Path
import pytest
from go2arx5_vla.world_model.native_state_snapshot import patch,remove_handoff,COPIES,POLICY_TIME,READ,PUBLISH


def candidate():
    return ('void RL_Sim::GetState(RobotState<float> *state)\n{\nREAD_ORIGINAL_SENSORS();\n'
        '// WM02_TRACE_END\n}\n\nvoid RL_Sim::SetCommand(const Command* command){}\n'
        'void RL_Sim::RobotControl(){this->GetState(&this->robot_state);ORIGINAL_PD();}\n'
        'void RL_Sim::RunModel(){\n'+POLICY_TIME+
        '        if(!wm02_applied_control_clock::clock().policy_due(time,period))return;\n'+
        ''.join(COPIES)+'        this->obs.actions = this->Forward();\nORIGINAL_OUTPUT();\n}\n')


def test_exact_inverse_and_read_precedes_policy_reservation_without_math_edits():
    old=candidate();new=patch(old)
    assert remove_handoff(new)==old
    assert new.index(READ)<new.index('clock().policy_due(')
    assert new.index(PUBLISH)<new.index('void RL_Sim::SetCommand(')
    assert 'READ_ORIGINAL_SENSORS();'in new and 'ORIGINAL_PD();'in new and 'ORIGINAL_OUTPUT();'in new
    assert not any(k in new for k in('lock_guard','unique_lock','clip_obs','filter_sensor'))
    with pytest.raises(ValueError):patch(new)
    with pytest.raises(ValueError):patch(old+next(iter(COPIES)))


def test_native_lineage_requires_exact_handoff_text_and_header_not_only_booleans(tmp_path):
    import json
    from go2arx5_vla.world_model.auto1_native import standing_repair_ancestor
    from go2arx5_vla.world_model.native_state_snapshot import VERSION
    from go2arx5_vla.world_model.storage import digest
    repo=Path(__file__).resolve().parents[2];root=tmp_path/'candidate';root.mkdir()
    old=tmp_path/'old.cpp';old.write_text(candidate());sdk=tmp_path/'old_sdk.cpp';sdk.write_text('ORIGINAL_SDK_MATH')
    (root/'rl_sim_mujoco.cpp').write_text(patch(old.read_text()));(root/'rl_sdk.cpp').write_text(sdk.read_text())
    header=repo/'src/go2arx5_vla/target_mujoco/state_snapshot.hpp'
    (root/'state_snapshot.hpp').write_bytes(header.read_bytes());binary=root/'not-executable-fixture';binary.write_bytes(b'LINEAGE_UNIT_TEST_ONLY')
    build=dict(source_sha='fixture',robot_state_handoff=VERSION,robot_state_handoff_predecessor_cpp=str(old),
        robot_state_handoff_predecessor_sdk=str(sdk),robot_state_handoff_predecessor_cpp_sha256=digest(old),
        robot_state_handoff_predecessor_sdk_sha256=digest(sdk),sdk_source=str(sdk),ancestor_binary_sha256='ancestor')
    receipt=dict(status='INIT_CANDIDATE_COMPILED_AND_LOADER_VERIFIED_NOT_NATIVE_QUALIFIED',source_sha='fixture',
        binary_sha256=digest(binary),cold_history_init='ORIGINAL_BUFFER_RESET_FIRST_REAL_OBSERVATION_V1',
        ongoing_control_math_unchanged=True,original_R1_math_assertions_pass=True,original_SDK_sha256=digest(sdk),
        robot_state_handoff=VERSION,state_snapshot_header_sha256=digest(header),
        robot_state_handoff_inverse_text_identical=True,robot_state_handoff_PD_waits=False,
        robot_state_handoff_sensor_filters=False)
    bp=tmp_path/'BUILD.json';bp.write_text(json.dumps(build));rp=root/'CANDIDATE.private.json';rp.write_text(json.dumps(receipt))
    spec=dict(native_binary=str(binary),source_root=str(repo),cold_history_build_spec=str(bp),
        cold_history_build_spec_sha256=digest(bp),cold_history_build_receipt=str(rp),cold_history_build_receipt_sha256=digest(rp))
    assert standing_repair_ancestor(spec)=='ancestor'
    with(root/'rl_sim_mujoco.cpp').open('a')as f:f.write('CHANGED_CONTROL_MATH();')
    with pytest.raises(PermissionError):standing_repair_ancestor(spec)
    (root/'rl_sim_mujoco.cpp').write_text(patch(old.read_text()));(root/'state_snapshot.hpp').write_text('UNVERIFIED_HEADER')
    with pytest.raises(PermissionError):standing_repair_ancestor(spec)


@pytest.mark.skipif(not shutil.which('g++'),reason='native CPU compiler required')
def test_concurrent_vector_publication_never_tears_or_accepts_stale_frame(tmp_path):
    header=Path(__file__).resolve().parents[2]/'src/go2arx5_vla/target_mujoco/state_snapshot.hpp'
    code=r'''
#include "state_snapshot.hpp"
#include <thread>
#include <atomic>
#include <cstring>
extern wm02_robot_state_snapshot::Snapshot* other();
int main(){using namespace wm02_robot_state_snapshot;Frame f,out;
 if(other()!=&snapshot())return 1;
 if(snapshot().try_read(out,7))return 2;
 f.sample_ns=7;for(int i=0;i<49;++i)f.values[i]=float(i)*.125f;
 snapshot().publish(f);
 if(snapshot().try_read(out,8)||!snapshot().try_read(out,7))return 3;
 if(std::memcmp(f.values.data(),out.values.data(),49*sizeof(float)))return 4;
 std::atomic<bool> done{false};std::atomic<int> bad{0};
 std::thread writer([&]{Frame x;x.sample_ns=7;
  for(int n=1;n<=100000;++n){for(int i=0;i<49;++i)x.values[i]=float(n*2+i);snapshot().publish(x);}done=true;});
 while(!done.load()){Frame x;if(snapshot().try_read(x,7)){
  for(int i=1;i<49;++i)if(x.values[i]-x.values[0]!=float(i)*(x.values[0]==0?.125f:1.f))bad=1;}}
 writer.join();if(bad.load())return 5;
 if(!snapshot().try_read(out,7)||out.values[48]!=200048)return 6;
 return 0;}
'''
    (tmp_path/'main.cpp').write_text(code)
    (tmp_path/'other.cpp').write_text('#include "state_snapshot.hpp"\nwm02_robot_state_snapshot::Snapshot* other(){return &wm02_robot_state_snapshot::snapshot();}\n')
    subprocess.run(['g++','-std=c++17','-O2','-pthread','-I'+str(header.parent),str(tmp_path/'main.cpp'),str(tmp_path/'other.cpp'),'-o',str(tmp_path/'check')],capture_output=True,check=True)
    subprocess.run([str(tmp_path/'check')],capture_output=True,check=True,timeout=10)


def test_historical_prefix_identity_requires_bound_old_binary_and_unchanged_libraries(tmp_path,monkeypatch):
    import json
    from go2arx5_vla.world_model import auto1_native as native
    from go2arx5_vla.world_model.native_state_snapshot import VERSION
    from go2arx5_vla.world_model.storage import digest
    old=tmp_path/'old';old.mkdir();new=tmp_path/'new';new.write_bytes(b'new executable')
    for n in('rl_sim_mujoco','rl_sim_mujoco.cpp','rl_sdk.cpp','observer.so','task_actuation_r1.so'):
        (old/n).write_text('preserved '+n)
    receipt=old/'CANDIDATE.private.json';receipt.write_text(json.dumps(dict(
        status='INIT_CANDIDATE_COMPILED_AND_LOADER_VERIFIED_NOT_NATIVE_QUALIFIED',
        candidate_sources={n:digest(old/n)for n in('rl_sim_mujoco.cpp','rl_sdk.cpp')},binary_sha256=digest(old/'rl_sim_mujoco'))))
    build=tmp_path/'build.json';build.write_text(json.dumps(dict(robot_state_handoff=VERSION,
        robot_state_handoff_predecessor_cpp=str(old/'rl_sim_mujoco.cpp'))))
    calls=[];monkeypatch.setattr(native,'standing_repair_ancestor',lambda s:calls.append(s)or'checked')
    s=dict(native_binary=str(new),phase='collect-target-task-learning-grasp-v1',cold_history_build_spec=str(build),
        observer_library=str(old/'observer.so'),outer_library=str(old/'task_actuation_r1.so'))
    assert native.native_prefix_predecessor(s)==digest(new)and not calls
    s.update(sensor_handoff_previous_candidate_receipt=str(receipt),sensor_handoff_previous_candidate_receipt_sha256=digest(receipt))
    assert native.native_prefix_predecessor(s)==digest(old/'rl_sim_mujoco')and calls
    bad=tmp_path/'other.so';bad.write_bytes(b'unrelated observer');s['observer_library']=str(bad)
    with pytest.raises(PermissionError):native.native_prefix_predecessor(s)
    s['observer_library']=str(old/'observer.so');(old/'rl_sim_mujoco').write_text('changed executable')
    with pytest.raises(PermissionError):native.native_prefix_predecessor(s)
