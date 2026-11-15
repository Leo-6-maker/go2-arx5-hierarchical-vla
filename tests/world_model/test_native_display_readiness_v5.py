from scripts.wm02.await_native_display_v5 import await_display, VERSION
from go2arx5_vla.world_model.target_sequential import native_display_command


def test_display_readiness_waits_without_launching_or_advancing_robot():
    now=[0.];calls=[]
    def probe():
        calls.append(now[0]);return (len(calls)==3,dict(message='X11 display not ready'))
    result=await_display(clock=lambda:now[0],sleep=lambda dt:now.__setitem__(0,now[0]+dt),probe=probe)
    assert result['passes'] and result['probe_attempts']==3
    assert result['simulator_starts']==result['physics_integrations']==0


def test_display_timeout_never_executes_native_and_keeps_first_error():
    now=[0.]
    result=await_display(timeout=.1,clock=lambda:now[0],
        sleep=lambda dt:now.__setitem__(0,now[0]+dt),probe=lambda:(False,dict(message='X11 unavailable')))
    assert not result['passes'] and result['first_error']['message']=='X11 unavailable'
    assert result['simulator_starts']==result['physics_integrations']==0


def test_frozen_preloads_follow_the_display_probe_only(tmp_path):
    spec=dict(xvfb_run='/xvfb',python='/native-python',native_binary='/frozen-simulator',
        trace_library='/trace',outer_library='/outer',observer_library='/observer',
        phase='collect-target-task-learning-grasp-v1',
        task_learning_role='CORELAB_EXPERT_TRAIN4_SEMANTICS_NOT_BC_V3',
        corelab_world_binding_v3={'sha256':'binding'},native_display_wait_v5=VERSION)
    argv,env=native_display_command(spec,tmp_path,dict(LD_PRELOAD='must-not-reach-display'))
    assert 'LD_PRELOAD' not in env
    assert argv.index('scripts.wm02.await_native_display_v5')<argv.index('/usr/bin/env')<argv.index('/frozen-simulator')
    assert argv.count('/frozen-simulator')==1 and 'LD_PRELOAD=/outer:/observer:/trace' in argv


def test_zero_packet_startup_failure_keeps_original_native_result(tmp_path,monkeypatch):
    import json
    from scripts.wm02 import commission_target
    from go2arx5_vla.world_model import target_sequential, native_replay
    class FailedDisplay:
        node=None
        reference_qualification=None
        exit_records=[dict(name='sim',exit_code=1)]
        def __init__(self,spec,output):
            (output/'state.bin.journal').touch()
        def start(self):raise RuntimeError('OWNED_NATIVE_PROCESS_EXIT_GLFW')
        def dynamic_phase(self):return False
        def reference_phase(self):return False
        def close(self):pass
    monkeypatch.setattr(target_sequential,'TargetSequentialAdapter',FailedDisplay)
    asset=tmp_path/'frozen';asset.write_bytes(b'unchanged')
    spec=dict(max_ticks=21703,phase='collect-target-task-learning-grasp-v1',campaign_root='/campaign',
        candidate_index=1,source_sha='source',candidate_source_sha='native',native_binary=str(asset),
        observer_library=str(asset),model_path=str(asset))
    assert commission_target.run(spec,tmp_path)==1
    result=json.loads((tmp_path/'RESULT.json').read_bytes())
    assert result['actual_target_integrations']==0 and result['status']=='NATIVE_STAGE_FAILED_PRESERVED'
    assert 'OWNED_NATIVE_PROCESS_EXIT_GLFW' in result['failure']
    import pytest
    with pytest.raises(ValueError,match='No original packets'):
        native_replay.packets(tmp_path/'state.bin.journal')
