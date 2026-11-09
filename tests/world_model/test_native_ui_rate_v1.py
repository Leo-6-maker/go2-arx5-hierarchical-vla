"""Viewer source seam tests, never qualification of a new native binary."""
import json
from pathlib import Path
import pytest
from go2arx5_vla.world_model import native_ui_rate_v1 as ui

SOURCE='''// original startup and controls remain outside the viewer
void OriginalControl() { unchanged_control(); }
void Simulate::Render() {
  original_scene_render();
}
void Simulate::RenderLoop() {
  original_initialization();
'''+ui.LOOP+'''    {
      const MutexLock lock(this->mtx);
      original_Sync();
    }
    original_Render();
  }
}
// add state to history buffer
void OriginalHistory() { unchanged_history(); }
'''


def test_only_viewer_is_paced_and_sleep_precedes_simulation_mutex():
    changed=ui.patch(SOURCE)
    assert changed[len(ui.INCLUDES):].replace(ui.THROTTLED,ui.LOOP).replace(ui.HEADLESS_RENDER,ui.RENDER)==SOURCE
    assert ui.HEADLESS_RENDER in changed
    assert changed.index('sleep_until')<changed.index('const MutexLock')
    assert 'milliseconds(200)'in changed


def test_changed_duplicate_or_already_patched_viewer_is_rejected():
    for source in(SOURCE.replace(ui.LOOP,'different loop'),SOURCE+ui.LOOP,ui.patch(SOURCE)):
        with pytest.raises(PermissionError):ui.patch(source)


def native_fixture(tmp_path,monkeypatch):
    """Synthetic closed metadata tests, with zero builds or native executions."""
    from go2arx5_vla.world_model.storage import digest
    root=tmp_path/'new-ui';candidate=root/'candidate-01';candidate.mkdir(parents=True)
    (root/'control').mkdir();base=tmp_path/'baseline';base.mkdir()
    (base/'rl_sim_mujoco').write_bytes(b'fixture-baseline-not-native')
    baseline=digest(base/'rl_sim_mujoco');monkeypatch.setattr(ui,'BASELINE',baseline)
    original=tmp_path/'original.cc';original.write_text(SOURCE)
    (candidate/'simulate.cc').write_text(ui.patch(SOURCE));(candidate/'rl_sim_mujoco').write_bytes(b'fixture-headless-not-native')
    code=tmp_path/'source';member=code/ui.MEMBER;member.parent.mkdir(parents=True);member.write_bytes(Path(ui.__file__).read_bytes())
    files={ui.MEMBER:digest(member)}
    c=dict(native_ui_rate=ui.VERSION,original_ui_source=str(original),baseline_candidate=str(base),
        frozen_files={str(original):digest(original),str(base/'rl_sim_mujoco'):baseline},source_files=files)
    config=root/'EXECUTION_SPEC.json';config.write_text(json.dumps(c))
    (root/'control/LEASE.json').write_text(json.dumps(dict(status='CLOSED',worker_exit_code=0,spec_sha256=digest(config))))
    (root/'WORKER_EXIT.private.json').write_text(json.dumps(dict(exit_code=0)))
    result=dict(status='HEADLESS_VIEWER_CANDIDATE_COMPILED_NOT_NATIVE_QUALIFIED',baseline_native_sha256=baseline,
        original_controller_and_leg_SDK_objects_reused=True,new_native_integrations=0,
        binary_sha256=digest(candidate/'rl_sim_mujoco'),patched_ui_source_sha256=digest(candidate/'simulate.cc'))
    path=candidate/'RESULT.json';path.write_text(json.dumps(result))
    spec=dict(native_ui_rate=ui.VERSION,native_ui_build=dict(path=str(path),sha256=digest(path)),
        native_binary=str(candidate/'rl_sim_mujoco'),phase='collect-target-task-learning-grasp-v1',
        source_root=str(code),source_files=files,untouched_other_control_metadata='fixture')
    return spec,root,base


def test_headless_build_preserves_control_ancestry_and_new_binary_identity(tmp_path,monkeypatch):
    spec,root,base=native_fixture(tmp_path,monkeypatch)
    bound=ui.baseline_spec(spec)
    assert bound['native_binary']==str(base/'rl_sim_mujoco')
    assert spec['native_binary']!=bound['native_binary']and 'native_ui_build'in spec
    assert bound['untouched_other_control_metadata']=='fixture'
    assert 'native_ui_build'not in bound
    assert ui.baseline_spec(bound)is bound


def test_pending_build_wrong_binary_or_collection_cannot_use_ancestry_bridge(tmp_path,monkeypatch):
    spec,root,base=native_fixture(tmp_path,monkeypatch)
    pending=json.loads((root/'control/LEASE.json').read_text());pending['status']='ACTIVE'
    (root/'control/LEASE.json').write_text(json.dumps(pending))
    with pytest.raises(PermissionError,match='CLOSED_UI_BUILD'):ui.baseline_spec(spec)
    pending['status']='CLOSED';(root/'control/LEASE.json').write_text(json.dumps(pending))
    bad=dict(spec,native_binary=str(base/'rl_sim_mujoco'))
    with pytest.raises(PermissionError,match='CLOSED_UI_BUILD'):ui.baseline_spec(bad)
    bad=dict(spec,task_learning_role='TASK_LEARNING_CONTINUOUS_TRAIN_COLLECTION')
    with pytest.raises(PermissionError,match='CONSTRUCTION_ONLY'):ui.baseline_spec(bad)
