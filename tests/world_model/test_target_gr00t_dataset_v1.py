"""Synthetic supervision/split/time-boundary tests; no native demonstrations."""
import copy,json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from go2arx5_vla.world_model import target_gr00t_dataset_v1 as ds
from go2arx5_vla.world_model import target_gr00t_collection_v1 as collect
from go2arx5_vla.world_model.storage import digest
from tests.world_model.test_stable_task_dataset_training_v1 import qualification


def test_only_consumed_commands_enter_target_action_labels():
    q=qualification()
    a=ds.actual_commands(q)
    for row in q['generated_segment']['windows']:row['actual_TCP'][0][3]=999.
    np.testing.assert_array_equal(ds.actual_commands(q),a)
    q['generated_segment']['windows'][-1]['solver_policies_with_exact_installed_target']=[]
    with pytest.raises(ValueError,match='SOLVER_GRIPPER'):ds.actual_commands(q)


def synthetic():
    state=np.tile([0,0,.3,1,0,0,0,1,0,1.],(27,1))
    command=np.tile(state[0],(40,1));command[:,0]=np.linspace(0,.004,40)
    return dict(packet=dict(actual_state10=state,committed_absolute10=command,
        native_command_anchor=np.asarray(collect.CAPTURE_ATOMS),
        valid_window=np.arange(27)==0),row=dict(split='TRAIN',purpose='DEMONSTRATION'),
        qualification_only=False)


def test_statistics_use_training_only_and_executed_window_grid():
    e=synthetic();s=ds.fit_statistics([e])
    assert s['new_embodiment']['state']['eef_9d']['mean'][2]==pytest.approx(.3)
    for patch in (dict(row=dict(split='DEV',purpose='DEMONSTRATION')),
                  dict(row=dict(split='TRAIN',purpose='QUALIFICATION')),
                  dict(qualification_only=True)):
        with pytest.raises(ValueError,match='TRAIN_ONLY'):ds.fit_statistics([dict(e,**patch)])


def dataset_fixture(tmp_path):
    root=tmp_path/'real-looking-software-fixture';root.mkdir()
    p=synthetic()['packet'];pic=root/'image.png'
    Image.fromarray(np.zeros((256,256,3),np.uint8)).save(pic)
    p.update(instruction=np.asarray(collect.LANGUAGES[collect.COMMANDS[0]]),
        rgb_paths=np.full((27,2),'image.png'),rgb_sha256=np.full((27,2),digest(pic)),
        label_representation=np.asarray('ABSOLUTE_COMMITTED_EEF10'))
    np.savez(root/'episode.npz',**p)
    (root/'statistics.json').write_text(json.dumps(ds.fit_statistics([synthetic()])))
    row=dict(id='SYNTHETIC_NOT_NATIVE',split='TRAIN',windows=1,file='episode.npz',sha256=digest(root/'episode.npz'))
    (root/'DATASET_MANIFEST.json').write_text(json.dumps(dict(schema=ds.SCHEMA,domain=ds.DOMAIN,profile='TARGET_GR00T_V1',
        statistics_sha256=digest(root/'statistics.json'),episodes=[row])))
    return root


def test_target_loader_preserves_50ms_actions_and_sparse_same_clock_rgb(tmp_path):
    root=dataset_fixture(tmp_path);d=ds.TargetNativeDataset(root,'unused','unused')
    assert len(d)==1
    _,commands,images,text=d.raw(0)
    assert commands.shape==(40,10) and commands[0,0]==0
    assert len(images)==2 and text==collect.LANGUAGES[collect.COMMANDS[0]]
    assert commands[-1,0]==pytest.approx(.004)
    with pytest.raises(IndexError):d.raw(1)
    with pytest.raises(FileNotFoundError):ds.TargetNativeDataset(root,'unused','unused',for_optimization=True)
    with pytest.raises(ValueError,match='NO_DEV'):ds.TargetNativeDataset(root,'unused','unused','DEV',for_optimization=True)


def test_old_source_or_wm_manifest_is_not_target_vla_data(tmp_path):
    root=dataset_fixture(tmp_path);path=root/'DATASET_MANIFEST.json';v=json.loads(path.read_text())
    for key,value in [('domain','FIXED_BASE_ARX5_TASK_SIM'),('schema','WM02_REAL_STABLE_TASK_SUPERVISION_V1')]:
        path.write_text(json.dumps(dict(v,**{key:value})))
        with pytest.raises(ValueError,match='NOT_OLD_SOURCE'):ds.TargetNativeDataset(root,'unused','unused')


def test_core_optimization_uses_original_saved_core_qualification_not_a_fake_vanda_receipt(tmp_path):
    root=dataset_fixture(tmp_path);saved=root/'target-processor-v1';saved.mkdir()
    (saved/'processor_config.json').write_text('{"synthetic":true}')
    admission=dict(schema='WM02_TARGET_GR00T_PROCESSOR_ADMISSION_V1',
        status='PASS_ACTUAL_OFFICIAL_PROCESSOR_ROUNDTRIP',manifest_sha256=digest(root/'DATASET_MANIFEST.json'),
        statistics_sha256=digest(root/'statistics.json'),TRAIN_only_statistics=True,
        processor_files={'processor_config.json':digest(saved/'processor_config.json')})
    (root/'PROCESSOR_TRAINING_ADMISSION.json').write_text(json.dumps(admission))
    d=ds.TargetNativeDataset(root,'unused','unused',for_optimization=True,optimization_site='CORELAB_4080S')
    assert d.for_optimization and len(d)==1
    assert not(root/'PROCESSOR_VANDA_RECONSTRUCTION.json').exists()
    with pytest.raises(FileNotFoundError):ds.TargetNativeDataset(root,'unused','unused',for_optimization=True)
    (saved/'processor_config.json').write_text('changed')
    with pytest.raises(ValueError,match='SAVED_QUALIFIED'):
        ds.TargetNativeDataset(root,'unused','unused',for_optimization=True,optimization_site='CORELAB_4080S')


def test_teacher_policy_boundary_undoes_no_grip_or_time_mapping_twice(tmp_path):
    from go2arx5_vla.world_model.stable_start_short_task_v1 import build_for_config
    actual=np.eye(4);cube=np.eye(4);cube[0,3]=.2;base=np.array([0,0,0,1,0,0,0.])
    program,proposal=build_for_config(actual,actual,base,cube,collect.config(collect.COMMANDS[0],'TRAIN'))
    n=tmp_path/'native-01';n.mkdir()
    def put(name,value):
        def plain(x):
            if isinstance(x,dict):return{k:plain(v)for k,v in x.items()}
            if isinstance(x,np.ndarray):return x.tolist()
            return x
        p=n/name;p.write_text(json.dumps(plain(value)));return digest(p)
    pin=put('STABLE_TASK_EXPERT_PROPOSAL.private.json',proposal)
    put('LIVE_SKILL_SEGMENT_DECISION.private.json',dict(raw_H40=program.decoded_H40,
        desired_path=program.desired_path,nominal_reference_path=program.reference_path))
    q=qualification();q['executed_reference_proposal_sha256']=pin;q['adaptation']=program.evidence()
    q['generated_segment']['original_raw_H40']=program.decoded_H40.tolist()
    for i,row in enumerate(q['generated_segment']['windows']):
        row['actual_installed_consumed_EEF10_BASE7']=np.r_[program.preview(i)[0],base].tolist()
    policy,proof=ds.policy_commands(tmp_path,q)
    assert len(policy)==40 and np.all(policy[:,9]==1)
    assert ds.actual_commands(q)[-1,9]==pytest.approx(.042/.044)
    assert proof['policy_decisions']==1 and proof['renewed_previews_as_additional_expert_decisions']==0
    assert proof['raw_policy_grip_preserved_in_label']is True
    assert 'start_readiness'not in proof  # Historical sealed episodes stay valid.
    initial=np.zeros(106);initial[34]=.0002
    decision=json.loads((n/'LIVE_SKILL_SEGMENT_DECISION.private.json').read_bytes())
    decision['start_step']=collect.B
    decision['actual_history']=dict(actual_state_history=[initial.tolist()],actual_history_native_steps=[collect.B])
    put('LIVE_SKILL_SEGMENT_DECISION.private.json',decision)
    ready=collect.start_readiness(dict(qvel=initial[34:66],native_step=collect.B,time_s=collect.B*.002))
    pin=put('TARGET_GR00T_START_READINESS.private.json',ready)
    q['decision_start_readiness']=dict(path=str(n/'TARGET_GR00T_START_READINESS.private.json'),sha256=pin)
    after,with_ready=ds.policy_commands(tmp_path,q,start_readiness=collect.start_readiness_contract())
    np.testing.assert_array_equal(after,policy)
    assert with_ready['start_readiness']['sha256']==pin
    # Observed NumPy2.5 collector vs NumPy1.26 exporter difference: a single
    # FP64 norm ULP, with the exact same recorded velocities and safe decision.
    rounded=copy.deepcopy(ready)
    rounded['BASE_linear_speed_m_s']=float(np.nextafter(ready['BASE_linear_speed_m_s'],np.inf))
    q['decision_start_readiness']['sha256']=put('TARGET_GR00T_START_READINESS.private.json',rounded)
    after,with_roundoff=ds.policy_commands(tmp_path,q,start_readiness=collect.start_readiness_contract())
    np.testing.assert_array_equal(after,policy)
    assert with_roundoff['start_readiness_numeric_verification']['raw_BASE_qvel6']=='EXACT'
    wrong=copy.deepcopy(ready);wrong['BASE_qvel6'][0]=.0001
    q['decision_start_readiness']['sha256']=put('TARGET_GR00T_START_READINESS.private.json',wrong)
    with pytest.raises(ValueError,match='ACTUAL_START_READINESS'):
        ds.policy_commands(tmp_path,q,start_readiness=collect.start_readiness_contract())
    q['decision_start_readiness']['sha256']=put('TARGET_GR00T_START_READINESS.private.json',ready)
    # Altering achieved TCP has no effect on the expert policy command.
    for row in q['generated_segment']['windows']:row['actual_TCP'][0][3]=999
    np.testing.assert_array_equal(ds.policy_commands(tmp_path,q)[0],policy)
    i=program.knot_indices[10]
    q['generated_segment']['windows'][i]['actual_installed_consumed_EEF10_BASE7'][0]+=.000001
    with pytest.raises(ValueError,match='ACTUALLY_COMPILED'):ds.policy_commands(tmp_path,q)


def test_readiness_roundoff_never_admits_other_states_or_crosses_original_threshold():
    velocity=np.zeros(32);velocity[:3]=[.0001,.00008,.00002]
    original=collect.start_readiness(dict(qvel=velocity,native_step=collect.B,time_s=collect.B*.002))
    rounded=copy.deepcopy(original)
    rounded['BASE_linear_speed_m_s']=float(np.nextafter(original['BASE_linear_speed_m_s'],np.inf))
    assert ds.verified_start_readiness(rounded,original)
    for field,value in [('native_step',collect.B+1),('passes',False),
                        ('BASE_qvel6',[0.]*6),('new_task_teacher_proposals_before_check',1)]:
        assert not ds.verified_start_readiness(dict(rounded,**{field:value}),original)
    for value in (original['BASE_linear_speed_m_s']*(1+1e-12),float('nan'),-1.,True):
        assert not ds.verified_start_readiness(dict(original,BASE_linear_speed_m_s=value),original)
    velocity[:3]=[.001,0.,0.]
    boundary=collect.start_readiness(dict(qvel=velocity,native_step=collect.B,time_s=collect.B*.002))
    assert boundary['passes']is True
    over=copy.deepcopy(boundary)
    over['BASE_linear_speed_m_s']=float(np.nextafter(.001,np.inf))
    # Both stored and recomputed norms must independently remain inside the
    # unchanged .001m/s bound, even if their difference is only one ULP.
    assert not ds.verified_start_readiness(over,boundary)
    assert not ds.verified_start_readiness(boundary,over)
    assert ds.verified_start_readiness(collect.start_readiness(dict(qvel=np.zeros(32),
        native_step=collect.B,time_s=collect.B*.002)),collect.start_readiness(dict(qvel=np.zeros(32),
        native_step=collect.B,time_s=collect.B*.002)))


def test_fit4_selection_can_change_train_subset_without_relabelling(tmp_path):
    rows=[dict(native_run=f'train{parent}-{command}',split='TRAIN',purpose='DEMONSTRATION',
               parent=parent,command=command,restart=1)
          for parent in ('ONE','THREE') for command in collect.COMMANDS]
    p=tmp_path/'selection.json'
    selection=dict(schema='WM02_TARGET_GR00T_FIT4_SELECTION_V1',
        purpose='DIAGNOSTIC_NOT_FORMAL_OR_DEV_SELECTION',native_runs=[r['native_run']for r in rows])
    p.write_text(json.dumps(selection))
    ids,proof=ds.fit4_selection(dict(assignments=rows),p)
    assert ids==set(selection['native_runs']) and proof['sha256']==digest(p)
    for patch in (dict(split='DEV'),dict(purpose='QUALIFICATION')):
        bad=copy.deepcopy(rows);bad[0].update(patch)
        with pytest.raises(ValueError,match='TRAIN_ONLY_BALANCED'):
            ds.fit4_selection(dict(assignments=bad),p)
    selection['native_runs'][-1]=selection['native_runs'][0];p.write_text(json.dumps(selection))
    with pytest.raises(ValueError,match='FOUR_UNIQUE'):ds.fit4_selection(dict(assignments=rows),p)


def test_fit4_keeps_nonselected_real_failure_exit_and_proof(tmp_path):
    (tmp_path/'control').mkdir()
    (tmp_path/'EXECUTION_SPEC.json').write_text('{"original":"fixture"}')
    (tmp_path/'WORKER_EXIT.private.json').write_text(json.dumps(dict(exit_code=1,failure='standing bound')))
    (tmp_path/'control/LEASE.json').write_text(json.dumps(dict(status='CLOSED',worker_exit_code=1,
        spec_sha256=digest(tmp_path/'EXECUTION_SPEC.json'))))
    result=ds.excluded_terminal_attempt(tmp_path,dict(native_run='failed-train'))
    assert result['status']=='EXCLUDED_RETAINED_FAILURE' and result['exit_code']==1
    assert result['failure']=='standing bound' and result['proof']['worker']['sha256']==digest(tmp_path/'WORKER_EXIT.private.json')
