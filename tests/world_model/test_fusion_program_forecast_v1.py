"""The actual online ABI must equal archived causal extraction at the anchor."""
import copy
import numpy as np
import pytest
from go2arx5_vla.world_model import fusion_program_forecast_v1 as forecast
from go2arx5_vla.world_model import fusion_live_program_replay_v1 as replay
from test_fusion_live_program_replay_v1 import artifacts


@pytest.mark.parametrize('atom',[0,7,14])
def test_forecast_has_no_future_observation_argument_and_matches_replay(atom):
    spec,prefix,decision,segment=artifacts()
    row=replay.extract(spec,prefix,decision,segment,install_atom=atom)
    expected,_=replay.flatten_features(row)
    history=(decision['actual_history'] if atom==0 else
             segment['windows'][atom]['actual_history_before_atom'])
    past=(prefix['windows']+segment['windows'][:atom])[-8:]
    actual=forecast.causal_record(history,past,
        decision['nominal_reference_path'],install_atom=atom)
    np.testing.assert_array_equal(actual['X'],expected)
    assert actual['future_observations_read'] is False
    assert actual['native_step']==34303+25*atom
    changed=copy.deepcopy(segment)
    changed['windows'][atom]['actual_state_vector'][86]+=1.
    again=forecast.causal_record(history,past,
        decision['nominal_reference_path'],install_atom=atom)
    np.testing.assert_array_equal(actual['X'],again['X'])


@pytest.mark.parametrize('fault',['clock','publication','missing_past'])
def test_no_synthetic_or_nonpublication_anchor(fault):
    spec,prefix,decision,segment=artifacts()
    history=copy.deepcopy(decision['actual_history']);past=prefix['windows'][-8:];atom=0
    if fault=='clock':history['actual_history_native_steps'][-1]+=1
    elif fault=='publication':atom=1
    elif fault=='missing_past':past=past[1:]
    with pytest.raises(ValueError):
        forecast.causal_record(history,past,decision['nominal_reference_path'],install_atom=atom)


def shadow_files(tmp_path,monkeypatch):
    """Synthetic file lifecycle only; no native worker or trained model claim."""
    import hashlib,json,subprocess
    from go2arx5_vla.world_model import pilot1_storage
    from go2arx5_vla.world_model.storage import digest
    from go2arx5_vla.world_model.fusion_program_response_data_v1 import ARMS,FEATURE_DIM
    monkeypatch.setattr(pilot1_storage,'local_path',lambda s,p:p)
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:'MainPID=0\n')
    root=tmp_path/'training';(root/'control').mkdir(parents=True);out=root/'fit-01';out.mkdir()
    registry=tmp_path/'registry.json';owner='SYNTHETIC_FIXTURE_OWNER'
    training=dict(owner=owner,registry=str(registry),kind='FUSION_LIVE_PROGRAM_RESPONSE_WM',
        data_roles=['TRAIN'],physics=False,source_sha='a'*40,source_manifest_sha256='b'*64,
        role='SYNTHETIC_TRAINER',endpoint=2)
    (root/'EXECUTION_SPEC.json').write_text(json.dumps(training));pin=digest(root/'EXECUTION_SPEC.json')
    (root/'control/LEASE.json').write_text(json.dumps(dict(status='CLOSED',spec_sha256=pin,
        run_uuid=root.name,worker_exit_code=0)))
    (root/'WORKER_EXIT.private.json').write_text(json.dumps(dict(exit_code=0,failure=None)))
    registry.write_text(json.dumps(dict(entries=[dict(path=str(root),source_sha='a'*40,
        role=training['role'],owner=owner,local_lease=str(root/'control/LEASE.json'))])))
    results=[];models=[]
    for arm in ARMS:
        folder=out/('1-'+arm);folder.mkdir();checkpoint=folder/'checkpoint-2.pt'
        checkpoint.write_bytes(b'FIXTURE_ONLY_NOT_TORCH_CHECKPOINT')
        arrays=dict(x_mean=np.zeros(FEATURE_DIM),x_scale=np.ones(FEATURE_DIM),
            y_mean=np.zeros((16,8)),y_scale=np.ones((16,8)),w0=np.zeros((32,FEATURE_DIM)),
            b0=np.zeros(32),w1=np.zeros((128,32)),b1=np.zeros(128),mask_future=np.array(arm=='NO_FUTURE_ACTION'))
        model=folder/'PROGRAM_RESPONSE_NUMPY.npz';np.savez_compressed(model,**arrays)
        models.append(dict(arm=arm,sha256=digest(model)))
        results.append(dict(arm=arm,seed=1,full_restore_error=0,endpoint_reload_error=0,
            numpy_sha256=digest(model),checkpoint_sha256=digest(checkpoint)))
    result=dict(status='REAL_TRAIN_OPTIMIZED_FULLY_RESTORED_AND_PORTABLE',DEV_read=False,TEST_read=False,
        source_sha='a'*40,source_manifest_sha256='b'*64,results=results)
    (out/'RESULT.private.json').write_text(json.dumps(result))
    spec=dict(owner=owner,registry=str(registry),phase=replay.PHASE,
        response_learning_assignment=dict(schema='WM02_PROGRAM_RESPONSE_ASSIGNMENT_V1',split='DEV',
            parent='SYNTHETIC_DEV',panel_sha256='c'*64),
        task_condition='BOUNDED_FULL_H40_MECHANICAL_TRACKING_AT34303_V1',
        live_skill_geometry_scale=1.,live_skill_time_stretch_factor=1,
        program_response_forecast=dict(schema='WM02_PROGRAM_RESPONSE_SHADOW_FORECAST_V1',
            mode='SHADOW_NO_REFERENCE_CHANGE',training_run=str(root),training_spec_sha256=pin,
            training_result_sha256=digest(out/'RESULT.private.json'),models=models,seed=1))
    output=tmp_path/'native';output.mkdir()
    return spec,output,root


def test_only_frozen_closed_train_models_make_saved_pre_effect_forecasts(tmp_path,monkeypatch):
    import json
    spec,output,root=shadow_files(tmp_path,monkeypatch)
    model=forecast.ShadowForecaster(spec,output)
    _,prefix,decision,_=artifacts()
    model.save(decision['actual_history'],prefix['windows'][-8:],decision['nominal_reference_path'],install_atom=0)
    saved=json.loads((output/'PROGRAM_FORECASTS/atom-000000.private.json').read_text())
    assert len(saved['models'])==3 and len(saved['predictions'])==6
    assert saved['native_step']==34303 and saved['native_effects_since_anchor']==0
    assert saved['reference_changes']==0 and saved['future_observations_read'] is False
    assert 'labels' not in saved


@pytest.mark.parametrize('fault',['active_model','tampered_model','TRAIN_forecast'])
def test_unsealed_or_wrong_role_shadow_inputs_fail_before_native_effect(tmp_path,monkeypatch,fault):
    import json
    spec,output,root=shadow_files(tmp_path,monkeypatch)
    if fault=='active_model':
        path=root/'control/LEASE.json';lease=json.loads(path.read_text());lease['status']='ACTIVE';path.write_text(json.dumps(lease))
    elif fault=='tampered_model':(root/'fit-01/1-RAW/PROGRAM_RESPONSE_NUMPY.npz').write_bytes(b'TAMPERED')
    elif fault=='TRAIN_forecast':spec['response_learning_assignment']['split']='TRAIN'
    with pytest.raises(ValueError):forecast.ShadowForecaster(spec,output)
    assert not(output/'PROGRAM_FORECASTS').exists()
