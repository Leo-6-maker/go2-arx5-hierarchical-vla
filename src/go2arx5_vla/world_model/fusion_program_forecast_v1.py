"""Pre-effect program forecasts using observed history and a declared program.

No observations after the anchor are accepted. File authentication, original
owner admission and actual consumption checks belong to the native caller.
"""
import hashlib
import numpy as np
from scripts.pi0294.task_clock import actual_source_age
from . import fusion_live_causal_features_v2 as feature
from .fusion_live_program_replay_v1 import SCHEMA, B, flatten_features
from .fusion_program_response_numpy_v1 import predict

RIDGE_ARMS=('TRAIN_RIDGE_FULL_PROGRAM','TRAIN_RIDGE_NO_FUTURE_ACTION')
COUNTERFACTUAL_CONFIG=dict(schema='WM02_PROGRAM_COUNTERFACTUAL_SHADOW_V1',
    candidates=['NOMINAL','X_PLUS_050','X_MINUS_050','Y_PLUS_050','Y_MINUS_050'],
    objective='H16_GENERATED_DESIRED_TCP_TRANSLATION',actual_reference_changes=0)


def causal_record(history, past_windows, reference_path, *, install_atom):
    if (type(install_atom) is not int or install_atom < 0 or install_atom % 7
            or history['actual_history_native_steps'][-1] != B+25*install_atom
            or len(past_windows) != 8
            or [w['native_start_step'] for w in past_windows]
               != list(history['actual_history_native_steps'][:-1])
            or [w['native_end_step'] for w in past_windows]
               != list(history['actual_history_native_steps'][1:])):
        raise ValueError('FORECAST_ONLY_CONTIGUOUS_OBSERVED_PAST_AT_PUBLICATION')
    path=np.asarray(reference_path, dtype=float)
    plan=feature.plan_from_complete_reference_path(path[min(install_atom,len(path)-1):])
    ages=[actual_source_age(w['clock']['native_start_s'],w['clock']['source_mpc_s'])
          for w in past_windows]
    # The next publication is fresh and has not happened yet. Its actual clock
    # and complete preview must subsequently match before labels are scored.
    features=feature.causal_inputs(history,past_source_age_s=ages,
        planned_H40=plan,first_install_source_age_s=0.)
    x,names=flatten_features(dict(schema=SCHEMA,features=features))
    return dict(native_step=B+25*install_atom,program_atom=install_atom,
        feature_names=list(names),X=x,planned_H40=plan,
        prospective_first_install_source_age_s=0.,future_observations_read=False,
        feature_sha256=hashlib.sha256(x.astype('<f8').tobytes()).hexdigest())


def forecasts(record, models):
    x=np.asarray(record['X'],dtype=float)[None]
    from .fusion_program_ridge_v1 import predict as ridge_predict
    result={name:(ridge_predict if 'weights' in arrays else predict)(arrays,x)[0]
            for name,arrays in models.items()}
    # Match the existing physical-SI baseline, using only the previous actual
    # velocity. Models and baselines see the same pre-effect observation.
    from .fusion_program_response_data_v1 import FEATURE_SLICES
    h=x[:,FEATURE_SLICES['measured_EEF_history']].reshape(9,8)
    velocity=np.r_[-h[-2,:6],h[-1,6:]-h[-2,6:]]
    result['ZERO_DISPLACEMENT']=np.zeros((16,8))
    result['PAST_CONSTANT_VELOCITY']=np.arange(1,17)[:,None]*velocity
    result['TRAIN_MEAN_RESPONSE']=np.asarray(next(iter(models.values()))['y_mean']).copy()
    return result


def counterfactual_program_choices(history,past_windows,program,models):
    """Score a fixed TRAIN-covered reference family before the first effect.

    These unexecuted alternatives are model estimates, never dynamics labels.
    Their ranking changes no reference and is not native action qualification.
    """
    from dataclasses import replace
    from .fusion_program_reference_intervention_v1 import SCHEMA,OFFSETS
    from .fusion_live_causal_features_v1 import _rotation6
    state=np.asarray(history['actual_state_history'][-1],float)
    rotation=_rotation6(state[89:95],columns=True)
    goal=rotation.T@(program.desired_at(16)[:3]-state[86:89])
    result={};raw=np.asarray(program.decoded_H40).copy();desired=np.asarray(program.desired_path).copy()
    for candidate in COUNTERFACTUAL_CONFIG['candidates']:
        intervention=dict(schema=SCHEMA,candidate=candidate,offset_target_xy_m=list(OFFSETS[candidate]),ramp_atoms=16)
        alternative=replace(program,reference_intervention=intervention)
        if (not np.array_equal(raw,alternative.decoded_H40)
                or not np.array_equal(desired,alternative.desired_path)):
            raise ValueError('COUNTERFACTUAL_SHARED_FULL_RAW_INTENT_AND_DESIRED_REQUIRED')
        record=causal_record(history,past_windows,alternative.reference_path,install_atom=0)
        predictions=forecasts(record,models)
        result[candidate]=dict(reference_intervention=intervention,
            reference_path_sha256=alternative.evidence()['reference_path']['sha256'],
            feature_sha256=record['feature_sha256'],predictions=predictions,
            predicted_H16_TCP_error_m={name:float(np.linalg.norm(y[15,:3]-goal))for name,y in predictions.items()})
    names=list(next(iter(result.values()))['predictions'])
    selected={name:min(result,key=lambda candidate:result[candidate]['predicted_H16_TCP_error_m'][name])for name in names}
    return dict(schema='WM02_PROGRAM_COUNTERFACTUAL_PRE_EFFECT_ESTIMATES_V1',native_step=B,
        goal_H16_TCP_displacement_in_current_TCP_frame_m=goal,candidates=result,
        model_selected_candidate=selected,actual_reference_changes=0,
        future_observations_read=False,unexecuted_estimates_are_actual_labels=False,
        candidate_scores_are_native_qualification=False)


def validate_registration(spec):
    config=spec.get('program_response_forecast')
    if config is None:return None
    from .fusion_program_response_data_v1 import ARMS
    from .fusion_live_program_replay_v1 import assignment,PHASE
    version=config.get('schema')
    if version=='WM02_PROGRAM_RESPONSE_SHADOW_FORECAST_V1':
        expected=list(ARMS);labels=[m.get('arm') for m in config.get('models',[])]
        valid_seed=type(config.get('seed'))is int
        valid_timing=spec.get('live_skill_time_stretch_factor')==1
    elif version=='WM02_PROGRAM_RESPONSE_SHADOW_FORECAST_V2':
        seeds=config.get('seeds',[])
        valid_seed=(isinstance(seeds,list)and bool(seeds)and all(type(s)is int and s>=0 for s in seeds)
                    and len(set(seeds))==len(seeds)and config.get('primary_seed')==seeds[0])
        expected=[str(s)+'-'+a for s in seeds for a in ARMS]+list(RIDGE_ARMS)
        labels=[m['arm'] if m.get('arm')in RIDGE_ARMS else str(m.get('seed'))+'-'+m.get('arm','')
                for m in config.get('models',[])]
        valid_timing=(config.get('time_stretch_factor')==spec.get('live_skill_time_stretch_factor')
                      and type(config.get('time_stretch_factor'))is int and config['time_stretch_factor']in(1,2))
    else:raise ValueError('VERSIONED_MATCHED_FROZEN_FORECAST_SET_REQUIRED')
    if (config.get('mode')!='SHADOW_NO_REFERENCE_CHANGE'
            or spec.get('phase')!=PHASE or assignment(spec)['split']!='DEV'
            or spec.get('task_condition')!='BOUNDED_FULL_H40_MECHANICAL_TRACKING_AT34303_V1'
            or spec.get('live_skill_geometry_scale')!=1.
            or not valid_timing or labels!=expected or not valid_seed):
        raise ValueError('EXPLICIT_NEW_DEV_SCOPE_FROZEN_SHADOW_FORECAST_REQUIRED')
    for key in ('training_spec_sha256','training_result_sha256'):
        if len(config.get(key,''))!=64:raise ValueError('PINNED_TRAINED_MODEL_IDENTITY_REQUIRED')
    return config


class ShadowForecaster:
    """Authenticate one CLOSED training result; save forecasts before effects."""
    def __init__(self,spec,output,*,program=None):
        import json,subprocess,time
        from pathlib import Path
        from .storage import digest
        from .pilot1_storage import local_path
        config=validate_registration(spec)
        if config is None:raise ValueError('SHADOW_FORECAST_NOT_REGISTERED')
        self.counterfactual_program=None;self.counterfactual_record=None
        if spec.get('program_response_counterfactual') is not None:
            if (spec['program_response_counterfactual']!=COUNTERFACTUAL_CONFIG
                    or config['schema']!='WM02_PROGRAM_RESPONSE_SHADOW_FORECAST_V2'
                    or program is None or program.geometry_scale!=1.
                    or program.time_stretch_factor!=2):
                raise ValueError('DECLARED_SHARED_TIME2_PROGRAM_COUNTERFACTUAL_SHADOW_ONLY')
            self.counterfactual_program=program
        root=Path(config['training_run']);self.output=Path(output)/'PROGRAM_FORECASTS'
        self.records=[];self.models={};self.identities=[];self.wall_s=0.
        def pinned(path,pin,*,json_file=True):
            path=Path(path);local_path(spec,path)
            if (not path.is_file()or path.is_symlink()or path.resolve()!=path
                    or path.stat().st_size>8*1024**2 or digest(path)!=pin):
                raise ValueError('EXACT_BOUNDED_FROZEN_TRAINING_ARTIFACT_REQUIRED')
            return json.loads(path.read_text()) if json_file else path
        training=pinned(root/'EXECUTION_SPEC.json',config['training_spec_sha256'])
        result=pinned(root/'fit-01/RESULT.private.json',config['training_result_sha256'])
        lease=json.loads((root/'control/LEASE.json').read_text())
        worker=json.loads((root/'WORKER_EXIT.private.json').read_text())
        pid=subprocess.check_output(['systemctl','--user','show','wm02-'+root.name,'-p','MainPID'],text=True,timeout=10)
        if (lease.get('status')!='CLOSED' or lease.get('spec_sha256')!=config['training_spec_sha256']
                or lease.get('run_uuid')!=root.name or lease.get('worker_exit_code')!=0
                or worker.get('exit_code')!=0 or worker.get('failure')is not None or pid.strip()!='MainPID=0'
                or training.get('owner')!=spec['owner'] or training.get('registry')!=spec['registry']
                or training.get('kind')!='FUSION_LIVE_PROGRAM_RESPONSE_WM'
                or training.get('data_roles')!=['TRAIN'] or training.get('physics')is not False
                or result.get('status')!='REAL_TRAIN_OPTIMIZED_FULLY_RESTORED_AND_PORTABLE'
                or result.get('DEV_read')is not False or result.get('TEST_read')is not False
                or result.get('source_sha')!=training['source_sha']
                or result.get('source_manifest_sha256')!=training['source_manifest_sha256']):
            raise ValueError('ORIGINAL_CLOSED_TRAIN_ONLY_MODEL_REQUIRED')
        registry=json.loads(Path(spec['registry']).read_text())
        if not any(e.get('path')==str(root)and e.get('source_sha')==training['source_sha']
                   and e.get('role')==training['role']and e.get('owner')==spec['owner']
                   and e.get('local_lease')==str(root/'control/LEASE.json')for e in registry['entries']):
            raise ValueError('ORIGINAL_MODEL_TRAINING_OWNER_ROW_REQUIRED')
        version2=config['schema']=='WM02_PROGRAM_RESPONSE_SHADOW_FORECAST_V2'
        if version2 and (config['seeds']!=training.get('seeds')
                         or config['primary_seed']!=training.get('primary_seed')):
            raise ValueError('ALL_PREDECLARED_TRAINING_SEEDS_NOT_POSTHOC_SUBSET_REQUIRED')
        for descriptor in config['models']:
            arm=descriptor['arm'];ridge=arm in RIDGE_ARMS
            seed=None if ridge else descriptor['seed'] if version2 else config['seed']
            label=arm if ridge or not version2 else str(seed)+'-'+arm
            if ridge:
                matches=[r for r in result.get('closed_form_ridge_fits',[])if r['name']==arm]
                if (len(matches)!=1 or matches[0]['ridge']!=1.
                        or matches[0]['normalizers_fit_only_TRAIN']is not True):
                    raise ValueError('EXACT_MATCHED_TRAIN_ONLY_RIDGE_CONTROL_REQUIRED')
                identity=matches[0];checkpoint=None
                path=root/'fit-01'/(arm+'_NUMPY.npz')
            else:
                matches=[r for r in result['results']if r['seed']==seed and r['arm']==arm]
                if len(matches)!=1:raise ValueError('ONE_PREDECLARED_TRAINED_ARM_REQUIRED')
                identity=matches[0];folder=root/'fit-01'/(str(seed)+'-'+arm)
                if identity['full_restore_error']!=0 or identity['endpoint_reload_error']!=0:
                    raise ValueError('EXACT_RESTORED_ARM_REQUIRED')
                checkpoint=identity['checkpoint_sha256']
                pinned(folder/('checkpoint-%d.pt'%training['endpoint']),checkpoint,json_file=False)
                path=folder/'PROGRAM_RESPONSE_NUMPY.npz'
            if descriptor['sha256']!=identity['numpy_sha256']:
                raise ValueError('EXACT_FROZEN_ARRAY_IDENTITY_REQUIRED')
            path=pinned(path,descriptor['sha256'],json_file=False)
            with np.load(path,allow_pickle=False)as arrays:self.models[label]={k:arrays[k]for k in arrays.files}
            if bool(self.models[label]['mask_future'])!=(arm in ('NO_FUTURE_ACTION','TRAIN_RIDGE_NO_FUTURE_ACTION')):
                raise ValueError('MATCHED_FUTURE_ACTION_ABLATION_REQUIRED')
            self.identities.append(dict(label=label,arm=arm,seed=seed,numpy_sha256=descriptor['sha256'],
                checkpoint_sha256=checkpoint,predictor_family='RIDGE'if ridge else'MATCHED_NEURAL'))
        first=next(iter(self.models.values()))
        for arrays in self.models.values():
            for key in ('x_mean','x_scale','y_mean','y_scale'):
                if not np.array_equal(arrays[key],first[key]):raise ValueError('MATCHED_TRAIN_ONLY_NORMALIZERS_REQUIRED')
        self.output.mkdir(exist_ok=False)

    def save(self,history,past_windows,reference_path,*,install_atom):
        import time
        from .storage import write_json,digest
        start=time.monotonic()
        record=causal_record(history,past_windows,reference_path,install_atom=install_atom)
        prediction=forecasts(record,self.models)
        if install_atom==0 and self.counterfactual_program is not None:
            if not np.array_equal(reference_path,self.counterfactual_program.reference_path):
                raise ValueError('COUNTERFACTUAL_PROGRAM_MUST_EQUAL_ACTUAL_PRE_EFFECT_PROGRAM')
            cf=counterfactual_program_choices(history,past_windows,self.counterfactual_program,self.models)
            cfpath=self.output/'candidate-choices-at-B.private.json'
            write_json(cfpath,dict(cf,models=self.identities))
            self.counterfactual_record=dict(filename=cfpath.name,sha256=digest(cfpath),
                actual_reference_changes=0,unexecuted_estimates_are_actual_labels=False)
        path=self.output/('atom-%06d.private.json'%install_atom)
        write_json(path,dict(schema='WM02_LIVE_PROGRAM_PRE_EFFECT_FORECAST_V1',**record,
            predictions=prediction,models=self.identities,reference_changes=0,
            native_effects_since_anchor=0,forecast_is_native_qualification=False))
        self.records.append(dict(program_atom=install_atom,native_step=record['native_step'],
            filename=path.name,sha256=digest(path),feature_sha256=record['feature_sha256']))
        self.wall_s+=time.monotonic()-start
