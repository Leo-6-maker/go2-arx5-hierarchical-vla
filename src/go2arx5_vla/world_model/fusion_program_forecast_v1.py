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
    result={name:predict(arrays,x)[0] for name,arrays in models.items()}
    # Match the existing physical-SI baseline, using only the previous actual
    # velocity. Models and baselines see the same pre-effect observation.
    from .fusion_program_response_data_v1 import FEATURE_SLICES
    h=x[:,FEATURE_SLICES['measured_EEF_history']].reshape(9,8)
    velocity=np.r_[-h[-2,:6],h[-1,6:]-h[-2,6:]]
    result['ZERO_DISPLACEMENT']=np.zeros((16,8))
    result['PAST_CONSTANT_VELOCITY']=np.arange(1,17)[:,None]*velocity
    result['TRAIN_MEAN_RESPONSE']=np.asarray(next(iter(models.values()))['y_mean']).copy()
    return result


def validate_registration(spec):
    config=spec.get('program_response_forecast')
    if config is None:return None
    from .fusion_program_response_data_v1 import ARMS
    from .fusion_live_program_replay_v1 import assignment,PHASE
    if (config.get('schema')!='WM02_PROGRAM_RESPONSE_SHADOW_FORECAST_V1'
            or config.get('mode')!='SHADOW_NO_REFERENCE_CHANGE'
            or spec.get('phase')!=PHASE or assignment(spec)['split']!='DEV'
            or spec.get('task_condition')!='BOUNDED_FULL_H40_MECHANICAL_TRACKING_AT34303_V1'
            or spec.get('live_skill_geometry_scale')!=1.
            or spec.get('live_skill_time_stretch_factor')!=1
            or [m.get('arm') for m in config.get('models',[])]!=list(ARMS)
            or type(config.get('seed'))is not int):
        raise ValueError('EXPLICIT_NEW_DEV_SCOPE_FROZEN_SHADOW_FORECAST_REQUIRED')
    for key in ('training_spec_sha256','training_result_sha256'):
        if len(config.get(key,''))!=64:raise ValueError('PINNED_TRAINED_MODEL_IDENTITY_REQUIRED')
    return config


class ShadowForecaster:
    """Authenticate one CLOSED training result; save forecasts before effects."""
    def __init__(self,spec,output):
        import json,subprocess,time
        from pathlib import Path
        from .storage import digest
        from .pilot1_storage import local_path
        config=validate_registration(spec)
        if config is None:raise ValueError('SHADOW_FORECAST_NOT_REGISTERED')
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
        for descriptor in config['models']:
            arm=descriptor['arm'];folder=root/'fit-01'/(str(config['seed'])+'-'+arm)
            matches=[r for r in result['results']if r['seed']==config['seed']and r['arm']==arm]
            if len(matches)!=1:raise ValueError('ONE_PREDECLARED_TRAINED_ARM_REQUIRED')
            identity=matches[0]
            if (identity['full_restore_error']!=0 or identity['endpoint_reload_error']!=0
                    or descriptor['sha256']!=identity['numpy_sha256']):
                raise ValueError('EXACT_RESTORED_ARM_REQUIRED')
            path=pinned(folder/'PROGRAM_RESPONSE_NUMPY.npz',descriptor['sha256'],json_file=False)
            pinned(folder/('checkpoint-%d.pt'%training['endpoint']),identity['checkpoint_sha256'],json_file=False)
            with np.load(path,allow_pickle=False)as arrays:self.models[arm]={k:arrays[k]for k in arrays.files}
            if bool(self.models[arm]['mask_future'])!=(arm=='NO_FUTURE_ACTION'):
                raise ValueError('MATCHED_FUTURE_ACTION_ABLATION_REQUIRED')
            self.identities.append(dict(arm=arm,seed=config['seed'],numpy_sha256=descriptor['sha256'],
                checkpoint_sha256=identity['checkpoint_sha256']))
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
        path=self.output/('atom-%06d.private.json'%install_atom)
        write_json(path,dict(schema='WM02_LIVE_PROGRAM_PRE_EFFECT_FORECAST_V1',**record,
            predictions=prediction,models=self.identities,reference_changes=0,
            native_effects_since_anchor=0,forecast_is_native_qualification=False))
        self.records.append(dict(program_atom=install_atom,native_step=record['native_step'],
            filename=path.name,sha256=digest(path),feature_sha256=record['feature_sha256']))
        self.wall_s+=time.monotonic()-start
