"""Verify a frozen completed response study and select its preregistered model."""
import json
from pathlib import Path
from .storage import digest


def validate_asset(spec):
    path=Path(spec['response_study_result']);value=json.loads(path.read_text())
    lease=json.loads((path.parent.parent/'control/LEASE.json').read_text())
    arm=spec['response_arm'];model=Path(spec['response_numpy'])
    matches=[r for r in value.get('results',[])if r.get('seed')==366301 and r.get('arm')==arm]
    if (arm not in ('ACTION_CONDITIONED','NO_FUTURE_ACTION') or digest(path)!=spec['response_study_result_sha256']
            or lease.get('status')!='CLOSED' or lease.get('worker_exit_code')!=0
            or value.get('status')!='TRAINED_EXACTLY_RESTORED_AND_NUMPY_EXPORTED'
            or value.get('schema')!='WM02_FUSION_MATCHED_RESPONSE_STUDY_RESULT_V1'
            or value.get('main_WM_updates')!=12000 or value.get('restore_probe_optimizer_calls')!=6
            or value.get('primary_seed')!=366301 or value.get('TEST_read')is not False
            or value.get('ARX_sha256')!=spec['ARX_sha256'] or len(matches)!=1
            or value.get('TRAIN_mean_response_rule_sha256')!=spec['train_mean_response_model_sha256']
            or {(r.get('seed'),r.get('arm'))for r in value.get('results',[])}!={(s,a)for s in (366301,366302,366303)for a in ('ACTION_CONDITIONED','NO_FUTURE_ACTION')}):
        raise PermissionError('ACTUAL_FROZEN_COMPLETED_RESPONSE_STUDY_REQUIRED')
    row=matches[0]
    if (digest(model)!=spec['response_numpy_sha256'] or row['numpy_sha256']!=digest(model)
            or model!=path.parent/(str(366301)+'-'+arm)/'RESPONSE_NUMPY.npz'
            or row['full_restore_error']!=0 or row['endpoint_reload_error']!=0 or row['numpy_max_error_m']>2e-7):
        raise PermissionError('PREREGISTERED_PRIMARY_SEED_AND_VERIFIED_NUMPY_MODEL_REQUIRED')
    return dict(arm=arm,model_sha256=digest(model),training_replay_sha256=value['replay_sha256'],
        output='WORLD_TCP_XYZ_H16_ONLY_NOT_COMPLETE_STATE',checkpoint_sha256=row['checkpoint_sha256'])
