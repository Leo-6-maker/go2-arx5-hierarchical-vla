"""A narrow FUSION supplement in the existing registry; no new scheduler/owner."""
from pathlib import Path
from contextlib import closing
import json, os, time, subprocess
from .fusion_contract import PLAN
from .storage import digest

SCOPE = 'WM02_FUSION_REFERENCE_SIM_ONLY'


def binding_contract(binding, now):
    expected = dict(schema='WM02_FUSION_NIGHT1_SCOPE_V1', plan_sha=PLAN, scope=SCOPE,
        owner='EXISTING_LOCAL_PC', host='corelab-4080s', uid=2009, max_native_slots=2,
        GR00T_updates=0, WBC_changes=False, physics_dt_s=.002,
        native_reference_domain='SINGLE_DECISION_LOCAL_XY4_RAMP8_H16_OPEN_GRIPPER')
    if (any(binding.get(k) != v for k, v in expected.items())
            or not binding['activation_epoch'] <= now < binding['deadline_epoch']
            or binding['deadline_epoch'] > binding['parent_deadline_epoch']-1200):
        raise PermissionError('FINITE_FUSION_SCOPE_OR_FROZEN_DOMAIN_INVALID')


def validate(spec, registry=None, now=None):
    from .pilot1_scope import validate as validate_parent
    now = time.time() if now is None else now
    validate_parent(spec, registry, now)
    path = Path(spec['fusion_binding']); binding = json.loads(path.read_text())
    binding_contract(binding, now)
    lease = json.loads((path.parent/'LEASE.json').read_text())
    registry = json.loads(Path(spec['registry']).read_text()) if registry is None else registry
    if (digest(path) != spec['fusion_binding_sha256']
            or binding['campaign_root'] != spec['campaign_root']
            or binding['pilot1_binding'] != spec['pilot1_binding']
            or binding['pilot1_binding_sha256'] != spec['pilot1_binding_sha256']
            or lease.get('status') != 'ACTIVE' or lease.get('binding_sha256') != digest(path)
            or lease.get('owner') != spec['owner'] or lease.get('scope') != SCOPE
            or spec['deadline_epoch'] > binding['deadline_epoch']
            or (path.parent/'STOP.json').exists()):
        raise PermissionError('ACTUAL_FUSION_SUPPLEMENT_LEASE_REQUIRED')
    if not any(e.get('path') == str(path.parent.parent) and e.get('scope') == SCOPE
               and e.get('owner') == spec['owner'] and e.get('local_lease') == str(path.parent/'LEASE.json')
               for e in registry['entries']):
        raise PermissionError('FUSION_SUPPLEMENT_NOT_REGISTERED')
    if not any(e.get('path') == spec['registered_root'] and e.get('source_sha') == spec['source_sha']
               and e.get('fusion_binding') == str(path) and e.get('scope') == SCOPE
               for e in registry['entries']):
        raise PermissionError('FRESH_FUSION_WORKER_NOT_REGISTERED')
    return binding


def validate_gpu(spec, actual):
    """One finite frozen-VLA shadow, after the actual four-arm calibration."""
    if spec.get('kind')=='FUSION_RESPONSE_WM':
        return validate_response_gpu(spec,actual)
    from .pilot1_scope import GPU_UUIDS
    if (spec.get('execution_site') != 'CORE_FUSION_NEURAL'
            or spec.get('profile') != 'wm02-fusion-vla-shadow' or spec.get('physics') is not False
            or spec.get('kind') != 'FROZEN_VLA_SHADOW' or spec.get('max_queries') != 1
            or spec.get('inference_mode') != 'OFFLINE_REAL_VLA_INFERENCE'
            or spec.get('GPU_uuid') not in GPU_UUIDS or actual['uuid'] != spec['GPU_uuid']
            or os.environ.get('CUDA_VISIBLE_DEVICES') != spec['GPU_uuid']
            or not 0 < spec.get('allocator_limit_bytes', 0) < actual['free_bytes']-1024**3
            or type(spec.get('checkpoint_parameter_bytes')) is not int
            or spec.get('checkpoint_parameter_bytes', 0) <= 0
            or spec.get('allocator_limit_bytes', 0) < spec.get('checkpoint_parameter_bytes', 0)
            or spec.get('numeric_mode') != 'ORIGINAL_FP32_WEIGHTS_BF16_AUTOCAST'
            or spec.get('GR00T_updates') != 0):
        raise PermissionError('FINITE_FROZEN_VLA_ORIGINAL_PRECISION_AND_ACTUAL_HEADROOM_REQUIRED')
    validate(spec)
    path = Path(spec['calibration_result']); result = json.loads(path.read_text())
    lease = json.loads((path.parent/'control/LEASE.json').read_text())
    if (digest(path) != spec['calibration_result_sha256'] or lease.get('status') != 'CLOSED'
            or result.get('schema') != 'WM02_FUSION_FIRST_ACTUAL_CALIBRATION_GROUP_V1'
            or result.get('exit_code') != 0 or result.get('actual_method_evaluations') != 4
            or result.get('actual_native_ticks') != 5*4103):
        raise PermissionError('ACTUAL_FIRST_NATIVE_CALIBRATION_BEFORE_VLA_REQUIRED')


def validate_response_gpu(spec,actual):
    """One small response-training worker, sharing measured local GPU headroom."""
    from collections import Counter
    from .pilot1_scope import GPU_UUIDS
    if (spec.get('execution_site')!='CORE_FUSION_NEURAL' or spec.get('profile')!='wm02-fusion-response-train'
            or spec.get('physics') is not False or spec.get('device')!='cuda'
            or spec.get('GPU_uuid')not in GPU_UUIDS or actual['uuid']!=spec['GPU_uuid']
            or os.environ.get('CUDA_VISIBLE_DEVICES')!=spec['GPU_uuid']
            or not 0<spec.get('allocator_limit_bytes',0)<=2*1024**3
            or spec['allocator_limit_bytes']+1024**3>=actual['free_bytes']
            or spec.get('GR00T_updates')!=0 or spec.get('numeric_mode')!='FP32_TF32_OFF'
            or spec.get('endpoint')!=2000 or spec.get('seeds')!=[366301,366302,366303]
            or spec.get('arms')!=['ACTION_CONDITIONED','NO_FUTURE_ACTION']):
        raise PermissionError('FINITE_SHARED_LOCAL_RESPONSE_WM_AND_ACTUAL_HEADROOM_REQUIRED')
    validate(spec)
    path=Path(spec['training_replay_result']);value=json.loads(path.read_text())
    lease=json.loads((path.parent.parent/'control/LEASE.json').read_text())
    if (digest(path)!=spec['training_replay_result_sha256'] or lease.get('status')!='CLOSED'
            or value.get('schema')!='WM02_FUSION_REAL_RESPONSE_REPLAY_V1'
            or value.get('eligible_windows')!=48 or value.get('real_post_choice_transitions')!=768
            or Counter(value.get('physical_parents',{}).values())!=dict(TRAIN=4,DEV=2)
            or value.get('excluded') or value.get('TEST_read')is not False
            or value.get('replay_sha256')!=spec.get('replay_sha256')):
        raise PermissionError('ACTUAL_COMPLETE_BALANCED_NEW_RESPONSE_DATA_REQUIRED')


def validate_structured_cpu(spec):
    """Finite two-arm NumPy fit under the existing FUSION owner and real data.

    It does not acquire or qualify a physical slot, invoke Torch or admit CUDA.
    The CLI separately checks sealed replay/task bytes before any fitting.
    """
    if (spec.get('execution_site') != 'CORE_FUSION_OFFLINE_CPU'
            or spec.get('profile') != 'wm02-fusion-structured-offline-cpu'
            or spec.get('kind') != 'FUSION_STRUCTURED_RESPONSE'
            or spec.get('physics') is not False or spec.get('device') != 'cpu'
            or spec.get('numeric_mode') != 'NUMPY_FLOAT64'
            or spec.get('arms') != ['ACTION_CONDITIONED', 'NO_FUTURE_ACTION']
            or spec.get('min_lopo_gain_mm') != .08124448952310876
            or spec.get('GR00T_updates') != 0
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ''):
        raise PermissionError('FINITE_TRAIN_ONLY_STRUCTURED_CPU_FUSION_REQUIRED')
    return validate(spec)


def validate_target_capture_cpu(spec):
    """Read-only, CPU-rendered pixels from a sealed actual target recording.

    Saved-state rendering is not target dynamic admission or online perception.
    Original scene/result/journal checks remain the capture CLI's responsibility.
    """
    if (spec.get('execution_site') != 'CORE_FUSION_TARGET_OBSERVATION_REPLAY_CPU'
            or spec.get('profile') != 'wm02-fusion-target-observation-replay-cpu'
            or spec.get('kind') != 'FUSION_TARGET_OBSERVATION_REPLAY'
            or spec.get('physics') is not False or spec.get('device') != 'cpu'
            or spec.get('capture_steps') != [3503, 3703, 4103]
            or spec.get('expected_native_ticks') != 4103
            or spec.get('GR00T_updates') != 0 or spec.get('WBC_changes') is not False
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or os.environ.get('MUJOCO_GL') != 'egl'
            or os.environ.get('PYOPENGL_PLATFORM') != 'egl'
            or os.environ.get('LIBGL_ALWAYS_SOFTWARE') != '1'
            or os.environ.get('__EGL_VENDOR_LIBRARY_FILENAMES') != '/usr/share/glvnd/egl_vendor.d/50_mesa.json'
            or 'MUJOCO_EGL_DEVICE_ID' in os.environ):
        raise PermissionError('READ_ONLY_SAVED_TARGET_CPU_CAPTURE_REQUIRED')
    return validate(spec)


def validate_corr_asset_cpu(spec):
    """Stage only the already published, fixed source CORR inference files."""
    if (spec.get('execution_site') != 'CORE_FUSION_ASSET_CPU'
            or spec.get('profile') != 'wm02-fusion-corr-asset-cpu'
            or spec.get('kind') != 'FUSION_CORR_ASSET'
            or spec.get('physics') is not False or spec.get('device') != 'cpu'
            or spec.get('GR00T_updates') != 0 or spec.get('WBC_changes') is not False
            or spec.get('original_checkpoint_manifest_sha256') !=
                '94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38'
            or spec.get('manifest_sha256') !=
                'ec46067a1179405d6c75081cfa333f0daa994f8e1c54bd3658eac0b5fb4eff61'
            or spec.get('data_revision') != 'b37a3d959335b548abb45f87ad48a5186ec662d9'
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ''):
        raise PermissionError('FIXED_SOURCE_CORR_INFERENCE_ASSET_CPU_ONLY_REQUIRED')
    return validate(spec)


def validate_contact_timing_cpu(spec):
    """Independent C-line event diagnostic; no contact execution grant."""
    if (spec.get('execution_site') != 'CORE_FUSION_CONTACT_TIMING_CPU'
            or spec.get('profile') != 'wm02-fusion-contact-timing-offline-cpu'
            or spec.get('kind') != 'FUSION_CONTACT_TIMING_BASELINE'
            or spec.get('recipe') != 'WM02_CONTACT_TIMING_MATCHED_TRAIN_FIR_V1'
            or spec.get('physics') is not False or spec.get('device') != 'cpu'
            or spec.get('numeric_mode') != 'NUMPY_FLOAT64' or spec.get('ridge') != 1.
            or spec.get('arms') != ['ACTION_CONDITIONED', 'NO_FUTURE_PROGRAM']
            or spec.get('information') != 'PREDECLARED_PROGRAM_CONDITIONAL'
            or spec.get('GR00T_updates') != 0 or spec.get('WBC_changes') is not False
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ''):
        raise PermissionError('FINITE_SAME_INFORMATION_CONTACT_CPU_EVENT_DIAGNOSTIC_REQUIRED')
    return validate(spec)


def validate_contact_full_history_cpu(spec):
    """New full927 CPU recipe; the historical FIR admission stays unchanged."""
    from .contact_timing_full_history import BLIND_SOURCE, FEATURE_RECIPE
    if (spec.get('execution_site') != 'CORE_FUSION_CONTACT_FULL_HISTORY_CPU'
            or spec.get('profile') != 'wm02-fusion-contact-full-history-offline-cpu'
            or spec.get('kind') != 'FUSION_CONTACT_TIMING_FULL_HISTORY_BASELINE'
            or spec.get('recipe') != 'WM02_CONTACT_TIMING_FULL927_MATCHED_TRAIN_RIDGE_V1'
            or spec.get('physics') is not False or spec.get('device') != 'cpu'
            or spec.get('numeric_mode') != 'NUMPY_FLOAT64'
            or type(spec.get('ridge')) not in (int, float) or spec['ridge'] != 1.
            or spec.get('arms') != ['ACTION_CONDITIONED', 'NO_FUTURE_PROGRAM']
            or spec.get('information') != 'PREDECLARED_PROGRAM_CONDITIONAL'
            or spec.get('feature_recipe') != FEATURE_RECIPE
            or spec.get('observed_history_features') != 927 or spec.get('blind_source') != BLIND_SOURCE
            or spec.get('GR00T_updates') != 0 or spec.get('WBC_changes') is not False
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ''):
        raise PermissionError('FINITE_FULL927_MATCHED_TRAIN_CPU_CONTACT_DIAGNOSTIC_REQUIRED')
    return validate(spec)


def register(path):
    """Register once under the same canonical locks, only after the sealed batch.

    Caller supplies actual immutable binding bytes. Missing or active predecessor
    is a wait, never a reason to interrupt it or mutate its capsule.
    """
    from scripts.pi0292.durable import OwnerLock, write
    from scripts.pi0294.night10.native_relay import snapshot, once
    path = Path(path); binding = json.loads(path.read_text()); binding_contract(binding, time.time())
    campaign = Path(binding['campaign_root']); window = path.parent.parent
    if os.uname().nodename != binding['host'] or os.getuid() != binding['uid']:
        raise PermissionError('ACTUAL_EXISTING_FUSION_HOST_UID_REQUIRED')
    with closing(OwnerLock(campaign/'OWNER.lock')), closing(OwnerLock(binding['physical_owner_lock'])):
        if (path.parent/'LEASE.json').exists(): raise FileExistsError('RECONCILE_EXISTING_FUSION_SCOPE')
        parent = json.loads((campaign/'control/LEASE.json').read_text())
        p1 = json.loads((Path(binding['pilot1_binding']).parent/'LEASE.json').read_text())
        prior = Path(binding['predecessor_result']); result = json.loads(prior.read_text())
        previous_lease = json.loads((prior.parent/'control/LEASE.json').read_text())
        process = subprocess.check_output(['systemctl', '--user', 'show', binding['predecessor_unit'],
                                           '-p', 'MainPID'], text=True).strip()
        if (parent.get('status') != 'ACTIVE' or parent.get('owner') != binding['owner']
                or parent['deadline_epoch'] != binding['parent_deadline_epoch']
                or p1.get('status') != 'ACTIVE'
                or digest(binding['pilot1_binding']) != binding['pilot1_binding_sha256']
                or p1.get('binding_sha256') != binding['pilot1_binding_sha256']
                or digest(prior) != binding['predecessor_result_sha256']
                or previous_lease.get('status') != 'CLOSED'
                or process != 'MainPID=0' or result.get('exit_code') != 0):
            raise PermissionError('CURRENT_OWNER_AND_ACTUALLY_SEALED_PREDECESSOR_REQUIRED')
        reg, before = snapshot(binding['registry'])
        reg['entries'].append(dict(role='WM02_FUSION_NIGHT1_SUPPLEMENT', path=str(window),
            run_uuid=window.name, owner=binding['owner'], executor_node=binding['host'], scope=SCOPE,
            source_sha=binding['source_sha'], plan_sha=PLAN, local_lease=str(path.parent/'LEASE.json')))
        once(Path(binding['registry']), reg, window.name+'-REGISTER', expected_before_sha256=before)
        value = dict(status='ACTIVE', scope=SCOPE, owner=binding['owner'], run_uuid=window.name,
            binding_sha256=digest(path), deadline_epoch=binding['deadline_epoch'],
            new_campaign=False, predecessor_closed_lease_reopened=False)
        write(path.parent/'LEASE.json', value)
        return dict(status='FUSION_SUPPLEMENT_REGISTERED_NOT_NATIVE_QUALIFIED',
                    binding_sha256=digest(path), native_integrations=0, VLA_inferences=0)
