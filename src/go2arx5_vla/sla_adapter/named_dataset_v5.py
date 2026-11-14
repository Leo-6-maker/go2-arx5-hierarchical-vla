"""Typed fresh TRAIN8 cohort; the old two-episode reader stays strict."""
from .dense_dataset_v4 import DensePilotDataset
from .gripper_v2 import VERSION

SCHEMA = 'SLA_FRESH_NAMED_PICK_HOLD_TRAIN8_N20_V5'
DATASET_ID = 'T_PICK_HOLD_TRAIN8_P10_N20_V5'
SCOPE = 'FRESH_TRAIN8_ONE_CUBE_PICK_HOLD_LIMITED_FITTING_NOT_D100'
LAYOUTS = ('TRAIN01', 'TRAIN04', 'TRAIN07', 'TRAIN10', 'TRAIN13', 'TRAIN16', 'TRAIN21', 'TRAIN25')
CONDITIONS = tuple((layout, restart) for restart in (5, 6, 7) for layout in LAYOUTS)[:20]
AUDIT_SCHEMA = 'SLA_FRESH_NAMED_N20_ALL_WINDOW_READBACK_V5'
RECIPE = 'SLA_FRESH_NAMED_N20_FM32_3K_V5'
OPERATION = 'NAMED_PICK_HOLD_N20_FM_3K'


def frame_cache_limit(spec):
    """Bounded host input cache for a declared future cold run only."""
    value=spec.get('frame_cache_bytes',512*1024**2)
    if type(value)is not int or not 512*1024**2<=value<=16*1024**3:
        raise ValueError('EXPLICIT_BOUNDED_HOST_FRAME_CACHE_BYTES_REQUIRED')
    return value


def manifest_contract(value):
    episodes = value.get('episodes') or []
    if (value.get('schema') != SCHEMA or value.get('state') != 'committed'
            or value.get('dataset_id') != DATASET_ID or value.get('scope') != SCOPE
            or value.get('tasks') != ['PICK_HOLD'] or value.get('objects') != 1
            or value.get('split') != 'TRAIN' or value.get('codec') != VERSION
            or value.get('D100_new') is not False or value.get('D300_new') is not False
            or value.get('PICK_PLACE_admitted') is not False
            or value.get('legacy_engineering_receipts_modified') is not False
            or value.get('old_2EP_reused') != 0 or value.get('old_engineering16_reused') != 0
            or value.get('phase_input_to_student') is not False
            or value.get('consumption_proven') is not True or value.get('actual_success_tail_s') != 2.
            or value.get('command_period_ticks') != 25 or value.get('frame_period_ticks') != 50
            or value.get('anchor_period_ticks') != 250 or value.get('physics_dt_s') != .002
            or value.get('supervised_windows') != 1220
            or value.get('selection_rule') != 'FIRST20_DECLARED_ASSIGNMENTS_ALL_COMPLETE_NO_SUBSTITUTION'
            or len(episodes) != 20 or len({ep['id'] for ep in episodes}) != 20
            or [(ep['layout'], ep['restart']) for ep in episodes] != list(CONDITIONS)
            or any(ep['split'] != 'TRAIN' or ep['task'] != 'PICK_HOLD'
                   or ep['prefix_atoms'] != 10 or ep['windows'] != 61 or ep['frames'] != 321
                   or ep['commands'] != 640 or ep['native_integrations'] != 21703 for ep in episodes)):
        raise ValueError('EXACT_FRESH_N20_TRAIN8_COMPLETE_COHORT_NOT_2EP_OR_D100')


class NamedTargetDataset(DensePilotDataset):
    validate_manifest = staticmethod(manifest_contract)
