"""Bind the one exploratory checkpoint to the two already-paid Core DEV6 blocks."""
from pathlib import Path
import time

from scripts.sla.core_hopper_expert import PLAN, POP
from scripts.sla.core_hopper_native_scope import AUTHORITY, SCOPE, read_pin
from scripts.sla.hopper_parent_custody import digest, write_once


ARM = 'MAIN-phase03800'
EXPLORATORY = 'MAIN3800_EXPLORATORY_MATCHED_DEV6_V1'
CHECKPOINT = '502d890d3ff314ac42f4c79c360942b9b626bf1fcc4b6fc488fbab15ba508361'
MANIFEST = 'd987be63fdf39ded1825fa37703511fd9449589c9b20d84814ac85d5b2854584'
EVENT = '90edf1e2e134695733d37fd973c4fed21418fc310ef4d018ad45b2bb0becb9f7'
REVISION = '1ca2fd64e989ee190c1e20345066d0939916a5c2'
TRAINING_SOURCE = 'b9581196d0acd02fb748222988e5a5543ce1f232'
RGB = 'd297e57464478493de2a6d46ff89f6c18d26efebf11e54897f5647353923c52a'
HISTORY = {
    'MAIN-phase00000': 'cc50a154e9f8d8a03effd99b6d787d4ec428c4bb62eddbdf438bfd967a0df1a4',
    'MAIN-phase02000': '4187d3e8f79c5fae078a7ee53bc5c74a9fd7b4efd8c8486d993e19fe88fe4735',
}
FROZEN_IDENTITY = ('action_sha256','foundation_sha256','frozen_model','panel',
                   'partition_sha256','state_sha256','tokenizer_sha256','training_source')
# Include the actual library and gate references that route the frozen execution.
CASE_IDENTITY = ('frozen_files','model_path','ROS_DOMAIN_ID','mobile_world_lock','native_binary',
                 'solver_binary','controller_binary','observer_library','outer_library','trace_library',
                 'mobile_safety_library','mobile_safety_binding','mobile_safety_library_sha256',
                 'mobile_guard_qualification','mobile_wbc_boundary_v1','physics_dt_s','max_ticks',
                 'policy_queries','execution_clock_profile','multitask_geometry','task_instruction_id',
                 'stage_budget_s','global_task_budget_s','language','startup_pause_stride_ticks')


def pin(path):
    return dict(path=str(path),sha256=digest(path))


def load_manifest(package, original_package):
    root = Path(package['path'])
    if root != root.resolve(strict=True) or package['sha256'] != MANIFEST:
        raise ValueError('Unexpected MAIN3800 local package identity')
    manifest = read_pin(dict(path=str(root/'CHECKPOINT.json'),sha256=MANIFEST))
    original = read_pin(dict(path=str(Path(original_package['path'])/'CHECKPOINT.json'),
                             sha256=original_package['sha256']))
    if (manifest['schema'] != 'SLA_GO2_EXPLORATORY_INFERENCE_CHECKPOINT_V1'
            or manifest['role'] != 'MAIN' or manifest['phase'] != 3800
            or manifest['checkpoint_sha256'] != CHECKPOINT or manifest['training_event_sha256'] != EVENT
            or manifest['training_source'] != TRAINING_SOURCE
            or any(manifest[k] is not False for k in ('physics','native_admission','hardware_allowed'))
            or any(manifest[k] != original[k] for k in FROZEN_IDENTITY)):
        raise ValueError('MAIN3800 frozen model or training identity differs')
    rows = manifest['arrays']
    if len(rows) != 29 or set(root.iterdir()) != {root/'parameters',root/'CHECKPOINT.json'}:
        raise ValueError('Expected exactly 29 Go2 weights')
    members = set()
    for row, baseline in zip(rows, original['arrays'], strict=True):
        rel = Path(row['file'])
        path = root/rel
        if (set(row) != set(baseline) or any(row[k] != baseline[k] for k in row if k != 'sha256')
                or len(rel.parts) != 2 or rel.parts[0] != 'parameters' or rel.suffix != '.npy'
                or not path.is_file() or path.is_symlink() or path.resolve() != path
                or digest(path) != row['sha256']):
            raise ValueError('A MAIN3800 weight or its original parameter identity differs')
        members.add(path)
    if len(members) != 29 or set((root/'parameters').iterdir()) != members:
        raise ValueError('Unexpected MAIN3800 weight member')
    return manifest


def publish(path, cfg):
    from scripts.sla.core_hopper_student_endpoint import cases

    population = read_pin(cfg['population'])
    panel = cases(population,ARM,'DEV')[:6]
    reference = Path(cfg['historic_run'])
    baselines = {}
    for arm, sha in HISTORY.items():
        folder = reference/'blocks'/f'{arm}-DEV-00'
        reconciled = read_pin(dict(path=str(folder/'RECONCILED.json'),sha256=sha))
        terminal = read_pin(reconciled['block'])
        if reconciled['requests'] != 6 or terminal['new_verified_native_requests'] != 6 or terminal['exit_code']:
            raise ValueError('Original first six denominator differs')
        roster = cases(population,arm,'DEV')[:6]
        cards = []
        for row, expected, result_sha in zip(panel, roster, reconciled['records'], strict=True):
            saved = folder/'cases'/expected['id']
            result = read_pin(dict(path=str(saved/'RESULT.private.json'),sha256=result_sha))
            card = pin(saved/'EXECUTION_SPEC.json')
            spec = read_pin(card)
            if (result['physical_requests'] != 1 or result['technical_failure'] or spec['arm'] != arm
                    or any(row[k] != expected[k] for k in row.keys()-{'id','arm'})
                    or spec['panel_entry_id'] != expected['id']):
                raise ValueError('Original matched condition differs')
            cards.append(card)
        baselines[arm] = cards
    original_package = read_pin(pin(reference/'endpoints/MAIN-phase00000.json'))['gate']['package']
    package = cfg['package']
    m = load_manifest(package,original_package)
    receiver = read_pin(cfg['receiver'])
    paired = read_pin(cfg['paired'])
    if (cfg['paired']['sha256'] != RGB or paired['model_identity']['manifest'] != MANIFEST
            or paired['model_identity']['checkpoint'] != CHECKPOINT
            or paired['model_identity']['revision'] != REVISION or paired['new_predictions'] != 16
            or paired['native_requests'] or paired['optimizer_updates']
            or receiver['action']['sha256'] != m['action_sha256'] or receiver['panel']['sha256'] != m['panel']
            or receiver['tokenizer']['sha256'] != m['tokenizer_sha256']
            or receiver['model']['parent_manifest']['sha256'] != m['foundation_sha256']
            or receiver['model']['state_sha256'] != m['state_sha256']):
        raise ValueError('Missing actual exploratory RGB16 or frozen Core identity')
    gate = dict(status=EXPLORATORY,arm=ARM,checkpoint=CHECKPOINT,manifest=MANIFEST,package=package,
                original_package=original_package,core_result=cfg['paired'],receiver=cfg['receiver'])
    proof = dict(gate=gate,plan=PLAN,source=cfg['source'],core_profile=receiver['gpu'],panels={'DEV':panel},
                 compiled_population=cfg['population'],baselines=baselines,hardware_allowed=False,
                 physical_requests=0,model_generations=0,check_epoch=time.time())
    write_once(path,proof)
    return pin(path)


def check_binding(spec, proof, gate, receiver):
    if (spec['arm'] != ARM or spec['plan_commit'] != PLAN or spec['scope'] != SCOPE
            or spec['source_sha'] != proof['source'] or proof['plan'] != PLAN or receiver['plan'] != PLAN
            or proof['hardware_allowed'] is not False or proof['physical_requests'] or proof['model_generations']
            or proof['core_profile'] != receiver['gpu'] or proof['compiled_population'] != spec['compiled_population']
            or proof['compiled_population']['sha256'] != POP or set(proof['panels']) != {'DEV'}
            or len(proof['panels']['DEV']) != 6 or spec['evaluation_split'] != 'DEV'
            or set(proof['baselines']) != set(HISTORY) or gate['status'] != EXPLORATORY or gate['arm'] != ARM
            or gate['receiver'] != spec['receiver'] or gate['checkpoint'] != CHECKPOINT or gate['manifest'] != MANIFEST
            or gate['package']['sha256'] != MANIFEST or spec['packages'] != {ARM:gate['package']}
            or gate['core_result']['sha256'] != RGB):
        raise PermissionError('Missing the independent matched MAIN3800 DEV6 binding')


def check_case(case, authority, proof):
    if (authority['source'] != AUTHORITY or authority['panel_start'] != 0 or authority['native_request_cap'] != 6
            or authority['arm'] != ARM or authority['split'] != 'DEV'):
        raise PermissionError('Only the first six exploratory conditions are registered')
    index = authority['case_ids'].index(case['panel_entry_id'])
    for arm in HISTORY:
        original = read_pin(proof['baselines'][arm][index])
        if (any(case[key] != original[key] for key in CASE_IDENTITY)
                or any(case['mobile_condition'][k] != v for k,v in original['mobile_condition'].items()
                       if k != 'collection_episode')):
            raise ValueError('Matched native condition or frozen evaluator changed at index '+str(index))
