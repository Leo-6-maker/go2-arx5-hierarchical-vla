"""One evidenced cold alloc04 boundary; old STOP evidence is never cleared."""
from pathlib import Path

from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha

AUTH='c8cdf60e41e31da5796b44a064ef2f22d594a491'
OLD_SOURCE='03646938ece5c6fa5f09b0fd00732f8159a51468'
STOP_SHA='713fef5c2d3c66912781fe7374a95f48fbbdb2370ff49dfc20b213af0bc4526c'
BINDING='plans/ALLOC04_COLD_BOUNDARY.json'
HISTORY='plans/ALLOC04_PRIOR_RUN_MANIFEST.json'
LEDGER='plans/ALLOC04_PRIOR_HOPPER_LEDGER.json'
EFFECTS='plans/ALLOC04_PARTIAL_EFFECT_CLOSURE.json'
OUTCOMES='plans/ALLOC04_PRIOR_ADJUDICATIONS.json'
NEW_STOP='RELAY_ENGINEERING_STOP_ALLOCATION04.json'
LEARNING_FILES=(
    'scripts/pi0294/mp1/train.py','scripts/pi0294/mp1/trainer.py','scripts/pi0294/mp1/dataset.py',
    'scripts/pi0294/e1/model.py','scripts/pi0294/e1/neural.py','scripts/pi0294/native_dataset.py',
    'scripts/pi0294/native_train.py','scripts/pi0294/numerics.py','scripts/pi0294/scene.py',
    'scripts/pi0294/episode.py','scripts/pi0294/s1_preflight.py','scripts/pi0294/score.py')


def synchronize_current_manifest(root,*,synchronize=None):
    import os
    if synchronize is None:
        from .directory_visibility_audit import force_server_attributes
        synchronize=force_server_attributes
    root=Path(root);path=root/'RUN_MANIFEST.json'
    if root.is_symlink() or path.is_symlink():raise PermissionError('REAL_COLD_MANIFEST_PATH')
    synchronize(root)
    with os.scandir(root) as entries:
        if sum(1 for _ in entries)>512:raise PermissionError('BOUNDED_COLD_MANIFEST_PARENT')
    synchronize(path)


def validate(root,worker_manifest=None,*,proposed_manifest=None):
    root=Path(root);m=read(root/'RUN_MANIFEST.json') if proposed_manifest is None else proposed_manifest
    relative=m['execution'].get('cold_engineering_binding_relative')
    if relative is None:return None
    from . import cold_boundary05 as fifth
    if relative==fifth.BINDING:return fifth.validate(root,worker_manifest,proposed_manifest=m)
    if relative!=BINDING or root.name!='source-pair-01':raise PermissionError('EXACT_COLD_ALLOC04_BOUNDARY_ONLY')
    required=[BINDING,HISTORY,LEDGER,EFFECTS,OUTCOMES,'RELAY_ENGINEERING_STOP.json']
    for name in required:
        p=root/name
        if (p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) or p.stat().st_mode&0o222
                or m['execution']['sealed_inputs'].get(name)!=sha(p)):
            raise PermissionError('SEALED_IMMUTABLE_COLD_BOUNDARY_EVIDENCE')
    q=read(root/BINDING);old=read(root/HISTORY);past=read(root/LEDGER);proof=read(root/EFFECTS);rows=read(root/OUTCOMES)
    a=read(root.parent.parent/'MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json')
    if (q['schema']!='MP1_SAME_ROOT_COLD_ALLOC04_BOUNDARY_V1' or q['authority_release']!=AUTH
            or q['run_uuid']!=root.name or q['parent_run_uuid']!=root.parent.parent.name
            or q['original_T0_epoch']!=a['T0_epoch'] or q['original_T0_epoch']!=old['limits']['original_MP1_T0_epoch'] or q['target_allocation_number']!=4
            or q['new_source_sha']!=m['code']['vla_commit'] or q['prior_manifest_sha256']!=sha(root/HISTORY)
            or q['old_STOP_sha256']!=STOP_SHA or sha(root/'RELAY_ENGINEERING_STOP.json')!=STOP_SHA
            or old['code']['vla_commit']!=OLD_SOURCE or q['new_budget_or_clock'] is not False
            or q['new_attempt_rights'] is not False or q['old_STOP_removed'] is not False):
        raise PermissionError('ORIGINAL_AUTHORITY_IDENTITY_CLOCK_AND_RETAINED_STOP')
    if ({k:v for k,v in m.items() if k not in ['code','execution']}
            !={k:v for k,v in old.items() if k not in ['code','execution']}):
        raise PermissionError('COLD_BOUNDARY_CANNOT_CHANGE_SCIENTIFIC_MANIFEST')
    if ({k:v for k,v in m['execution'].items() if k not in ['sealed_inputs','cold_engineering_binding_relative']}
            !={k:v for k,v in old['execution'].items() if k!='sealed_inputs'}):
        raise PermissionError('COLD_BOUNDARY_CANNOT_CHANGE_ENTRY_IDENTITY')
    variable={'vla_commit','capsule_relative','source_seal_relative','source_seal_sha256','unchanged_learning_files'}
    if ({k:v for k,v in m['code'].items() if k not in variable}
            !={k:v for k,v in old['code'].items() if k not in variable}):
        raise PermissionError('COLD_BOUNDARY_CANNOT_CHANGE_UPSTREAM_OR_FROZEN_CODE')
    expected=dict(old['execution']['sealed_inputs'],**{name:sha(root/name) for name in required})
    if m['execution']['sealed_inputs']!=expected:raise PermissionError('NO_REMOVED_OR_UNDECLARED_SCIENTIFIC_INPUT')
    if (past['run_uuid']!=root.name or past['GPU_allocated_s']!=23728 or past['GPU_reserved_s'] or past['pending_updates'] or past['optimizer_updates']!=2560
            or past['generation_queries']!=1277 or len(past['PBS_submissions'])!=3
            or [(v['job_id'],v['status'],v['GPU_conservative_charge_s']) for v in past['PBS_submissions']]!=[
                ('672080.hopper-m-02','NATIVE_TERMINAL',11795),('672471.hopper-m-02','NATIVE_TERMINAL',11521),
                ('672829.hopper-m-02','NATIVE_TERMINAL',412)]):
        raise PermissionError('ALL_PRIOR_ORIGINAL_SOURCE_COSTS_STILL_APPLY')
    if (proof['status']!='PASS_KNOWN_PARTIAL_PREFIX_AND_ORPHAN_QUERY_CLOSED_NOT_A_TASK_RESULT'
            or proof['valid_policy_outcome'] is not False or proof['rerun_authority'] is not False
            or proof['old_STOP_markers_modified'] is not False or proof['new_model_queries'] or proof['new_physics']
            or proof['all_local_mutation_journal_entries_complete']!=1926
            or proof['core']['executed_ACKs']!=26 or proof['core']['integrations']!=5200
            or proof['hopper']['native_draws']!=26 or proof['hopper']['orphan_sequence']!=26
            or proof['hopper']['orphan_request_sha256']!='0b9a988b9053c972da640d966855c2a6d4e81b4a7bb6187c8dc4e08a68a91ab1'
            or proof['hopper']['old_STOP_sha256']!=STOP_SHA or proof['hopper']['all_array_hashes_pass'] is not True):
        raise PermissionError('ONLY_ACTUALLY_RECONCILED_TERMINAL_PARTIAL_EFFECTS')
    if len(rows)!=20 or len({v['attempt_id'] for v in rows})!=20 or any(v['live_native_failures'] for v in rows):
        raise PermissionError('ALL_ORIGINAL_OUTCOMES_AND_NO_NATIVE_SAFETY_STOP')
    from .plans import remaining_panel
    plan=read(root/'plans/SOURCE_EVALUATION_PLAN.json')
    remaining=[c for spec in plan['panels'].values() for c in remaining_panel(rows,spec)]
    if len(remaining)!=13 or any('F1-000-T1' in c['id'] and c['mp1_model_label']=='S-CORR-512-TRAIN4' for c in remaining):
        raise PermissionError('NO_NEW_VALID_NEGATIVE_OR_EXHAUSTED_ENGINEERING_ATTEMPT')
    oldseal=read(root/old['code']['source_seal_relative'])
    if sha(root/old['code']['source_seal_relative'])!=old['code']['source_seal_sha256']:
        raise PermissionError('OLD_SOURCE_SEAL_PRESERVED')
    hashes={n:oldseal['files'][n] for n in LEARNING_FILES}
    for manifest in [m]+([worker_manifest] if worker_manifest is not None else []):
        sealpath=root/manifest['code']['source_seal_relative'];seal=read(sealpath)
        if (manifest['code'].get('unchanged_learning_files')!=hashes
                or sha(sealpath)!=manifest['code']['source_seal_sha256']
                or any(seal['files'].get(n)!=digest for n,digest in hashes.items())):
            raise PermissionError('ORIGINAL_MODEL_TRAINING_DATA_NUMERICS_AND_PLANT_BYTES')
    for name in ['SOURCE_PHYSICS_STOP.json','SOURCE_SAFETY_STOP_FROM_CORE.json']:
        if (root/name).exists():raise PermissionError('COLD_BOUNDARY_NEVER_RESOLVES_NATIVE_SAFETY_STOP')
    return q


class Guard:
    """Fully validate once; retain stop/input identity checks on every guard call."""
    def __init__(self,root,allocation_number):
        self.root=Path(root);self.allocation_number=allocation_number;self.binding=validate(root)
        from . import cold_boundary05 as fifth
        self.fifth=self.binding is not None and self.binding['target_allocation_number']==5
        names=(fifth.REQUIRED if self.fifth else [BINDING,HISTORY,LEDGER,EFFECTS,OUTCOMES,'RELAY_ENGINEERING_STOP.json'])+['RUN_MANIFEST.json']
        if self.fifth:
            names+=list(read(self.root/fifth.HISTORY)['execution']['sealed_inputs'])
        self.identities={n:sha(self.root/n) for n in names} if self.binding is not None else {}

    def blocked(self):
        from . import cold_boundary05 as fifth
        if (self.root/(fifth.NEW_STOP if self.fifth else NEW_STOP)).exists():return True
        if self.binding is None:return (self.root/'RELAY_ENGINEERING_STOP.json').exists()
        if self.allocation_number!=(5 if self.fifth else 4):return True
        # Byte hashes retain the admission evidence without reparsing all prior traces.
        if any(sha(self.root/n)!=digest for n,digest in self.identities.items()):
            raise PermissionError('COLD_BOUNDARY_EVIDENCE_CHANGED_AFTER_ADMISSION')
        return False


def blocked(root,allocation_number):
    root=Path(root)
    from . import cold_boundary05 as fifth
    if read(root/'RUN_MANIFEST.json')['execution'].get('cold_engineering_binding_relative')==fifth.BINDING:
        return Guard(root,allocation_number).blocked()
    if (root/NEW_STOP).exists():return True
    if not (root/'RELAY_ENGINEERING_STOP.json').exists():
        if read(root/'RUN_MANIFEST.json')['execution'].get('cold_engineering_binding_relative'):
            raise PermissionError('RETAINED_OLD_STOP_WAS_REMOVED')
        return False
    if type(allocation_number) is not int or allocation_number!=4:return True
    return validate(root) is None


def stop_target(root,allocation_number):
    root=Path(root)
    from . import cold_boundary05 as fifth
    if allocation_number==5 and validate(root) is not None and validate(root)['target_allocation_number']==5:
        return root/fifth.NEW_STOP
    return root/(NEW_STOP if allocation_number==4 and validate(root) is not None else 'RELAY_ENGINEERING_STOP.json')
