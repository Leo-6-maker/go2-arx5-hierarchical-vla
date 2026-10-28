"""Keep typed read-side failures excluded without replaying physical successes."""
import math
import os
from pathlib import Path
import re

from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.scene_cards import digest


def contact_force_only(result,audit,s1):
    error=re.fullmatch(r'Independent reconstruction error ([0-9.eE+-]+) exceeds 1e-05',audit.get('error',''))
    return bool(error and math.isfinite(float(error[1])) and float(error[1])>1e-5
        and audit.get('status')=='AUDIT_FAIL_PRESERVED' and audit.get('error_type')=='ValueError'
        and 'near(recorded["normal_force_N"], force[0], 1e-5)' in audit.get('traceback','')
        and all(audit.get(k)==0 for k in ('integrations','render_calls','model_loads'))
        and result.get('status')=='VALID_COMPLETE' and result.get('physics_steps')==16000
        and result.get('rgb_pairs')==641 and result.get('model_calls')==0
        and result.get('score',{}).get('safety_valid') is True
        and result.get('score',{}).get('complete_32s') is True
        and s1.get('live_native_failures')==s1.get('pre_execution_rejections')==0
        and s1.get('shadow_invocations')==80 and s1.get('shadow_integrations')==16000
        and s1.get('actions_modified') is False and 0<=s1.get('maximum_live_shadow_difference',math.inf)<=1e-10)


def saved_case(binding,source,*,unpublished=False):
    source=Path(source);card=binding['card'];folder=source/'episodes'/binding['id']
    if card!=read(folder/'CARD.json') or digest(card)!=binding['card_sha256']:
        raise PermissionError('PV1_REJECTED_RAW_EXACT_ORIGINAL_CARD')
    if unpublished:
        worker=read(source/'worker_processes/capture.json')
        p=Path('/proc',str(worker['pid']),'stat');f=p.read_text().split() if p.exists() else None
        if f and f[2]!='Z' and f[21]==str(worker['proc_start']):
            raise PermissionError('PV1_UNPUBLISHED_REJECTED_RAW_REQUIRES_NATIVE_SOURCE_TERMINAL')
    result,audit,s1=(read(folder/n) for n in ('RESULT.json','AUDIT.json','S1_RESULT.json'))
    if not contact_force_only(result,audit,s1):
        raise PermissionError('PV1_UNCLASSIFIED_RAW_FAILURE_STILL_REQUIRES_DIAGNOSIS')
    m=read(source/'RUN_MANIFEST.json');worker=read(folder/'WORKER_RECEIPT.json')
    if (result['source_sha']!=m['code']['vla_commit'] or worker['source_sha']!=result['source_sha']
            or result['samples_sha256']!=sha(folder/'samples.json')
            or any(not (folder/r['path']).resolve().is_relative_to(folder.resolve())
                or (folder/r['path']).is_symlink() or (folder/r['path']).stat().st_size!=r['bytes']
                or sha(folder/r['path'])!=r['sha256'] for r in read(folder/'FILE_MANIFEST.json'))):
        raise PermissionError('PV1_REJECTED_RAW_IDENTITY_OR_MEMBER_HASH_CHANGED')
    return dict(id=card['id'],status='VALID_NEGATIVE_NOT_TRAINING',task=card['task'],scene_group=card['group_id'],
        nominal_clearance_m=card['nominal_clearance_m'],raw_manifest_sha256=sha(folder/'FILE_MANIFEST.json'),
        audit_sha256=sha(folder/'AUDIT.json'),S1_rejections=0,native_faults=0,
        data_readside_rejection=True,independent_raw_admitted=False,
        raw_task_score_success=result['score']['persistent_success'],scientific_task_failure=False,
        reason='ORIGINAL_CONTACT_FORCE_RECONSTRUCTION_AUDIT_FAILED_NO_TOLERANCE_CHANGE',
        valid_failure_not_engineering_replacement=True,new_episodes=0,new_integrations=0)


def retired_alternate(binding,root,active_route):
    root=Path(root);source=Path(binding['retired_unattempted_alternate_source']);card=binding['card']
    if (active_route!='COVERAGE_FIRST' or card['task']!='T2' or card['nominal_clearance_m']!=.012
            or source.is_symlink() or source.parent.resolve()!=root.parent.resolve()
            or digest(card)!=binding['card_sha256']):
        raise PermissionError('PV1_ONLY_UNUSED_ALTERNATE_AFTER_PREDECLARED_COVERAGE_FALLBACK')
    b=read(source/'OUTPUT_BINDING.private.json');l=read(source/'RESOURCE_LEDGER.json')
    old=read(source/'plans/CAPTURE_PLAN.json')
    if (b['uid']!=os.getuid() or Path(b['run_root']).resolve()!=source.resolve()
            or read(source/'RUN_MANIFEST.json')['scope']!='PI0294_PV1_CORE_TRAIN_CAPTURE_V1'
            or read(b['lease'])['status']!='CLOSED' or l['unknown_effects'] or card not in old['cards']
            or card['id'] in l['attempted_cells'] or (source/'episodes'/card['id']).exists()):
        raise PermissionError('PV1_NEVER_RETIRE_AN_ALREADY_ATTEMPTED_ALTERNATE')
    return dict(id=card['id'],scene_group=card['group_id'],task='T2',nominal_clearance_m=.012,
        status='NOT_ATTEMPTED_ALTERNATE_RETIRED_AFTER_COVERAGE_FALLBACK',admitted_for_training=False,
        scientific_task_failure=False,physical_attempted=False,new_episodes=0,new_integrations=0,
        reason='OPTIONAL_OTHER_REFERENCE_NO_LONGER_PART_OF_SELECTED_COVERAGE_COMPARISON')
