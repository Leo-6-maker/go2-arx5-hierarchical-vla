"""One finite CPU worker: reproduce first, then one sealed range and paired readers."""
import copy
import importlib.metadata
import os
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np

from scripts.pi0291_arx5_data.native_dataset import cpu_guard, validate_samples
from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0292.train_adapter import build_processor, OFFICIAL_SHA
from scripts.pi0294.e1.scope import STATS
from scripts.pi0294.fd1 import put, sha
from scripts.pi0294.native_dataset import NativeDataset
from scripts.pi0294.pv1.export import committed_packet, batch_inventory
from scripts.pi0294.pv1.event_exposure import analyze as event_exposure
from scripts.pi0294.scene_cards import digest
from .numeric import active, relative, scan, serial, reproduction_matches
from .rules import Envelope, candidate_statistics, TRACE_ANCHORS, CONTROL_ANCHORS, isclose_defaults
from .scope import validate, install_import_guard, ZERO


def load_case(case, root, guard):
    folder = Path(case['source_episode']); card = read(folder/'CARD.json')
    if folder.is_symlink() or digest(card) != case['card_sha256'] or card['split'] != 'TRAIN':
        raise PermissionError('CR1_EXACT_SAVED_TRAIN_CARD')
    source = Path(case['source_registry_root']); binding = read(source/'OUTPUT_BINDING.private.json')
    if sha(binding['lease']) != case['source_lease_sha256'] or read(binding['lease'])['status'] != 'CLOSED':
        raise PermissionError('CR1_ONLY_CLOSED_EXACT_SOURCE_LEASE')
    actual_root=folder.parent.parent; ab=read(actual_root/'OUTPUT_BINDING.private.json')
    matches=[e for e in read(ab['registry'])['entries'] if e.get('role')==ab['role']]
    if (read(ab['lease'])['status'] != 'CLOSED' or len(matches)!=1
            or Path(matches[0]['path']).resolve()!=actual_root.resolve()):
        raise PermissionError('CR1_ACTUAL_RAW_ROOT_MUST_ALSO_BE_REGISTERED_CLOSED')
    checks = {'FILE_MANIFEST.json':'raw_manifest_sha256','RESULT.json':'result_sha256',
              'S1_RESULT.json':'s1_sha256','SCORE.json':'score_sha256','samples.json':'samples_sha256','physics.npz':'physics_sha256'}
    for name,key in checks.items():
        guard()
        if sha(folder/name)!=case[key]: raise ValueError('CR1_SAVED_PARENT_IDENTITY_CHANGED')
    audit=read(case['audit_file']); result=read(folder/'RESULT.json'); s1=read(folder/'S1_RESULT.json')
    if (sha(case['audit_file']) != case['audit_sha256'] or audit['status']!='PASS_INDEPENDENT_RAW_AUDIT'
            or not audit['admitted'] or result['status']!='VALID_COMPLETE' or result['physics_steps']!=16000
            or s1['live_native_failures'] or s1['pre_execution_rejections'] or s1['actions_modified']
            or s1['maximum_live_shadow_difference']>1e-10):
        raise PermissionError('CR1_FULL_RAW_ADMITTED_SAFE_COMPLETE_SUCCESS_REQUIRED')
    for member in read(folder/'FILE_MANIFEST.json'):
        guard(); p=folder/member['path']
        if (Path(member['path']).is_absolute() or '..' in Path(member['path']).parts or p.is_symlink()
                or not p.resolve().is_relative_to(folder.resolve()) or p.stat().st_size!=member['bytes']
                or sha(p)!=member['sha256']): raise ValueError('CR1_FULL_RAW_MEMBER_CHANGED_OR_ESCAPED')
    samples=read(folder/'samples.json');validate_samples(samples)
    arrays=committed_packet(samples,card)
    path=root/'numeric'/(case['id']+'.npz');np.savez_compressed(path,**arrays)
    with np.load(folder/'physics.npz',allow_pickle=False) as archive:
        physics={k:archive[k].copy() for k in archive.files}
    events=event_exposure(physics,card)
    put(root/'evidence'/(case['id']+'-events.json'),events)
    return arrays,dict(numeric_file=str(path.relative_to(root)),numeric_sha256=sha(path),
                       actual_source_lease_sha256=sha(ab['lease']),raw_member_count=len(read(folder/'FILE_MANIFEST.json')),
                       events=events,physics_arrays_read=True,actual_physics_reconstruction=False)


def reader(case, arrays, processor, root, source, cosmos, statistics, guard):
    """Use the existing production DatasetLoader, real two-camera pixels and collator."""
    folder=root/'reader'/case['id'];folder.mkdir(parents=True,exist_ok=False)
    stats=folder/'statistics.json';write(stats,statistics);stats.chmod(0o400)
    numeric=root/'numeric'/(case['id']+'.npz')
    row=dict(id=case['id'],split='TRAIN',windows=601,file='numeric.npz',sha256=sha(numeric),raw_root='PRIVATE_BINDING')
    os.link(numeric,folder/'numeric.npz')
    write(folder/'DATASET_MANIFEST.json',dict(domain='FIXED_BASE_ARX5_TASK_SIM',statistics_sha256=sha(stats),
          purpose='CR1_ISOLATED_CPU_READER_NOT_BANK_NOT_TRAINING',episodes=[row]))
    dataset=NativeDataset(folder,source,cosmos,for_optimization=True,raw_bindings={case['id']:case['source_episode']})
    dataset.processor=processor
    processed=[];inventories=[]
    for anchor in (0,80,160,320,480,600):
        guard();p=dataset[anchor]['processed'];processed.append(p)
        inventories.append(batch_inventory(processor.collator([p]),1))
    repeat=batch_inventory(processor.collator([dataset[0]['processed']]),1)
    if repeat!=inventories[0]:raise ValueError('CR1_OFFICIAL_REAL_RGB_REPEAT_MUST_BE_DETERMINISTIC')
    return dict(id=case['id'],actual_anchors=[0,80,160,320,480,600],real_dual_RGB=True,
                deterministic_repeat=True,statistics_sha256=sha(stats),
                batch_inventory_sha256=[digest(v) for v in inventories],first_batch_inventory=inventories[0]),processed


def run(root):
    r=Path(root);lock=OwnerLock(r/'CR1_CPU_WORKER.lock')
    a,m,inputs,plan=validate(r);ledger=read(r/'RESOURCE_LEDGER.json')
    if ledger['status']!='ADOPTED_CONTROL_ONLY_RUNTIME_PENDING' or ledger['unknown_effects']:
        raise PermissionError('CR1_RUN_ONCE_FROM_SEALED_CONTROL_WINDOW')
    start=time.process_time();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    base_cpu=ledger['CPU_s'];budget=ledger['CPU_reserved_s'];failure=None;conclusion=None
    os.sched_setaffinity(0,set(a['limits']['cpus']));os.nice(15)
    resource.setrlimit(resource.RLIMIT_CPU,(int(budget),int(budget)))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    def guard(new_batch=False):
        if time.time()>=a['stop_compute_epoch'] or time.process_time()-start>=budget-30 or (r/'STOP.json').exists():
            raise PermissionError('CR1_FINITE_CPU_WALL_OR_EXPLICIT_STOP')
        if new_batch:
            memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
            if (time.time()>=a['stop_new_batch_epoch'] or os.getloadavg()[0]>=16
                    or memory['MemAvailable']<8*1024**3 or os.statvfs(r).f_bavail*os.statvfs(r).f_frsize<96*1024**3):
                raise PermissionError('CR1_NO_NEW_BATCH_AFTER_LIMIT_OR_PV1_RESOURCE_PRESSURE')
            logical=sum(p.stat().st_size for p in r.rglob('*') if p.is_file())
            ledger['output_peak_logical_bytes']=max(ledger.get('output_peak_logical_bytes',0),logical)
            if logical>8*1024**3:raise PermissionError('CR1_OUTPUT_BOUND')
    def checkpoint(phase,**extra):
        ledger.update(status='RUNNING',phase=phase,worker_CPU_s=time.process_time()-start,**extra)
        write(r/'RESOURCE_LEDGER.json',ledger)
    def alarm(*unused):raise TimeoutError('CR1_FINITE_COMPUTE_DEADLINE')
    try:
        signal.signal(signal.SIGALRM,alarm);signal.alarm(max(1,int(a['stop_compute_epoch']-time.time())))
        install_import_guard();cpu_guard()
        statistics=read(r/'plans/statistics.json');processor=build_processor(inputs['official_source'],inputs['cosmos'],statistics)
        params=serial(processor.state_action_processor.norm_params)
        original_params=copy.deepcopy(params)
        dependencies={name:importlib.metadata.version(name) for name in ('numpy','scipy','torch','transformers','pillow')}
        put(r/'evidence/R0_EFFECTIVE_ORIGINAL_PARAMS.json',dict(norm_params=params,dependencies=dependencies,
            original_statistics_sha256=STATS,upstream=OFFICIAL_SHA,numpy_isclose_defaults=isclose_defaults(),
            modality='EEF9_RELATIVE_XYZ_ROT6D_ROWS_PLUS_ABSOLUTE_GRIPPER',transport='FLOAT32',physical_metric='FLOAT64_TRACE_ARCCOS'))
        # R1 is a hard gate, before any fit-relative extrema or candidate construction.
        checkpoint('R1_ORIGINAL_REPRODUCTION')
        cached={};identities={};baseline={};diagnostics=[]
        for index,case in enumerate(plan['known']):
            guard(new_batch=True);arrays,identity=load_case(case,r,guard);cached[case['id']]=arrays;identities[case['id']]=identity
            anchors=TRACE_ANCHORS if index==0 else CONTROL_ANCHORS if index==1 else ()
            row=scan(arrays,processor,guard,diagnostic_anchors=anchors,destination=r/'arrays',prefix=case['id'])
            baseline[case['id']]=row;diagnostics.extend(row['observation']['diagnostics'])
        reproduced=reproduction_matches(baseline[plan['known'][0]['id']])
        control=baseline[plan['known'][1]['id']]['status']=='PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP'
        put(r/'CR1_REPRODUCTION_AND_CAUSAL_TRACE.json',dict(status='PASS_EXACT_REPRODUCTION' if reproduced and control else 'REPRODUCTION_MISMATCH',
            original_failure=baseline[plan['known'][0]['id']],original_PASS_control=control,
            diagnostic_windows=len(diagnostics),isolated_2x2=diagnostics,original_thresholds_changed=False,
            private_arrays_not_published=True,no_clamp_is_not_production_decoder=True,
            effective_params_file_sha256=sha(r/'evidence/R0_EFFECTIVE_ORIGINAL_PARAMS.json')))
        if not reproduced or not control:
            conclusion='NUMERIC_OR_INTERFACE_DISCREPANCY'
            put(r/'CR1_CANDIDATE_STATISTICS_AND_EFFECTIVE_PARAMS.json',dict(status='NOT_FITTED_REPRODUCTION_GATE_FAILED'))
            put(r/'CR1_PAIRED_RESULTS_AND_READER.json',dict(status='NOT_RUN_REPRODUCTION_GATE_FAILED',rows=[]))
            return dict(conclusion=conclusion,reproduction=False)
        checkpoint('R2_SEALED_TRAIN_FIT_ENVELOPE')
        envelope=Envelope(np.asarray(active(processor)['min']).shape)
        for case in plan['main']:
            guard(new_batch=True);arrays,identity=load_case(case,r,guard)
            cached[case['id']]=arrays;identities[case['id']]=identity
            if case['role']=='FIT':
                for anchor in range(601):
                    guard();envelope.add(relative(processor.state_action_processor,arrays['actual_state10'][anchor],
                                                 arrays['committed_absolute10'][anchor:anchor+40]))
            checkpoint('R2_SEALED_TRAIN_FIT_ENVELOPE',parents_loaded=len(cached),fit_windows=envelope.windows)
        candidate,range_report=candidate_statistics(statistics,active(processor),envelope)
        write(r/'plans/candidate_statistics.json',candidate);(r/'plans/candidate_statistics.json').chmod(0o400)
        if os.stat(r/'plans/candidate_statistics.json').st_ino==os.stat(r/'plans/statistics.json').st_ino:
            raise PermissionError('CR1_CANDIDATE_CANNOT_SHARE_ORIGINAL_INODE')
        other=build_processor(inputs['official_source'],inputs['cosmos'],candidate)
        effective=serial(other.state_action_processor.norm_params)
        expected=copy.deepcopy(original_params)
        for key in ('min','max'):expected['new_embodiment']['action']['eef_9d'][key]=candidate['new_embodiment']['relative_action']['eef_9d'][key]
        if effective!=expected or serial(processor.state_action_processor.norm_params)!=original_params:
            raise ValueError('CR1_NEW_INSTANCE_EFFECTIVE_CANDIDATE_ONLY_NO_CACHE_POLLUTION')
        put(r/'CR1_CANDIDATE_STATISTICS_AND_EFFECTIVE_PARAMS.json',dict(status='ONE_CANDIDATE_FIT_FROM_SEALED_FIT_ONLY',
            statistics_sha256=sha(r/'plans/candidate_statistics.json'),original_effective_params=original_params,
            candidate_effective_params=effective,range_report=range_report,only_minmax_changed=True,
            inactive_fields_original_canonical_sha256=digest({k:v for k,v in statistics['new_embodiment']['relative_action']['eef_9d'].items() if k not in ('min','max')})))
        rows=[];reader_proofs=[];mixed_first={};mixed_id=None
        for case in plan['main']+plan['known']:
            guard(new_batch=True);arrays=cached[case['id']]
            left=baseline.get(case['id']) or scan(arrays,processor,guard)
            right=scan(arrays,other,guard)
            lp=left['status']=='PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP';rp=right['status']=='PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP'
            item={k:case[k] for k in ('id','task','direction','family','scene_group','lineage','role','frozen_139_member')}
            item.update(transition=('PASS' if lp else 'FAIL')+'->'+('PASS' if rp else 'FAIL'),R0=left,R1=right,
                        identity=identities[case['id']])
            for label,p,stats in (('R0',processor,statistics),('R1',other,candidate)):
                # Reader views never become BANK/READY and never modify input arrays.
                target=r/label;target.mkdir(exist_ok=True)
                (target/'numeric').mkdir(exist_ok=True)
                os.link(r/'numeric'/(case['id']+'.npz'),target/'numeric'/(case['id']+'.npz'))
                proof,processed=reader(case,arrays,p,target,inputs['official_source'],inputs['cosmos'],stats,guard)
                reader_proofs.append(dict(variant=label,**proof))
                if label not in mixed_first:
                    mixed_first[label]=processed[0];mixed_id=case['id']
                elif len(rows)==1:
                    if case['id']==mixed_id:raise ValueError('CR1_CROSS_PARENT_READER_MUST_BE_DISTINCT')
                    reader_proofs.append(dict(variant=label,cross_parent_ids=[mixed_id,case['id']],
                        real_mixed_batch=batch_inventory(p.collator([mixed_first[label],processed[-1]]),2)))
            rows.append(item);put(r/'evidence'/(case['id']+'-paired.json'),item)
            checkpoint('R3_PAIRED_ALL_WINDOWS_AND_ACTUAL_READERS',paired_parents=len(rows),fit_windows=envelope.windows)
        def count(role,transition):return sum(x['role']==role and x['transition']==transition for x in rows)
        fit_good=all(x['transition'].endswith('->PASS') for x in rows if x['role']=='FIT')
        diagnostic_recovered=count('DIAGNOSTIC','FAIL->PASS')>0
        regressions=[x['id'] for x in rows if x['transition']=='PASS->FAIL' and (x['role']=='DIAGNOSTIC' or x['frozen_139_member'])]
        supported=fit_good and diagnostic_recovered and not regressions
        conclusion='RANGE_SUPPORTED' if supported else 'RANGE_NOT_SUPPORTED'
        qualification='INDEPENDENT_TRAIN_DIAGNOSTIC_SUPPORT' if supported else (
            'KNOWN_CASE_ONLY' if count('KNOWN_CHALLENGE','FAIL->PASS') and not diagnostic_recovered else 'NOT_SUPPORTED')
        put(r/'CR1_PAIRED_RESULTS_AND_READER.json',dict(status='COMPLETE_PAIRED_CPU_ONLY',rows=rows,reader=reader_proofs,
            unique_parents=len(rows),all_original_139_checked=sum(x['frozen_139_member'] for x in rows)==139,
            fit_all_window_pass=fit_good,diagnostic_independent_group_recovered=diagnostic_recovered,
            diagnostic_or_retention_regressions=regressions,conclusion=conclusion,qualification=qualification,
            known_challenges_never_fit=True,diagnostic_feedback_refits=0,old_results_relabelled=False))
        return dict(conclusion=conclusion,qualification=qualification,parents=len(rows))
    except BaseException as exc:
        failure=type(exc).__name__+': '+str(exc)
        put(r/'evidence/CR1_FAILURE_PRESERVED.json',dict(error=failure,phase=ledger.get('phase'),conclusion=conclusion))
        raise
    finally:
        signal.alarm(0);children=resource.getrusage(resource.RUSAGE_CHILDREN)
        used=time.process_time()-start+children.ru_utime+children.ru_stime-before.ru_utime-before.ru_stime
        import torch
        forbidden=[n for n in sys.modules if n.split('.')[0] in ('mujoco','mujoco_py','isaaclab','isaacsim')
                   or n.startswith('gr00t.model.') and '.modeling_' in n]
        ledger.update(status='FAILED_PRESERVED' if failure else 'TERMINAL_CR1_CPU',CPU_s=base_cpu+used,CPU_reserved_s=0,
            worker_CPU_s=used,failure=failure,conclusion=conclusion,forbidden_modules=forbidden,
            CUDA_initialized=torch.cuda.is_initialized(),finished_epoch=time.time())
        if forbidden or ledger['CUDA_initialized']:ledger['status']='INTEGRITY_STOP';ledger['unknown_effects'].append('FORBIDDEN_RUNTIME')
        write(r/'RESOURCE_LEDGER.json',ledger);lock.close()


if __name__=='__main__':
    import json
    print(json.dumps(run(sys.argv[1])),flush=True)
