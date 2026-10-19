"""Read a presealed capture order; freeze one admitted bank, never hot-inject it."""
import os
from pathlib import Path
import resource
import signal
import sys
import time

from scripts.pi0291_arx5_data.native_dataset import cpu_guard
from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0292.train_adapter import build_processor
from scripts.pi0294.e1.scope import STATS
from scripts.pi0294.fd1 import put,sha
from scripts.pi0294.scene_cards import digest
from .dataset import POPULATION,TRAIN_GROUPS,admit_new_support
from .export import export_case
from .scope import EXPORT_ROLE,validate

MASTER='c4a71d4ebb518a58c3417e7d3c5aba4641eda39a2e9e9a0ddf77404866a99d28'


def bank_manifest(rows,route,prefix_count):
    if (route not in ('CLEARANCE_PAIR','COVERAGE_FIRST') or not 1<=prefix_count<=288 or len(rows)>prefix_count
            or any(r.get('status')!='PASS_SAVED_TEACHER_EXPORT_AND_OFFICIAL_CHAIN'
                or r.get('admitted_for_training') is not True
                or r['codec']['status']!='PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP' for r in rows)):
        raise PermissionError('PV1_BANK_ONLY_ACTUAL_FULL_OFFICIAL_CHAIN_ADMITTED_ROWS')
    all_admitted=len(rows)
    if route=='COVERAGE_FIRST':
        rows=[r for r in rows if r['nominal_clearance_m']==.004]
    else:
        support={margin:{r['scene_group'] for r in rows if r['task']=='T2' and r['nominal_clearance_m']==margin}
            for margin in (.004,.012)}
        common=support[.004]&support[.012]
        rows=[r for r in rows if r['task']=='T1' or r['scene_group'] in common]
    manifest=dict(schema='PI0294_PV1_ADMITTED_TRAIN_BANK_V1',domain='FIXED_BASE_ARX5_TASK_SIM',
        statistics_sha256=STATS,statistics_refit=False,population_plan_sha256=POPULATION,
        master_capture_plan_sha256=MASTER,elementwise_codec_admission='PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP',
        evaluation_rows_used_for_training=False,episodes=rows,route=route,
        ordered_master_first_attempt_prefix=prefix_count,later_captures_hot_injected=False,
        all_admitted_rows_before_route_filter=all_admitted,
        observations_and_labels='REAL_TWO_RGB_ACTUAL_STATE_AND_ACTUALLY_COMMITTED_ABSOLUTE_EEF10')
    support=admit_new_support(manifest,route)
    manifest['independent_support']=support
    return manifest


def common_support_upper_bound(records):
    failed={r['scene_group'] for r in records if r['task']=='T2' and r.get('admitted_for_training') is not True}
    possible=set(TRAIN_GROUPS)-failed
    directions={d:sum(TRAIN_GROUPS[g]==d for g in possible) for d in ('+Y','-Y')}
    return dict(total=len(possible),directions=directions,
        impossible_groups=sorted(failed),can_still_meet_margin_gate=len(possible)>=64 and min(directions.values())>=16)


def ready_case(binding,root):
    """Only a completed raw-audit publication can turn a future input into data."""
    root=Path(root)
    source=Path(binding['source_root'])
    if not source.resolve().is_relative_to(root.parent.resolve()) or source.is_symlink():
        raise PermissionError('PV1_STREAM_ONLY_SIBLING_REGISTERED_SOURCE_ROOTS')
    m=read(source/'RUN_MANIFEST.json');b=read(source/'OUTPUT_BINDING.private.json')
    if (m['code']['vla_commit']!=binding['source_sha'] or b['run_root']!=str(source)
            or b['uid']!=os.getuid() or b['host']!=os.uname().nodename
            or m['scope']!='PI0294_PV1_CORE_TRAIN_CAPTURE_V1'):
        raise PermissionError('PV1_STREAM_NATIVE_CAPTURE_SOURCE_IDENTITY')
    registry=read(b['registry'])['entries'];match=[e for e in registry if e.get('role')==b['role']]
    if len(match)!=1 or Path(match[0]['path']).resolve()!=source.resolve():
        raise PermissionError('PV1_STREAM_UNIQUE_REGISTERED_CAPTURE_SOURCE')
    audit=source/'evidence/audits'/binding['id']/'CAPTURE_AUDIT.json'
    if not audit.exists():
        lease=read(b['lease']);ledger=read(source/'RESOURCE_LEDGER.json')
        folder=source/'episodes'/binding['id']
        if ledger['status']=='FAILED_PRESERVED' and (folder/'AUDIT.json').exists():
            from .data_rejection import saved_case
            if ledger['unknown_effects']:raise PermissionError('PV1_UNKNOWN_CAPTURE_EFFECT_NOT_A_TYPED_DATA_REJECTION')
            return saved_case(binding,source,unpublished=True)
        if lease['status']=='CLOSED' or ledger['status'] in ('FAILED_PRESERVED','TERMINAL_CAPTURE_BATCH'):
            raise PermissionError('PV1_TERMINAL_CAPTURE_WITHOUT_EXPECTED_RAW_AUDIT_NO_BLIND_REPLAY')
        return None
    receipt=read(audit);folder=source/'episodes'/binding['id'];card=read(folder/'CARD.json')
    if (receipt['id']!=binding['id'] or card!=binding['card'] or digest(card)!=binding['card_sha256']
            or receipt['raw_manifest_sha256']!=sha(folder/'FILE_MANIFEST.json')
            or receipt['native_faults'] or receipt['split']!='TRAIN'):
        raise PermissionError('PV1_STREAM_ONLY_EXACT_PRESEALED_SAFETY_VALID_RAW_ARRIVAL')
    if receipt['task_success']:
        if receipt['audit_status']!='PASS_INDEPENDENT_RAW_AUDIT':
            raise PermissionError('PV1_STREAM_FULL_SUCCESS_NEEDS_INDEPENDENT_RAW_AUDIT')
        return dict(id=binding['id'],source_episode=str(folder),card_sha256=binding['card_sha256'],
            raw_manifest_sha256=receipt['raw_manifest_sha256'],audit_file=str(folder/'AUDIT.json'),
            audit_sha256=receipt['audit_sha256'],capture_publication_sha256=sha(audit))
    if receipt.get('data_readside_rejection'):
        from .data_rejection import saved_case
        value=saved_case(binding,source)
        if value['audit_sha256']!=receipt['audit_sha256']:
            raise PermissionError('PV1_PUBLISHED_DATA_REJECTION_AUDIT_CHANGED')
        return value
    if receipt['audit_status'] not in ('PASS_INDEPENDENT_RAW_AUDIT','PASS_INDEPENDENT_PARTIAL_NEGATIVE_WITNESS'):
        raise PermissionError('PV1_STREAM_NEGATIVE_NEEDS_ORIGINAL_AUDIT_WITNESS')
    return dict(id=binding['id'],status='VALID_NEGATIVE_NOT_TRAINING',task=receipt['task'],
        scene_group=card['group_id'],nominal_clearance_m=card['nominal_clearance_m'],
        capture_publication_sha256=sha(audit),raw_manifest_sha256=receipt['raw_manifest_sha256'],
        audit_sha256=receipt['audit_sha256'],S1_rejections=receipt['S1_rejections'],native_faults=0,
        valid_failure_not_engineering_replacement=True)


def build_bank(root,manifest,sources,guard):
    root=Path(root);out=root/'datasets/FROZEN_TRAIN_BANK'
    if out.exists():raise FileExistsError('PV1_FROZEN_BANK_CANNOT_BE_REPLACED_OR_HOT_UPDATED')
    admit_new_support(manifest,manifest['route'])
    out.mkdir();os.link(root/'plans/statistics.json',out/'statistics.json')
    for row in manifest['episodes']:
        guard();source=Path(sources[row['id']]);numeric=source/row['file']
        if sha(numeric)!=row['sha256'] or sha(source/'statistics.json')!=STATS:
            raise ValueError('PV1_STREAM_QUALIFIED_NUMERIC_OR_STATISTICS_CHANGED')
        target=out/row['file'];target.parent.mkdir(exist_ok=True);os.link(numeric,target)
        raw=source/'raw'/row['id'];target_raw=out/'raw'/row['id']
        import numpy as np
        with np.load(numeric,allow_pickle=False) as z:
            if z['actual_state10'].shape!=(641,10) or z['committed_absolute10'].shape!=(640,10) or int(z['valid_window'].sum())!=601:
                raise ValueError('PV1_BANK_MUST_RETAIN_ACTUAL_COMPLETE_H40_NUMERIC_SHAPES')
            for names,hashes in zip(z['rgb_paths'],z['rgb_sha256']):
                for name,expected in zip(names,hashes):
                    guard();name=Path(str(name));p=raw/name
                    if name.is_absolute() or '..' in name.parts or p.is_symlink() or not p.resolve().is_relative_to(raw.resolve()) or sha(p)!=str(expected):
                        raise ValueError('PV1_STREAM_QUALIFIED_RGB_CHANGED_OR_ESCAPED')
                    target=target_raw/name;target.parent.mkdir(parents=True,exist_ok=True);os.link(p,target)
    put(out/'DATASET_MANIFEST.json',manifest)
    bindings={r['id']:str(out/'raw'/r['id']) for r in manifest['episodes']}
    put(root/'plans/FROZEN_TRAIN_RAW_BINDINGS.private.json',bindings)
    files=[]
    for p in sorted(out.rglob('*')):
        if p.is_file():
            guard();files.append(dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)))
    put(out/'PAYLOAD_MANIFEST.json',files)
    ready=dict(schema='PI0294_PV1_FROZEN_TRAIN_BANK_READY_V1',status='SEALED_FULL_SUCCESS_OFFICIAL_CODEC_ADMITTED',
        dataset_manifest_sha256=sha(out/'DATASET_MANIFEST.json'),payload_manifest_sha256=sha(out/'PAYLOAD_MANIFEST.json'),
        statistics_sha256=STATS,files=len(files),logical_payload_bytes=sum(f['bytes'] for f in files),
        native_input_bank_only_not_a_model_or_GPU_allocation=True,later_rows_hot_injected=False)
    put(out/'READY.json',ready)
    return ready


def run(root):
    root=Path(root);lock=OwnerLock(root/'PV1_STREAM_OWNER.lock')
    m,inputs=validate(root,role=EXPORT_ROLE);plan=read(root/'plans/SAVED_EXPORT_PLAN.private.json')
    ledger=read(root/'RESOURCE_LEDGER.json')
    if (plan['schema']!='PI0294_PV1_ORDERED_CAPTURE_STREAM_PLAN_V1' or len(plan['cards'])!=288
            or plan['master_sha256']!=MASTER or plan['population_sha256']!=POPULATION
            or ledger['status']!='RESERVED' or ledger['unknown_effects'] or sha(root/'plans/statistics.json')!=STATS
            or any(m['limits'][k]!=0 for k in ('engine_starts','physics_ticks','GR00T_queries','training_updates'))):
        raise PermissionError('PV1_STREAM_SEALED_288_FIRST_ATTEMPTS_ZERO_NATIVE_OR_NEURAL_EFFECTS')
    if len({b['id'] for b in plan['cards']})!=288:raise PermissionError('PV1_STREAM_UNIQUE_PRESEALED_CAPTURE_ORDER')
    if (sha(plan['master_path'])!=MASTER or sha(plan['population_path'])!=POPULATION
            or [b['card'] for b in plan['cards']]!=read(plan['master_path'])['cards']):
        raise PermissionError('PV1_STREAM_EXACT_ORIGINAL_MASTER_CARD_ORDER_AND_POPULATION')
    started=time.process_time();before=resource.getrusage(resource.RUSAGE_CHILDREN);failure=None;records=[];positive=[];sources={}
    os.sched_setaffinity(0,set(m['limits']['cpus']))
    resource.setrlimit(resource.RLIMIT_CPU,(m['limits']['CPU_s'],m['limits']['CPU_s']));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    def guard():
        if (time.time()>=m['limits']['operation_deadline_epoch'] or time.process_time()-started>=m['limits']['CPU_s']-30
                or (root/'STOP.json').exists() or (root.parent.parent/'STOP.json').exists()):
            raise PermissionError('PV1_STREAM_ORIGINAL_FINITE_WALL_CPU_AND_STOP')
    def expire(*a):raise TimeoutError('PV1_STREAM_FINITE_DEADLINE')
    ready=None;active_route=plan['route'];fallback=None
    try:
        signal.signal(signal.SIGALRM,expire);signal.alarm(max(1,int(m['limits']['operation_deadline_epoch']-time.time())))
        cpu_guard();processor=build_processor(inputs['official_source'],inputs['cosmos'],read(root/'plans/statistics.json'))
        ledger['status']='RUNNING_WAITING_ONLY_PRESEALED_CAPTURE_ORDER';write(root/'RESOURCE_LEDGER.json',ledger)
        for index,binding in enumerate(plan['cards']):
            guard()
            if 'retired_unattempted_alternate_source' in binding:
                from .data_rejection import retired_alternate
                row=retired_alternate(binding,root,active_route)
                records.append(row);put(root/'evidence/rows'/(binding['id']+'.json'),row)
                ledger.update(completed_prefix=index+1,admitted=len(positive),waiting_cell=None,worker_CPU_s=time.process_time()-started)
                write(root/'RESOURCE_LEDGER.json',ledger)
                continue
            if 'reuse_export_root' in binding:
                source=Path(binding['reuse_export_root']);receipt=source/'EXPORT_RECEIPT.json'
                if (not source.resolve().is_relative_to(root.parent.resolve())
                        or sha(receipt)!=binding['reuse_export_receipt_sha256']):
                    raise PermissionError('PV1_STREAM_ONLY_BOUND_QUALIFIED_PRIOR_EXPORT_BYTES')
                row=read(receipt)
                if row['id']!=binding['id'] or row['source_card_sha256']!=binding['card_sha256']:
                    raise PermissionError('PV1_STREAM_REUSE_SAME_SEALED_TRAIN_PARENT')
            else:
                case=ready_case(binding,root)
                while case is None:
                    ledger.update(waiting_cell=binding['id'],worker_CPU_s=time.process_time()-started)
                    write(root/'RESOURCE_LEDGER.json',ledger);time.sleep(30);guard();case=ready_case(binding,root)
                put(root/'evidence/arrivals'/(binding['id']+'.private.json'),case)
                if case.get('status')=='VALID_NEGATIVE_NOT_TRAINING':
                    records.append(case);put(root/'evidence/rows'/(binding['id']+'.json'),case)
                    ledger.update(completed_prefix=index+1,admitted=len(positive),waiting_cell=None,
                        worker_CPU_s=time.process_time()-started)
                    write(root/'RESOURCE_LEDGER.json',ledger)
                    row=None
                else:
                    source=root/'datasets'/binding['id'];source.mkdir(parents=True,exist_ok=False)
                    os.link(root/'plans/statistics.json',source/'statistics.json')
                    row=export_case(case,source,processor,inputs['official_source'],inputs['cosmos'],guard)
                    put(source/'EXPORT_RECEIPT.json',row)
            if row is not None:
                records.append(row);put(root/'evidence/rows'/(binding['id']+'.json'),row)
                if row.get('admitted_for_training') is True:
                    positive.append(row);sources[row['id']]=str(source)
                elif (row['status']!='BLOCKED_FROZEN_CODEC_PHYSICAL_ERROR'
                        or row['codec']['H40_windows_actually_checked']!=601
                        or row['codec']['statistics_refit'] is not False
                        or row['codec']['encode_clip_components']!=0):
                    raise PermissionError('PV1_STREAM_UNKNOWN_DATA_FAILURE_REQUIRES_DIAGNOSIS_NO_RETRY')
            upper=common_support_upper_bound(records)
            if active_route=='CLEARANCE_PAIR' and not upper['can_still_meet_margin_gate']:
                active_route='COVERAGE_FIRST'
                fallback=dict(reason='QUALIFIED_COMMON_SUPPORT_MATHEMATICALLY_CANNOT_REACH_PREFIXED_GATE',
                    observed_prefix=index+1,common_support_upper_bound=upper,
                    original_FD2_physical_success_not_erased=True,correct_clipped_labels_not_used_for_training=True,
                    normalizer_and_original_decoder_unchanged=True)
                put(root/'evidence/COVERAGE_FALLBACK_DECISION.json',fallback)
            ledger.update(completed_prefix=index+1,admitted=len(positive),waiting_cell=None,worker_CPU_s=time.process_time()-started)
            write(root/'RESOURCE_LEDGER.json',ledger)
            try:manifest=bank_manifest(positive,active_route,index+1)
            except PermissionError as exc:
                if str(exc) not in ('PV1_MINIMUM_INDEPENDENT_SUPPORT_NOT_MET','PV1_MARGIN_PAIR_NEEDS_64_COMMON_SUCCESS_GROUPS',
                        'PV1_COMMON_T2_SUPPORT_MUST_RETAIN_BOTH_DIRECTION_MINIMA'):raise
                continue
            manifest['data_route_fallback']=fallback
            ready=build_bank(root,manifest,sources,guard);break
        if ready is None:
            # Fallback is preauthorized only after the complete original cohort, never based on a lucky early subset.
            if active_route=='COVERAGE_FIRST' and len(records)==288:
                try:manifest=bank_manifest(positive,'COVERAGE_FIRST',288)
                except PermissionError as exc:
                    if str(exc)!='PV1_MINIMUM_INDEPENDENT_SUPPORT_NOT_MET':raise
                else:
                    manifest['data_route_fallback']=fallback
                    ready=build_bank(root,manifest,sources,guard)
        import torch
        forbidden=[n for n in sys.modules if n.startswith('gr00t') and ('modeling_' in n or '.backbone.' in n or '.action_head.' in n)]
        if forbidden or torch.cuda.is_initialized():raise PermissionError('PV1_CPU_STREAM_NO_NEURAL_OR_CUDA')
        result=dict(status='FROZEN_TRAIN_BANK_READY' if ready else 'BLOCKED_FINITE_COHORT_SUPPORT_NOT_MET',
            ready=ready,ordered_prefix_observed=len(records),positive_rows=len(positive),records=records,
            active_data_route=active_route,data_route_fallback=fallback,
            new_episodes=0,new_integrations=0,new_engine_starts=0,GR00T_queries=0,training_updates=0,
            complete_capture_cohort_may_continue_without_hot_injection=True)
        put(root/'evidence/STREAM_RESULT.private.json',result)
        return {k:v for k,v in result.items() if k!='records'}
    except BaseException as exc:
        failure=type(exc).__name__+': '+str(exc)
        put(root/'evidence/STREAM_FAILURE.private.json',dict(error=failure,completed_prefix=len(records)))
        raise
    finally:
        signal.alarm(0);after=resource.getrusage(resource.RUSAGE_CHILDREN)
        ledger.update(status='FAILED_PRESERVED' if failure else ('FROZEN_TRAIN_BANK_READY' if ready else 'BLOCKED_FINITE_COHORT_SUPPORT_NOT_MET'),
            CPU_s=time.process_time()-started+after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
            CPU_reserved_s=0,failure=failure,completed_prefix=len(records))
        write(root/'RESOURCE_LEDGER.json',ledger);lock.close()
