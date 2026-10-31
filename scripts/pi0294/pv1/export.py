"""Saved committed Teacher data -> unchanged official EEF10 CPU qualification."""
import os
import hashlib
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np

from scripts.pi0291_arx5_data.native_dataset import validate_samples,pose10,matrix_from10,cpu_guard
from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0292.train_adapter import build_processor
from scripts.pi0294.e1.artifacts import atomic_npz
from scripts.pi0294.e1.scope import STATS
from scripts.pi0294.fd1 import put,sha
from scripts.pi0294.native_dataset import NativeDataset
from scripts.pi0294.scene_cards import digest
from .dataset import TRAIN_GROUPS,POPULATION
from .scope import EXPORT_ROLE,validate

GATES=dict(XYZ_m=1e-5,SO3_rad=1e-4,gripper=1e-5,relative_SE3_max_abs=1e-8)


def packet(samples,card):
    validate_samples(samples)
    if (card['split']!='TRAIN' or card['group_id'] not in TRAIN_GROUPS
            or card['task'] not in ('T1','T2')
            or card['nominal_clearance_m'] not in (.004,.012)
            or (card['task']=='T1' and card['nominal_clearance_m']!=.004)
            or any(s['policy_obs']['instruction']!=card['instruction'] for s in samples)):
        raise PermissionError('PV1_SAVED_EXPORT_ONLY_SEALED_TRAIN_COMMITTED_LANGUAGE')
    return committed_packet(samples,card)


def committed_packet(samples,card):
    """Shared label construction; each caller must separately admit its scope."""
    validate_samples(samples)
    if any(s['policy_obs']['instruction']!=card['instruction'] for s in samples):
        raise PermissionError('SAVED_EXPORT_ONLY_ACTUALLY_OBSERVED_TASK_LANGUAGE')
    states=np.asarray([pose10(s['policy_obs']['actual_tcp'],s['policy_obs']['actual_gripper']) for s in samples])
    commands=np.asarray([pose10(s['supervision']['committed'],s['supervision']['committed_gripper']) for s in samples[:640]])
    return dict(actual_state10=states,committed_absolute10=commands,
        valid_window=np.arange(640)<601,
        rgb_paths=np.asarray([[s['policy_obs'][c]['path'] for c in ('body_rgb','wrist_rgb')] for s in samples]),
        rgb_sha256=np.asarray([[s['policy_obs'][c]['sha256'] for c in ('body_rgb','wrist_rgb')] for s in samples]),
        instruction=np.asarray(card['instruction']),label_representation=np.asarray('ABSOLUTE_COMMITTED_EEF10'))


def physical_errors(expected,actual):
    expected,actual=np.asarray(expected),np.asarray(actual)
    if expected.shape!=actual.shape or expected.shape!=(40,10) or not np.isfinite(actual).all():
        raise ValueError('PV1_OFFICIAL_DECODE_MUST_RETURN_40_FINITE_EEF10')
    left,right=matrix_from10(expected),matrix_from10(actual)
    cosine=np.clip((np.trace(left[:,:3,:3]@np.swapaxes(right[:,:3,:3],-1,-2),axis1=-2,axis2=-1)-1)/2,-1,1)
    return dict(XYZ_m=float(np.max(np.abs(expected[:,:3]-actual[:,:3]))),
        SO3_rad=float(np.max(np.arccos(cosine))),gripper=float(np.max(np.abs(expected[:,9]-actual[:,9]))))


def codec_packet(arrays,processor,guard=lambda:None,*,diagnostic_sink=None,anchors=None):
    from gr00t.data.types import ActionType,ActionFormat
    from gr00t.data.embodiment_tags import EmbodimentTag
    sap=processor.state_action_processor
    if sap.clip_outliers or not sap.use_relative_action or sap.use_percentiles:
        raise PermissionError('PV1_ORIGINAL_NO_ENCODE_CLIP_RELATIVE_MINMAX_REQUIRED')
    maxima=dict(XYZ_m=0.,SO3_rad=0.,gripper=0.,relative_SE3_max_abs=0.)
    preclip=inverseclip=state_preclip=0
    per_channel=np.zeros(10,dtype=np.int64)
    anchors=tuple(range(601))if anchors is None else tuple(anchors)
    if (not anchors or len(set(anchors))!=len(anchors)
            or any(type(a)is not int or a<0 or a+40>len(arrays['committed_absolute10'])
                   or a>=len(arrays['actual_state10'])or not arrays['valid_window'][a]for a in anchors)):
        raise ValueError('ONLY_DISTINCT_COMPLETE_VALID_COMMITTED_H40_ANCHORS')
    for anchor in anchors:
        guard()
        state=arrays['actual_state10'][anchor]
        target=arrays['committed_absolute10'][anchor:anchor+40]
        relative=sap._convert_to_relative_action(target[:,:9],state[:9],ActionType.EEF,ActionFormat.XYZ_ROT6D)
        expected=pose10(np.linalg.inv(matrix_from10(state))@matrix_from10(target),target[:,9])
        maxima['relative_SE3_max_abs']=max(maxima['relative_SE3_max_abs'],float(np.max(np.abs(relative-expected[:,:9]))))
        state_dict={'eef_9d':state[None,:9],'gripper_position':state[None,9:]}
        values=sap.apply_action({'eef_9d':target[:,:9],'gripper_position':target[:,9:]},'new_embodiment',state=state_dict)
        normalized=np.concatenate([values[k] for k in ('eef_9d','gripper_position')],axis=-1)
        normalized_state=sap.apply_state(state_dict,'new_embodiment')
        if not np.isfinite(normalized).all() or any(not np.isfinite(v).all() for v in normalized_state.values()):
            raise ValueError('PV1_NONFINITE_OFFICIAL_ENCODE')
        state_preclip+=sum(int(np.count_nonzero(np.abs(v)>1)) for v in normalized_state.values())
        preclip+=int(np.count_nonzero(np.abs(normalized)>1))
        packed=normalized.astype(np.float32)
        per_channel+=np.count_nonzero(np.abs(packed)>1,axis=0)
        inverseclip+=int(np.count_nonzero(np.abs(packed)>1))
        decoded=processor.decode_action(np.pad(packed,((0,0),(0,122)))[None],EmbodimentTag.NEW_EMBODIMENT,
            {k:v[None] for k,v in state_dict.items()})
        actual=np.concatenate([decoded[k] for k in ('eef_9d','gripper_position')],axis=-1)[0]
        for name,error in physical_errors(target,actual).items():maxima[name]=max(maxima[name],error)
        if diagnostic_sink is not None:
            diagnostic_sink(anchor=anchor,state=state.copy(),target=target.copy(),expected_relative=expected.copy(),
                native_relative=relative.copy(),normalized=normalized.copy(),packed=packed.copy(),actual=actual.copy())
    passed=all(maxima[k]<=v for k,v in GATES.items())
    return dict(status='PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP' if passed else 'BLOCKED_FROZEN_CODEC_PHYSICAL_ERROR',
        H40_windows_actually_checked=len(anchors),committed_label_elements_checked=len(anchors)*40*10,
        maximum_error=maxima,frozen_numeric_gates=GATES,action_preclip_components=preclip,
        native_inverse_clip_components_float32=inverseclip,inverse_clip_by_EEF10_channel=per_channel.tolist(),
        state_preclip_components=state_preclip,encode_clip_components=0,statistics_refit=False,
        native_decode_unchanged=True,nonzero_out_of_range_counts_explicit_not_silently_discarded=True)


def batch_inventory(batch,count):
    inputs=batch['inputs']
    action,mask,state=(inputs[k].numpy() for k in ('action','action_mask','state'))
    if (action.shape!=(count,40,132) or mask.shape!=action.shape or state.shape!=(count,1,132)
            or not np.all(mask[...,:10]==1) or not np.all(mask[...,10:]==0)
            or not np.all(state[...,10:]==0)):
        raise ValueError('PV1_OFFICIAL_EEF10_MASK_H40_AND_ZERO_PADDING')
    def inventory(value):
        if hasattr(value,'detach'):value=value.detach().cpu().numpy()
        if isinstance(value,np.ndarray):
            if not np.isfinite(value).all():raise ValueError('PV1_NONFINITE_OFFICIAL_BATCH')
            return dict(shape=list(value.shape),dtype=str(value.dtype),sha256=hashlib.sha256(value.tobytes()).hexdigest())
        if isinstance(value,dict) or hasattr(value,'items'):
            return {str(k):inventory(v) for k,v in sorted(value.items())}
        if isinstance(value,(tuple,list)):return [inventory(v) for v in value]
        return value
    return inventory(batch)


def export_case(case,out,processor,source,cosmos,guard):
    return export_committed_case(case,out,processor,source,cosmos,guard,
        packet_builder=packet,codec_auditor=codec_packet,
        direction_for_card=lambda c:TRAIN_GROUPS[c['group_id']])


def export_committed_case(case,out,processor,source,cosmos,guard,*,packet_builder,codec_auditor,
        direction_for_card,origin='PV1_ACTUAL_COMMITTED_TEACHER_TRAIN',reuse_registered_source_rgb=False):
    """Shared full raw/official-chain reader; scope-specific callers bind both gates."""
    folder=Path(case['source_episode']);card=read(folder/'CARD.json')
    audit=read(case['audit_file']);result=read(folder/'RESULT.json');s1=read(folder/'S1_RESULT.json')
    if (card['id']!=case['id'] or digest(card)!=case['card_sha256']
            or sha(folder/'FILE_MANIFEST.json')!=case['raw_manifest_sha256']
            or sha(case['audit_file'])!=case['audit_sha256']
            or audit['status']!='PASS_INDEPENDENT_RAW_AUDIT' or not audit['admitted']
            or result['status']!='VALID_COMPLETE' or result['physics_steps']!=16000
            or s1['live_native_failures'] or s1['pre_execution_rejections'] or s1['actions_modified']
            or s1['maximum_live_shadow_difference']>1e-10):
        raise PermissionError('PV1_EXPORT_REQUIRES_ACTUAL_FULL_SUCCESS_AND_INDEPENDENT_RAW_IDENTITY')
    members=read(folder/'FILE_MANIFEST.json')
    for row in members:
        guard();p=folder/row['path']
        if (Path(row['path']).is_absolute() or '..' in Path(row['path']).parts or p.is_symlink()
                or not p.resolve().is_relative_to(folder.resolve())
                or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']):
            raise ValueError('PV1_SAVED_RAW_MEMBER_CHANGED_OR_ESCAPED')
    samples=read(folder/'samples.json');validate_samples(samples,verify_images=folder)
    arrays=packet_builder(samples,card)
    codec=codec_auditor(arrays,processor,guard)
    record=dict(id=card['id'],scene_group=card['group_id'],task=card['task'],split='TRAIN',family=card['family'],
        direction=direction_for_card(card),nominal_clearance_m=card['nominal_clearance_m'],
        full_task_success=True,independent_raw_admitted=True,windows=601,source_raw_manifest_sha256=case['raw_manifest_sha256'],
        source_full_audit_sha256=case['audit_sha256'],source_card_sha256=case['card_sha256'],
        capture_source_sha=result['source_sha'],codec=codec,
        raw_root='EXPLICIT_PRIVATE_BINDING_REQUIRED',origin=origin)
    if codec['status']!='PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP':
        record.update(status='BLOCKED_FROZEN_CODEC_PHYSICAL_ERROR',admitted_for_training=False)
        return record
    path=out/'numeric'/(card['id']+'.npz');path.parent.mkdir(exist_ok=True)
    atomic_npz(path,**arrays)
    raw=materialize_rgb(folder,out,card,arrays,guard,reuse_registered_source_rgb=reuse_registered_source_rgb)
    if reuse_registered_source_rgb:
        record['RGB_materialization']='EXPLICIT_GEN1_REGISTERED_IMMUTABLE_SOURCE_INPUT'
    record.update(file=str(path.relative_to(out)),sha256=sha(path),bytes=path.stat().st_size)
    # The same production reader, processor and official collator see actual images/labels.
    write(out/'DATASET_MANIFEST.json',dict(domain='FIXED_BASE_ARX5_TASK_SIM',statistics_sha256=STATS,episodes=[record]))
    dataset=NativeDataset(out,source,cosmos,for_optimization=True,raw_bindings={card['id']:str(raw)})
    dataset.processor=processor
    processed=[dataset[i]['processed'] for i in (0,80,160,320,480,600)]
    inventories=[batch_inventory(processor.collator([p]),1) for p in processed]
    mixed=batch_inventory(processor.collator([processed[0],processed[-1]]),2)
    repeat=batch_inventory(processor.collator([dataset[0]['processed']]),1)
    if inventories[0]!=repeat:raise ValueError('PV1_FIXED_IMAGE_PROCESSING_MUST_BE_DETERMINISTIC')
    record.update(status='PASS_SAVED_TEACHER_EXPORT_AND_OFFICIAL_CHAIN',admitted_for_training=True,
        official_chain=dict(actual_anchors=[0,80,160,320,480,600],single_batches=inventories,
            mixed_first_terminal_anchor_batch=mixed,deterministic_repeat=True,
            full_641_two_camera_raw_RGB256_verified=True,image_crop_fraction=1.,image_target_size=[256,256],
            model_forward=False,privileged_policy_metadata=False))
    return record


def materialize_rgb(folder,out,card,arrays,guard,*,reuse_registered_source_rgb=False):
    """Legacy links by default; GEN1 may bind its sole registered raw input.

    No generated image is substituted. Both modes verify every actual frame.
    The caller's native scope must establish closed custody before either.
    """
    if type(reuse_registered_source_rgb)is not bool:
        raise PermissionError('SAVED_EXPORT_EXPLICIT_RGB_MATERIALIZATION_POLICY')
    folder,out=Path(folder),Path(out)
    if reuse_registered_source_rgb:
        if (card.get('study_split')!='G_TRAIN' or card.get('split')!='G_TRAIN'
                or not card.get('id','').startswith('GEN1-PILOT-') or folder.is_symlink()):
            raise PermissionError('SAVED_EXPORT_RAW_REUSE_ONLY_EXPLICIT_GEN1_SOURCE_INPUT')
        raw=folder
    else:
        raw=out/'raw'/card['id'];raw.mkdir(parents=True)
    for names,digests in zip(arrays['rgb_paths'],arrays['rgb_sha256']):
        for name,digest_value in zip(names,digests):
            guard();p=folder/str(name);target=raw/str(name)
            if Path(str(name)).is_absolute() or '..' in Path(str(name)).parts or sha(p)!=str(digest_value):
                raise ValueError('PV1_EXPORTED_RGB_HASH_AND_RELATIVE_PATH')
            if not reuse_registered_source_rgb:
                target.parent.mkdir(exist_ok=True);os.link(p,target)
    return raw


def run(root):
    root=Path(root);lock=OwnerLock(root/'PV1_SAVED_EXPORT_OWNER.lock')
    m,inputs=validate(root,role=EXPORT_ROLE);plan=read(root/'plans/SAVED_EXPORT_PLAN.private.json')
    ledger=read(root/'RESOURCE_LEDGER.json')
    if (ledger['status']!='RESERVED' or ledger['unknown_effects'] or not plan['cases']
            or len(plan['cases'])>288 or sha(root/'plans/statistics.json')!=STATS
            or m['limits']['engine_starts']!=0 or m['limits']['physics_ticks']!=0
            or m['limits']['GR00T_queries']!=0 or m['limits']['training_updates']!=0):
        raise PermissionError('PV1_SAVED_EXPORT_NO_NATIVE_ENGINE_NEURAL_OR_OPTIMIZER')
    started=time.process_time();child=resource.getrusage(resource.RUSAGE_CHILDREN);failure=None;records=[]
    os.sched_setaffinity(0,set(m['limits']['cpus']))
    resource.setrlimit(resource.RLIMIT_CPU,(m['limits']['CPU_s'],m['limits']['CPU_s']))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    def guard():
        if (time.time()>=m['limits']['operation_deadline_epoch']
                or time.process_time()-started>=m['limits']['CPU_s']-10
                or (root/'STOP.json').exists() or (root.parent.parent/'STOP.json').exists()):
            raise PermissionError('PV1_SAVED_EXPORT_FINITE_CPU_WALL_AND_STOP')
    def expire(*a):raise TimeoutError('PV1_SAVED_EXPORT_FINITE_WALL')
    try:
        signal.signal(signal.SIGALRM,expire);signal.alarm(max(1,int(m['limits']['operation_deadline_epoch']-time.time())))
        cpu_guard()
        processor=build_processor(inputs['official_source'],inputs['cosmos'],read(root/'plans/statistics.json'))
        ledger['status']='RUNNING';write(root/'RESOURCE_LEDGER.json',ledger)
        for case in plan['cases']:
            guard();out=root/'datasets'/case['id'];out.mkdir(parents=True,exist_ok=False)
            os.link(root/'plans/statistics.json',out/'statistics.json')
            row=export_case(case,out,processor,inputs['official_source'],inputs['cosmos'],guard)
            put(out/'EXPORT_RECEIPT.json',row);records.append(row)
            ledger.update(completed=len(records),admitted=sum(r['admitted_for_training'] for r in records),
                blocked=sum(not r['admitted_for_training'] for r in records),worker_CPU_s=time.process_time()-started)
            write(root/'RESOURCE_LEDGER.json',ledger)
        import torch
        forbidden=[n for n in sys.modules if n.startswith('gr00t') and ('modeling_' in n or '.backbone.' in n or '.action_head.' in n)]
        if forbidden or torch.cuda.is_initialized():raise PermissionError('PV1_CPU_EXPORT_NEURAL_OR_CUDA_FORBIDDEN')
        result=dict(status='TERMINAL_SAVED_TRAIN_EXPORT',rows=records,planned=len(plan['cases']),
            admitted=sum(r['admitted_for_training'] for r in records),statistics_sha256=STATS,
            population_plan_sha256=POPULATION,new_episodes=0,integrations=0,constructors=0,GR00T_queries=0,
            training_updates=0,neural_modules_imported=False,CUDA_initialized=False)
        put(root/'evidence/SAVED_EXPORT_RESULT.private.json',result)
        return dict(status=result['status'],completed=len(records),admitted=result['admitted'],new_physics=0)
    except BaseException as exc:
        failure=type(exc).__name__+': '+str(exc)
        put(root/'evidence/SAVED_EXPORT_FAILURE.private.json',dict(error=failure,completed=len(records)))
        raise
    finally:
        signal.alarm(0);now=resource.getrusage(resource.RUSAGE_CHILDREN)
        ledger.update(status='FAILED_PRESERVED' if failure else 'TERMINAL_SAVED_TRAIN_EXPORT',
            CPU_s=time.process_time()-started+now.ru_utime+now.ru_stime-child.ru_utime-child.ru_stime,
            CPU_reserved_s=0,failure=failure,completed=len(records))
        write(root/'RESOURCE_LEDGER.json',ledger);lock.close()
