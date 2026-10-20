"""Full original codec/data chain for separately sealed DATA-COV1 TRAIN cards."""
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np

from scripts.pi0291_arx5_data.native_dataset import cpu_guard
from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0292.train_adapter import build_processor
from scripts.pi0294.e1.scope import STATS
from scripts.pi0294.fd1 import put,sha
from scripts.pi0294.scene_cards import digest,task_card
from scripts.pi0294.pv1.export import committed_packet,export_committed_case
from scripts.pi0294.pv1.event_exposure import analyze as event_exposure,_edges,_coverage
from .codec import audit as codec_audit
from .plan import validate_group
from .scope import sealed_file,validate


def packet(samples,card):
    validate_group(card)
    if (card['split']!='TRAIN' or card['nominal_clearance_m']!=.004 or card['task'] not in ('T1','T2')
            or card['instruction']!=task_card(card,card['task'])['instruction']
            or not card['id'].startswith('PI0294-DATA_COV1-TRAIN-')
            or any(s['causal_history']['episode_id']!=card['id'] for s in samples)):
        raise PermissionError('DATA_COV1_ORIGINAL_REFERENCE_EXACT_TRAIN_LABELS')
    return committed_packet(samples,card)


def physical_events(folder,card,geometry):
    with np.load(folder/'physics.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    result=event_exposure(data,card)
    micro=data['contact_audit'];times=data['contact_time_s'];initial=data['object_pose'][0,2,3]
    lift=_edges(times,(micro[:,1].astype(bool))&(micro[:,3]-initial>=.02),True)
    result['events']['lift_2cm_while_native_grasp']=dict(count=len(lift),first_s=lift[0] if lift else None,
        last_s=lift[-1] if lift else None,prefix8_0_to_0p35s=_coverage(lift,.35),H40_0_to_1p95s=_coverage(lift,1.95))
    result.update(schema='PI0294_DATA_COV1_PHYSICAL_EVENTS_V1',
        original_score=read(folder/'SCORE.json'),lift_event_threshold_m=.02,
        native_grasp_end_is_not_complete_release_success=True,terminal_hold_from_original_unchanged_score=True)
    info=read(geometry.with_suffix('.json'));sealed_file(geometry.with_suffix('.json'))
    sealed_file(geometry,info['sha256'])
    with np.load(geometry,allow_pickle=False) as z:rows=z['rows'].copy()
    if rows.shape!=(16000,5) or not np.isfinite(rows).all() or not np.allclose(rows[:,0],times,atol=1e-8,rtol=0):
        raise ValueError('DATA_COV1_COMPLETE_ACTUAL_PAD_GEOMETRY_CLOCK_REQUIRED')
    result['actual_pad_plane_clearance']=dict(minimum_left_m=float(rows[:,1].min()),minimum_right_m=float(rows[:,2].min()),
        measured_ticks=16000,projected_overlap_left_ticks=int(rows[:,3].sum()),projected_overlap_right_ticks=int(rows[:,4].sum()),
        geometry_sha256=info['sha256'],vertical_plane_gap_not_exact_geom_distance=True)
    return result


def join_codec_event(codec,folder,card):
    trace=codec['diagnostics']['first_error']
    if trace is None:return
    with np.load(folder/'physics.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    index=trace['window_anchor']+trace['action_index'];t=index/20
    micro_index=max(int(round(t/.002))-1,0);micro=data['contact_audit'][micro_index]
    obj=data['object_pose'][index];tcp=data['tcp_world'][index]
    import itertools
    corners=np.asarray(list(itertools.product((-.02,.02),repeat=3)))@obj[:3,:3].T+obj[:3,3]
    trace['actual_event_join']=dict(time_s=t,observation_index=index,native_contact_index=micro_index,
        contact_500Hz=bool(micro[0]),native_grasp_500Hz=bool(micro[1]),
        lift_from_initial_m=float(obj[2,3]-data['object_pose'][0,2,3]),
        horizontal_displacement_m=float(np.linalg.norm(obj[:2,3]-data['object_pose'][0,:2,3])),
        all_object_corners_in_zone_20Hz=bool((np.abs(corners[:,:2]-card['goal_xy'])<=.07).all()),
        TCP_clear_20Hz=bool(np.linalg.norm(tcp[:3,3]-obj[:3,3])>=.08),
        t0_native_contact_uses_first_2ms_not_unrecorded_initial_state=index==0,
        actual_saved_state_before_that_command_not_counterfactual_next_state=True,
        Teacher_phase_used=False)


def horizon_kinematics(samples,card):
    from scripts.pi0291_arx5_data.native_dataset import matrix_from10
    commands=packet(samples,card)['committed_absolute10']
    matrices=matrix_from10(commands)
    cosine=np.clip((np.trace(matrices[:-1,:3,:3]@np.swapaxes(matrices[1:,:3,:3],-1,-2),axis1=-2,axis2=-1)-1)/2,-1,1)
    values=dict(linear_speed_mps=np.linalg.norm(np.diff(commands[:,:3],axis=0),axis=-1)/.05,
        angular_speed_radps=np.arccos(cosine)/.05,gripper_fraction_change=np.abs(np.diff(commands[:,9])))
    return dict(legal_windows=601,sample_interval_s=.05,committed_labels_not_unexecuted_plans=True,
        regions={name:{key:dict(maximum=float(max(np.max(array[a+start:a+stop]) for a in range(601))),
            per_window_max=[float(np.max(array[a+start:a+stop])) for a in range(601)]) for key,array in values.items()}
            for name,start,stop in [('H8_INTERNAL',0,7),('H8_H9_BOUNDARY',7,8),('H9_H40_INTERNAL',8,39)]},
        diagnostic_only_not_a_shorter_safety_gate=True)


def run(root):
    root=Path(root);lock=OwnerLock(root/'DATA_COV1_EXPORT.lock');m,inputs=validate(root,'EXPORT')
    plan=read(root/'plans/WORK_PLAN.private.json');ledger=read(root/'RESOURCE_LEDGER.json')
    if (ledger['status']!='RESERVED' or ledger['unknown_effects'] or not plan['cases']
            or sha(root/'plans/statistics.json')!=STATS
            or any(m['limits'][k]!=0 for k in ('engine_starts','physics_ticks','Teacher_episodes'))):
        raise PermissionError('DATA_COV1_SAVED_EXPORT_NO_NEW_NATIVE_OR_NEURAL_EFFECTS')
    began=time.process_time();before=resource.getrusage(resource.RUSAGE_CHILDREN);rows=[];failure=None
    os=__import__('os');os.sched_setaffinity(0,set(m['limits']['cpus']))
    resource.setrlimit(resource.RLIMIT_CPU,(m['limits']['CPU_s'],m['limits']['CPU_s']))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    def guard():
        if (time.time()>=m['limits']['operation_deadline_epoch'] or time.process_time()-began>=m['limits']['CPU_s']-30
                or (root/'STOP.json').exists() or (root.parent.parent/'STOP.json').exists()):
            raise TimeoutError('DATA_COV1_EXPORT_CPU_WALL_STOP')
    def expire(*args):raise TimeoutError('DATA_COV1_EXPORT_WALL')
    try:
        signal.signal(signal.SIGALRM,expire);signal.alarm(max(1,int(m['limits']['operation_deadline_epoch']-time.time())))
        cpu_guard();processor=None
        ledger['status']='RUNNING';write(root/'RESOURCE_LEDGER.json',ledger)
        for case in plan['cases']:
            guard();source=Path(case['source_episode']);card=read(source/'CARD.json')
            binding=read(source.parent.parent/'OUTPUT_BINDING.private.json')
            if (read(binding['lease'])['status']!='CLOSED' or card!=case['card']
                    or read(source.parent.parent/'RESOURCE_LEDGER.json')['unknown_effects']):
                raise PermissionError('DATA_COV1_CLOSED_CAPTURE_AND_EXACT_STATIC_CARD')
            if case.get('kind')=='SAVED_UNSCORABLE_FAILURE_WITNESS':
                from .failed_witness import score_input_failure_witness
                if sha(source/'FILE_MANIFEST.json')!=case['raw_manifest_sha256']:
                    raise ValueError('DATA_COV1_FAILED_RAW_EXACT_BOUND_MANIFEST')
                row=score_input_failure_witness(source,card,guard)
                row['admitted_for_training']=False
                put(root/'evidence'/(card['id']+'-FAILED_RAW_WITNESS.json'),row);rows.append(row)
                continue
            packet(read(source/'samples.json'),card)
            if processor is None:
                processor=build_processor(inputs['official_source'],inputs['cosmos'],read(root/'plans/statistics.json'))
            out=root/'datasets'/case['id'];out.mkdir(parents=True)
            os.link(root/'plans/statistics.json',out/'statistics.json')
            row=export_committed_case(case,out,processor,inputs['official_source'],inputs['cosmos'],guard,
                packet_builder=packet,codec_auditor=codec_audit,direction_for_card=lambda c:c['direction'],
                origin='DATA_COV1_ACTUAL_ORIGINAL_TEACHER_TRAIN')
            join_codec_event(row['codec'],source,card)
            row.update(appearance_profile=card['appearance_profile'],distance_bin=card['distance_bin'],
                actual_physical_event_receipt=physical_events(source,card,Path(case['geometry_file'])),
                committed_horizon_kinematics=horizon_kinematics(read(source/'samples.json'),card))
            if row['admitted_for_training']:(out/row['file']).chmod(0o400)
            put(out/'EXPORT_RECEIPT.json',row);rows.append(row)
            ledger.update(completed=len(rows),admitted=sum(r['admitted_for_training'] for r in rows),
                worker_CPU_s=time.process_time()-began);write(root/'RESOURCE_LEDGER.json',ledger)
        import torch
        forbidden=[n for n in sys.modules if n.startswith('gr00t') and any(k in n for k in ('modeling_','.backbone.','.action_head.'))]
        if forbidden or torch.cuda.is_initialized():raise PermissionError('DATA_COV1_CPU_ONLY_NO_NEURAL_OR_CUDA')
        result=dict(status='TERMINAL_DATA_COV1_SAVED_EXPORT',rows=rows,planned=len(plan['cases']),
            statistics_sha256=STATS,neural_modules_imported=False,CUDA_initialized=False,
            new_episodes=0,integrations=0,constructors=0,GR00T_queries=0,training_updates=0)
        put(root/'evidence/EXPORT_RESULT.private.json',result)
        return dict(status=result['status'],completed=len(rows),admitted=sum(r['admitted_for_training'] for r in rows))
    except BaseException as exc:
        failure=type(exc).__name__+': '+str(exc);put(root/'evidence/EXPORT_FAILURE.private.json',dict(error=failure,completed=len(rows)))
        raise
    finally:
        signal.alarm(0);after=resource.getrusage(resource.RUSAGE_CHILDREN)
        ledger.update(status='FAILED_PRESERVED' if failure else 'TERMINAL_DATA_COV1_SAVED_EXPORT',CPU_reserved_s=0,
            CPU_s=time.process_time()-began+after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
            failure=failure,completed=len(rows))
        write(root/'RESOURCE_LEDGER.json',ledger);lock.close()
