"""Finite P1 source diagnostics or explicit target-blocked readback; no training."""
import argparse,json,time
from pathlib import Path


def qualify_source(spec,output):
    import numpy as np
    from copy import deepcopy
    from go2arx5_vla.world_model.native_adapter import SourceAdapter
    from go2arx5_vla.world_model.contracts import command_labels,FRAME,TCP
    from go2arx5_vla.world_model.action_compiler import compile_reference,SOURCE
    from go2arx5_vla.world_model.transitions import TransitionRecorder
    from go2arx5_vla.world_model.storage import write_json,dump_private,load_private,digest
    cards=json.loads(Path(spec['diagnostic_cards']).read_text())
    if len(cards)!=8 or any(c['split']!='TRAIN'for c in cards):raise ValueError('Fixed8 source TRAIN cases required')
    began,cpu=time.monotonic(),time.process_time();results=[];total_ticks=shadow_ticks=atoms=0
    def guard():
        if time.time()>=spec['deadline_epoch']or(Path(spec['registered_root'])/'STOP.json').exists():
            raise TimeoutError('FINITE_WM02_P1_SOURCE_ENDPOINT')
    for index,card in enumerate(cards):
        guard();folder=output/card['id'];folder.mkdir();recorder=TransitionRecorder(native_dt=.002)
        adapter=SourceAdapter(card,spec['asset_root'],executor_id=spec['executor_id'],seed=card['seed'])
        case_cpu=time.process_time();case_wall=time.monotonic();live=0;restore_ticks=0;failure=None
        try:
            write_json(folder/'CARD.json',card)
            # Physically construct the state with original Teacher, no teleport.
            for macro in range(card['prefix_macros']):
                guard();plan=adapter.teacher_plan();raw,ok=command_labels(plan['poses'][:8],plan['opening'][:8],adapter.snapshot()['tcp'])
                if not ok:raise ValueError('Teacher prefix outside primitive')
                plan['action_raw']=raw
                ref=adapter.save_complete(folder/f'prefix-{macro:02d}-restore.pkl.gz')
                if adapter.gate(plan)['decision']=='REJECT'or adapter.preflight(plan,guard)['decision']!='PASS':
                    recorder.reject(episode_id=card['id'],reason='PREFIX_S1_REJECT',state=adapter.snapshot(),action=raw);failure='PREFIX_S1_REJECT';break
                for point in range(8):
                    row=adapter.advance_recorded(plan,point,run_id=output.name,actor_version='ORIGINAL_SOURCE_TEACHER_PREFIX',restore_ref=ref)
                    recorder.append(row);live+=row['physics_ticks']
                    if row['terminal']or row['truncated']:raise RuntimeError('Native prefix terminated')
            if failure:raise RuntimeError(failure)
            native_before=adapter.snapshot();saved_ref=adapter.save_complete(folder/'diagnostic-restore.pkl.gz');saved=load_private(saved_ref['path'])
            if card['category']in('hold','small_free_space_motion'):
                raw=np.zeros((8,7));raw[:,6]=native_before['gripper_fraction']
                if card['category']=='small_free_space_motion':raw[:,0]=.0005
                plan=compile_reference(raw,native_before['tcp'],native_before['time_s'],backend=SOURCE,frame=FRAME,tcp=TCP)
            else:
                plan=adapter.teacher_plan();raw,ok=command_labels(plan['poses'][:8],plan['opening'][:8],adapter.snapshot()['tcp'])
                if not ok:raise ValueError('Diagnostic command not representable')
                plan['action_raw']=raw
                # Capture after original Teacher-plan snapshot cadence.
                saved=adapter.capture();dump_private(folder/'planned-diagnostic-restore.pkl.gz',saved)
                saved_ref=dict(path=str(folder/'planned-diagnostic-restore.pkl.gz'),sha256=digest(folder/'planned-diagnostic-restore.pkl.gz'))
            branches=[]
            for branch in range(2):
                adapter.restore(saved);guard()
                gate=adapter.gate(plan);forecast=adapter.preflight(plan,guard)if gate['decision']!='REJECT'else None
                if gate['decision']=='REJECT'or forecast['decision']!='PASS':
                    recorder.reject(episode_id=card['id'],reason='DIAGNOSTIC_S1_REJECT',state=adapter.snapshot(),action=raw)
                    failure='DIAGNOSTIC_S1_REJECT';break
                rows=[]
                for point in range(8):
                    row=adapter.advance_recorded(plan,point,run_id=output.name,actor_version='BOUNDED_SOURCE_P1_REFERENCE',restore_ref=saved_ref)
                    rows.append(row);restore_ticks+=row['physics_ticks']
                    if row['terminal']or row['truncated']:break
                branches.append(rows);dump_private(folder/f'branch-{branch}.pkl.gz',dict(rows=rows,final_restore=adapter.capture()))
            error=None;contact_equal=terminal_equal=False
            if len(branches)==2:
                if len(branches[0])!=len(branches[1]):raise RuntimeError('Restore branch length differs')
                error=max(float(np.max(np.abs(a['native_after']['integration']-b['native_after']['integration'])))for a,b in zip(*branches))
                contact_equal=all(a['state_after']['contacts']==b['state_after']['contacts']for a,b in zip(*branches))
                terminal_equal=all((a['terminal'],a['truncated'],a['safety_reason'])==(b['terminal'],b['truncated'],b['safety_reason'])for a,b in zip(*branches))
                if error>1e-10 or not contact_equal or not terminal_equal:raise RuntimeError('Full restore mismatch')
            after=adapter.snapshot()
            case=dict(index=index,id=card['id'],category=card['category'],source_sha=spec['source_sha'],domain='S_FIXED_BASE_DIAGNOSTIC',
                prefix_atomic_transitions=len(recorder.rows),diagnostic_atomic_transitions=sum(map(len,branches)),
                rejection_events=len(recorder.events),prefix_ticks=live,restore_branch_ticks=restore_ticks,shadow_ticks=adapter.shadow_ticks,
                restore_max_abs_error=error,contacts_equal=contact_equal,terminal_equal=terminal_equal,
                before_grasp=native_before['native_grasp'],after_grasp=after['native_grasp'],
                before_object_height_m=native_before['object_pose'][2,3],after_object_height_m=after['object_pose'][2,3],
                before_gripper_fraction=native_before['gripper_fraction'],after_gripper_fraction=after['gripper_fraction'],
                before_stage=saved['teacher']['stage'],hard_safety=list(adapter.engine.safety),failure=failure,
                source_S1_max_abs_error=adapter.max_shadow_error,CPU_s=time.process_time()-case_cpu,wall_s=time.monotonic()-case_wall,
                full_task_success_claimed=False,target_qualified=False)
            dump_private(folder/'PREFIX_RECORDS.pkl.gz',dict(rows=recorder.rows,events=recorder.events))
            write_json(folder/'RESULT.json',case);results.append(case)
            total_ticks+=live+restore_ticks;shadow_ticks+=adapter.shadow_ticks;atoms+=len(recorder.rows)+sum(map(len,branches))
            print(json.dumps(case),flush=True)
        except BaseException:
            dump_private(folder/'FAILED_PARTIAL.pkl.gz',dict(rows=recorder.rows,events=recorder.events,restore=adapter.capture()))
            write_json(folder/'FAILURE.json',dict(error=__import__('traceback').format_exc(),native_step=adapter.engine.steps,
                live_ticks=live,restore_branch_ticks=restore_ticks,shadow_ticks=adapter.shadow_ticks,CPU_s=time.process_time()-case_cpu))
            raise
        finally:adapter.close()
    wall=time.monotonic()-began
    result=dict(status='SOURCE_P1_DIAGNOSTICS_COMPLETE_TARGET_BLOCKED',domain='S_FIXED_BASE_DIAGNOSTIC',source_sha=spec['source_sha'],
        cases=results,attempts=8,actual_atomic_transitions=atoms,live_physics_ticks=total_ticks,shadow_physics_ticks=shadow_ticks,
        CPU_s=time.process_time()-cpu,wall_s=wall,end_to_end_transitions_per_s=atoms/wall,
        environment_instances=8,simultaneous_environments=1,parallel_target_profiles={'1':'NOT_RUN_TARGET_BLOCKED','4':'NOT_RUN_TARGET_BLOCKED','16':'NOT_RUN_TARGET_BLOCKED'},
        model_updates=0,actor_updates=0,target_transitions=0,target_task_successes=0,TEST_read=False,
        reward_validation='NOT_COMPLETE_NO_LEARNING_PERMITTED',full_target_restore='NOT_IMPLEMENTED_REQUIRES_MPC_WBC_EXPORTER')
    write_json(output/'RESULT.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--domain',choices=('S','T'),required=True);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--run-id',required=True);p.add_argument('--profile',required=True)
    p.add_argument('--parser',type=Path)
    a=p.parse_args(argv)
    from go2arx5_vla.world_model.storage import write_json
    if a.domain=='T':
        resolved=json.loads(a.spec.read_text())
        # No live target bridge/exporter is implemented in this P1 version.
        # Never fabricate admission from a caller-provided boolean.
        result=dict(status='TARGET_BLOCKED',blockers=resolved.get('target_blockers',[]),
            implementation_gap='No qualified native MPC/WBC full-state exporter or rebuild-prefix adapter',physics_ticks=0,model_updates=0)
        a.output.mkdir(exist_ok=False);write_json(a.output/'RESULT.json',result);print(json.dumps(result));return 78
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(a.spec,a.run_id,a.profile,a.output,physics=True)as spec:
        if a.parser:
            from scripts.pi0294.dec1.target_time_wire_cpu import qualify_parser_file
            write_json(a.output/'TARGET_PARSER_CPU.json',qualify_parser_file(a.parser))
        qualify_source(spec,a.output)
    return 0


if __name__=='__main__':raise SystemExit(main())
