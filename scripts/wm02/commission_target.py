"""Run one registered AUTO1 cold-start/hold stage with actual native trace evidence."""
import argparse,json,time,traceback
from pathlib import Path

def run(spec,output):
    from go2arx5_vla.world_model.target_sequential import TargetSequentialAdapter
    from go2arx5_vla.world_model.init_trace_reader import audit_trace
    from go2arx5_vla.world_model.native_replay import packets
    from go2arx5_vla.world_model.storage import digest,write_json
    start=time.monotonic();cpu=time.process_time();adapter=TargetSequentialAdapter(spec,output);failure=None;reached=False
    try:
        adapter.start()
        if not spec.get('standby_shutdown_diagnostic',False):adapter.step_to(spec['max_ticks']);reached=True
        if spec.get('reference_readiness_probe',False):adapter.inspect_reference_readiness()
        elif spec['phase']=='g2-reference-qualification':adapter.qualify_reference()
    except BaseException:failure=traceback.format_exc()
    finally:adapter.close()
    output=Path(output);trace=output/'init.trace';audit=None;audit_error=None
    try:audit=audit_trace(trace,max_native_ticks=spec['max_ticks'])if trace.exists()else None
    except BaseException:audit_error=traceback.format_exc()
    rows=packets(output/'state.bin.journal')if(output/'state.bin.journal').exists()else[]
    physics=[json.loads(l)for l in(output/'physics.jsonl').read_text().splitlines()]if(output/'physics.jsonl').exists()else[]
    stops=[r for r in physics if r['kind']=='stop'];samples=[r for r in physics if r['kind']=='sample']
    warnings=sum(bool(r.get('positive_bound')or any(r.get('native_warnings',[])))for r in samples)
    shutdown_clean=bool(adapter.exit_records)and all(p['exit_code']==0 for p in adapter.exit_records)
    reference_ok=spec['phase']!='g2-reference-qualification'or adapter.reference_qualification is not None
    passed=reached and not failure and not stops and not warnings and len(rows)==spec['max_ticks']and audit is not None and audit['status']=='RECORDED_TRACE_CHECKS_PASS'and shutdown_clean and reference_ok
    result=dict(schema='WM02_AUTO1_NATIVE_STAGE_V1',status='NATIVE_STAGE_PASSED'if passed else'NATIVE_STAGE_FAILED_PRESERVED',campaign_uuid=Path(spec['campaign_root']).name,candidate_index=spec['candidate_index'],phase=spec['phase'],source_sha=spec['source_sha'],candidate_source_sha=spec['candidate_source_sha'],native_binary_sha256=digest(spec['native_binary']),observer_sha256=digest(spec['observer_library']),scene_sha256=digest(spec['model_path']),actual_target_integrations=len(rows),complete_atomic_transitions=len(rows)//25,partial_native_ticks=len(rows)%25,reached_exact_pause=reached,physics_stops=stops,original_joint_or_warning_samples=warnings,trace_audit=audit,trace_audit_error=audit_error,failure=failure,owned_process_exits=adapter.exit_records,reference_qualification=adapter.reference_qualification,shutdown_clean=shutdown_clean,operator_shutdown='SIGNAL_ONLY_OWNED_PROCESSES_AFTER_NATIVE_PAUSE'if reached else'ERROR_CLOSEOUT',safety_STOP_qualification='NOT_TESTED'if not stops else'RECORDED_SEPARATELY',WM_updates=0,Actor_updates=0,full_tasks=0,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)
    write_json(output/'RESULT.json',result);print(json.dumps({k:result[k]for k in('status','phase','source_sha','actual_target_integrations','complete_atomic_transitions','failure','wall_s')}));return 0 if passed else 1

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in('spec','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--run-id',required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.auto1_native import admitted_target
    with admitted_target(a.spec,a.run_id,a.output)as spec:return run(spec,a.output)
if __name__=='__main__':raise SystemExit(main())
