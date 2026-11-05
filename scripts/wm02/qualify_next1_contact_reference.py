"""Static original reference certificates for one preregistered graded path."""
def main(argv=None):
    import argparse,json,time
    from pathlib import Path
    import numpy as np
    from go2arx5_vla.world_model.native_replay import load,apply_frame,forbid_integration
    from go2arx5_vla.world_model.next1_contact import planned_path,chunk,GRADES
    from go2arx5_vla.world_model.storage import write_json,digest
    from scripts.pi0294.task_state import ActualState
    from scripts.pi0294.task_path_model import certify_chunk
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--grade',choices=list(GRADES),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv);start=time.monotonic();cpu=time.process_time();cfg=json.loads(a.config.read_text());mj,m,d,rows=load(cfg);native=json.loads(Path(cfg['native_result']['path']).read_text());proof=json.loads(Path(cfg['steady_prefix']['path']).read_text())
    if digest(cfg['steady_prefix']['path'])!=cfg['steady_prefix']['sha256']or not proof['qualified']or native['reference_qualification']['steady_prefix']!=proof:raise ValueError('ACTUAL_STEADY_RECORDING_REQUIRED')
    initial=next(r for r in rows if r['step']==2503);steady=next(r for r in rows if r['step']==proof['actual_native_step']);actual=ActualState(m);checks=[];failure=None
    with forbid_integration(mj):
        initial_state=actual.update(initial);sample=actual.update(steady);path=planned_path(initial_state['tcp'],sample['object_pose'],a.grade)
        for offset in range(0,GRADES[a.grade]['task_atoms']-48,7):
            # Static witness uses the pinned actual root/legs, never a prediction
            # of future closed-loop state; every live installation is recertified.
            sample=dict(sample,native_step=steady['step']+25*offset)
            history=[dict(end_native_step=sample['native_step']-25,pose=path[max(0,min(offset-2,len(path)-1))]),dict(end_native_step=sample['native_step'],pose=path[max(0,min(offset-1,len(path)-1))])]
            try:certificate=certify_chunk(m,sample,chunk(path,offset),stage='approach',history=history)
            except ValueError as e:failure=str(e);checks.append(dict(offset=offset,failure=failure));break
            checks.append(dict(offset=offset,certificate=certificate))
    value=dict(schema='WM02_NEXT1_GRADED_STATIC_REFERENCE_V1',grade=a.grade,all_static_chunks_feasible=failure is None,first_blocker=failure or'ACTUAL_NATIVE_GRADED_RESPONSE_REQUIRED',checks=checks,pinned_steady_native_step=steady['step'],path_sha256=__import__('hashlib').sha256(path.astype('<f8').tobytes()).hexdigest(),config_sha256=digest(a.config),new_physical_integrations=0,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)
    write_json(a.output,value);print(json.dumps({k:v for k,v in value.items()if k!='checks'}));return 0 if failure is None else 2

if __name__=='__main__':raise SystemExit(main())
