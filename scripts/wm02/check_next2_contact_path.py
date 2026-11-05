"""Zero-integration nominal descent checks from an actual sealed C1 endpoint."""
def run(spec):
    import json
    from pathlib import Path
    import numpy as np,mujoco
    from go2arx5_vla.world_model.native_replay import packets,forbid_integration
    from go2arx5_vla.world_model.next2_contact import stage_path
    from go2arx5_vla.world_model.next1_contact import chunk
    from go2arx5_vla.world_model.storage import digest
    from scripts.pi0294.task_state import ActualState
    from scripts.pi0294.task_path_model import certify_chunk
    root=Path(spec['from_run']);r=root/'native-01';native=json.loads((r/'RESULT.json').read_text());proof=json.loads((r/'PRECONTACT_CONTINUITY.private.json').read_text());execution=json.loads((root/'EXECUTION_SPEC.json').read_text())
    if native['status']!='NATIVE_STAGE_PASSED'or not proof['precontact_continuity_verified']or not native['shutdown_clean']or native['physics_stops']:raise ValueError('ACTUAL_NATIVE_AND_CONTINUITY_PASS_REQUIRED')
    model=mujoco.MjModel.from_binary_path(execution['model_path']);engine=ActualState(model)
    with forbid_integration(mujoco):
        state=engine.update(packets(r/'state.bin.journal')[-1]);above=np.asarray(json.loads((r/'CONTACT_PLANNED_REFERENCE.private.json').read_text())['path'])[-1];end=above.copy();end[:3,3]=state['object_pose'][:3,3]+[0,0,.02];descend=stage_path(above,end);withdraw=stage_path(end,above);checks=[]
        for label,path in [('DESCEND_80_TO20',descend),('WITHDRAW_20_TO80',withdraw)]:
            for offset in range(0,51,7):
                try:checks.append(dict(stage=label,offset=offset,passed=True,certificate=certify_chunk(model,state,chunk(path,offset),stage='align')))
                except ValueError as error:checks.append(dict(stage=label,offset=offset,passed=False,reason=str(error),contact_details=getattr(error,'native_contact_details',None)))
    return dict(schema='WM02_NEXT2_CONTACT_STAGE_STATIC_V1',all_nominal_stage_chunks_feasible=all(c['passed']for c in checks),actual_C1_native_result_sha256=digest(r/'RESULT.json'),actual_C1_continuity_sha256=digest(r/'PRECONTACT_CONTINUITY.private.json'),state_journal_sha256=digest(r/'state.bin.journal'),checks=checks,stage_atoms=50,planning_only_not_actual_future_history=True,new_integrations=0,new_optimizer_updates=0,remaining_time_rule=dict(DESCEND_s=10.,CLOSE_s=6.25,WITHDRAW_s=3.),native_original_certificates_still_required_at_every_real_install=True)

def main(argv=None):
    import argparse,json
    from pathlib import Path
    from go2arx5_vla.world_model.storage import write_json
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv);v=run(json.loads(a.spec.read_text()));write_json(a.output,v);print(json.dumps({k:v[k]for k in ['all_nominal_stage_chunks_feasible','stage_atoms','new_integrations']}));return 0 if v['all_nominal_stage_chunks_feasible']else 2
if __name__=='__main__':raise SystemExit(main())
