"""Export actual post-choice FUSION response windows without physical integration."""
import argparse,json,time
from pathlib import Path


def build(manifest,output):
    from go2arx5_vla.world_model.storage import digest,write_json
    from go2arx5_vla.world_model.native_replay import packets,forbid_integration
    from go2arx5_vla.world_model.next2_episode import state_vector
    from go2arx5_vla.world_model.fusion_replay import response_window
    from scripts.pi0294.task_state import ActualState
    import numpy as np,mujoco
    rows=json.loads(manifest.read_text())['episodes'];records=[];identities=[];excluded=[];seen=set()
    start=time.monotonic();cpu=time.process_time();output.mkdir(exist_ok=False)
    for row in rows:
        root=Path(row['root'])
        if root.name in seen:raise ValueError('DUPLICATE_PHYSICAL_FUSION_RUN')
        seen.add(root.name)
        required={'EXECUTION_SPEC.json','EPISODE_MANIFEST.private.json','native-01/RESULT.json',
            'native-01/FUSION_DECISION.private.json','native-01/FUSION_SCORE.private.json',
            'native-01/DYNAMIC_DIAGNOSTIC.json','native-01/state.bin.journal'}
        if not required <= set(row['files']) or any(Path(n).is_absolute()or '..'in Path(n).parts for n in row['files']):
            raise ValueError('EXACT_HASHED_RELATIVE_FUSION_EVIDENCE_INVENTORY_REQUIRED')
        for rel,pin in row['files'].items():
            if digest(root/rel)!=pin:raise ValueError('SEALED_FUSION_EVIDENCE_CHANGED:'+rel)
        s=json.loads((root/'EXECUTION_SPEC.json').read_text());e=json.loads(Path(s['episode_manifest']).read_text())
        if Path(s['episode_manifest']) != root/'EPISODE_MANIFEST.private.json':raise ValueError('BOUND_FUSION_EPISODE_PATH_DIFFERS')
        lease=json.loads(Path(s['lease']).read_text())
        if e['split']=='TEST':raise PermissionError('TEST_NEVER_ENTERS_FUSION_REPLAY')
        if lease['status']!='CLOSED':raise PermissionError('ACTUAL_FUSION_RUN_NOT_SEALED')
        n=root/'native-01';native=json.loads((n/'RESULT.json').read_text());decision=json.loads((n/'FUSION_DECISION.private.json').read_text());score=json.loads((n/'FUSION_SCORE.private.json').read_text());diag=json.loads((n/'DYNAMIC_DIAGNOSTIC.json').read_text())
        if native['source_sha']!=s['source_sha']or decision['source_sha']!=s['source_sha']:raise ValueError('ACTUAL_FUSION_EXECUTION_SOURCE_DIFFERS')
        if digest(s['model_path'])!=native['scene_sha256']:raise ValueError('ACTUAL_SCENE_BYTES_CHANGED')
        journal=packets(n/'state.bin.journal');by_step={p['step']:p for p in journal}
        selected=[];steps=3503+25*np.arange(25)
        if any(int(step)not in by_step for step in steps):
            excluded.append(dict(run=root.name,reason='PARTIAL_RUN_NO_COMPLETE_RESPONSE_WINDOW'));continue
        with forbid_integration(mujoco)as calls:
            state=ActualState(mujoco.MjModel.from_binary_path(s['model_path']))
            for step in steps:selected.append(state_vector(state.update(by_step[int(step)])))
            try:record=response_window(e,decision,native,score,selected,steps,diag['windows'])
            except ValueError as error:
                excluded.append(dict(run=root.name,reason=str(error)));continue
            if calls:raise RuntimeError('REPLAY_MUST_NEVER_INTEGRATE')
        records.append(record);identities.append(dict(run=root.name,parent=e['parent'],split=e['split'],
            native_source_sha=native['source_sha'],native_binary_sha256=native['native_binary_sha256'],scene_sha256=native['scene_sha256'],files=row['files']))
    if not records:raise ValueError('NO_COMPLETE_VALID_REAL_FUSION_WINDOWS')
    parent_splits={}
    for r in records:
        if r['parent']in parent_splits and parent_splits[r['parent']]!=r['split']:raise ValueError('WHOLE_PARENT_SPLIT_REQUIRED')
        parent_splits[r['parent']]=r['split']
    arrays={key:np.stack([r[key]for r in records])for key in ['state','goal','history_goal','source_age','source_age_valid','desired_xy_m']}
    for key in ['future_prefix_remaining','window_start_s','observed_action','task_success','actual_XY_error_m','nominal_action']:
        arrays[key]=np.asarray([r[key]for r in records])
    arrays['split']=np.asarray([r['split']for r in records]);arrays['parent']=np.asarray([r['parent']for r in records])
    data=output/'REAL_RESPONSE_WINDOWS.npz'
    with data.open('xb')as f:np.savez_compressed(f,**arrays)
    result=dict(schema='WM02_FUSION_REAL_RESPONSE_REPLAY_V1',manifest_sha256=digest(manifest),
        replay_sha256=digest(data),eligible_windows=len(records),real_post_choice_transitions=16*len(records),
        physical_parents=parent_splits,identities=identities,excluded=excluded,negative_task_outcomes=sum(not r['task_success']for r in records),
        actual_native_integrations_by_builder=0,optimizer_updates=0,TEST_read=False,
        expert_action_labels=False,source_age_only_anchor8_valid=True,exact_full_controller_restore=False,
        wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu)
    write_json(output/'RESULT.private.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    print(json.dumps(build(a.manifest,a.output)));return 0


if __name__=='__main__':raise SystemExit(main())
