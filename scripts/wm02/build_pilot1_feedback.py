"""Build real TRAIN-only native feedback, preserving the original DEV split."""
import argparse
import json
from pathlib import Path


def merge(base, incoming, output):
    import numpy as np
    from go2arx5_vla.world_model.target_replay_v2 import materialize
    from go2arx5_vla.world_model.storage import digest,write_json
    values=[]
    for path in (base,incoming):
        with np.load(path,allow_pickle=False)as archive:raw={k:archive[k].copy()for k in archive.files}
        meta=json.loads(raw.pop('metadata_utf8').tobytes());materialize(raw,meta);values.append((raw,meta))
    (old,om),(new,nm)=values
    for key in ('schema','state_fields','joints','robot_fingerprint','executor_sha256','collection_protocol','action'):
        if om[key]!=nm[key]:raise ValueError('NO_SILENT_FEEDBACK_EXECUTOR_OR_ACTION_MIGRATION:'+key)
    if any(e['split']!='TRAIN'or not e.get('pilot1_feedback')for e in nm['episodes']):raise ValueError('ONLY_ACTUAL_NEW_TRAIN_FEEDBACK_MAY_BE_APPENDED')
    old_names={e['episode']for e in om['episodes']}
    if any(e['episode']in old_names for e in nm['episodes']):raise ValueError('DUPLICATED_PHYSICAL_EPISODE')
    state_offset=len(old['flat_state']);atom_offset=len(old['flat_source_age']);episode_offset=len(om['episodes'])
    raw={k:np.concatenate((old[k],new[k]),axis=0)for k in old if k!='window_index'}
    shifted=new['window_index'].copy();shifted[:,0]+=episode_offset;raw['window_index']=np.concatenate((old['window_index'],shifted),axis=0)
    meta=dict(om);meta['episodes']=list(om['episodes'])+[dict(e,state_start=e['state_start']+state_offset,atom_start=e['atom_start']+atom_offset)for e in nm['episodes']]
    meta.update(task_atoms=om['task_atoms']+nm['task_atoms'],windows=om['windows']+nm['windows'],
                scene_assets_sha256=sorted(set(om['scene_assets_sha256'])|set(nm['scene_assets_sha256'])),
                pilot1_feedback_replay_sha256=digest(incoming),parent_replay_sha256=digest(base),actual_new_TRAIN_windows=nm['windows'])
    materialize(raw,meta)
    raw['metadata_utf8']=np.frombuffer(json.dumps(meta,sort_keys=True,allow_nan=False).encode(),dtype=np.uint8)
    with Path(output).open('xb')as stream:np.savez_compressed(stream,**raw)
    return dict(status='PILOT1_ACTUAL_TRAIN_FEEDBACK_APPENDED',replay_sha256=digest(output),parent_replay_sha256=digest(base),
                incoming_replay_sha256=digest(incoming),new_TRAIN_windows=nm['windows'],new_TRAIN_atoms=nm['task_atoms'],
                old_DEV_episodes_preserved=sum(e['split']=='DEV'for e in om['episodes']),new_physics_integrations=0)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in('manifest','base','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(argv);args.output.mkdir(exist_ok=False)
    from go2arx5_vla.world_model.target_replay_v2 import build
    from go2arx5_vla.world_model.storage import write_json
    built=build(args.manifest,args.output/'new-native-01')
    result=merge(args.base,args.output/'new-native-01/REPLAY.npz',args.output/'REPLAY.npz')
    result['native_build']=built;write_json(args.output/'RESULT.json',result);print(json.dumps(result));return 0


if __name__=='__main__':raise SystemExit(main())
