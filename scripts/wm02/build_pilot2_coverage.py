"""Append preregistered TRAIN/DEV coverage without changing historical splits."""
import argparse
import json
from pathlib import Path


def merge(base,incoming,output):
    import numpy as np
    from go2arx5_vla.world_model.target_replay_v2 import materialize
    from go2arx5_vla.world_model.storage import digest
    values=[]
    for path in(base,incoming):
        with np.load(path,allow_pickle=False)as a:raw={k:a[k].copy()for k in a.files}
        meta=json.loads(raw.pop('metadata_utf8').tobytes());materialize(raw,meta);values.append((raw,meta))
    (old,om),(new,nm)=values
    if set(old)!=set(new):raise ValueError('EXACT_REPLAY_ARRAY_ABI_REQUIRED')
    if (nm.get('manifest_schema')!='WM02_PILOT2_COVERAGE_REPLAY_MANIFEST_V1'
            or any(e['split']not in('TRAIN','DEV')for e in nm['episodes'])):
        raise ValueError('ONLY_PREDECLARED_PILOT2_COVERAGE_NO_TEST_IMPORT')
    for key in('schema','state_fields','joints','robot_fingerprint','executor_sha256','collection_protocol','action'):
        if om[key]!=nm[key]:raise ValueError('NO_SILENT_EXECUTOR_OR_ACTION_MIGRATION:'+key)
    names={e['episode']for e in om['episodes']}
    if any(e['episode']in names for e in nm['episodes']):raise ValueError('NO_DUPLICATE_PHYSICAL_EPISODE')
    raw={k:np.concatenate((old[k],new[k]),axis=0)for k in old if k!='window_index'}
    shifted=new['window_index'].copy();shifted[:,0]+=len(om['episodes']);raw['window_index']=np.concatenate((old['window_index'],shifted))
    meta=dict(om);meta['episodes']=list(om['episodes'])+[dict(e,state_start=e['state_start']+len(old['flat_state']),
        atom_start=e['atom_start']+len(old['flat_source_age']))for e in nm['episodes']]
    meta.update(task_atoms=om['task_atoms']+nm['task_atoms'],windows=om['windows']+nm['windows'],
        scene_assets_sha256=sorted(set(om['scene_assets_sha256'])|set(nm['scene_assets_sha256'])),
        pilot2_coverage_replay_sha256=digest(incoming),parent_replay_sha256=digest(base),
        historical_DEV_not_relabelled=True,TEST_imported=False)
    materialize(raw,meta)
    raw['metadata_utf8']=np.frombuffer(json.dumps(meta,sort_keys=True,allow_nan=False).encode(),dtype=np.uint8)
    with Path(output).open('xb')as f:np.savez_compressed(f,**raw)
    return dict(status='ACTUAL_PILOT2_COVERAGE_APPENDED',replay_sha256=digest(output),
        parent_replay_sha256=digest(base),incoming_replay_sha256=digest(incoming),
        new_TRAIN_windows=sum(e['window_count']for e in nm['episodes']if e['split']=='TRAIN'),
        new_DEV_windows=sum(e['window_count']for e in nm['episodes']if e['split']=='DEV'),
        existing_DEV_episodes_preserved=sum(e['split']=='DEV'for e in om['episodes']),
        native_integrations=0,model_updates=0)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in('manifest','base','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args(argv);a.output.mkdir(exist_ok=False)
    from go2arx5_vla.world_model.target_replay_v2 import build
    from go2arx5_vla.world_model.storage import write_json
    built=build(a.manifest,a.output/'new-native-01')
    result=merge(a.base,a.output/'new-native-01/REPLAY.npz',a.output/'REPLAY.npz');result['native_build']=built
    write_json(a.output/'RESULT.json',result);print(json.dumps(result));return 0


if __name__=='__main__':raise SystemExit(main())
