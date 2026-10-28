"""Render every required saved state of one complete family, with exact-state reuse."""
import isaacgym
from pathlib import Path
import json
import hashlib
import sys
import os
import numpy as np
import pi0259_renderer as module
from pi0259_calibration_render import save_packet
from pi0255_motion3 import sha,write

ROOT=Path(os.environ.get('PI0260_ROOT','/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0260'));module.ROOT=ROOT


def run(family_id):
    reg=json.loads((ROOT/'design/registry.json').read_text());freeze=json.loads((ROOT/'state/design_freeze.json').read_text())
    assert sha(ROOT/'design/registry.json')==freeze['registry_sha256'];reg['renderer']['selected_rig']=freeze['selected_rig']
    family=next(f for f in reg['families'] if f['family_id']==family_id);assert family['split'] in ('DEVELOPMENT','TRAIN','KIN-VAL')
    assert not family.get('exclusion'),'excluded geometry cannot be rendered'
    folder=ROOT/'proposals'/family_id;assert not (folder/'observation.json').exists()
    write(folder/'render_started.json',dict(source_sha256=sha(__file__),family=family_id,design_sha256=sha(ROOT/'state/design_freeze.json')))
    cache={};rows=[];renderer=module.Renderer(reg,family)
    try:
        for seq in [s for s in reg['sequences'] if s['family_id']==family_id]:
            path=folder/seq['sequence_id']/'geometry.npz'
            with np.load(path) as z:a={k:z[k].copy() for k in z.files}
            assert np.isfinite(a['joints']).all()
            indices=sorted(set(a['active_indices'].tolist())|{round(t/.05) for t in reg['renderer']['calibration_times'][seq['task']]})
            observations=[]
            for i in indices:
                q=a['joints'][i];t=float(a['synthetic_reference_time'][i]);key=hashlib.sha256(q.tobytes()).hexdigest()
                if key not in cache:
                    assert len(list((ROOT/'proposals').glob('*/frames/*.npz')))<(117866 if ROOT.name=='R2' else 120000)
                    info,_=save_packet(folder/'frames',key,renderer,q,reg,dict(family_id=family_id,rig_index=freeze['selected_rig'],
                        sequence=seq['sequence_id'],task=seq['task'],command=seq['command'],synthetic_reference_time=t,
                        geometry_index=i,geometry_sha256=sha(path),requested_position=a['desired_position'][i].tolist(),requested_rotation=a['desired_rotation'][i].tolist()))
                    cache[key]=info
                info=cache[key]
                # Contour sector requirement is only at each sequence's common start.
                from pi0259_observation import visibility
                with np.load(folder/'frames'/(key+'.npz')) as z:
                    frames={('body' if k=='body_rgb' else 'wrist' if k=='wrist_rgb' else k):z[k] for k in z.files}
                    check=visibility(frames,renderer.scene_ids,seq['task'],seq['command'],common_start=t==0)
                    assert np.array_equal(z['arm_joints'],q) and np.max(abs(z['FK_world']-a['FK_world'][i]))<1e-9
                observations.append(dict(index=i,time=t,frame=key,array_sha256=info['array_sha256'],check=check,
                    active=t<reg['experts'][seq['task']]['motion_end_s'],same_state_reuse=info['sequence']!=seq['sequence_id'] or info['geometry_index']!=i))
            active=[r for r in observations if r['active']];end=reg['experts'][seq['task']]['motion_end_s']
            good=lambda r:r['check']['pass_necessary'];fraction=sum(map(good,active))/len(active)
            quarters=[any(good(r) and k*end/4<=r['time']<(k+1)*end/4 for r in active) for k in range(4)]
            required=[r for r in observations if any(abs(r['time']-t)<1e-8 for t in reg['renderer']['calibration_times'][seq['task']])]
            result=dict(sequence=seq['sequence_id'],rows=observations,fraction=fraction,quartiles=quarters,
                status='PASS' if fraction>=.95 and all(quarters) and all(map(good,required)) else 'VALID_OBSERVATION_FAIL')
            write(folder/seq['sequence_id']/'observations.json',result);rows.append(result)
        start_frames={r['rows'][0]['frame'] for r in rows};assert len(start_frames)==1,'paired common state/input differs'
        write(folder/'observation.json',dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'VALID_OBSERVATION_FAIL',
            sequences=[dict(sequence=r['sequence'],status=r['status'],fraction=r['fraction'],quartiles=r['quartiles']) for r in rows],
            unique_image_pairs=len(cache),shared_start_frame=next(iter(start_frames)),physics_steps=0))
    finally:renderer.close()


if __name__=='__main__':run(sys.argv[1])
