"""PI0260 camera adapter: unchanged renderer and pixel rules, new frozen design."""
import isaacgym  # Preserve the native runtime import order.
from pathlib import Path
import copy
import json
import numpy as np
import pi0259_renderer as renderer_module
from pi0259_calibration_render import save_packet
from pi0255_motion3 import sha,write

ROOT=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0260')
# The inherited renderer writes private link views only under this isolated root.
renderer_module.ROOT=ROOT


def run():
    reg=json.loads((ROOT/'design/registry.json').read_text());output=ROOT/'calibration_rigs'
    assert not output.exists(),'first-valid calibration already started'
    output.mkdir();write(output/'started.json',dict(source_sha256=sha(__file__),registry_sha256=sha(ROOT/'design/registry.json')))
    arrays={}
    for s in reg['calibration_sequences']:
        p=ROOT/'calibration'/s['sequence_id']/'geometry.npz'
        with np.load(p) as a:arrays[s['sequence_id']]={k:a[k].copy() for k in a.files}
    complete=[f['family_id'] for f in reg['calibration_families'] if all(
        np.all(arrays[s['sequence_id']]['sample_qualified']) for s in reg['calibration_sequences'] if s['family_id']==f['family_id'])]
    assert len(complete)>=4,('INSUFFICIENT_PAIRED_OLD_GEOMETRY',complete)
    results=[];pairs=0;selected=None
    for rig in range(3):
        configured=copy.deepcopy(reg);configured['renderer']['selected_rig']=rig;rigroot=output/('rig%d'%rig)
        starts=[];later=[];missing=[]
        for family in reg['calibration_families']:
            renderer=renderer_module.Renderer(configured,family)
            try:
                folder=rigroot/family['family_id']
                metadata=dict(rig_index=rig,family_id=family['family_id'],synthetic_reference_time=0.,role='COMMON_START_ALL_TASKS_COMMANDS')
                info,frames=save_packet(folder,'common_start',renderer,np.array(reg['arm_default']),configured,metadata)
                _,again=save_packet(folder,'common_start_repeat',renderer,np.array(reg['arm_default']),configured,
                    dict(metadata,role='PAIR_INPUT_REPRODUCTION'))
                pairs+=2;assert all(np.array_equal(frames[k],again[k]) for k in frames)
                starts.append(dict(family_id=family['family_id'],checks={t:info['visibility'][t] for t in ('M1','M3')}))
                for seq in [s for s in reg['calibration_sequences'] if s['family_id']==family['family_id']]:
                    case=seq['sequence_id'];data=arrays[case];geometry=ROOT/'calibration'/case/'geometry.npz'
                    for t in reg['renderer']['calibration_times'][seq['task']][1:]:
                        i=int(round(t/.05));row=dict(sequence=case,task=seq['task'],command=seq['command'],time=t)
                        if not data['sample_qualified'][i]:
                            missing.append(dict(row,reason='NO_ACCEPTED_OLD_GEOMETRY',camera_measurement='EXCLUDED_NEITHER_PASS_NOR_FAIL'));continue
                        info,_=save_packet(folder,case+'-t%04d'%i,renderer,data['joints'][i],configured,
                            dict(rig_index=rig,family_id=family['family_id'],sequence=case,task=seq['task'],command=seq['command'],
                                synthetic_reference_time=t,geometry_sha256=sha(geometry),geometry_index=i,
                                requested_position=data['desired_position'][i].tolist(),requested_rotation=data['desired_rotation'][i].tolist()))
                        pairs+=1;assert pairs<=512
                        later.append(dict(row,check=info['visibility'][seq['task']][str(seq['command'])]))
            finally:renderer.close()
        start_ok=all(x['checks'][t][str(c)]['pass_necessary'] for x in starts for t in ('M1','M3') for c in (0,1))
        later_ok=bool(later) and all(x['check']['pass_necessary'] for x in later)
        outcome=dict(rig_index=rig,starts=starts,later=later,missing_geometry=missing,
            complete_paired_family_ids=complete,start_pass=start_ok,later_pass=later_ok,qualified=start_ok and later_ok,pairs_so_far=pairs)
        results.append(outcome);write(rigroot/'outcome.json',outcome)
        print(json.dumps(dict(rig=rig,start_pass=start_ok,later_pass=later_ok,pairs=pairs)),flush=True)
        if outcome['qualified']:selected=rig;break
    write(ROOT/'evidence/camera_calibration.json',dict(schema='PI0260_CAMERA_CALIBRATION_V1',rigs=results,
        selected_rig=selected,image_pairs=pairs,status='H0_REVIEW_PENDING' if selected is not None else 'PI0260_CAMERA_DESIGN_BLOCKED',
        complete_paired_family_ids=complete,new_IK_calls=0,physics_steps=0,model_queries=0,
        missing_geometry_rule='excluded from camera measurement only; not eligible for future data'))


if __name__=='__main__':run()
