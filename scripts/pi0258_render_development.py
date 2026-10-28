"""Render the remaining fixed development references, retaining all raw QA outcomes."""
import isaacgym
from pathlib import Path
import json
import sys
import numpy as np
from pi0258_renderer import Renderer,ROOT
from pi0255_motion3 import sha,write
from go2arx5_vla.contracts import TrajectoryChunkV1
from go2arx5_vla.safety import TrajectorySafetyGateV1,GateDecision
from go2arx5_vla.trajectory.so3 import matrix_to_quat


def run():
    reg=json.loads((ROOT/'design/registry.json').read_text());seal=json.loads((ROOT/'design/development_render_seal.json').read_text())
    assert seal['source_sha256']==sha(__file__) and seal['canonical_comment'] and seal['commit']
    geometry=json.loads((ROOT/'evidence/development_geometry_summary.json').read_text())
    generation=json.loads((ROOT/'design/generation_seal.json').read_text())
    for name,digest in generation['source_files'].items():assert sha(ROOT/'runtime'/name)==digest,name
    families={f['family_id']:f for f in reg['families']};sequences={s['sequence_id']:s for s in reg['sequences']}
    anchor=np.asarray(reg['geometry']['nominal_tcp_world']);gate=TrajectorySafetyGateV1();results=[]
    for summary in geometry['sequences']:
        if not summary.get('geometry_pass',False):continue
        seq=sequences[summary['sequence_id']];case=seq['sequence_id'];family=families[seq['family_id']]
        folder=ROOT/'development'/seq['family_id'];out=folder/(case+'-render.json')
        if out.exists():results.append(json.loads(out.read_text()));continue
        path=folder/(case+'-geometry.npz');assert sha(path)==summary['payload_sha256']
        with np.load(path) as a:data={k:a[k].copy() for k in a.files}
        renderer=Renderer(reg,family);body=[];wrist=[];states=[];readback=[];gates=[];eligible=[]
        try:
            for i in data['observation_indices']:
                frames,info=renderer.place(data['joints'][i]);body.append(frames['body']);wrist.append(frames['wrist'])
                pose=data['FK_world'][i];assert np.max(abs(np.array(info['FK_TCP_world'])-pose[:3,3]))<1e-5
                state=np.r_[pose[:3,3]-anchor,pose[:2,:3].reshape(-1),.5];states.append(state)
                if i==0:
                    with np.load(folder/'common_start.npz') as previous:
                        assert np.array_equal(frames['body'],previous['body_rgb'] if 'body_rgb' in previous else previous['body'])
                        assert np.array_equal(frames['wrist'],previous['wrist_rgb'] if 'wrist_rgb' in previous else previous['wrist'])
                    assert np.max(abs(state[:3]))<1e-12
                required=generation['visibility_required'][seq['task']][str(seq['command'])]
                missing=[n for n in required if sum(info['visible_pixels'][n].values())==0]
                info.update(synthetic_reference_time=float(data['synthetic_reference_time'][i]),missing_command_cues=missing)
                readback.append(info)
                if i not in data['active_indices']:continue
                t=float(data['synthetic_reference_time'][i]);indices=np.minimum(i+np.arange(40),len(data['joints'])-1)
                chunk=TrajectoryChunkV1(case+'-'+str(i),'KINEMATIC_RENDERED_REFERENCE',t,t+np.arange(40)*.05,
                    data['desired_position'][indices]-anchor,data['desired_quaternion'][indices],
                    'NATIVE_WBC_RELATIVE_REFERENCE','x5_link6_gripper_center',t,t+1.95,gripper=np.full(40,.5))
                verdict=gate.evaluate(chunk,now=t,achieved_position=state[:3],achieved_orientation=matrix_to_quat(pose[:3,:3]))
                gates.append(dict(index=int(i),verdict=verdict.as_dict()))
                geometry_ok=bool(data['active_geometry_qualified'][np.flatnonzero(data['active_indices']==i)[0]])
                eligible.append(geometry_ok and not missing and verdict.decision is not GateDecision.REJECT)
            payload=folder/(case+'-render.npz')
            np.savez_compressed(payload,body_rgb=np.asarray(body),wrist_rgb=np.asarray(wrist),state_v1=np.asarray(states),
                synthetic_reference_time=data['synthetic_reference_time'][data['observation_indices']],
                T_world_native=np.block([[np.eye(3),anchor[:,None]],[np.zeros((1,3)),np.ones((1,1))]]))
            write(folder/(case+'-render-readback.json'),readback);write(folder/(case+'-raw-gate.json'),gates)
            fraction=float(np.mean(eligible));times=data['synthetic_reference_time'][data['active_indices']];end=reg['experts'][seq['task']]['motion_end_s']
            quartiles=[bool(np.any(np.asarray(eligible)[(times>=q*end/4)&(times<(q+1)*end/4)])) for q in range(4)]
            row=dict(sequence_id=case,task=seq['task'],frames=len(body),active_windows=len(eligible),
                missing_cue_windows=sum(bool(x['missing_command_cues']) for x in readback[:len(eligible)]),
                raw_gate_rejects=sum(x['verdict']['decision']=='REJECT' for x in gates),
                geometry_visibility_raw_gate_fraction=fraction,quartile_coverage=quartiles,
                geometry_visibility_raw_gate_pass=fraction>=.95 and all(quartiles),qualified=False,
                processor='NOT_BUILT_NO_TRAIN_DATA_ADMISSION',render_sha256=sha(payload),physics_steps=0,
                state_origin='KINEMATIC_FK_NOT_DYNAMIC_MEASUREMENT',state_frame='WORLD_ALIGNED_WITH_IMMUTABLE_NOMINAL_TCP_TRANSLATION_ANCHOR')
            write(out,row);results.append(row);print(json.dumps(row),flush=True)
        finally:renderer.close()
    write(ROOT/'evidence/development_render_summary.json',dict(sequences=results,physics_steps=0,GR00T=0,
        state_frame_note='Original preflight common_start.npz state_v1 contains world FK position. Retained unchanged as raw geometry evidence, not model input. Full render export explicitly subtracts immutable nominal TCP anchor and retains T_world_native.'))


if __name__=='__main__':run()
