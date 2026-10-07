"""CPU readout of the six frozen fixtures; no launch or controller changes."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0281')


def read(version,name):
    out=ROOT/'wbc'/(version+'_'+name)
    if version=='repair' and (ROOT/'wbc/REPAIR_INSTALLATION_DEFECT.json').exists():
        out=ROOT/'wbc'/(version+'_'+name+'-engineering-001')
    if not (out/'RESULT.json').exists():return dict(status='DEPENDENCY_BLOCKER',gate=None)
    result=json.loads((out/'RESULT.json').read_text());rows=json.loads((out/'rows.json').read_text())
    if version=='repair':
        binding=json.loads((out/'REPAIR_RUNTIME_BINDING.json').read_text())
        assert binding['bound'] and binding['calls']>0,'Repair not consumed by actual runtime'
    if result['status'] not in ('WBC_FIXTURE_EXECUTED','VALID_NATIVE_TERMINATION','VALID_GATE_REJECT'):
        return dict(status='EXECUTOR_BLOCKER',gate=None,result=result)
    origin=result.get('initialization_s',0.);tail=[r for r in rows if r['time']>=origin+8-1e-7]
    metrics=None
    if tail:
        ep=np.array([np.linalg.norm(np.asarray(r['native_TCP'])-r['installed']) for r in tail])
        er=np.array([(Rotation.from_quat(r['installed_quaternion'])*Rotation.from_quat(r['native_quaternion']).inv()).magnitude() for r in tail])
        metrics=dict(samples=len(tail),position_rms_m=float(np.sqrt(np.mean(ep**2))),position_max_m=float(ep.max()),
                     orientation_rms_rad=float(np.sqrt(np.mean(er**2))),orientation_max_rad=float(er.max()))
    hard=max((max(r['hard_penetration'].values(),default=0.) for r in rows),default=0.)
    complete=result['status']=='WBC_FIXTURE_EXECUTED' and result.get('actual_elapsed_s',0.)>=9.99
    gate=bool(complete and metrics and metrics['samples']>=100 and metrics['position_rms_m']<=.02 and metrics['position_max_m']<=.04
              and metrics['orientation_rms_rad']<=.15 and metrics['orientation_max_rad']<=.30 and hard<=1e-6)
    actor_chain=[]
    layout=json.loads((out/'observation_layout.json').read_text()) if (out/'observation_layout.json').exists() else None
    if layout and (out/'actor_inputs.jsonl').exists():
        parts=layout['parts'];width=sum(parts.values());offsets={};offset=0
        for key,n in parts.items():offsets[key]=offset;offset+=n
        previewkey=next((k for k in parts if 'preview' in k),None)
        for line in (out/'actor_inputs.jsonl').read_text().splitlines():
            a=json.loads(line);state=a['state'];current=np.array(a['obs_history'][-width:])
            world=Rotation.from_quat(a['preview_quaternion']);heading=Rotation.from_quat(state['heading_quaternion'])
            pack=(heading.inv()*world).as_matrix()[:,:,:2].reshape(-1,6)
            preview_error=None
            if previewkey:
                preview=current[offsets[previewkey]:offsets[previewkey]+parts[previewkey]].reshape(-1,14)
                preview_error=float(np.max(abs(preview[:,3:9]-pack)))
            target=Rotation.from_quat(state['target_body_quaternion']).as_matrix()[:,:2].reshape(6)
            index=offsets['ee_target_rot6d_body']
            actor_chain.append(dict(tick=a['tick'],preview_pack_max_error=preview_error,
                body_target_pack_max_error=float(np.max(abs(current[index:index+6]-target))),
                native_link6_TCP_rotation_error_rad=float((Rotation.from_quat(state['native_quaternion'])*Rotation.from_quat(state['actual_link6_quaternion']).inv()).magnitude()),
                FK_same_TCP_rotation_error_rad=float((Rotation.from_matrix(state['rotation'])*Rotation.from_quat(state['native_quaternion']).inv()).magnitude())))
    return dict(status='PASS' if gate else 'FAIL',gate=gate,result=result,complete_10s=complete,final_2s=metrics,
       hard_joint_limit_penetration_max=hard,actor_chain=actor_chain,
       soft_joint_limit_negatives='Retained unchanged in per-tick native telemetry; not substituted for hard-limit gate.')


if __name__=='__main__':
    panels={version:{name:read(version,name) for name in ('hold','monotonic','return')} for version in ('native','repair')}
    eligible={v:all(r['gate'] is True for r in p.values()) for v,p in panels.items()}
    receipt=dict(panels=panels,fixture_eligible=eligible,selected_controller='repair' if eligible['repair'] else ('native' if eligible['native'] else None),
        target_expert_dependency=None,target_recorded_model_dependency=None,
        dependency_note='Null until owner-controlled downstream attempts exist; failed qualification is not a zero-success model result.')
    p=ROOT/'wbc/FIXTURE_READOUT.json';p.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(fixture_eligible=eligible,selected_controller=receipt['selected_controller'])))
