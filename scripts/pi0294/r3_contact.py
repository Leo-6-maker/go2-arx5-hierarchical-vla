"""Saved contact identity first; reconstructed fields are not actual old forces."""
import argparse,hashlib,json,os,resource,sys,time
from pathlib import Path
if not __package__:
    sys.path[:0]=[str(Path(__file__).resolve().parents[2]),str(Path(__file__).resolve().parents[2]/'src')]
    __package__='scripts.pi0294'
from scripts.pi0292.durable import OwnerLock,read,write
from .continuation import validate_entry,RUN
from .gripper_safety import saved_inputs,create_original


def saved_contacts(root,output,binding):
    import mujoco
    import numpy as np
    card,initial,old,nq,nv,commands=saved_inputs(binding)
    engine=create_original(card,initial,binding)
    def annotate(c):
        names=[mujoco.mj_id2name(engine.model,mujoco.mjtObj.mjOBJ_GEOM,c[k]) for k in ('geom0','geom1')]
        bodies=[mujoco.mj_id2name(engine.model,mujoco.mjtObj.mjOBJ_BODY,int(engine.model.geom_bodyid[c[k]])) for k in ('geom0','geom1')]
        return dict(c,geom_names=names,body_names=bodies)
    try:
        retired=read(Path(binding['R2_root'])/'evidence/G2-failure_prefix_hold-R2_VEL_C01/RESULT.json')
        observed=[annotate(c) for c in retired['final_contacts']]
        path=Path(binding['R2_root'])/'evidence/G1-B/actual_replay.npz'
        with np.load(path,allow_pickle=False) as f:row=f['rows'][-1]
        engine.data.qpos[:]=row[1:1+nq];engine.data.qvel[:]=row[1+nq:1+nq+nv]
        engine.data.ctrl[:]=row[1+nq+nv:1+nq+nv+engine.model.nu];engine.data.time=row[0]
        mujoco.mj_forward(engine.model,engine.data)
        _,_,_,illegal,contacts=engine.contacts()
        result=dict(run_uuid=RUN,status='SAVED_CONTACT_IDENTITY_AND_RECONSTRUCTED_B_COMPLETE',
            original_model_xml_sha256=initial['model_xml_sha256'],
            retired_candidate_final_contacts_observed=observed,
            G1_B=dict(time_s=float(row[0]),recorded_safety_flag=float(row[-1]),
                contacts_RECONSTRUCTED=[annotate(c) for c in contacts],illegal_RECONSTRUCTED=illegal,
                original_warmstart='UNOBSERVED',original_contact_force='UNOBSERVED',
                replay_npz_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
            model_constructions=1,reconstruction_forward_calls=1,physics_integrations=0,RGB_pairs=0,
            actual_contact_force_causal_claim=False,selected='NONE',deployment='NONE')
        write(output/'CONTACT_EVENT_ATLAS_INITIAL.json',result)
        return result
    finally:engine.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root);validate_entry(root,profile='4080-simulation')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU_ONLY')
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    owner=OwnerLock(root/'CONTACT_OWNER.lock');output=root/'evidence/contact-A0'
    output.mkdir(exist_ok=False);started=time.process_time()
    result=None
    try:result=saved_contacts(root,output,read(root/'INPUT_BINDINGS.private.json'))
    finally:
        cpu=time.process_time()-started;ledger=read(root/'RESOURCE_LEDGER.json')
        ledger['Core_CPU_s']+=cpu;ledger['stages'].append(dict(phase='contact-A0',CPU_s=cpu,status='PASS_SAVED_CONTACT_READ' if result else 'ENGINEERING_FAILURE_PRESERVED'))
        write(root/'RESOURCE_LEDGER.json',ledger);owner.close()
    print(json.dumps(dict(status=result['status'],observed_contacts=result['retired_candidate_final_contacts_observed'],reconstructed_B=result['G1_B'],CPU_s=cpu)))


if __name__=='__main__':main()
