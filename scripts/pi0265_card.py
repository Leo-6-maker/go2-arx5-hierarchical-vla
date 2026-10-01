"""Freeze new geometry/reset/scenario cards and actual inference noise before physics."""
import copy
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from pi0265_source import ROOT, BASE, sha, write

NODE=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5')
AUTHORITY='edec521de2087d0ab96abfa8a13afc79d108912c'
KEYS=('x_shift','y_shift','z_shift','radius')


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'))


def geometry_key(f,reg):
    center=f.get('center_world')
    if center is None:center=np.asarray(reg['geometry']['nominal_center_world'])+[f['factors'][k] for k in KEYS[:3]]
    return tuple(round(float(x),10) for x in [*center,f['factors']['radius']])


def main():
    path=ROOT/'design/SCENE_CARD.json';assert not path.exists(),'already frozen: inspect instead of regenerating'
    source=NODE/'PI0260/design/registry.json';reg=copy.deepcopy(json.loads(source.read_text()))
    freeze=json.loads((NODE/'PI0260/state/design_freeze.json').read_text())
    assert freeze['registry_sha256']==sha(source)
    reg['renderer']['selected_rig']=freeze['selected_rig']
    prior_paths=[BASE/'PI0255/design/task_registry_v1.json',NODE/'PI0258/design/registry.json',
        NODE/'PI0259/design/registry.json',source,NODE/'PI0260/R2/design/registry.json']
    prior=[];excluded=set()
    for old_path in prior_paths:
        assert old_path.is_file(),str(old_path)
        old=json.loads(old_path.read_text());values=[]
        for f in old.get('families',[])+old.get('calibration_families',[]):
            key=geometry_key(f,old);excluded.add(key)
            values.append(dict(family_id=f['family_id'],split=f['split'],key=key))
        prior.append(dict(path=str(old_path),sha256=sha(old_path),families=values,metadata_only=True))
    proposals=[]
    for xyz in itertools.product((-.03,-.01,.01,.03),(-.03,-.01,.01,.03),(.0075,.0225),(.04,.05)):
        factors=dict(zip(KEYS,xyz));f=dict(factors=factors)
        key=geometry_key(f,reg);proposals.append(dict(factors=factors,key=key,excluded=key in excluded,
            order_sha256=hashlib.sha256(('PI0265:26501:scene:'+canonical(factors)).encode()).hexdigest()))
    eligible=sorted((p for p in proposals if not p['excluded']),key=lambda p:p['order_sha256'])
    assert len(eligible)>=36,('ELIGIBLE_SCENE_SHORTFALL',len(eligible))
    colors=list(itertools.permutations(('red','green','blue'),2));families=[];ranks={}
    for index,p in enumerate(eligible):
        split='CONTROLLER_DEV' if index<4 else 'FEEDBACK_TRAIN' if index<28 else 'NEW_SIM_EVAL' if index<36 else 'UNUSED'
        rank=ranks.get(split,0);ranks[split]=rank+1;fid='pi0265-family-%03d'%index
        seed=int.from_bytes(hashlib.sha256(('PI0265:RESET:26531:'+fid).encode()).digest()[:8],'big')%(2**31)
        factors=dict(p['factors'],colors=list(colors[rank%6]))
        families.append(dict(family_id=fid,split=split,rank=rank,factors=factors,reset_seed=seed,
            center_world=(np.asarray(reg['geometry']['nominal_center_world'])+[factors[k] for k in KEYS[:3]]).tolist(),
            geometry_key=p['key'],source_order_sha256=p['order_sha256']))
    cases=[]
    for version in ('LEGACY','ROW6D'):
        for route in ('DIRECT_NATIVE','REFERENCE_PROVIDER'):
            cases.append(dict(case_id='C0-'+version+'-'+route,stage='C0',version=version,route=route,
                              task='C0',command=0,reset_seed=22301,max_task_s=5.,execute=True))
    for version in ('LEGACY','ROW6D'):
        for family in families[:2]:
            for task in ('M1','M3'):
                for command in (0,1):
                    cases.append(dict(case_id=family['family_id']+'-'+task+'-c'+str(command)+'-'+version,
                        stage='DEV',version=version,route='REFERENCE_PROVIDER',family=family,
                        task=task,command=command,reset_seed=family['reset_seed'],max_task_s=20.,execute=True))
    reg.update(authority=AUTHORITY,schema='PI0265_NEW_LOCAL_GEOMETRY_CARD_V1',families=families,sequences=[],
        state_origin='ACTUAL_WBC_MEASUREMENT',selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')
    card=dict(schema='PI0265_BEFORE_PHYSICS_SCENE_RESET_CARD_V1',authority=AUTHORITY,source=prior,
        source_registry_sha256=sha(source),sensor_design_sha256=sha(NODE/'PI0260/state/design_freeze.json'),
        proposals=proposals,families=families,registry=reg,G1_initial_cases=cases,
        confirmation_rule='third DEV only:4 cases chosen by registered qualification; fourth never executed',
        reset=dict(normal_native_reset=True,source='PINNED_LEGGED_ROBOT_RESET_FUNCTIONS',initialization_max_s=1.,
            root_DOF_generator='actual pinned defaults and native random reset; exact seed registered, actual bytes retained before first physics',
            task_anchor='measured reset FK; reference formula fixed before outcomes; no robot teleport',
            target_axis_orientation='source reference R0 is measured reset FK; cue axes use that same R0 @ original yaw, bound once before first physics; camera unchanged'),
        source_reference='PI0259_KIN_REF_V2',motion={'M1':6.,'M3_approach':4.,'M3_arc':12.,'horizon':20.},
        physics_starts_before_freeze=0,model_queries_before_freeze=0)
    write(path,card)
    noise=[]
    for f in families[28:36]:
        for task in ('M1','M3'):
            for name in ('N0','N1'):
                for index in range(50):
                    key='PI0265:EVAL:26541:%s:%s:%s:%d'%(f['family_id'],task,name,index)
                    seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big')
                    generator=torch.Generator(device='cpu').manual_seed(seed)
                    packet=torch.randn((1,40,132),dtype=torch.float32,generator=generator)
                    relative='noise/%s-%s-%s-%02d.npz'%(f['family_id'],task,name,index)
                    target=ROOT/relative;target.parent.mkdir(exist_ok=True)
                    np.savez_compressed(target,epsilon_fp32=packet.numpy(),epsilon_bf16_uint16=packet.to(torch.bfloat16).view(torch.uint16).numpy())
                    noise.append(dict(key=key,seed=seed,file=relative,sha256=sha(target),shape=[1,40,132],
                        generator='torch CPU standard normal FP32; actual BF16 bits retained',command_and_model_independent=True))
    write(ROOT/'design/NOISE_CARD.json',dict(status='ACTUAL_1600_PACKETS_SEALED_BEFORE_PHYSICS',rows=noise,
        SCENE_CARD_sha256=sha(path),training_namespace='PI0265:TRAIN:26511:<unit>; mature conversion unchanged'))
    print(json.dumps(dict(status='PASS_FIXED_SCENES_RESET_NOISE',survivors=len(eligible),splits=ranks,
        fixed_initial_G1_cases=len(cases),actual_noise_packets=len(noise),scene_sha256=sha(path),noise_sha256=sha(ROOT/'design/NOISE_CARD.json'))))


if __name__=='__main__':
    main()
