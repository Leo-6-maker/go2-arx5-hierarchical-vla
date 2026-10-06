"""Package existing local policy identities and measured results, without new calls."""
import argparse,json,shlex
from pathlib import Path
from evaluation_card import ROOT,RAW,read,sha,write

def handoff(point):
    cpu=read(ROOT/'evidence'/f'CPU_REPLAY_{point}_TRAIN_PHYSICS.json')
    assert cpu['status']=='PASS_SAVED_S4_DECODE_FRESH_INPUT_NATIVE_PREDICATES'
    binding=read(ROOT/'model_bindings'/f'{point}.json');view=Path(binding['view'])
    ready=read(view/'READY.json');assert ready['point']==point and not ready['resumable']
    # Full readback and real model loading are already in admission/panel receipts.
    for r in ready['files']:assert (view/r['path']).stat().st_size==r['bytes']
    admission=ROOT/'evidence'/f'POLICY_ADMISSION_{point}.json'
    assert read(admission)['status']=='PASS_PI0277_FIXED_POLICY_IDENTITY'
    evidence=[]
    for p in sorted((ROOT/'evidence').glob('CPU_REPLAY_*_*.json')):
        v=read(p);assert v['status']=='PASS_SAVED_S4_DECODE_FRESH_INPUT_NATIVE_PREDICATES'
        evidence.append(dict(point=v['point'],lane=v['lane'],path=str(p),sha256=sha(p),task=v['task']))
    dev=next((e for e in evidence if e['point']==point and e['lane']=='TEST'),None)
    confirm=next((e for e in evidence if e['point']==point and e['lane']=='CONFIRM'),None)
    qualified=cpu['task']['persistent_stacks']>=3 and dev is not None and dev['task']['persistent_stacks']>=6
    status='SOURCE_PICKPLACE_PILOT_CONFIRMED' if qualified and confirm is not None and confirm['task']['persistent_stacks']>=12 else ('LOCAL_DEV_PICKPLACE_PROTOTYPE' if qualified else 'NOT_USABLE')
    fit=read(ROOT/'evidence'/f'FIT16_{point}.json')['results'][0]
    source=Path(fit['output']);saved=read(source.with_suffix('.receipt.json'))
    assert saved['binding']==binding and saved['output_sha256']==sha(source)
    noises=read(ROOT/'design/EVALUATION_NOISE.json')['rows']
    noise=next(r for r in noises if r['lane']=='FIT' and r['condition']==fit['uid'] and r['index']==0)
    request=dict(input=str(source.with_name('input.npz')),output=str(source),noise_file=noise['file'],noise_name=noise['name'],noise_sha256=noise['sha256'])
    target=ROOT/'handoff'/point;write(target/'INPUT_REQUEST.json',request)
    python=ROOT.parent.parent/'PI0250/runtime/venv/bin/python'
    command='CUDA_VISIBLE_DEVICES=0 OPENBLAS_CORETYPE=Prescott OMP_NUM_THREADS=4 '+shlex.join([str(python),'-u',str(ROOT/'code/infer.py'),'--point',point])+' < '+shlex.quote(str(target/'INPUT_REQUEST.json'))
    videos=read(ROOT/'evidence'/f'VIDEOS_{point}_TRAIN_PHYSICS.json')
    package=dict(status=status,handoff_model=point,view=str(view),view_type='MODEL_ONLY_NOT_RESUME',model_ready=ready,binding=binding,
        policy_admission=dict(path=str(admission),sha256=sha(admission)),runtime_fileset=read(ROOT/'design/EVALUATION_FILESET.json'),
        inference_command=command,input_example=request,example_behavior='Reload policy and reuse one existing first-valid FIT output; zero new queries',
        input_schema=dict(body_rgb='uint8[96,96,3]',wrist_rgb='uint8[96,96,3]',state_v1='float32[10], measured world TCP/row6D/opening',language='Stack the red cube on the green cube and release it.'),
        output_schema='Original FP32 physical H40x10 EEF/gripper decode; install only through unchanged native gate/controller',
        selection_trail=evidence,saved_RGB_videos=videos,task_success_example='AVAILABLE' if any(r['persistent_success'] for r in cpu['task']['rows']) else 'NONE_IN_THIS_MEASURED_PANEL',
        scope='ManiSkill StackCube source-Panda simulation; non-real-time inference pauses; not Go2/WBC or hardware/generalization qualification',
        runtime='S4 BF16 quality query; FP32 physical decode; 100Hz physics,20Hz controls,0.4s replan,full32s native episodes',
        archive='DEFERRED_USER_PRIORITY_LOCAL_ORIGINALS_PRESERVED',selected_checkpoint='NONE',deployment='NONE',
        new_model_queries=0,new_physics_steps=0)
    write(target/'POLICY_HANDOFF.json',package);(target/'RUN_INFERENCE.sh').write_text('#!/bin/sh\nset -eu\n'+command+'\n')
    print(json.dumps(dict(path=str(target),status=status,point=point,new_model_queries=0,new_physics_steps=0)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('point');handoff(p.parse_args().point)
