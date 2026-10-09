"""Complete native-label roundtrip and real language/two-worker CPU admission."""
import functools,json,os
from pathlib import Path
from contract import ROOT,sha,write


def check_inputs(root,samples=None):
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    import numpy as np,torch
    from data import Dataset,codec
    from cpu_inputs import tensor_hash
    root=Path(root);torch.set_num_threads(4);bank=json.loads((root/'SOURCE_BANK.json').read_text());encoder=codec(str(root))
    if samples is None:
        samples=[];seen=set()
        for ep in bank['admitted_episodes']:
            key=(ep['task'],ep['split'])
            if key in seen:continue
            seen.add(key);name=ep['episode'];phase=np.load(root/'action_target'/name/'phase.npy',mmap_mode='r')
            start=[int(x) for x in np.unique(phase[20:621],return_index=True)[1]+20]
            for a in sorted(set(start+[620])):samples.append(dict(index=len(samples),episode=name,anchor=a,task=ep['task'],split=ep['split'],instruction=__import__('contract').INSTRUCTIONS[ep['task']],uid=f'{name}-{a}'))
    path=root/'CPU_INPUT_CHECK_PLAN.jsonl';payload=''.join(json.dumps(r,sort_keys=True)+'\n' for r in samples)
    if path.exists():assert path.read_text()==payload
    else:path.write_text(payload)
    dataset=Dataset(root,path);features=[];checks=[]
    for i,row in enumerate(samples):
        raw=dataset.raw(row);feature=encoder.feature(*raw,row['instruction']);inputs=encoder.batch([feature])['inputs'];features.append(feature)
        expected=encoder.observation(raw[0],raw[1],raw[2],row['instruction']);shared=set(inputs)-{'action','action_mask'}
        assert shared==set(expected)-{'action_mask'}
        for key in shared:assert torch.equal(inputs[key],expected[key]),key
        assert tuple(inputs['action'].shape)==(1,40,132) and bool((inputs['action_mask'][:,:,:10]==1).all()) and not torch.count_nonzero(inputs['action_mask'][:,:,10:])
        checks.append(dict(uid=row['uid'],task=row['task'],instruction=row['instruction'],input_sha256=tensor_hash(inputs),tokens=inputs['input_ids'][0][inputs['attention_mask'][0].bool()].tolist()))
    # Native collation must own language padding; never concatenate fixed-text tensors.
    batches=[]
    for start in range(0,len(features),4):
        batch=encoder.batch(features[start:start+4])['inputs'];batches.append(batch)
        for j,row in enumerate(checks[start:start+4]):
            assert batch['input_ids'][j][batch['attention_mask'][j].bool()].tolist()==row['tokens']
    if 'T2' in bank['tasks']:
        assert next(r['tokens'] for r in checks if r['task']=='T1')!=next(r['tokens'] for r in checks if r['task']=='T2')
        assert len(next(r['tokens'] for r in checks if r['task']=='T1'))!=len(next(r['tokens'] for r in checks if r['task']=='T2'))
        instructions=__import__('contract').INSTRUCTIONS
        assert encoder.cache_key('T1',instructions['T1'],'paired-source')!=encoder.cache_key('T2',instructions['T2'],'paired-source')
    for split in ('TRAIN','DEV'):
        paired=[r for r in samples if r['split']==split and r['anchor']==20]
        if len(paired)==2:
            raw=[dataset.raw(r) for r in paired]
            assert all(np.array_equal(raw[0][i],raw[1][i]) for i in range(3)),'Paired initial RGB/state differ'
            for key in ('state','state_mask','images'):
                if key in features[paired[0]['index']]:assert np.array_equal(features[paired[0]['index']][key],features[paired[1]['index']][key]),key
    # A module-level collator is spawn-safe and shares the worker's native codec.
    from data import collate
    loader=torch.utils.data.DataLoader(dataset,batch_size=4,num_workers=2,multiprocessing_context='spawn',collate_fn=functools.partial(collate,root=str(root)),shuffle=False,generator=torch.Generator().manual_seed(28203))
    worker=[]
    for i,batch in enumerate(loader):
        inputs=batch['inputs'];assert tensor_hash(inputs)==tensor_hash(batches[i])
        assert [r['index'] for r in batch['_pi0282_visits']]==list(range(i*4,min((i+1)*4,len(samples))))
        worker.append(tensor_hash(inputs))
    bindings={n:sha(root/n) for n in ('SOURCE_BANK.json','STATISTICS_RELEASE.json','processor_config.json','statistics.json','embodiment_id.json','PLACE4096.jsonl','SHARED4096.jsonl','WINDOW_INDEX.json')}
    value=dict(status='PASS_ACTUAL_NATIVE_INPUTS_ONLY_NOT_CODEC_ADMISSION',bindings=bindings,inputs=checks,two_worker_native_batch_hashes=worker,variable_language_native_padding=True,paired_initial_RGB_state_checked=True,leaked_policy_fields=[],physics=0,neural_queries=0,optimizer_updates=0,TRAIN_statistics_only=True,launch_allowed=False)
    write(root/'NATIVE_INPUTS_CHECK.json',value);return value


def check(root):
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    import numpy as np,torch
    from data import Dataset,codec
    from cpu_inputs import physical_errors,tensor_hash
    root=Path(root);torch.set_num_threads(4);bank=json.loads((root/'SOURCE_BANK.json').read_text());encoder=codec(str(root))
    samples=[];roundtrip=[];maxima=dict(XYZ_max_m=0.,SO3_max_rad=0.,gripper_max_fraction=0.);clamps=0;values=0
    for ep in bank['admitted_episodes']:
        name=ep['episode'];states=np.load(root/'policy_observation'/name/'state_v1.npy',mmap_mode='r');commands=np.load(root/'action_target'/name/'action.npy',mmap_mode='r')
        start=[int(x) for x in np.unique(np.load(root/'action_target'/name/'phase.npy',mmap_mode='r')[20:621],return_index=True)[1]+20]
        anchors=range(20,621) if ep['split']=='TRAIN' else sorted(set(start+[620]))
        worst=dict(XYZ_max_m=0.,SO3_max_rad=0.,gripper_max_fraction=0.)
        for a in anchors:
            target=commands[a:a+40].copy();state=states[a].copy();normalized=encoder.encode_actions(target,state);decoded=encoder.decode_actions(normalized,state)
            error=physical_errors(target,decoded);assert not error['nonfinite'] and not error['decoded_rot6d_validity']['degenerate_count']
            for key in maxima:worst[key]=max(worst[key],error[key]);maxima[key]=max(maxima[key],error[key])
            audit=encoder.action_audit(target,state);clamps+=audit['native_preclip_out_of_range_count'];values+=audit['active_values']
            assert error['XYZ_max_m']<=1e-5 and error['SO3_max_rad']<=1e-4 and error['gripper_max_fraction']<=1e-5,(name,a,error)
        roundtrip.append(dict(episode=name,split=ep['split'],labels_tested=len(anchors),maxima=worst))
        # One real episode per task/split exercises every native input and text path.
        if not any(r['task']==ep['task'] and r['split']==ep['split'] for r in samples):
            for a in sorted(set(start+[620])):samples.append(dict(index=len(samples),episode=name,anchor=a,task=ep['task'],split=ep['split'],instruction=__import__('contract').INSTRUCTIONS[ep['task']],uid=f'{name}-{a}'))
    inputs=check_inputs(root,samples);bindings=inputs['bindings'];checks=inputs['inputs'];worker=inputs['two_worker_native_batch_hashes']
    value=dict(status='PASS_NEW_COMMON_SCENE_FULL_NATIVE_CODEC_AND_TWO_WORKERS',bindings=bindings,roundtrip=roundtrip,maxima=maxima,native_preclip_out_of_range_count=clamps,native_preclip_values=values,inputs=checks,two_worker_native_batch_hashes=worker,variable_language_native_padding=True,leaked_policy_fields=[],physics=0,neural_queries=0,optimizer_updates=0,TRAIN_statistics_only=True)
    write(root/'DATA_CODEC_ADMISSION.json',value);print(json.dumps(dict(status=value['status'],maxima=maxima,episodes=len(roundtrip),inputs=len(checks))))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT/'data'));p.add_argument('--inputs-only',action='store_true');a=p.parse_args();check_inputs(a.root) if a.inputs_only else check(a.root)
