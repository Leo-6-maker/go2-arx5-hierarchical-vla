"""Current storage/donor and complete actual processor admission; zero models."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                  OPENBLAS_CORETYPE='Prescott',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NO_ALBUMENTATIONS_UPDATE='1')
import argparse
import ctypes
import inspect
import json
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from train import ROOT,old,paired,read,sha,np,torch,Pairs,hooks
from prepare import STATS_SHA


def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);text=json.dumps(value,indent=2,sort_keys=True)+'\n'
    if p.exists():assert p.read_text()==text
    else:p.write_text(text)


def provision():
    import importlib.util
    spec=importlib.util.spec_from_file_location('pi0270_provision',ROOT.with_name('PI0270')/'code/provision.py')
    previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
    prior=read(ROOT.with_name('PI0270')/'evidence/AGG/checkpoint-4096.full_readback.json')
    assert prior['status']=='PASS_FULL_FILE_READBACK' and prior['complete_trainer']
    assert prior['revision']=='f862abb26041c8e7d355470d6e9d48899f78b9ce'
    assert prior['tree_sha256']=='0427b2be852f572ab4f3c4fa343cc56104acbbf0cca9db2e61d60abad3b15f66'
    cp=ROOT.with_name('PI0270')/'AGG/main/training/checkpoint-4096';checked=[]
    for row in prior['files']:
        p=cp/row['path'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes']
        if row['path'] in ('config.json','model.safetensors.index.json') or p.suffix=='.safetensors':
            assert previous.sha(p)==row['sha256'];checked.append(row)
    assert len(checked)==4
    import struct
    tensors={}
    for row in checked:
        if not row['path'].endswith('.safetensors'):continue
        with (cp/row['path']).open('rb') as f:
            size=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(size))
        tensors.update({k:v for k,v in header.items() if k!='__metadata__'})
    config=read(cp/'experiment_cfg/final_model_config.json')
    assert all(config[k] is False for k in ('tune_llm','tune_visual','tune_vlln'))
    assert config['tune_projector'] and config['tune_diffusion_model']
    prefixes=tuple('action_head.'+s+'.' for s in ('model','state_encoder','action_encoder','action_decoder','position_embedding'))
    scope={kind:[dict(name=n,shape=v['shape'],stored_dtype=v['dtype']) for n,v in sorted(tensors.items()) if n.startswith(prefixes)==enabled]
           for kind,enabled in (('trainable',True),('frozen',False))}
    assert len(scope['trainable'])==471 and len(scope['frozen'])==559
    write(ROOT/'DONOR_PARAMETER_SCOPE.json',scope)
    write(ROOT/'DONOR_LOCAL_READBACK.json',dict(status='PASS_PI0272_AGG4096_WEIGHT_IDENTITY',
        path=str(cp),revision=prior['revision'],tree_sha256=prior['tree_sha256'],files=prior['files'],
        complete_trainer_bytes=sum(r['bytes'] for r in prior['files']),fresh_model_hashes=checked,
        optimizer_inherited=False,source_full_readback_sha256=sha(ROOT.with_name('PI0270')/'evidence/AGG/checkpoint-4096.full_readback.json')))
    source=ROOT.with_name('PI0270')/'contracts/ROT/source/dataset/statistics.json';assert sha(source)==STATS_SHA
    target=ROOT/'contracts/ROT/source/dataset/statistics.json';target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert target.read_bytes()==source.read_bytes()
    else:target.write_bytes(source.read_bytes())
    if not (ROOT/'STORAGE_ADMISSION.json').exists():
        quota=subprocess.run(['/usr/bin/hpc','quota'],capture_output=True,text=True,timeout=30)
        text=quota.stdout+quota.stderr;q=previous.quota_bytes(text)
        assert q is not None,('USER_QUOTA_UNKNOWN',text)
        df=int(subprocess.check_output(['/usr/bin/df','-B1','--output=avail',str(ROOT)],text=True).splitlines()[-1])
        unique={}
        for p in ROOT.rglob('*'):
            if p.is_file() and not p.is_symlink():
                s=p.stat();unique[s.st_dev,s.st_ino]=s.st_size
        size=sum(r['bytes'] for r in prior['files']);required=11*size+10*2**30
        # Retain all six canary checkpoints and four formal checkpoints; one write overlap.
        assert min(q,df)>=required,(q,df,required)
        write(ROOT/'STORAGE_ADMISSION.json',dict(available_bytes_at_freeze=min(q,df),owned_bytes_at_freeze=sum(unique.values()),
            quota_report=text,parsed_quota_headroom_bytes=q,df_available_bytes=df,required_bytes=required,
            retained_canary_checkpoints=6,formal_checkpoints=4,overlap=1,reserve_bytes=10*2**30))
    print('PASS_CURRENT_DONOR_MODEL_HASHES_AND_PERSONAL_QUOTA',flush=True)


def cpu():
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(16*2**30,16*2**30));os.nice(19)
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-4:])
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    native_call=torch.nn.Module.__call__
    def guard(self,*a,**kw):
        if type(self).__module__.startswith('torchvision.transforms'):return native_call(self,*a,**kw)
        raise RuntimeError('PI0272_CPU_NO_NEURAL_QUERY')
    def forbidden(*a,**kw):raise RuntimeError('PI0272_CPU_NO_OPTIMIZER')
    torch.nn.Module.__call__=guard;torch.optim.Optimizer.__init__=forbidden
    assert not torch.cuda.is_available()
    library=next((Path(np.__file__).parent.parent/'numpy.libs').glob('*openblas*'))
    core=ctypes.CDLL(str(library)).openblas_get_corename64_;core.restype=ctypes.c_char_p
    assert core().decode()=='Prescott'
    from pi0260_r3_inputs import build
    stats=ROOT/'contracts/ROT/source/dataset/statistics.json';assert sha(stats)==STATS_SHA
    processor=build(read(stats),paired.COSMOS);audits=[];ranges={};plan=read(ROOT/'design/FITTING_CARD.json')['stream']
    for arm in ('S4FIT','DIRECT'):
        output=ROOT/'evidence/input-audit'/arm;output.mkdir(parents=True,exist_ok=True)
        assert not (output/'sample_audit.jsonl').exists(),'PRESERVE_PRIOR_PARTIAL_CPU_AUDIT'
        counts=Counter()
        for n,value in enumerate(Pairs(processor,output,arm,0),1):
            for name,shape in (('state',(1,1,132)),('action',(1,40,132))):
                x=value['inputs'][name].detach().cpu().numpy()
                assert x.shape==shape and x.dtype==np.float32 and np.isfinite(x).all() and not x[...,10:].any()
                ranges[name]=max(ranges.get(name,0.),float(np.abs(x).max()))
            mask=value['inputs']['action_mask'];assert torch.all(mask[...,:10]==1) and not torch.any(mask[...,10:])
            counts[plan[n-1]['input_id']]+=1
            if n%512==0:print('ACTUAL_CPU_MICROBATCHES',arm,n,flush=True)
        assert n==2048 and Counter(counts.values())=={7:272,6:24}
        rows=[json.loads(s) for s in (output/'sample_audit.jsonl').read_text().splitlines()]
        for row in rows:row.pop('arm')
        audits.append(rows)
    assert audits[0]==audits[1]
    write(ROOT/'evidence/HOPPER_CPU_ADMISSION.json',dict(status='PASS_PI0272_BOTH_FULL_2048_ACTUAL_PROCESSOR_STREAMS',
        microbatches_per_arm=2048,unique_conditions=296,exposure_histogram={7:272,6:24},paired_inputs_labels_equal=True,
        normalized_abs_max=ranges,statistics_sha256=sha(stats),fitting_card_sha256=sha(ROOT/'design/FITTING_CARD.json'),
        kernel=core().decode(),kernel_sha256=sha(library),model_queries=0,optimizer_updates=0,
        sample_audits={arm:sha(ROOT/'evidence/input-audit'/arm/'sample_audit.jsonl') for arm in ('S4FIT','DIRECT')}))
    print('PASS_PI0272_BOTH_FULL_2048_ACTUAL_PROCESSOR_STREAMS',flush=True)


def binding(arm):
    assert read(ROOT/'evidence/HOPPER_CPU_ADMISSION.json')['status']=='PASS_PI0272_BOTH_FULL_2048_ACTUAL_PROCESSOR_STREAMS'
    import pi0264_compare_canary,pi0264_training_audit,pi0264_archive_trainer,pi0260_r3_inputs,pi0260_r2_codec,pi0260_r3_physical,temporal
    previous={r['module']:r for r in read(ROOT.with_name('PI0270')/'evidence/RUNTIME_BINDING_AGG.json')['modules']}
    modules=[]
    for module in (paired,paired.r3,pi0264_compare_canary,pi0264_training_audit,pi0264_archive_trainer,pi0260_r3_inputs,pi0260_r2_codec,pi0260_r3_physical,temporal):
        path=Path(inspect.getfile(module));row=dict(module=module.__name__,path=str(path),sha256=sha(path))
        assert row==previous[module.__name__];modules.append(row)
    def forbidden(*a,**kw):raise RuntimeError('PI0272_BINDING_NO_MODEL_OR_OPTIMIZER')
    torch.nn.Module.__call__=forbidden;torch.optim.Optimizer.__init__=forbidden
    output=ROOT/'runtime'/f'cpu-hook-{arm}';output.mkdir(parents=True,exist_ok=True)
    audit=hooks(SimpleNamespace(resume=0,limit=8,arm=arm,phase='continuous'),output)
    assert audit['forwards']==0 and not torch.cuda.is_available()
    write(ROOT/'evidence'/f'RUNTIME_BINDING_{arm}.json',dict(status='PASS_PI0272_ACTUAL_FROZEN_HELPERS_AND_HOOKS',arm=arm,
        modules=modules,model_queries=0,gradient_queries=0,optimizer_updates=0,readout_source_sha256=sha(ROOT/'code/readout.py')))
    print('PASS_PI0272_ACTUAL_FROZEN_HELPERS_AND_HOOKS',arm,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('provision','cpu','binding'));p.add_argument('--arm',choices=('S4FIT','DIRECT'));a=p.parse_args()
    if a.stage=='provision':provision()
    elif a.stage=='cpu':cpu()
    else:assert a.arm;binding(a.arm)
