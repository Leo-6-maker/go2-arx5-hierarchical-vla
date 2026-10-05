"""New-input comparison and narrow lifecycle/source checks; no learned calls."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_CORETYPE='Prescott')
import argparse,ctypes,inspect,json,resource
from pathlib import Path
from types import SimpleNamespace
from train import ROOT,CONTROL,previous,old,paired,read,sha,np,torch,Pairs,hooks,seed_for,ADAPTATIONS,RELEASE
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);text=json.dumps(value,indent=2,sort_keys=True)+'\n'
    if p.exists():assert p.read_text()==text,str(p)
    else:p.write_text(text)
def guard():
    resource.setrlimit(resource.RLIMIT_AS,(16*2**30,16*2**30));os.nice(19)
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-4:])
    torch.set_num_threads(4);torch.set_num_interop_threads(1);assert not torch.cuda.is_available()
    original=torch.nn.Module.__call__
    def forbidden(self,*a,**kw):
        if type(self).__module__.startswith('torchvision.transforms'):return original(self,*a,**kw)
        raise RuntimeError('PI0274_CPU_NO_NEURAL_QUERY')
    torch.nn.Module.__call__=forbidden
    def no_optimizer(*a,**kw):raise RuntimeError('PI0274_CPU_NO_OPTIMIZER')
    torch.optim.Optimizer.__init__=no_optimizer
def cpu():
    guard();assert read(ROOT/'evidence/DATA_LOCAL_STAGE.json')['status']=='PASS_FIXED_NEW_DATA_ALL_MEMBERS_LOCAL'
    from pi0260_r3_inputs import build
    stats=ROOT/'contracts/ROT/source/dataset/statistics.json';processor=build(read(stats),paired.COSMOS)
    library=next((Path(np.__file__).parent.parent/'numpy.libs').glob('*openblas*'));core=ctypes.CDLL(str(library)).openblas_get_corename64_;core.restype=ctypes.c_char_p;assert core().decode()=='Prescott'
    prior=read(CONTROL/'evidence/HOPPER_CPU_ADMISSION.json');assert sha(library)==prior['kernel_sha256']
    dataset=Pairs(processor,ROOT/'runtime','EQUIL',0);card=dataset.card
    original=read(CONTROL/'design/FITTING_CARD.json');assert card['base']==original['base']
    assert card['base_source_card_sha256']==sha(CONTROL/'design/FITTING_CARD.json')
    corelab=read(ROOT/'evidence/DATA_ADMISSION.json');assert corelab['base_encoding_proof_sha256']==sha(CONTROL/'evidence/BASE_ENCODING8192.json')
    proof=[]
    for item in card['new_inputs']:
        sample=dataset.supplement(item['pose_id'],item['sample_sha256']);packet={k:sample[k] for k in ('body_rgb','wrist_rgb','state_v1')};packet['language']=item['language']
        value=old.batch(processor,packet,sample['raw_action'])['inputs']
        assert old.identity({k:v for k,v in value.items() if k not in ('action','action_mask')})==item['condition_sha256']
        assert old.identity({k:value[k] for k in ('action','action_mask')})==item['labels_masks_sha256']
        assert value['state'].dtype==value['action'].dtype==torch.float32 and not value['state'][...,10:].any() and not value['action'][...,10:].any()
        proof.append(item)
        if len(proof)%200==0:print('HOPPER_ACTUAL_NEW_INPUTS',len(proof),flush=True)
    records=[]
    for visit,control in zip(card['streams']['EQUIL'],dataset.closed):
        record=dataset.record(visit)
        for k in ('unit','micro','seed','stream','family','task','command','language'):assert record[k]==control[k]
        if visit['stream']=='BASE':
            for k in ('source_index','condition_sha256','labels_masks_sha256','nonlanguage_sha256','source_sample_sha256'):assert record[k]==control[k]
        records.append(record)
    for resume in (0,4,8,2560,5119):
        dataset.reset_seed(27303+resume);assert dataset.cursor==2*resume
        assert card['streams']['EQUIL'][dataset.cursor]['unit']==resume+1 and card['streams']['EQUIL'][dataset.cursor]['seed']==seed_for(resume+1)
    assert len(records)==10240 and len({r['pose_id'] for r in records if r['stream']=='SUPPLEMENT'})==672
    path=ROOT/'evidence/input-audit/EQUIL/sample_audit.jsonl';path.parent.mkdir(parents=True,exist_ok=True);text=''.join(json.dumps(r,sort_keys=True)+'\n' for r in records)
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)
    write(ROOT/'evidence/CPU_FIXTURES.json',dict(status='PASS_PI0274_SCHEDULE_CURSOR_AND_CLOSED_PHYSICAL_FIXTURES',source_physical_fixture_sha256=sha(CONTROL/'evidence/CPU_FIXTURES.json'),training_source_sha256=sha(ROOT/'code/train.py'),scope_unchanged=True,model_queries=0,optimizer_updates=0))
    write(ROOT/'evidence/HOPPER_CPU_ADMISSION.json',dict(status='PASS_PI0274_NEW_INPUTS_AND_REUSED8192_BASE',actual_new_processed_conditions=len(proof),base_occurrences=8192,base_reencoded=False,
        base_encoding_receipt_sha256=sha(CONTROL/'evidence/BASE_ENCODING8192.json'),control_cpu_admission_sha256=sha(CONTROL/'evidence/HOPPER_CPU_ADMISSION.json'),
        fitting_card_sha256=sha(ROOT/'design/FITTING_CARD.json'),new_input_admission_sha256=sha(ROOT/'evidence/DATA_ADMISSION.json'),kernel='Prescott',kernel_sha256=sha(library),sample_audit_sha256=sha(path),model_queries=0,optimizer_updates=0))
    print('PASS_PI0274_NEW_INPUTS_AND_REUSED8192_BASE',flush=True)
def binding(repair=False):
    guard();assert read(ROOT/'evidence/HOPPER_CPU_ADMISSION.json')['status']=='PASS_PI0274_NEW_INPUTS_AND_REUSED8192_BASE'
    import pi0264_compare_canary,pi0264_training_audit,pi0264_archive_trainer,pi0260_r3_inputs,pi0260_r2_codec,pi0260_r3_physical,temporal
    prior={r['module']:r for r in read(CONTROL/'evidence/RUNTIME_BINDING_BROAD.json')['modules']};modules=[]
    for m in (paired,paired.r3,pi0264_compare_canary,pi0264_training_audit,pi0264_archive_trainer,pi0260_r3_inputs,pi0260_r2_codec,pi0260_r3_physical,temporal):
        path=Path(inspect.getfile(m));row=dict(module=m.__name__,path=str(path),sha256=sha(path));assert row==prior[m.__name__];modules.append(row)
    output=ROOT/'runtime/cpu-hook-EQUIL';output.mkdir(parents=True,exist_ok=True)
    audit=hooks(SimpleNamespace(resume=0,limit=8,arm='EQUIL',phase='continuous'),output);assert audit['forwards']==0
    write(ROOT/('evidence/RUNTIME_BINDING_R1_EQUIL.json' if repair else 'evidence/RUNTIME_BINDING_EQUIL.json'),dict(status='PASS_PI0274_ACTUAL_FROZEN_HELPERS_AND_HOOKS',modules=modules,adaptations=ADAPTATIONS,qualified_trainer_sha256=sha(CONTROL/'code/train.py'),training_source_sha256=sha(ROOT/'code/train.py'),model_queries=0,gradient_queries=0,optimizer_updates=0))
    print('PASS_PI0274_ACTUAL_FROZEN_HELPERS_AND_HOOKS',flush=True)
def seal():
    assert read(ROOT/'evidence/RUNTIME_BINDING_EQUIL.json')['status']=='PASS_PI0274_ACTUAL_FROZEN_HELPERS_AND_HOOKS'
    archive=read(ROOT/'evidence/RESULT_ARCHIVE_data.json');assert archive['status']=='PASS_PUBLIC_FIXED_REVISION_FILES_AND_ALL_MEMBERS' and archive['revision']=='a786d91e501d241eb8af29d9666f6b8fd3c7a6be'
    assert archive['tree_sha256']==read(ROOT/'evidence/DATA_LOCAL_STAGE.json')['tree_sha256']
    paths=list((ROOT/'code').glob('*'))+list((ROOT/'authority').glob('*'))+list((ROOT/'design').glob('*.json'))
    for row in read(CONTROL/'EXECUTION_FILESET.json')['files']:
        p=Path(row['path'])
        if p.suffix=='.py':assert sha(p)==row['sha256'];paths.append(p)
    paths.extend(Path(r['path']) for r in read(ROOT/'evidence/RUNTIME_BINDING_EQUIL.json')['modules'])
    paths.extend(ROOT/p for p in ('DONOR_LOCAL_READBACK.json','DONOR_PARAMETER_SCOPE.json','STORAGE_ADMISSION.json','contracts/ROT/source/dataset/statistics.json','evidence/PRECISION_CONTRACT_RESOLUTION.json',
        'evidence/DATA_LOCAL_STAGE.json','evidence/RESULT_ARCHIVE_data.json','evidence/EXPOSURE_AND_ACTUAL_SEEN.json','evidence/A0_VISIBILITY_SOURCE_BINDING.json','evidence/DATA_ADMISSION.json','evidence/CPU_FIXTURES.json','evidence/HOPPER_CPU_ADMISSION.json','evidence/RUNTIME_BINDING_EQUIL.json','evidence/input-audit/EQUIL/sample_audit.jsonl'))
    paths.extend((CONTROL/'code/train.py',CONTROL/'design/FITTING_CARD.json',CONTROL/'evidence/BASE_ENCODING8192.json',CONTROL/'BROAD/main/training/step_rng.jsonl',CONTROL/'BROAD/main/training/sample_audit.jsonl'))
    assert sha(CONTROL/'BROAD/main/training/step_rng.jsonl')=='c7306e4bd9f14dcaee9203e12f6274f612b4d731b1e0aaddd3e963186d13bdb3'
    paths.extend((ROOT/'source/data').glob('*/sample.npz'));paths=sorted({p for p in paths if p.is_file()})
    write(ROOT/'EXECUTION_FILESET.json',dict(schema='PI0274_IMMUTABLE_NEW_DATA_S4_RUNTIME_V1',release=RELEASE,files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in paths],canary_still_required=True,model_queries=0))
    print('PI0274_RUNTIME_SEALED',len(paths),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('cpu','binding','binding-r1','seal'));a=p.parse_args();dict(cpu=cpu,binding=binding,seal=seal,**{'binding-r1':lambda:binding(True)})[a.action]()
