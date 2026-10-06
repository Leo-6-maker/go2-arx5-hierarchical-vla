"""Actual task input and frozen helper admission, without neural queries."""
import os,sys
assert os.environ.get('PBS_JOBID') or (len(sys.argv)==2 and sys.argv[1] in ('binding','seal')),'Dataset CPU qualification requires PBS; login permits zero-query configuration binding/seal only'
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_CORETYPE='Prescott')
import argparse,ast,ctypes,inspect,json,resource
from pathlib import Path
from types import SimpleNamespace
from train import ROOT,CONTROL,previous,old,paired,read,sha,np,torch,Pairs,hooks,ADAPTATIONS,RELEASE
from run import write

def guard():
    resource.setrlimit(resource.RLIMIT_AS,(16*2**30,16*2**30));os.nice(19)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);assert not torch.cuda.is_available()
    original=torch.nn.Module.__call__
    def forbidden(self,*a,**kw):
        if type(self).__module__.startswith('torchvision.transforms'):return original(self,*a,**kw)
        raise RuntimeError('PI0276_CPU_NO_NEURAL_QUERY')
    torch.nn.Module.__call__=forbidden
    def no_optimizer(*a,**kw):raise RuntimeError('PI0276_CPU_NO_OPTIMIZER')
    torch.optim.Optimizer.__init__=no_optimizer

def cpu():
    guard();admission=read(ROOT/'evidence/DATA_ADMISSION.json')
    assert admission['status']=='PASS_PI0276_ACTUAL_REPLAY_INPUTS_CODEC_AND_TASK_STATISTICS' and admission['launch_allowed']
    from pi0260_r3_inputs import build
    processor=build(read(ROOT/'contracts/TASK/source/dataset/statistics.json'),paired.COSMOS)
    library=next((Path(np.__file__).parent.parent/'numpy.libs').glob('*openblas*'))
    core=ctypes.CDLL(str(library)).openblas_get_corename64_;core.restype=ctypes.c_char_p;assert core().decode()=='Prescott'
    source=Pairs(processor,ROOT/'runtime','TASK',0);proof={};records=[]
    for i,visit in enumerate(source.plan):
        if visit['uid'] not in proof:
            value=source[i]['inputs']
            for name,shape in [('state',(1,1,132)),('action',(1,40,132))]:
                a=value[name].detach().numpy();assert a.shape==shape and a.dtype==np.float32 and np.isfinite(a).all() and not a[...,10:].any()
            mask=value['action_mask'];assert torch.all(mask[...,:10]==1) and not torch.any(mask[...,10:])
            proof[visit['uid']]=source.proof[visit['uid']]
        assert (visit['unit'],visit['micro'],visit['group'])==(i//2+1,i%2,i%4)
        records.append(dict(visit,**{k:proof[visit['uid']][k] for k in ('condition_sha256','labels_masks_sha256')}))
        if (i+1)%512==0:print('HOPPER_TASK_INPUTS',i+1,len(proof),flush=True)
    assert len(records)==16384
    # Check the actual map-style batching order and the independent loader RNG.
    from torch.utils.data import DataLoader
    class Indices(torch.utils.data.Dataset):
        def __len__(self):return 16384
        def __getitem__(self,i):return i
    node=next(n for n in ast.walk(ast.parse((ROOT/'code/train.py').read_text())) if isinstance(n,ast.FunctionDef) and n.name=='get_train_dataloader')
    for resume in (0,4,4096):
        scope=dict(torch=torch,DataLoader=DataLoader,args=SimpleNamespace(resume=resume))
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(ROOT/'code/train.py'),'exec'),scope)
        trainer=SimpleNamespace(state=SimpleNamespace(global_step=0),args=SimpleNamespace(dataloader_num_workers=0,dataloader_pin_memory=False),
            train_dataset=Indices(),_train_batch_size=1,data_collator=lambda rows:rows[0])
        before=torch.get_rng_state().clone();loader=scope['get_train_dataloader'](trainer)
        assert list(loader)==list(range(2*resume,16384)) and torch.equal(before,torch.get_rng_state()) and trainer.args.ignore_data_skip
    for start in (0,8,16,8192,16382):assert source.plan[start]['unit']==start//2+1
    write(ROOT/'evidence/INPUT_AUDIT16384.json',dict(records=records,actual_unique_encodings=len(proof)))
    write(ROOT/'evidence/HOPPER_CPU_ADMISSION.json',dict(status='PASS_PI0276_ACTUAL_TASK_INPUTS_AND16384_SCHEDULE',
        actual_unique_encodings=len(proof),microbatches=16384,statistics_sha256=sha(ROOT/'contracts/TASK/source/dataset/statistics.json'),
        corelab_admission_sha256=sha(ROOT/'evidence/DATA_ADMISSION.json'),input_audit_sha256=sha(ROOT/'evidence/INPUT_AUDIT16384.json'),
        kernel=core().decode(),kernel_sha256=sha(library),sequential_map_style=True,loader_global_rng_unchanged=True,model_queries=0,optimizer_updates=0))

def binding():
    guard();admission=read(ROOT/'evidence/HOPPER_CPU_ADMISSION.json')
    assert admission['status']=='PASS_PI0276_ACTUAL_TASK_INPUTS_AND16384_SCHEDULE'
    assert admission['statistics_sha256']==sha(ROOT/'contracts/TASK/source/dataset/statistics.json')
    assert admission['corelab_admission_sha256']==sha(ROOT/'evidence/DATA_ADMISSION.json')
    assert admission['input_audit_sha256']==sha(ROOT/'evidence/INPUT_AUDIT16384.json')
    import pi0264_compare_canary,pi0264_training_audit,pi0264_archive_trainer,pi0260_r3_inputs,pi0260_r2_codec,pi0260_r3_physical,temporal
    prior={r['module']:r for r in read(CONTROL/'evidence/RUNTIME_BINDING_BROAD.json')['modules']};modules=[]
    for m in (paired,paired.r3,pi0264_compare_canary,pi0264_training_audit,pi0264_archive_trainer,pi0260_r3_inputs,pi0260_r2_codec,pi0260_r3_physical,temporal):
        p=Path(inspect.getfile(m));r=dict(module=m.__name__,path=str(p),sha256=sha(p));assert r==prior[m.__name__];modules.append(r)
    output=ROOT/'runtime/cpu-hook-TASK';output.mkdir(parents=True,exist_ok=True)
    audit=hooks(SimpleNamespace(resume=0,limit=8,arm='TASK',phase='continuous'),output);assert audit['forwards']==0
    write(ROOT/'evidence/RUNTIME_BINDING_TASK.json',dict(status='PASS_PI0276_ACTUAL_FROZEN_HELPERS_AND_HOOKS',modules=modules,
        adaptations=ADAPTATIONS,qualified_trainer_sha256=sha(CONTROL/'code/train.py'),training_source_sha256=sha(ROOT/'code/train.py'),model_queries=0,gradient_queries=0,optimizer_updates=0))

def seal():
    assert read(ROOT/'evidence/RUNTIME_BINDING_TASK.json')['status']=='PASS_PI0276_ACTUAL_FROZEN_HELPERS_AND_HOOKS'
    paths=list((ROOT/'code').glob('*'))+list((ROOT/'authority').glob('*'))+list((ROOT/'design').glob('*.json'))
    for row in read(CONTROL/'EXECUTION_FILESET.json')['files']:
        p=Path(row['path'])
        if p.suffix=='.py':assert sha(p)==row['sha256'];paths.append(p)
    paths.extend(ROOT/p for p in ('DONOR_LOCAL_READBACK.json','DONOR_PARAMETER_SCOPE.json','STORAGE_ADMISSION.json',
        'contracts/TASK/source/dataset/statistics.json','evidence/PRECISION_CONTRACT_RESOLUTION.json','evidence/DATA_ADMISSION.json',
        'evidence/HOPPER_CPU_ADMISSION.json','evidence/RUNTIME_BINDING_TASK.json','evidence/INPUT_AUDIT16384.json'))
    paths.extend((ROOT/'source/data').glob('*/*.npz'));paths=sorted(set(p for p in paths if p.is_file()))
    write(ROOT/'EXECUTION_FILESET.json',dict(schema='PI0276_ACTUAL_TASK_S4_RUNTIME_V1',release=RELEASE,
        files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in paths],real_canary_still_required=True,model_queries=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('cpu','binding','seal'));globals()[p.parse_args().stage]()
