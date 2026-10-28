"""CPU-only saved-export seal/pack prelude within the one training allocation.

No Teacher replay, re-export, codec refit, model import or CUDA permission.
Original failed files/modes remain untouched; bounded new copies are sealed.
"""
import json,os,time
from pathlib import Path
from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.fd1 import put
from . import amd_bank as bank,amd_training_accounting as a

def copy_sealed(source,target,expected,*,maximum_bytes=8*1024**2):
    source=Path(source);target=Path(target);before=source.lstat()
    a.need(not source.is_symlink()and source.is_file()and before.st_uid==os.getuid()
        and 0<before.st_size<=maximum_bytes and sha(source)==expected,'TRAIN_PACK_EXACT_OWN_SAVED_FILE')
    target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
    with os.fdopen(fd,'wb')as f:
        with source.open('rb')as s:
            for chunk in iter(lambda:s.read(1024**2),b''):f.write(chunk)
        f.flush();os.fsync(f.fileno())
    after=source.stat()
    a.need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)
        ==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
        and sha(target)==expected and target.stat().st_mode&0o777==0o400,
        'TRAIN_PACK_UNCHANGED_SOURCE_AND_SEALED_IDENTICAL_NEW_COPY')
    return dict(bytes=target.stat().st_size,sha256=expected)

def prepare(root,guard):
    """Caller has already proven its registered CPU/PBS allocation and scope."""
    from .amd_export_repair_runtime import tools
    from .amd_export_runtime import assemble
    from scripts.pi0291_arx5_data.native_dataset import cpu_guard
    root=Path(root);inputs=read(root/'plans/GEN1_TRAIN_INPUTS.private.json')
    evidence=read(root/'plans/GEN1_CLOSED_EXPORT_INPUT.private.json');source=Path(evidence['root'])
    packet=(root/'plans/GEN1_V5_PREVIEW_V2.private.json').read_bytes()
    value=read(root/'plans/GEN1_EXPECTED_BANK_MANIFEST.json');bank.validate(value,packet)
    a.need(evidence['status']=='PASS_CLOSED_GEN1_SAVED_EXPORTS_FOR_PREMODEL_PACK_ONLY'
        and inputs['bank_manifest_sha256']==evidence['bank_manifest_sha256']
        and sha(root/'plans/GEN1_EXPECTED_BANK_MANIFEST.json')==evidence['bank_manifest_sha256'],
        'TRAIN_PACK_EXACT_SAVED_INPUT_AND_PROSPECTIVE_BANK')
    guard();cpu_guard();tools(run_git=True);began=time.process_time()
    from scripts.pi0292.train_adapter import build_processor
    b=read(root/'NEURAL_BINDINGS.private.json')
    processor=build_processor(b['source'],b['cosmos'],read(root/'plans/statistics.json'))
    for row in value['episodes']:
        guard();old=source/'datasets'/row['id'];out=root/'datasets'/row['id']
        a.need(not out.exists(),'TRAIN_PACK_NO_PARTIAL_OUTPUT_REPLAY');out.mkdir(mode=0o700)
        a.need(read(old/'EXPORT_RECEIPT.json')==row,'TRAIN_PACK_SAME_ORIGINAL_OFFICIAL_RECEIPT')
        for n,h in [(row['file'],row['sha256']),('EXPORT_RECEIPT.json',sha(old/'EXPORT_RECEIPT.json')),
            ('DATASET_MANIFEST.json',sha(old/'DATASET_MANIFEST.json')),('statistics.json',a.original.STATS)]:
            copy_sealed(old/n,out/n,h)
        a.need(sha(old/'DATASET_MANIFEST.json')==evidence['saved_dataset_manifest_sha256'][row['id']],
            'TRAIN_PACK_EXACT_SAVED_SINGLE_PARENT_MANIFEST')
    raw=read(inputs['raw_bindings']);a.need(set(raw)=={r['id']for r in value['episodes']},'TRAIN_PACK_EXACT_ALL_RAW_BINDINGS')
    # assemble consumes this original named sealed binding, never materializes RGB.
    a.need(read(root/'plans/GEN1_FROZEN_RAW_BINDINGS.private.json')==raw,'TRAIN_PACK_NO_NEW_RAW_NAMESPACE')
    ready=assemble(root,value,processor,b['source'],b['cosmos'],guard,raw)
    a.need(sha(root/'datasets/FROZEN_GEN1_BANK/DATASET_MANIFEST.json')==evidence['bank_manifest_sha256'],
        'TRAIN_PACK_FINAL_BANK_EXACT_EXPECTED_IDENTITY')
    import sys,torch
    a.need(not torch.cuda.is_initialized()and not any(n.startswith('gr00t')and
        ('modeling_'in n or'.backbone.'in n or'.action_head.'in n)for n in sys.modules),
        'TRAIN_PACK_NO_MODEL_OR_CUDA_BEFORE_BANK_PASS')
    result=dict(status='PASS_ACTUAL_SAVED_GEN1_PREMODEL_BANK_PACK',ready=ready,
        CPU_s=time.process_time()-began,model_loads=0,queries=0,updates=0,constructors=0,
        physics_ticks=0,Teacher_episodes=0,old_output_unchanged=True)
    put(root/'evidence/GEN1_PREMODEL_BANK_READY.json',result);return result

if __name__=='__main__':
    import argparse,resource
    from .amd_training_runtime import guard,root_context
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--request',required=True);v=p.parse_args()
    need=a.need;need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','TRAIN_PACK_ISOLATED_CPU_CHILD_ONLY')
    resource.setrlimit(resource.RLIMIT_CPU,(590,600));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    root_context(v.root,v.request,full=False)
    allocation=read(Path(v.request).parent/'ALLOCATION_ADMISSION.json')
    request=read(v.request)
    def bounded_guard():
        guard(v.root,allocation);need(time.time()<request['CPU_bank_deadline_epoch'],'TRAIN_ORIGINAL_CPU_PACK_DEADLINE')
    prepare(v.root,bounded_guard)
