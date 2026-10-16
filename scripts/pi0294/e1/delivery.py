"""New private correction prefix only, fixed revision/hash and READY-last transfer."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tarfile
import time

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0292.data_consumer import extract_checked
from scripts.pi0294.native_dataset import sha
from .scope import validate,Ledger,STATS


def publish(root,dataset,repo,token_file):
    root=Path(root);validate(root,'4080-simulation');ledger=Ledger(root,'core');started=time.process_time()
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]));os.environ['HF_HUB_OFFLINE']='0'
    from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
    dataset=Path(dataset)
    if not dataset.resolve().is_relative_to((root/'datasets').resolve()):raise PermissionError('OWN_E1_DATASET_ONLY')
    card=read(dataset/'CORRECTION_DATASET_CARD.json')
    if card['status']!='ACTUAL_NONEMPTY_RECOVERY_BANK_INDEPENDENT_READER_PASS' or card['statistics_sha256']!=STATS:raise PermissionError('E1_REAL_BANK_READER_REQUIRED')
    output=root/'datasets'/('delivery-'+dataset.name);output.mkdir(mode=0o700)
    owner=OwnerLock(root/'E1_HF_PUBLISH.lock');receipt=output/'TRANSFER.private.json';state=dict(status='PAYLOAD_PREPARING',READY_claim=False)
    try:
        token=Path(token_file).read_text().strip();api=HfApi(token=token);info=api.repo_info(repo,repo_type='dataset')
        if not info.private:raise PermissionError('E1_RAW_CORRECTION_PRIVATE_HF_ONLY')
        prefix='PI0294/'+root.name+'/'+dataset.name
        if any(n.startswith(prefix+'/') for n in api.list_repo_files(repo,repo_type='dataset')):
            raise FileExistsError('PRESERVE_EXISTING_IMMUTABLE_HF_PREFIX')
        members={str(p.relative_to(dataset)):sha(p) for p in dataset.rglob('*') if p.is_file() and '.private.' not in p.name}
        if any(p.is_symlink() for p in dataset.rglob('*')):raise PermissionError('NO_DATASET_SYMLINKS')
        archive=output/'correction.tar.gz'
        with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
            for name in sorted(members):
                ledger.guard();tar.add(dataset/name,arcname=name,recursive=False)
        write(output/'MEMBERS.json',members)
        files={n:dict(bytes=(output/n).stat().st_size,sha256=sha(output/n)) for n in ['correction.tar.gz','MEMBERS.json']}
        state.update(repo_id=repo,prefix=prefix,files=files,status='UPLOAD_INTENT_UNKNOWN_NO_BLIND_RETRY');write(receipt,state)
        revision=api.create_commit(repo,repo_type='dataset',parent_commit=info.sha,
            commit_message='E1 actual TRAIN correction bytes before READY',operations=[CommitOperationAdd(path_in_repo=prefix+'/'+n,path_or_fileobj=str(output/n)) for n in files]).oid
        state.update(status='PAYLOAD_COMMIT_CREATED',data_revision=revision);write(receipt,state)
        for name,row in files.items():
            downloaded=hf_hub_download(repo,prefix+'/'+name,repo_type='dataset',revision=revision,token=token,local_dir=output/'readback')
            if sha(downloaded)!=row['sha256']:raise ValueError('E1_HF_PAYLOAD_READBACK_HASH')
        ready=dict(status='READY_BYTES_AND_INDEPENDENT_NPZ_JSONL_RGB_VERIFIED',run_uuid=root.name,repo_id=repo,private=True,
            prefix=prefix,data_revision=revision,files=files,manifest_sha256=sha(dataset/'DATASET_MANIFEST.json'),
            statistics_sha256=STATS,dataset_card_sha256=sha(dataset/'CORRECTION_DATASET_CARD.json'),
            READY_uploaded_last=True,actual_Hopper_processor='PENDING',origin=card['origin'])
        write(output/'READY.json',ready)
        final=api.create_commit(repo,repo_type='dataset',parent_commit=revision,commit_message='E1 READY after fixed revision byte verification',
            operations=[CommitOperationAdd(path_in_repo=prefix+'/READY.json',path_or_fileobj=str(output/'READY.json'))]).oid
        actual=hf_hub_download(repo,prefix+'/READY.json',repo_type='dataset',revision=final,token=token,local_dir=output/'final-readback')
        if sha(actual)!=sha(output/'READY.json'):raise ValueError('E1_HF_READY_READBACK_HASH')
        for name,row in files.items():
            actual=hf_hub_download(repo,prefix+'/'+name,repo_type='dataset',revision=final,token=token,local_dir=output/'final-readback')
            if sha(actual)!=row['sha256']:raise ValueError('E1_FINAL_REVISION_PAYLOAD_READBACK_HASH')
        state.update(status='PASS_PRIVATE_FIXED_REVISION_READY_LAST',revision=final,READY_claim=True,
            READY_sha256=sha(output/'READY.json'),manifest_sha256=ready['manifest_sha256'],statistics_sha256=STATS)
        write(receipt,state);return state
    finally:ledger.cpu(time.process_time()-started);owner.close()


def consume(root,transfer,token_file):
    root=Path(root);validate(root,'hopper-neural');ledger=Ledger(root,'hopper');started=time.process_time()
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]));os.environ['HF_HUB_OFFLINE']='0'
    from huggingface_hub import HfApi,hf_hub_download
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('E1_DATA_CONSUMER_CPU_ONLY')
    t=read(transfer)
    if t['status']!='PASS_PRIVATE_FIXED_REVISION_READY_LAST' or not t['READY_claim'] or t['statistics_sha256']!=STATS:raise PermissionError('E1_FIXED_READY_LAST_REQUIRED')
    output=root/'datasets'/('received-'+t['prefix'].split('/')[-1]);output.mkdir(mode=0o700)
    try:
        token=Path(token_file).read_text().strip();api=HfApi(token=token)
        info=api.repo_info(t['repo_id'],repo_type='dataset',revision=t['revision'])
        if not info.private or info.sha!=t['revision']:raise PermissionError('E1_PRIVATE_EXACT_REVISION_REQUIRED')
        def get(name,digest):
            p=Path(hf_hub_download(t['repo_id'],t['prefix']+'/'+name,repo_type='dataset',revision=t['revision'],token=token,local_dir=output/'download'))
            if sha(p)!=digest:raise ValueError('E1_FIXED_REVISION_DATA_HASH_CHANGED')
            return p
        ready=read(get('READY.json',t['READY_sha256']))
        if ready['run_uuid']!=root.name or ready['data_revision']!=t['data_revision'] or ready['files']!=t['files']:raise ValueError('E1_READY_PAYLOAD_BINDING')
        members=read(get('MEMBERS.json',t['files']['MEMBERS.json']['sha256']));archive=get('correction.tar.gz',t['files']['correction.tar.gz']['sha256'])
        def guard(extra=0):
            ledger.guard()
            if extra>40*1024**3 or shutil.disk_usage(root).free<extra+20*1024**3:raise PermissionError('E1_DATA_CONSUMER_SPACE')
        extract_checked(archive,output/'native',members,guard)
        data=output/'native';manifest=read(data/'DATASET_MANIFEST.json')
        if sha(data/'DATASET_MANIFEST.json')!=t['manifest_sha256'] or sha(data/'statistics.json')!=STATS:raise ValueError('E1_RECEIVED_BANK_CHANGED')
        bindings={r['id']:str(data/'raw'/r['id']) for r in manifest['episodes']};write(output/'RAW_BINDINGS.private.json',bindings)
        from .correction import independent_read,CorrectionDataset
        audit=independent_read(data,bindings)
        b=read(root/'NEURAL_BINDINGS.private.json')
        from scripts.pi0294.native_dataset import official_import,ProcessedCollator
        official_import(b['source'])
        import torch
        torch.set_num_threads(1)
        from scripts.pi0292.eval_contract import expert_decode
        dataset=CorrectionDataset(data,b['source'],b['cosmos'],for_optimization=True,raw_bindings=bindings);dataset.initialize()
        selected=sorted({0,len(dataset)//2,len(dataset)-1});checks=[]
        for index in selected:
            state,commands,images,text=dataset.raw(index);value=dataset[index]
            batch=ProcessedCollator(dataset.processor)([value]);action=batch['inputs']['action'];mask=batch['inputs']['action_mask']
            if tuple(action.shape)!=(1,40,132) or not bool((mask[...,:10]==1).all()) or not bool((mask[...,10:]==0).all()):raise ValueError('E1_ACTUAL_CORRECTION_NATIVE_MASK')
            decoded,clip=expert_decode(dataset.processor,state,commands)
            checks.append(dict(identity=dataset.identity(index),roundtrip_maximum_absolute_difference=float(__import__('numpy').max(__import__('numpy').abs(decoded-commands))),native_decode=clip))
        from .mixture import MixedDataset
        from .trainer import make_trainer_class,check_trace
        from scripts.pi0294.native_train import tree_identity
        from types import SimpleNamespace
        loader_checks=[]
        for resume_step in [0,1]:
            trace=output/('mixed-trace-'+str(resume_step))
            mixed=MixedDataset(b['dataset'],b['source'],b['cosmos'],for_optimization=True,raw_bindings=read(b['raw_bindings']),
                trace_root=trace,selected_ids=[x['id'] for x in read(root/'plans/SMALL8_BINDING.json')['episodes']],
                correction_root=data,correction_bindings=bindings)
            mixed.initialize();trainer=object.__new__(make_trainer_class())
            trainer.args=SimpleNamespace(ignore_data_skip=False,max_steps=2);trainer.state=SimpleNamespace(global_step=resume_step)
            trainer.train_dataset=mixed;trainer._train_batch_size=2;trainer.data_collator=ProcessedCollator(mixed.processor)
            loader=trainer.get_train_dataloader();batches=[tree_identity(batch) for batch in loader]
            identities=[mixed.identity(trainer.sampler.index_at(c)) for c in range(resume_step*32,64)]
            original=sum(row.get('origin')=='new' for row in identities)
            if original!=(64-resume_step*32)//2:raise ValueError('ACTUAL_MIXED_LOADER_HALF_EXPOSURE')
            loader_checks.append(dict(resume_step=resume_step,batches=batches,
                trace=check_trace(trace,mixed,resume_step*32,64),original_samples=original,correction_samples=len(identities)-original))
        if loader_checks[0]['batches'][16:]!=loader_checks[1]['batches']:raise ValueError('ACTUAL_MIXED_LOADER_RESUME_MISMATCH')
        if torch.cuda.is_initialized():raise PermissionError('E1_CONSUMER_INITIALIZED_CUDA')
        result=dict(status='PASS_ACTUAL_HOPPER_CORRECTION_PROCESSOR_LOADER',dataset=str(data),raw_bindings=str(output/'RAW_BINDINGS.private.json'),
            revision=t['revision'],prefix=t['prefix'],manifest_sha256=t['manifest_sha256'],statistics_sha256=STATS,
            independent_reader=audit,actual_first_middle_last=checks,model_loaded=False,optimizer_updates=0,CPU_s=time.process_time()-started)
        result['actual_mixed_loader']=[{k:v for k,v in item.items() if k!='batches'} for item in loader_checks]
        write(output/'CONSUMPTION.private.json',result);return result
    finally:ledger.cpu(time.process_time()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['publish','consume']);p.add_argument('--root',required=True);p.add_argument('--dataset');p.add_argument('--repo');p.add_argument('--transfer');p.add_argument('--existing-token-file',required=True)
    a=p.parse_args();value=publish(a.root,a.dataset,a.repo,a.existing_token_file) if a.mode=='publish' else consume(a.root,a.transfer,a.existing_token_file)
    print(json.dumps({k:v for k,v in value.items() if k in ['status','revision','prefix','manifest_sha256','CPU_s']}),flush=True)
