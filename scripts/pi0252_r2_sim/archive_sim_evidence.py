#!/usr/bin/env python3
"""Public fixed-revision archival with full byte-stream readback; no models."""
import hashlib
import json
import shutil
import sys
from pathlib import Path
import requests
from huggingface_hub import HfApi,hf_hub_url

ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0252')
CODE=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/reference-317a')
sys.path.insert(0,str(CODE/'scripts'))
from pi0243_checkpoint_persistence import inventory

def main():
    payload=ROOT/'pre_full_sim'
    replay=json.loads((payload/'cpu_replay.json').read_text())
    assert replay['status']=='PASS_INDEPENDENT_CPU_REPLAY' and replay['decision']=='PI0252_R1_PRE_FULL_TRAIN_SIM_FAIL'
    source=payload/'source_identity'
    source.mkdir(exist_ok=True)
    for p in [ROOT/'evidence/current_wbc_readback.json', *[ROOT/'evidence/G2000'/n for n in ('run_card.json','hf_readback.json','training_integrity.json','readonly_revalidation.json')]]:
        shutil.copy2(p,source/p.name)
    for p in [*(ROOT/'runtime').glob('*.py'),*(ROOT/'runtime').glob('*.sh')]:
        shutil.copy2(p,source/p.name)
    for name in ('stage21_native_receding_horizon.py','stage21_native_model_receding_horizon.py','stage21_same_scene_groot_worker.py','stage21_same_scene_groot_inference.py'):
        shutil.copy2(CODE/'scripts'/name,source/name)
    local=inventory(payload,label='PI0252_R2_PRE_FULL_SIM')
    token=Path('/home/liuyu-nfs/.config/huggingface/token').read_text().strip()
    api=HfApi(token=token)
    repo='leoyuer/pi0246-gate-pass-campaign'
    prefix='evidence/PI0252/r2-pre-full-train-sim-20260928'
    assert api.model_info(repo).private is False
    receipt_path=ROOT/'evidence/PI0252_R2_PRE_FULL_SIM_HF_READBACK.json'
    if receipt_path.exists():
        old=json.loads(receipt_path.read_text())
        assert old['local']['tree_sha256']==local['tree_sha256']
        print(json.dumps(old),flush=True)
        return
    commit=api.upload_folder(repo_id=repo,repo_type='model',folder_path=str(payload),path_in_repo=prefix,
        commit_message='PI0252 R2 first-valid G TRAIN-A physical FAIL evidence; no retraining')
    revision=commit.oid
    observed=[]
    for entry in local['entries']:
        url=hf_hub_url(repo,filename=prefix+'/'+entry['path'],revision=revision,repo_type='model')
        h=hashlib.sha256(); size=0
        with requests.get(url,headers={'Authorization':'Bearer '+token},stream=True,timeout=(30,180)) as response:
            response.raise_for_status()
            for b in response.iter_content(8*1024*1024): h.update(b); size+=len(b)
        assert size==entry['bytes'] and h.hexdigest()==entry['sha256'],entry['path']
        observed.append(dict(entry,readback_sha256=h.hexdigest()))
    result=dict(schema='PI0252_R2_PRE_FULL_SIM_DURABLE_READBACK_V1',status='PASS_FULL_FILE_READBACK',
        repo_id=repo,path_prefix=prefix,revision=revision,public=True,local=local,readback=observed,
        model_queries=1,head_calls=4,optimizer_updates=0,selected_checkpoint='NONE',scientific_decision=replay['decision'])
    receipt_path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':result['status'],'revision':revision,'files':len(observed),'bytes':local['total_bytes'],'tree_sha256':local['tree_sha256']}),flush=True)

if __name__=='__main__': main()
