#!/usr/bin/env python3
"""Small evidence archive; fixed revision full streaming readback."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import requests
from huggingface_hub import HfApi, hf_hub_url

ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0253')
sys.path.insert(0,'/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/reference-317a/scripts')
from pi0243_checkpoint_persistence import inventory


def main():
    p=argparse.ArgumentParser();p.add_argument('--payload',type=Path,required=True);p.add_argument('--prefix',required=True)
    p.add_argument('--receipt',type=Path,required=True);args=p.parse_args()
    assert args.payload.resolve().is_relative_to(ROOT) and args.receipt.resolve().is_relative_to(ROOT)
    assert args.prefix.startswith('evidence/PI0253/')
    local=inventory(args.payload,label='PI0253')
    token=Path('/home/liuyu-nfs/.config/huggingface/token').read_text().strip()
    api=HfApi(token=token);repo='leoyuer/pi0246-gate-pass-campaign'
    assert api.model_info(repo).private is False
    if args.receipt.exists():
        old=json.loads(args.receipt.read_text());assert old['local']==local
        print(json.dumps(dict(status=old['status'],revision=old['revision'])),flush=True);return
    commit=api.upload_folder(repo_id=repo,repo_type='model',folder_path=str(args.payload),path_in_repo=args.prefix,
                             commit_message='PI0253 fixed bridge and zero-model TRAIN support evidence')
    revision=commit.oid;observed=[]
    for row in local['entries']:
        h=hashlib.sha256();size=0
        url=hf_hub_url(repo,filename=args.prefix+'/'+row['path'],revision=revision,repo_type='model')
        with requests.get(url,headers={'Authorization':'Bearer '+token},stream=True,timeout=(30,180)) as r:
            r.raise_for_status()
            for b in r.iter_content(8<<20):h.update(b);size+=len(b)
        assert size==row['bytes'] and h.hexdigest()==row['sha256'],row['path']
        observed.append(dict(row,readback_sha256=h.hexdigest()))
    result=dict(schema='PI0253_EVIDENCE_FULL_HF_READBACK_V1',status='PASS_FULL_FILE_READBACK',repo_id=repo,
                prefix=args.prefix,revision=revision,public=True,local=local,readback=observed)
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(status=result['status'],revision=revision,files=len(observed),bytes=local['total_bytes'],tree=local['tree_sha256'])),flush=True)


if __name__=='__main__':main()
