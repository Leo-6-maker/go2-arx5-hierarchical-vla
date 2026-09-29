"""Archive sealed H0 public evidence; private dependency binding stays on Corelab."""
from pathlib import Path
import gzip
import hashlib
import json
import tarfile
import requests
from huggingface_hub import HfApi,hf_hub_url
from pi0260_design import ROOT
from pi0255_motion3 import sha,write


def run():
    manifest_path=ROOT/'handoff/H0_v1/manifest.json';manifest=json.loads(manifest_path.read_text())
    assert sha(manifest_path)=='dd75ddfc38c480a200e3853cd991cdf451ea86d87295c8cd0c1d1ef556e4da6e'
    out=ROOT/'archive/H0_v1_public';out.mkdir(parents=True,exist_ok=True)
    entries=[];excluded=[]
    for row in manifest['entries']:
        p=ROOT/row['path'];assert p.resolve().is_relative_to(ROOT) and not p.is_symlink()
        assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
        if row['path']=='design/FK_source_binding.json':excluded.append(dict(row,reason='Private native dependency binding retained on Corelab for independent V access'));continue
        entries.append(row)
    for p in [manifest_path,ROOT/'state/H0_ACCEPTED.json',ROOT/'state/design_freeze.json',ROOT/'evidence/R1_route_cpu.json',ROOT/'evidence/review_negative_tests.json']:
        entries.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
    entries.sort(key=lambda x:x['path'])
    tree=hashlib.sha256(json.dumps(entries,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    write(out/'PUBLIC_MEMBER_MANIFEST.json',dict(entries=entries,excluded=excluded,tree_sha256=tree,
        producer_H0_manifest_sha256=sha(manifest_path),rights='Owned numerical evidence and rendered rasters only; no private mesh/URDF/WBC source, tokenizer or model weights.'))
    archive=out/'PI0260_H0_calibration_evidence.tar.gz'
    with archive.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as compressed,tarfile.open(fileobj=compressed,mode='w|') as tar:
        for row in entries:
            info=tarfile.TarInfo(row['path']);info.size=row['bytes'];info.mode=0o644;info.mtime=0
            with (ROOT/row['path']).open('rb') as data:tar.addfile(info,data)
    repo='leoyuer/pi0246-gate-pass-campaign';prefix='evidence/PI0260/H0-motion2-r1-20260930'
    api=HfApi(token=Path('/home/liuyu-nfs/.config/huggingface/token').read_text().strip());assert api.model_info(repo).private is False
    record=ROOT/'archive/H0_upload_commit.json'
    if record.exists():
        prior=json.loads(record.read_text());assert prior['tree_sha256']==tree;revision=prior['revision']
    else:
        revision=api.upload_folder(repo_id=repo,folder_path=str(out),path_in_repo=prefix,commit_message='PI0260 H0 actual calibration rasters and geometry; no model or physical success').oid
        write(record,dict(repo_id=repo,prefix=prefix,revision=revision,tree_sha256=tree))
    client=requests.Session();client.trust_env=False;verified=[]
    for p in sorted(out.iterdir()):
        h=hashlib.sha256();count=0;download=ROOT/'archive'/('anonymous-'+p.name)
        with client.get(hf_hub_url(repo,filename=prefix+'/'+p.name,revision=revision),stream=True,timeout=(30,180)) as response:
            response.raise_for_status();assert 'Authorization' not in response.request.headers
            with download.open('wb') as stream:
                for block in response.iter_content(8<<20):stream.write(block);h.update(block);count+=len(block)
        assert count==p.stat().st_size and h.hexdigest()==sha(p)
        verified.append(dict(path=p.name,bytes=count,sha256=h.hexdigest(),anonymous=True))
    expected={x['path']:x for x in entries}
    with tarfile.open(ROOT/'archive'/('anonymous-'+archive.name),'r|gz') as tar:
        for member in tar:
            assert member.isfile() and member.name in expected
            row=expected.pop(member.name);h=hashlib.sha256();size=0
            data=tar.extractfile(member)
            for block in iter(lambda:data.read(1<<20),b''):h.update(block);size+=len(block)
            assert size==row['bytes'] and h.hexdigest()==row['sha256']
    assert not expected
    receipt=dict(status='PASS_PUBLIC_ANONYMOUS_FULL_FILE_AND_MEMBER_READBACK',repo_id=repo,prefix=prefix,revision=revision,
        tree_sha256=tree,members=len(entries),source_bytes=sum(x['bytes'] for x in entries),files=verified,
        excluded_private_bindings=excluded,producer_H0_manifest_sha256=sha(manifest_path),physics_steps=0,model_queries=0)
    write(ROOT/'archive/H0_PUBLIC_RECEIPT.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':run()
