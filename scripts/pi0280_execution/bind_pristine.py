"""Verify original local assets and isolate the pinned official source; zero queries."""
import hashlib,json,os,subprocess,sys,urllib.request
from pathlib import Path

ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0280')
BASE=Path('/var/cache/users/liuyu-nfs/go2-arx5-vla-stage10/assets/GR00T-N1.7-3B')
SOURCE=Path('/home/liuyu-nfs/Projects/isaac-gr00t')
CODE_SHA='51d4c89f72fda44cbf77285c6a8114b52676b8a1'
REVISION='2fc962b973bccdd5d8ce4f67cc63b264d6886495'
RELEASE='9c0172ced33c6052f7cb037b4a6d91910f9fe69e'

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def main():
    assert json.loads((ROOT.parent/'PI0279/runtime/SCIENTIFIC_PAUSE.json').read_text())['launch_allowed'] is False
    out=ROOT/'evidence';out.mkdir(parents=True,exist_ok=True)
    upstream=json.load(urllib.request.urlopen('https://huggingface.co/api/models/nvidia/GR00T-N1.7-3B/revision/'+REVISION+'?blobs=true',timeout=30))
    assert upstream['sha']==REVISION
    (out/'OFFICIAL_HF_METADATA.json').write_text(json.dumps(upstream,indent=2)+'\n')
    entries={r['rfilename']:r for r in upstream['siblings']}
    names=['config.json','processor_config.json','statistics.json','embodiment_id.json','model.safetensors.index.json',*sorted(set(json.loads((BASE/'model.safetensors.index.json').read_text())['weight_map'].values()))]
    files=[]
    for name in names:
        p=BASE/name;r=entries[name];sha=digest(p)
        actual=sha if r.get('lfs') else hashlib.sha1(b'blob '+str(p.stat().st_size).encode()+b'\0'+p.read_bytes()).hexdigest()
        expected=r['lfs']['sha256'] if r.get('lfs') else r['blobId']
        files.append(dict(path=name,bytes=p.stat().st_size,sha256=sha,upstream_digest=expected,upstream_match=actual==expected))
        print(name,actual==expected,sha,flush=True)
    receipt=dict(status='PRISTINE_LOCAL_FILES_VERIFIED' if all(r['upstream_match'] for r in files) else 'LOCAL_ASSET_MISMATCH',release=RELEASE,repo='nvidia/GR00T-N1.7-3B',revision=REVISION,original_local_root=str(BASE),files=files,new_training_updates=0,new_model_queries=0)
    (out/'PRISTINE_FILES.json').write_text(json.dumps(receipt,indent=2)+'\n')
    assert all(r['upstream_match'] for r in files),'Local assets differ from fixed upstream: retain evidence and resolve before loading'
    target=ROOT/'official'
    if not target.exists():subprocess.run(['git','-C',str(SOURCE),'worktree','add','--detach',str(target),CODE_SHA],env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1'),check=True)
    assert subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()==CODE_SHA
    assert not subprocess.check_output(['git','-C',str(target),'status','--porcelain'],text=True).strip()
    routes=json.loads((BASE/'processor_config.json').read_text())['processor_kwargs']['modality_configs'];ids=json.loads((BASE/'embodiment_id.json').read_text())
    rows=[dict(tag=tag,projector=ids[tag],video=v['video'],state=v['state'],action=v['action'],language=v['language'],target_compatible=False,reason='DROID requires real Panda joints; all other serialized routes require dual-arm or humanoid-specific state absent on Go2+ARX5') for tag,v in routes.items()]
    matrix=dict(status='BLOCKED_TARGET_PRETRAINED_INPUT_CONTRACT',routes=rows,source_tag='oxe_droid_relative_eef_relative_joint',source_supported=True,target_missing_semantics=['DROID Panda seven-joint state','other routes dual wrists/hands/joints/waist'],no_placeholder_or_invented_state=True,remaining_authorized=['native-source','source-physics-if-contract','frozen-WBC-reference-controls'])
    (out/'SUPPORTED_INPUT_MATRIX.json').write_text(json.dumps(matrix,indent=2)+'\n')
    print('FILES_VERIFIED; OFFICIAL_SOURCE_ISOLATED; TARGET_INPUT_CONTRACT_BLOCKED',flush=True)

if __name__=='__main__':main()
