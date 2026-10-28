"""Bind existing local assets; no new data, model construction, or archive downloads."""
import hashlib,json,os,re,shutil,subprocess
from pathlib import Path

RELEASE='63151a4548243ab0fb4ba172eea6f7d92e02c576'
ROOT=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0277')
OLD=ROOT.with_name('PI0276')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())

def freeze(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert read(p)==value,p;return
    temp=p.with_suffix('.partial');temp.write_text(json.dumps(value,sort_keys=True)+'\n');temp.replace(p)

def storage():
    """The site's hpc quota utility exists on the login node, not compute nodes."""
    proof=read(OLD/'evidence/TASK/checkpoint-8192.full_readback.json')
    query=subprocess.run(['hpc','quota'],capture_output=True,text=True);quota=query.stdout
    match=re.search(r'e1192548\s*\|\|\s*([\d.]+)([GT])B\s*\|\s*([\d.]+)([GT])B',quota)
    assert match,'Actual user quota unavailable; shared df is insufficient'
    used=float(match[1])*2**(30 if match[2]=='G' else 40);limit=float(match[3])*2**(30 if match[4]=='G' else 40)
    available=min(limit-used,shutil.disk_usage(ROOT).free);required=4*sum(x['bytes'] for x in proof['files'])+48*2**30+10*2**30
    assert available>=required,('QUOTA_HEADROOM',available,required)
    owned=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())
    freeze(ROOT/'STORAGE_ADMISSION.json',dict(available_bytes_at_freeze=int(available),owned_bytes_at_freeze=owned,required_bytes=required,
        quota_report=quota,quota_returncode=query.returncode,quota_stderr=query.stderr,df_available_bytes=shutil.disk_usage(ROOT).free,qualification_payloads_retired_after_comparison=True,formal_checkpoints=4,
        observation_host=__import__('socket').gethostname(),observed_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()))

def main():
    assert os.environ.get('PBS_JOBID'),'Preparation hashes run within the legal allocation'
    root=ROOT;old=OLD
    proof=read(old/'evidence/TASK/checkpoint-8192.full_readback.json')
    assert proof['status']=='PASS_FULL_FILE_READBACK' and proof['revision']=='7687ad4c04341b30c6a2a0959aeadf299fa3ba0e'
    assert proof['tree_sha256']=='b78e300af57772c22ab8327918463afb1ea4b176689dbeba2ab243d4a27ae175'
    checkpoint=old/'TASK/main/training/checkpoint-8192';weights=[]
    for r in proof['files']:
        if r['path'] in ('config.json','model.safetensors.index.json') or r['path'].endswith('.safetensors'):
            p=checkpoint/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];weights.append(r)
    freeze(root/'DONOR_LOCAL_READBACK.json',dict(status='PASS_PI0277_REUSED_TASK8192_LOCAL_WEIGHTS',path=str(checkpoint),revision=proof['revision'],
        tree_sha256=proof['tree_sha256'],source_full_readback_sha256=sha(old/'evidence/TASK/checkpoint-8192.full_readback.json'),
        complete_trainer_bytes=sum(x['bytes'] for x in proof['files']),optimizer_inherited=False,files=weights))
    for name in ('DONOR_PARAMETER_SCOPE.json',):
        dest=root/name
        if not dest.exists():shutil.copyfile(old/name,dest)
        assert dest.read_bytes()==(old/name).read_bytes()
    previous=read(old/'EXECUTION_FILESET.json')['files']
    sources=[]
    for row in previous:
        p=Path(row['path'])
        # Actual TRAIN arrays are verified on original construction, never reread just for a new label.
        if '/source/data/' in str(p) or p.suffix in ('.safetensors','.pt','.pth'):continue
        assert sha(p)==row['sha256'],p;sources.append(row)
    overlay=Path('/scratch/e1192548/pi0247_generation_to_sim_20260926/execution_code_m64_39b3718a0a1e')
    groot=Path('/scratch/e1192548/stage22_h200_runtime_deb920_20260902/src/isaac-gr00t')
    r3=root.with_name('PI0260')/'R3/execution'
    paths=[str(root/'code'),str(root.with_name('PI0269')/'code'),str(root.with_name('PI0267')/'code'),str(root.with_name('PI0264')/'code'),str(r3),
        '/scratch/e1192548/pi0244_fixed16_10k_v1/scripts','/scratch/e1192548/pi0244_fixed16_10k_v1/src',str(overlay/'scripts'),str(overlay/'src'),str(groot)]
    bound=[]
    for p in [r3/'pi0260_r3_inputs.py',r3/'pi0260_r2_inputs.py',r3/'pi0260_r2_codec.py',old/'design/DATASET_CARD.json',old/'design/PROCESSED_INPUTS.json',
              old/'contracts/TASK/source/dataset/statistics.json',groot/'gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py']:
        bound.append(dict(path=str(p),sha256=sha(p)))
    freeze(root/'design/INPUT_BINDING.json',dict(old_root=str(old),sources=bound,module_paths=paths,input_helper=str(r3/'pi0260_r3_inputs.py'),
        cosmos='/scratch/e1192548/stage22_h200_cosmos_v2_20260902/hf_home/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561',
        resize='UNCHANGED_SAVED_NATIVE128_TO96_BILINEAR',statistics_sha256='b3da3d500d514ce3d1ce83001de30846d9ab76d699dc48831db8e4367ce98572'))
    assert read(root/'design/TRAIN_PLAN.json')['stream']==read(old/'design/TRAIN_PLAN.json')['stream'][:8192]
    assert read(root/'design/TRAIN_PLAN.json')['release']==RELEASE
    for arm in ('CONTROL','GRIP'):
        contract=read(old/'design/TASK_READOUT_R1.json')
        contract.update(schema='PI0277_NATIVE_S4_GENERATED_GRIPPER_V1',release=RELEASE,arm=arm,
            initial_weights_revision=proof['revision'],initial_weights_prefix=proof['prefix'],
            generated_gripper=dict(coefficient=.1 if arm=='GRIP' else 0.,reference_fraction=.10,prefix_points=8,prefix_mass=.8,suffix_points=32,suffix_mass=.2),
            unchanged_predecessor_readout_sha256=sha(old/'design/TASK_READOUT_R1.json'))
        freeze(root/'design'/f'{arm}_READOUT.json',contract)
    freeze(root/'design/NOISE_CARD.json',dict(namespace='PI0277:TRAIN:27703',construction='sha256(namespace+":"+str(unit))[:8] big-endian mod 2**31',
        seed=27703,updates=4096,microbatches=8192,actual_epsilon_tau_S4_byte_pairing_required=True,loader_rng_seed=27703001))
    snapshot=read(root/'STORAGE_ADMISSION.json')
    owned=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
    assert snapshot['available_bytes_at_freeze']-max(0,owned-snapshot['owned_bytes_at_freeze'])>=snapshot['required_bytes']
    extra=list((root/'code').glob('*.py'))+list((root/'code').glob('*.pbs'))+list((root/'design').glob('*.json'))
    extra += [groot/'gr00t/model/gr00t_n1d7/gr00t_n1d7.py',root.with_name('PI0269')/'code/temporal.py']
    extra += [overlay/'scripts'/n for n in ('pi0247_u_rollout_overlay.py','pi0245_physical_delta.py','pi0246_physical_position.py','pi0248_rotation_cpu.py','pi0252_speed_margin.py') if (overlay/'scripts'/n).exists()]
    merged={r['path']:r for r in sources}
    for p in extra:merged[str(p)]=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
    freeze(root/'EXECUTION_FILESET.json',dict(release=RELEASE,files=sorted(merged.values(),key=lambda r:r['path']),qualified_math_changed=False))
    freeze(root/'evidence/LOCAL_BINDING.json',dict(status='PASS_ACTUAL_LOCAL_ASSET_AND_SOURCE_BINDING',release=RELEASE,
        fileset_sha256=sha(root/'EXECUTION_FILESET.json'),optimizer_updates=0,model_queries=0,canary='NOT_YET_EXECUTED'))
    print('LOCAL_BINDING_COMPLETE',flush=True)

if __name__=='__main__':
    import sys
    storage() if sys.argv[1:]==['storage'] else main()
