"""Fixed closed48 collection package; no full-goal terminal claim or recursive dataset walk."""
import argparse,gzip,hashlib,json,os,resource,shutil,tarfile,time
from pathlib import Path

ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0286')
OUT=ROOT/'archive/correction-closed/bundle'
VIDEO=ROOT/'evidence/video/V2-native-contract'
PREFIX='evidence/PI0286/recovery-vs-replay-20261009/correction-closed'
REPO='leoyuer/pi0246-gate-pass-campaign'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def write(p,v):
    Path(p).write_text(json.dumps(v,indent=2)+'\n')


def prepare():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    resource.setrlimit(resource.RLIMIT_AS,(16<<30,16<<30));os.nice(10)
    OUT.mkdir(parents=True,exist_ok=True);assert OUT.resolve().is_relative_to(ROOT.resolve())
    delivery=json.loads((VIDEO/'VIDEO_DELIVERY.json').read_text())
    assert delivery['covered_episodes']==48 and delivery['status']=='PASS_ALL48_SAVED_RGB_MONTAGES_FULL_DECODE_AND_COMPLETE_CONDITION_INDEX'
    index=json.loads((VIDEO/'ALL48_SOURCE_INDEX.json').read_text())
    assert delivery['source_index_sha256']==sha(VIDEO/'ALL48_SOURCE_INDEX.json')
    bank=json.loads((ROOT/'correction/BANK_NATIVE_CONTRACT_V2.json').read_text())
    plan=json.loads((ROOT/'correction/C1_PLAN_NATIVE_CONTRACT_V2.json').read_text())
    gate=json.loads((ROOT/'evidence/OPTIMIZER_START_DEADLINE_CLOSED.json').read_text())
    assert bank['valid_branches']==27 and bank['physical_success_branches']==28
    assert delivery['correction_codec_admitted']==27 and delivery['C1']=='NO_TRAINING_DEADLINE_PARTIAL'
    assert gate['formal_updates']==gate['discarded_canary_updates']==gate['new_checkpoints']==0 and gate['launch_allowed'] is False
    slots=[row['slot'] for rows in index['groups'].values() for row in rows];assert len(slots)==len(set(slots))==48
    files={};raw_index=[];excluded=[]
    def add(path,name,expected=None):
        path=Path(path);assert path.is_file() and (path.resolve().is_relative_to(ROOT.resolve()) or path.resolve().is_relative_to((ROOT/'source').resolve()))
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        h=sha(path)
        if expected is not None:assert h==expected,(name,'Frozen member changed')
        row=dict(path=name,bytes=path.stat().st_size,sha256=h)
        if name in files:assert files[name][1]==row
        files[name]=(path,row)
    for slot in slots:
        proof=ROOT/'evidence/offload'/(slot+'.json');p=json.loads(proof.read_text())
        assert p['status']=='PASS_CLOSED_SLOT_DURABLE_READBACK_AND_EXACT_SSD_OFFLOAD_LOGICAL_PATH_PRESERVED'
        folder=ROOT/'durable/episodes'/slot
        assert (ROOT/'source/episodes'/slot).is_symlink() and (ROOT/'source/episodes'/slot).resolve()==folder.resolve()
        add(proof,'offload/'+proof.name)
        for name,v in sorted(p['members'].items()):
            raw_index.append(dict(slot=slot,path='source/episodes/'+slot+'/'+name,bytes=v['size'],sha256=v['sha256']))
            if Path(name).name=='observations.npz':
                excluded.append(dict(path='source/episodes/'+slot+'/'+name,bytes=v['size'],sha256=v['sha256'],reason='FULL_SAVED_RGB_RETAINED_DURABLE_NFS_AND_LINKED_BY_FULL_HASH_INDEX'))
                continue
            assert Path(name).suffix in ('.json','.jsonl','.npz','.npy'),('Unexpected raw member',name)
            add(folder/name,'source/episodes/'+slot+'/'+name,v['sha256'])
    required=(
        'source/evidence/PARENT_DOSE6144_LEARNER_TRAIN16.json','source/evidence/CORRECTIONS32.json',
        'audit/PARENT_DOSE6144_LEARNER_TRAIN16_CPU.json','correction/COLLECTION_CLOCK.json',
        'correction/CORRECTION_BANK.json','correction/C1_PLAN_AUDIT.json',
        'correction/BANK_NATIVE_CONTRACT_V2.json','correction/C1_PLAN_NATIVE_CONTRACT_V2.json',
        'correction/NATIVE_CONTRACT_V2_BOUNDARY_CHECK.json','correction/NATIVE_CONTRACT_V2_DERIVATION_RECEIPT.json',
        'evidence/OPTIMIZER_START_DEADLINE_CLOSED.json','evidence/OFFLOAD_ALL48_READBACK_CPU.json',
        'design/PROTOCOL.md','design/PLAN_AND_LIMITS.json','design/PUBLICATION_READBACK.json',
        'design/CANONICAL_READBACK.json','design/SOURCE_FILESET.json','design/SCENE_V1.json',
        'design/CODE_SEAL.json','design/RESOURCE_RESERVATIONS.json','design/RECORDING_TIER.json',
        'evidence/PARENT_MODEL_BINDING.json','evidence/CORE_FINAL_SEAL.json',
        'evidence/video/ALL48_SOURCE_INDEX.json','evidence/video/VIDEO_DELIVERY.json','evidence/video/index.html',
        'evidence/video/video_saved_montage.py','evidence/video/video_saved_montage_v2.py',
        'evidence/video/V2-native-contract/ALL48_SOURCE_INDEX.json',
        'evidence/video/V2-native-contract/VIDEO_DELIVERY.json','evidence/video/V2-native-contract/index.html',
        'evidence/video/VIDEO_V2_ERRATUM.json')
    for name in required:add(ROOT/name,name)
    for name in ('correction/CORRECTION_TRANSFER_MANIFEST.json','correction/PRECLIP_FAILED15_DETAILS.json'):
        if (ROOT/name).exists():add(ROOT/name,name)
    seal=json.loads((ROOT/'design/CODE_SEAL.json').read_text())
    for name,h in seal['files'].items():add(ROOT/'code'/name,'code/'+name,h)
    for group in ('LEARNER16','CORRECTION03S16','CORRECTION12S16'):
        for suffix in ('_SOURCE_INDEX.json','_VIDEO.json','_DELIVERY.json'):
            name='evidence/video/'+group+suffix;add(ROOT/name,name)
        receipt=json.loads((VIDEO/(group+'_VIDEO.json')).read_text())
        add(VIDEO/(group+'_VIDEO.json'),'evidence/video/V2-native-contract/'+group+'_VIDEO.json')
        video=VIDEO/(group+'.mp4')
        add(video,'evidence/video/V2-native-contract/'+video.name,receipt['sha256'])
        target=OUT/video.name
        if target.exists():assert sha(target)==receipt['sha256']
        else:shutil.copyfile(video,target)
    inventory=dict(status='FIXED_CLOSED_COLLECTION48_COMPLETE_MEMBER_HASH_INDEX_AND_COMPACT_PAYLOAD',scope='C1_CORRECTION_COLLECTION_CLOSED_ONLY',
        repo_id=REPO,prefix=PREFIX,fixed_slots=slots,covered_valid_episodes=48,learner_physical_success=12,correction_physical_success=28,
        source_contract_codec_valid_branches=27,original_V1_extra_gate_valid_branches=15,
        V1_extra_zero_preclip_gate_superseded=True,bank_status=bank['status'],
        implemented_sampler_status=plan['status'],implemented_global_equal_branch_constraint_not_release_requirement=True,
        C1='NO_TRAINING_DEADLINE_PARTIAL',optimizer_gate_reason=gate['reason'],
        formal_updates=0,discarded_canary_updates=0,new_checkpoints=0,selected='NONE',deployment='NONE',
        members=[v for _,v in sorted(files.values(),key=lambda a:a[1]['path'])],all_original_raw_members=raw_index,excluded_payload=excluded,
        original64_dataset_included=False,weights_included=False,credentials_or_hf_cache_included=False,
        original_NFS_raw_retained=True,new_model_calls=0,new_physics=0,new_render=0)
    write(OUT/'INVENTORY.json',inventory)
    package=OUT/'evidence.tar.gz';partial=OUT/'evidence.partial.tar.gz';assert not package.exists() and not partial.exists()
    with partial.open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',compresslevel=1,mtime=0) as gz,tarfile.open(fileobj=gz,mode='w') as tar:
        for path,row in sorted(files.values(),key=lambda a:a[1]['path']):
            info=tarfile.TarInfo(row['path']);info.size=row['bytes'];info.mode=0o644;info.mtime=0
            with path.open('rb') as f:tar.addfile(info,f)
    partial.replace(package)
    expected={v['path']:v for v in inventory['members']};seen=set()
    with tarfile.open(package,'r:gz') as tar:
        for member in tar:
            assert member.name in expected and member.name not in seen and member.isfile();row=expected[member.name]
            h=hashlib.sha256()
            with tar.extractfile(member) as f:
                for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
            assert member.size==row['bytes'] and h.hexdigest()==row['sha256'],member.name;seen.add(member.name)
    assert seen==set(expected)
    (OUT/'README.md').write_text('''# PI0286 correction collection closed evidence

Scope:16 frozen parent learner TRAIN episodes and32 fixed3s/12s teacher correction branches, source Panda native PD, NON-REALTIME.
Parent physical success12/16. Correction physical success28/32; original physical codec valid27/32 under versioned native contract V2. No training: protocol section6 latest arm-start deadline elapsed and no replacement lease was authorized. Formal updates, canary updates, new checkpoints:0. Overall PI0286 scientific work remains managed by the owner. Selected and deployment:NONE.

Erratum: V1 added an extra zero-preclip gate absent from the release protocol and reported15/32. V1 receipts and original source evidence are preserved; V2 reclassifies12 preclip-only cases using the saved error values. The remaining SO3 failure and four physical failures are preserved. The one implemented global equal-branch sampler flow was infeasible (hold-only minimum1517 versus phase4quota205); this does not establish that protocol within-bucket equal weighting is infeasible. Data codec validity, sampler implementation feasibility, and the optimizer time gate are separate evidence fields.

Three videos include every fixed condition, both actual saved cameras, physical and codec outcomes. Each video has four33s pages of four conditions; playback follows recorded simulation time.
INVENTORY.json includes the complete original raw member hash index. Full observations.npz RGB payloads remain on durable Core NFS; their hashes and sizes are included. The compact tar includes original saved results, cards, contacts, physics, commands, installed actions, query evidence, independent CPU proofs, frozen code/card identity, and the three videos.
The original interrupted engineering prefix and its same-condition replacement proof are preserved alongside the48 valid episodes. Three final V2 videos are included; V1 video receipt hashes and indices are retained while V1 MP4 originals remain on Core NFS.
Original64 dataset, weights, credentials, runtime keys, and HF caches are excluded. Original NFS evidence is retained. New model, rendering, and physics calls:0.
''',encoding='utf8')
    payloads={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in OUT.iterdir() if p.is_file()}
    result=dict(status='PASS_EXACT_CLOSED_COLLECTION_PACKAGE_TAR_ALL_MEMBERS_REHASHED_READY_FOR_PUBLICATION',scope='C1_CORRECTION_COLLECTION_CLOSED_ONLY',
        repo_id=REPO,prefix=PREFIX,files=payloads,total_bundle_bytes=sum(v['bytes'] for v in payloads.values()),included_source_bytes=sum(v['bytes'] for v in inventory['members']),
        complete_original_raw_index_bytes=sum(v['bytes'] for v in raw_index),excluded_full_RGB_bytes=sum(v['bytes'] for v in excluded),tar_members=len(expected),original_raw_members=len(raw_index),
        fixed_valid_episodes=48,source_contract_codec_valid_branches=27,correction_physical_success=28,
        C1='NO_TRAINING_DEADLINE_PARTIAL',formal_updates=0,discarded_canary_updates=0,new_checkpoints=0,
        new_model_calls=0,new_physics=0,new_render=0)
    write(ROOT/'archive/correction-closed/PACKAGE_READY.json',result);print(json.dumps(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare',));a=p.parse_args();prepare()
