"""Prepare C0 from immutable original arrays and DOSE6144 parent; no model load."""
import argparse,json,os,time,hashlib
from pathlib import Path
from contract import ROOT,guard,sha,write
from authority import RELEASE,CANONICAL,stamp
from data import plan,immutable

FROZEN={'statistics.json':'c3ec9c6b797bb735e3ac38d611b409262de04bbb965cb68b7426df749ba7fb81',
    'processor_config.json':'fae35db9d84083cc1faac16442c062c74a22691fa3358cfc58efb759e01977d3',
    'embodiment_id.json':'fb47934b4bd4b5e11ec1c61ade0c5e03ca9110ec5f16abce9633b10cfd9edfbb'}


def alias(path,target):
    path=Path(path);target=Path(target);assert target.exists()
    if path.exists() or path.is_symlink():
        assert path.resolve()==target.resolve() or path.is_file() and target.is_file() and sha(path)==sha(target),'Own data path conflicts with original immutable data'
    else:path.symlink_to(target,target_is_directory=target.is_dir())


def prepare(root,manifest):
    root=Path(root);assert root.resolve()==ROOT.resolve();guard();prior=root.parent/'PI0285'
    parent=prior/'learning/REPAIR_A2048/main/checkpoint-2048';export=prior/'delivery/REPAIR_A2048-2048/EXPORT.json'
    expected=json.loads(Path(manifest).read_text());assert expected['status']=='PASS_FULL_FILE_READBACK' and expected['complete_trainer']
    assert expected['revision']=='75432eb91a2c602930a9e1e979aaebbeaebff02b' and expected['tree_sha256']=='fb17b65effc3ae8dcbefa784adc8f7e6fab916821d29e4b2539e5be685ed8469'
    assert len(expected['files'])==14 and {p.name for p in parent.iterdir() if p.is_file()}=={x['path'] for x in expected['files']}
    files={}
    for item in expected['files']:
        p=parent/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],item['path']
        files[item['path']]=dict(item,stamp=stamp(p));files[item['path']].pop('path')
    recovery=json.loads((parent/'PI0282_RECOVERY.json').read_text());assert recovery['cursor']==16384
    assert json.loads((parent/'trainer_state.json').read_text())['global_step']==2048
    audit=json.loads((prior/'evidence/REPAIR_A2048-2048_TRAINING_CPU.json').read_text())
    assert audit['status']=='PASS_ACTUAL_NATIVE_TRAINER_INPUT_PLAN_NOISE_LOSS_MODES_SCOPE_AND_COMPLETE_LOCAL_STATE' and audit['point']=='REPAIR_A2048-2048'
    assert audit['export_sha256']==sha(export) and audit['checkpoint']==str(parent)
    e=json.loads(export.read_text());assert all(files[n]['sha256']==h for n,h in e['model_files'].items())
    immutable(root/'design/PARENT_READBACK.json',dict(status='PASS_PI0286_EXACT_SEALED_DOSE6144_FULL_PARENT',parent=str(parent),files=files,
        HF_revision=expected['revision'],HF_tree=expected['tree_sha256'],export_sha256=sha(export),full_master_sha256=e['full_state_sha256'],
        original_full_CPU_audit_sha256=sha(prior/'evidence/REPAIR_A2048-2048_TRAINING_CPU.json'),Adam_global_update=6144))
    data=root/'data';data.mkdir(exist_ok=True)
    names=['SOURCE_BANK.json','STATISTICS_RELEASE.json','WINDOW_INDEX.json',
        'NATIVE_INPUTS_CHECK.json','CANARY_INPUT.npz',*FROZEN]
    for name in names:alias(data/name,prior/'data'/name)
    for name,h in FROZEN.items():assert sha(data/name)==h and files[name]['sha256']==h
    inherited=json.loads((prior/'data/R2_SCOPED_DATA_ADMISSION.json').read_text())
    assert inherited['status']=='PASS_SUPERVISED_TRAIN_AND_INPUTS_WITH_DISCLOSED_DEV_EXTRAPOLATION'
    assert inherited['TRAIN_assertions_adopted']==76928 and inherited['original_full_check_status']=='FAIL_PRESERVED'
    bank=json.loads((data/'SOURCE_BANK.json').read_text());eps=[v for v in bank['admitted_episodes'] if v['task']=='T1' and v['split']=='TRAIN']
    assert sorted(v['layout'] for v in eps)==list(range(285100,285164))
    members={}
    for ep in eps:
        for name,item in ep['members'].items():
            p=data/name;assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],name
            members[name]=dict(sha256=item['sha256'],stamp=stamp(p))
    immutable(data/'EXPERT_DATA_ADMISSION.json',dict(status='PASS_PI0286_ORIGINAL_EXPERT_TRAIN_INPUTS_AND_FROZEN_CODEC_ADOPTED',
        TRAIN_layouts=list(range(285100,285164)),supervised_tasks=['T1'],old_CONFIRM_or_DEV_supervision=0,original_full_codec_status='FAIL_PRESERVED',
        inherited_scope_receipt_sha256=sha(prior/'data/R2_SCOPED_DATA_ADMISSION.json'),array_members=members,
        bindings={name:sha(data/name) for name in names}))
    experts=json.loads((data/'WINDOW_INDEX.json').read_text())['rows'];rows=plan(experts,['T1'],1024)
    assert len(rows)==8192 and all(v['source']=='expert' and v['task']=='T1' and v['split']=='TRAIN' for v in rows)
    planpath=data/'C0_REPLAY1024.jsonl';payload=''.join(json.dumps(v,sort_keys=True)+'\n' for v in rows)
    if planpath.exists():assert planpath.read_text()==payload
    else:planpath.write_text(payload)
    from learning import seed_for
    schedule=[dict(phase_update=update+1,micro=micro,global_seed_index=6144+update,native_seed=seed_for(6144+update,micro)) for update in range(1024) for micro in range(2)]
    immutable(root/'design/PAIRED_NATIVE_RNG.json',dict(status='FROZEN_PAIRED_NATIVE_EPSILON_TIME_SEEDS_BEFORE_ARM_RESULTS',
        arms=['C0_REPLAY1024','C1_CORRECT1024'],lineage_offset=6144,effective_batch=8,microbatch=4,accumulation=2,
        native_learning_sha256=sha(root.parent/'PI0283/code/learning.py'),rows=schedule))
    card=dict(release=RELEASE,canonical=CANONICAL,stage='C0_REPLAY1024',parent=str(parent),parent_files=files,parent_global_update=6144,parent_phase_step=2048,
        parent_export=str(export),parent_export_sha256=sha(export),parent_full_master_sha256=e['full_state_sha256'],parent_readback_sha256=sha(root/'design/PARENT_READBACK.json'),
        updates=1024,warmup=52,peak_LRs=[5e-5,5e-6],plan_path=str(planpath),plan_sha256=sha(planpath),RNG_lineage_offset=6144,
        paired_RNG_sha256=sha(root/'design/PAIRED_NATIVE_RNG.json'),launch_allowed=False,canary='NEW_REAL_CONTINUOUS8_VS_FRESH4_RESTORE4_REQUIRED',selected='NONE',deployment='NONE')
    immutable(root/'design/C0_REPLAY1024_CARD.prospective.json',card)
    receipt=dict(status='PASS_PREPARED_PI0286_C0_DATA_PARENT_AND_PAIRED_RNG_NOT_LAUNCHED',card=card,model_queries=0,optimizer_updates=0,physics=0)
    write(root/'evidence/C0_TRAINING_PREPARATION.json',receipt);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT));p.add_argument('--manifest',required=True);a=p.parse_args();print(json.dumps(prepare(a.root,a.manifest)))
