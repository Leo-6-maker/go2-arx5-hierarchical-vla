"""Explicit R2 admission and absolute phase stops; original codec FAIL is immutable."""
import json,time
from pathlib import Path
from contract import ROOT,sha,write

RELEASE='97773a036ebaffe59285bd2ecbee0c155f12822b'
CANONICAL=6065031954
STATUS='PASS_SUPERVISED_TRAIN_AND_INPUTS_WITH_DISCLOSED_DEV_EXTRAPOLATION'
UPDATE_STOP=1791505800 # 2026-10-09 08:30 SGT
MODEL_STOP=1791510300 # 09:45 SGT


def publication(root=ROOT):
    root=Path(root);v=json.loads((root/'design/R2_PUBLICATION_READBACK.json').read_text())
    assert v['release']==RELEASE and v['canonical']==CANONICAL and v['status']=='PASS_FIXED_RELEASE_FILES_AND_CANONICAL_READBACK'
    for name,h in v['files'].items():assert sha(root/name)==h,name
    return json.loads((root/'design/R2_DEADLINE_SCOPE_AND_LIMITS.json').read_text())


def phase_allowed(kind,reserve=0,now=None):
    assert kind in ('update','model','physics')
    assert (time.time() if now is None else now)+reserve<(UPDATE_STOP if kind=='update' else MODEL_STOP),('R2_ABSOLUTE_PHASE_STOP',kind)


def admission(root=ROOT):
    root=Path(root);publication(root);v=json.loads((root/'data/R2_SCOPED_DATA_ADMISSION.json').read_text())
    assert v['status']==STATUS and v['release']==RELEASE and v['canonical']==CANONICAL
    assert v['original_full_check_status']=='FAIL_PRESERVED' and v['TRAIN_assertions_adopted']==76928 and v['TRAIN_aggregate_maxima']=='UNOBSERVED'
    assert v['DEV_coverage']=='190/192_representative_windows_NOT_full_DEV_coverage' and len(v['known_DEV_failures'])==2
    assert v['inputs_count']==24 and v['two_worker_batches']==6
    for name,h in v['bindings'].items():assert sha(root/'data'/name)==h,name
    assert sha(root/'data/CANARY_INPUT.npz')==v['canary_input_sha256']
    return v


def create():
    import os,numpy as np
    assert os.environ['CUDA_VISIBLE_DEVICES']=='';publication()
    from data import Dataset
    d=ROOT/'data';rt=ROOT/'runtime';failed=rt/'codec_check-BANK.CPU_COMPLETE.json';log=rt/'codec_check-BANK.CPU.log'
    assert json.loads(failed.read_text())['returncode']==1
    bank=json.loads((d/'SOURCE_BANK.json').read_text());eps=bank['admitted_episodes'];first=next(i for i,e in enumerate(eps) if e['split']=='DEV')
    assert first==128 and all(e['split']=='TRAIN' and e['eligible_H40']==601 for e in eps[:first]) and eps[first]['episode']=='T1-DEV-285200'
    original=ROOT/'design/CODE_SEAL_BANK.json';seal=json.loads(original.read_text());checker=ROOT/'runtime/original-codec-checker.py'
    assert sha(checker)==seal['files']['codec_check.py']=='f838eb74ca67adc880e244d5ad51028f5577df8e25d6a7c2c49f6af680224e0a'
    assert "AssertionError: ('T1-DEV-285200', 20," in log.read_text()
    assert json.loads((rt/'data-BANK.CPU_COMPLETE.json').read_text())['returncode']==0
    inputs=json.loads((d/'NATIVE_INPUTS_CHECK.json').read_text());gap=json.loads((ROOT/'audit/codec-gap/REPORT.json').read_text())
    assert inputs['status']=='PASS_ACTUAL_NATIVE_INPUTS_ONLY_NOT_CODEC_ADMISSION' and len(inputs['inputs'])==24 and len(inputs['two_worker_native_batch_hashes'])==6
    assert json.loads((rt/'NATIVE_INPUTS_ONLY.CPU_COMPLETE.json').read_text())['returncode']==0
    assert all(inputs['bindings'][n]==h for n,h in gap['bindings'].items())
    for name,h in inputs['bindings'].items():assert sha(d/name)==h,name
    for name,h in [('statistics.json','c3ec9c6b797bb735e3ac38d611b409262de04bbb965cb68b7426df749ba7fb81'),('processor_config.json','fae35db9d84083cc1faac16442c062c74a22691fa3358cfc58efb759e01977d3'),('embodiment_id.json','fb47934b4bd4b5e11ec1c61ade0c5e03ca9110ec5f16abce9633b10cfd9edfbb')]:assert sha(d/name)==h
    members=[dict(path=n,bytes=v['bytes'],sha256=v['sha256']) for ep in eps for n,v in ep['members'].items()]
    assert all((d/m['path']).is_file() and (d/m['path']).stat().st_size==m['bytes'] for m in members)
    dataset=Dataset(d,d/'PLACE4096.jsonl');row=dataset.rows[0];assert row['split']=='TRAIN' and row['task']=='T1'
    base,wrist,state,_=dataset.raw(row);path=d/'CANARY_INPUT.npz'
    content=dict(actual_state=state,base_raw=base,wrist_raw=wrist,instruction=np.asarray(row['instruction']))
    if not path.exists():np.savez_compressed(path,**content)
    with np.load(path) as f:assert set(f.files)==set(content) and all(np.array_equal(f[n],v) for n,v in content.items())
    proof_paths=['design/CODE_SEAL_BANK.json','runtime/original-codec-checker.py','runtime/codec_check-BANK.CPU_COMPLETE.json','runtime/codec_check-BANK.CPU.log','runtime/data-BANK.CPU_COMPLETE.json','runtime/NATIVE_INPUTS_ONLY.CPU_COMPLETE.json','audit/codec-gap/REPORT.json']
    proof_bindings={}
    for n in proof_paths:
        dst=d/'R2_PROVENANCE'/n;dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():dst.write_bytes((ROOT/n).read_bytes())
        assert sha(dst)==sha(ROOT/n);proof_bindings[str(dst.relative_to(d))]=sha(dst)
    v=dict(status=STATUS,release=RELEASE,canonical=CANONICAL,original_full_check_status='FAIL_PRESERVED',TRAIN_assertions_adopted=76928,TRAIN_aggregate_maxima='UNOBSERVED',DEV_coverage='190/192_representative_windows_NOT_full_DEV_coverage',known_DEV_failures=[r for r in gap['rows'] if not r['gate_pass']],bindings=dict(inputs['bindings'],**proof_bindings,NATIVE_INPUTS_CHECK=sha(d/'NATIVE_INPUTS_CHECK.json')),inputs_count=24,two_worker_batches=6,canary_source_UID=row['uid'],canary_input_sha256=sha(path),proofs={n:sha(ROOT/n) for n in proof_paths},members_available=len(members),source_member_bytes=sum(m['bytes'] for m in members),destination_full_readback_required=True,new_model_queries=0,new_optimizer_updates=0,new_physics=0,selected='NONE',deployment='NONE')
    v['bindings']['NATIVE_INPUTS_CHECK.json']=v['bindings'].pop('NATIVE_INPUTS_CHECK')
    out=d/'R2_SCOPED_DATA_ADMISSION.json'
    if out.exists():assert json.loads(out.read_text())==v
    else:write(out,v)
    assert admission()==v
    write(rt/'R2_ADMISSION_COMPLETE.json',dict(status=STATUS,release=RELEASE,canonical=CANONICAL,data_codec_sha256=sha(out),canary_input_sha256=sha(path),canary_source_UID=row['uid'],new_model_queries=0,new_optimizer_updates=0,new_physics=0,selected='NONE',deployment='NONE'))
    return v


def self_check():
    phase_allowed('update',now=UPDATE_STOP-1);phase_allowed('model',now=MODEL_STOP-1)
    for kind,stop in [('update',UPDATE_STOP),('model',MODEL_STOP),('physics',MODEL_STOP)]:
        try:phase_allowed(kind,now=stop)
        except AssertionError:pass
        else:raise AssertionError('Missing absolute stop')
    print('PASS_R2_PHASE_BOUNDARIES_NO_MODEL_QUERY')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--self-check',action='store_true');a=p.parse_args()
    self_check() if a.self_check else print(json.dumps(create()))
