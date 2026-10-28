"""Seal deterministic TRAIN/DEV probe identities and expert positive controls."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys

from .durable import read,write
from .train_adapter import NativeDataset,sha
from .eval_contract import uniform_train_ids,stage_anchors,noise_seed,gate_configuration,expert_decode,gate_action,error_metrics


def seal(args):
    root=Path(args.root);target=root/'evidence/OPEN_LOOP_PROBE_SEAL.json'
    if target.exists():raise FileExistsError('Never adapt the probe to model outputs')
    if read(root/'BANK_V2_ADMISSION.json')['status']!='ARX5_BANK_V2_QUALIFIED':raise ValueError('BANK_V2_RAW_GATE_REQUIRED')
    if read(root/'evidence/DATALOADER_SMOKE.json')['status']!='NATIVE_DATALOADER_PASS':raise ValueError('ALL_WINDOW_NATIVE_CODEC_GATE_REQUIRED')
    sys.path.insert(0,str(Path(args.frozen_code)/'src'))
    code_seal=read(root/'CODE_SEAL.json')
    files=['src/go2arx5_vla/contracts/trajectory.py','src/go2arx5_vla/safety/trajectory_gate.py','src/go2arx5_vla/trajectory/so3.py']
    for path in files:
        if sha(Path(args.frozen_code)/path)!=code_seal['files'][path]:raise ValueError('FROZEN_GATE_CODE_CHANGED')
    manifest=read(Path(args.dataset)/'DATASET_MANIFEST.json')
    train=uniform_train_ids([r['id'] for r in manifest['episodes'] if r['split']=='TRAIN'])
    dev=sorted(r['id'] for r in manifest['episodes'] if r['split']=='DEV')
    if not 14<=len(dev)<=16:raise ValueError('QUALIFIED_DEV_REQUIRED')
    rows=[];experts=[];omissions=[]
    for split,identifiers in (('TRAIN',train),('DEV',dev)):
        dataset=NativeDataset(args.dataset,args.source,args.cosmos,split)
        dataset.initialize()
        first={r['id']:i for i,(r,anchor) in enumerate(dataset.rows) if anchor==0}
        records={r['id']:r for r in manifest['episodes']}
        for identifier in identifiers:
            row=records[identifier];raw=Path(row['raw_root'])
            if sha(raw/'samples.json')!=row['samples_sha256']:raise ValueError('PROBE_RAW_SOURCE_CHANGED')
            anchors,missing=stage_anchors(read(raw/'samples.json'))
            omissions.append(dict(id=identifier,missing_stages=missing,no_replacement=True))
            for item in anchors:
                anchor=item['anchor'];state,commands,images,text=dataset.raw(first[identifier]+anchor)
                decoded,decode_receipt=expert_decode(dataset.processor,state,commands)
                gate=gate_action(decoded,state,anchor/20,identifier+':'+str(anchor))
                probe=dict(id=identifier,split=split,anchor=anchor,stages=item['stages'],
                    samples_sha256=row['samples_sha256'],numeric_sha256=row['sha256'],
                    noise_seeds={str(bank):str(noise_seed(bank,identifier,anchor)) for bank in (29201,29202)})
                rows.append(probe)
                experts.append(dict(id=identifier,anchor=anchor,split=split,decode=decode_receipt,gate=gate,
                    errors=error_metrics(decoded,commands,state),RGB_sha256=[hashlib.sha256(image.tobytes()).hexdigest() for image in images]))
    if len(rows)>192:raise ValueError('PROBE_WINDOW_BUDGET')
    positive=all(row['gate']['decision']!='REJECT' for row in experts)
    u=resource.getrusage(resource.RUSAGE_SELF)
    result=dict(status='SEALED_BEFORE_MAIN' if positive else 'EXPERT_POSITIVE_CONTROL_FAILED',
        schema='PI0292_OPEN_LOOP_PROBE_V1',TRAIN_ids=train,DEV_ids=dev,probes=rows,omissions=omissions,
        selection='Numeric sort; TRAIN index j*(N-1)//15, j=0..15; all DEV',
        anchors='0 and first observed close/lift/carry/release/hold stage at index<=600; deduplicate; never replace missing',
        noise_banks=[29201,29202],noise_derivation='uint64_be(SHA256(PI0292/OFFLINE/{bank}/{id}/{anchor})[:8]) mod 2**63',
        per_profile_queries=len(rows)*2,total_three_profile_queries=len(rows)*6,native_inference_steps=4,
        statistics_sha256=manifest['statistics_sha256'],manifest_sha256=sha(Path(args.dataset)/'DATASET_MANIFEST.json'),
        gate=gate_configuration(),gate_code_hashes={p:sha(Path(args.frozen_code)/p) for p in files},
        evaluation_script_hashes={name:sha(Path(__file__).parent/name) for name in
            ('seal_probe.py','eval_contract.py','generate_probes.py')},
        expert_positive_controls=experts,expert_rejects=sum(r['gate']['decision']=='REJECT' for r in experts),
        retain_rejected_errors=True,per_episode_macro_required=True,models_loaded=0,physics_steps=0,
        CPU_s=u.ru_utime+u.ru_stime)
    write(target,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','dataset','source','cosmos','frozen-code'):p.add_argument('--'+key,required=True)
    result=seal(p.parse_args());print(json.dumps({k:result[k] for k in ('status','per_profile_queries','expert_rejects','CPU_s')}))
    if result['status']!='SEALED_BEFORE_MAIN':raise SystemExit(2)
