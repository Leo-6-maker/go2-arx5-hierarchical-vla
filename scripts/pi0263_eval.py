"""Freeze and execute the unchanged PI0261/2 panel union, with first-valid rows."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0263')
OLD = ROOT.with_name('PI0261')
CROSS = ROOT.with_name('PI0262')
DATA = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0260/R2')
sys.path[:0] = [str(OLD), str(CROSS), str(DATA/'R3')]
import pi0261_eval as legacy
import pi0262_crossed as crossed
from pi0263_plan import read, sha, write, AUTHORITY
from pi0260_r3_inputs import batch, identity
r3 = legacy.r3


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def source(row):
    if row['kind'] == 'crossed':
        packet, expert = crossed.packet(row['source_row'])
    else:
        card, pairs = legacy.plan()
        motion = read(r3.R3/'motion2_eval_cases.json')['cases']
        original = read(r3.R3/'O64_eval_cases.json')['rows']
        packet, expert, _ = legacy.source(row['source_row'],pairs,motion,original)
    return packet, expert


def tuple_identity(policy, packet, expert, noise):
    return dict(condition_sha256=identity(policy), state_sha256=digest(packet['state_v1']),
                reference_sha256=digest(expert), body_sha256=digest(packet['body_rgb']),
                wrist_sha256=digest(packet['wrist_rgb']), noise_sha256=digest(noise))


def freeze():
    proc = crossed.processor()
    binding = read(OLD/'model_bindings/P2000.json')
    assert binding['checkpoint_tree_sha256'] == '882931d754cb25653105a9213fe0dbeb4d2b14e463efb27e49c6101fd5fc5e4b'
    assert '@875abdde2cc7c9e6858fac008e53f2f4b3375255/' in binding['source_revision']
    assert read(CROSS/'SOURCE_BINDING_AUDIT.json')['models']['P2000']['binding'] == binding
    noise = {}
    for name in ('B0','B1'):
        with np.load(r3.R3/'noise'/(name+'.npz'),allow_pickle=False) as z:
            noise[name] = z['float32'].copy()
    rows, lookup, aliases = [], {}, []
    card, _ = legacy.plan()
    originals = [r for r in card['rows'] if r['point']=='P2000']
    crossrows = [r for r in read(CROSS/'CROSSED_EVAL_CARD.json')['rows'] if r['point']=='P2000']
    assert (len(originals),len(crossrows)) == (680,640)
    for kind, panel in (('legacy',originals),('crossed',crossrows)):
        for index, old in enumerate(panel):
            spec = dict(kind=kind,source_row=old)
            packet, expert = source(spec)
            policy = batch(proc,packet)['inputs']
            identities = tuple_identity(policy,packet,expert,noise[old['noise']])
            key = json.dumps(identities,sort_keys=True)
            if kind == 'legacy':
                if old['cohort']=='TRAIN_ENTRY':
                    oldid=f"TRAIN_ENTRY-{old['pair_index']:02d}-c{old['command']}-{old['noise']}"
                else:
                    oldid=f"{old['cohort']}-{old['case_index']:03d}-{old['noise']}"
                cell=OLD/'eval/P2000/cells'/oldid
            else:
                cell=CROSS/'eval/P2000/cells'/old['row_id']
            record=read(cell/'record.json')
            assert record['status']=='VALID' and sha(cell/'tensors.npz')==record['tensor_sha256']
            if kind=='legacy':
                assert record['model_binding_sha256']==sha(OLD/'model_bindings/P2000.json')
                assert record['eval_card_sha256']==sha(OLD/'FIXED_EVAL_CARD.json')
                assert record['condition_sha256']==identities['condition_sha256']
            else:
                assert record['row']==old and record['eval_card_sha256']==sha(CROSS/'CROSSED_EVAL_CARD.json')
                assert old['condition_sha256']==identities['condition_sha256']
            with np.load(cell/'tensors.npz',allow_pickle=False) as z:
                assert np.array_equal(z['expert'],expert) and np.array_equal(z['state'],packet['state_v1'])
                assert np.array_equal(z['noise'],noise[old['noise']])
            if kind=='crossed' and key in lookup:
                uid=lookup[key]
                primary=rows[int(uid.split('-')[1])]
                with np.load(Path(primary['donor_cell'])/'tensors.npz',allow_pickle=False) as a, np.load(cell/'tensors.npz',allow_pickle=False) as b:
                    assert np.array_equal(a['raw'],b['raw']) and np.array_equal(a['decoded'],b['decoded'])
            else:
                uid=f'u-{len(rows):04d}'
                rows.append(dict(spec,uid=uid,noise=old['noise'],identities=identities,
                    donor_cell=str(cell), donor_record_sha256=sha(cell/'record.json'),
                    donor_tensor_sha256=sha(cell/'tensors.npz')))
                if kind=='legacy':
                    assert key not in lookup, ('DUPLICATE_ORIGINAL_TUPLE',index)
                    lookup[key]=uid
            aliases.append(dict(panel=kind,index=index,uid=uid,source_row=old))
    assert len(rows)==1152 and len(aliases)==1320
    assert sum(a['panel']=='crossed' and int(a['uid'].split('-')[1])<680 for a in aliases)==168
    write(ROOT/'UNION_EVAL_CARD.json',dict(schema='PI0263_FIXED_UNION_3456_V1',authority_commit=AUTHORITY,
        legacy_card_sha256=sha(OLD/'FIXED_EVAL_CARD.json'),crossed_card_sha256=sha(CROSS/'CROSSED_EVAL_CARD.json'),
        statistics_sha256=sha(r3.ROOT/'dataset/statistics.json'),noise_manifest_sha256=sha(r3.R3/'noise_manifest.json'),
        donor_binding=binding,rows=rows,aliases=aliases,points=['P2000','F2000','X2000'],
        rows_per_point=1152,logical_rows=3456,planned_new_queries=2304,new_attempt_cap=2432,inference_head_cap=9728,
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print(json.dumps(dict(status='PASS_PI0263_UNION_FROZEN_AND_DONOR_1152_REUSE',rows=1152,overlap=168)))


def append(path,row):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(row,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())


def count(path):
    return sum(1 for _ in path.open()) if path.exists() else 0


def run(point):
    import fcntl
    import torch
    from transformers import AutoModel
    from gr00t.data.embodiment_tags import EmbodimentTag
    import pi0242_train_infer_bridge as bridge
    import stage21_same_scene_groot_inference as stage21
    import pi0243_staged_learning_to_sim as stage43
    import pi0228_train_calib_checkpoint_selection as selector
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat,rot6d_rows_to_matrix
    card=read(ROOT/'UNION_EVAL_CARD.json')
    assert sha(r3.ROOT/'dataset/statistics.json')==card['statistics_sha256']
    lock=(ROOT/'dispatcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    proc=crossed.processor()
    binding=read(ROOT/'model_bindings'/f'{point}.json') if point!='P2000' else card['donor_binding']
    model=None
    if point!='P2000':
        assert point in ('F2000','X2000')
        for name,value in binding['model_files_sha256'].items():
            assert sha(Path(binding['view'])/name)==value
    attempts,headsfile=ROOT/'attempts.jsonl',ROOT/'head_forwards.jsonl'
    heads=count(headsfile)
    cache={}
    valid_fresh=0
    for path in (ROOT/'eval').glob('*/cells/*/record.json'):
        saved=read(path)
        if saved['point'] in ('F2000','X2000') and saved.get('forward_reused') is None:
            valid_fresh+=1
    for row in card['rows']:
        cell=ROOT/'eval'/point/'cells'/row['uid'];recordpath=cell/'record.json'
        if recordpath.exists():
            prior=read(recordpath)
            assert prior['row']==row and prior['point']==point and prior['status']=='VALID'
            assert prior['eval_card_sha256']==sha(ROOT/'UNION_EVAL_CARD.json')
            assert prior['model_binding']==binding and sha(cell/'tensors.npz')==prior['tensor_sha256']
            cache.setdefault((row['identities']['condition_sha256'],row['identities']['noise_sha256']),cell)
            continue
        packet,expert=source(row);policy=batch(proc,packet)['inputs']
        with np.load(r3.R3/'noise'/(row['noise']+'.npz'),allow_pickle=False) as z:
            noise=z['float32'].copy()
        assert tuple_identity(policy,packet,expert,noise)==row['identities']
        cell.mkdir(parents=True,exist_ok=True);tensor=cell/'tensors.npz'
        src=Path(row['donor_cell'])/'tensors.npz' if point=='P2000' else tensor
        forward_reused=None
        forward_key=(row['identities']['condition_sha256'],row['identities']['noise_sha256'])
        if point!='P2000' and not src.exists() and forward_key in cache:
            prior_cell=cache[forward_key];prior=read(prior_cell/'record.json')
            assert prior['model_binding']==binding and prior['status']=='VALID'
            assert sha(prior_cell/'tensors.npz')==prior['tensor_sha256']
            with np.load(prior_cell/'tensors.npz',allow_pickle=False) as z:
                assert np.array_equal(z['state'],packet['state_v1']) and np.array_equal(z['noise'],noise)
                values={k:z[k].copy() for k in ('raw','trace_x','trace_v','trace_t','decoded')}
            values.update(expert=expert,state=packet['state_v1'],noise=noise)
            forward_reused=dict(uid=prior_cell.name,record_sha256=sha(prior_cell/'record.json'),tensor_sha256=prior['tensor_sha256'])
        journal=cell/'FORWARD_RECEIPT.json'
        if point=='P2000':
            assert sha(src)==row['donor_tensor_sha256'] and sha(src.parent/'record.json')==row['donor_record_sha256']
        if forward_reused:
            pass
        elif src.exists():
            with np.load(src,allow_pickle=False) as z:
                values={k:z[k].copy() for k in z.files}
            assert np.array_equal(values['expert'],expert) and np.array_equal(values['state'],packet['state_v1'])
            assert np.array_equal(values['noise'],noise)
            if point!='P2000':
                saved=read(journal)
                assert saved['model_binding']==binding and saved['identities']==row['identities']
                forward_reused=saved['forward_reused']
        else:
            if model is None:
                model=AutoModel.from_pretrained(binding['view'],local_files_only=True,trust_remote_code=True).eval().to('cuda:0',dtype=torch.bfloat16)
                assert model.action_head.num_inference_timesteps==4
            ordinal=count(attempts)+1
            assert ordinal<=card['new_attempt_cap']
            assert count(attempts)-valid_fresh<=128, 'NO_VALID_RESULT_TECHNICAL_RESERVE'
            append(attempts,dict(ordinal=ordinal,point=point,uid=row['uid'],identities=row['identities']))
            def hook(_module,_inputs):
                nonlocal heads
                heads+=1;assert heads<=card['inference_head_cap']
                append(headsfile,dict(ordinal=heads,attempt_ordinal=ordinal,point=point,uid=row['uid']))
            handle=model.action_head.model.register_forward_pre_hook(hook)
            try:
                with torch.inference_mode():stock=bridge._stock_query(model,policy,stage21,noise)
            finally:handle.remove()
            assert len(stock['trace'])==4 and np.array_equal(stock['noise'],noise)
            values=dict(raw=np.asarray(stock['raw'],np.float32),expert=expert,state=packet['state_v1'],noise=noise,
                trace_x=np.stack([x['x'] for x in stock['trace']]),trace_v=np.stack([x['v'] for x in stock['trace']]),
                trace_t=np.stack([x['t'] for x in stock['trace']]))
            values['decoded']=bridge._decode_physical(proc,EmbodimentTag.NEW_EMBODIMENT,values['raw'],packet['state_v1'])
        decoded=bridge._decode_physical(proc,EmbodimentTag.NEW_EMBODIMENT,values['raw'],packet['state_v1'])
        assert np.max(np.abs(decoded-values['decoded']))<=1e-7
        if point!='P2000' and not journal.exists():
            write(journal,dict(model_binding=binding,identities=row['identities'],forward_reused=forward_reused))
        if not tensor.exists():
            tmp=cell/'tensors.partial.npz';np.savez_compressed(tmp,**values)
            with tmp.open('rb') as stream:os.fsync(stream.fileno())
            os.replace(tmp,tensor)
        metrics=stage43._screen_metrics(decoded[0].astype(np.float64),expert.astype(np.float64),None)
        state=packet['state_v1']
        gate=TrajectorySafetyGateV1().evaluate(selector._build_chunk(decoded,chunk_id=f'pi0263-{point}-{row["uid"]}'),now=0.,
            achieved_position=state[:3],achieved_orientation=matrix_to_quat(rot6d_rows_to_matrix(state[None,3:9]))[0])
        write(recordpath,dict(status='VALID',point=point,row=row,metrics=metrics,gate=gate.as_dict(),
            tensor_sha256=sha(tensor),eval_card_sha256=sha(ROOT/'UNION_EVAL_CARD.json'),model_binding=binding,
            forward_reused=forward_reused))
        if point!='P2000' and forward_reused is None:
            valid_fresh+=1
        cache.setdefault(forward_key,cell)
        print(json.dumps(dict(status='PI0263_VALID',point=point,uid=row['uid'])),flush=True)
    print(json.dumps(dict(status='PI0263_POINT_COMPLETE',point=point,attempts=count(attempts),heads=count(headsfile))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--point',choices=('P2000','F2000','X2000'))
    args=p.parse_args();freeze() if args.freeze else run(args.point)
