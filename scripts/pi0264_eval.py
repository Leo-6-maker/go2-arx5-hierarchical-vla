"""PI0264 first-valid fixed evaluation; no quality admission until both finals freeze."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from pi0264_plan import read,sha,write
from pi0264_freeze_eval import source,noise as noise_for,previous
from pi0260_r3_inputs import batch

ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0264')
r3=previous.r3
crossed=previous.crossed
tuple_identity=previous.tuple_identity
append=previous.append
count=previous.count


def assert_release():
    path=ROOT/'MAIN_TRAINING_AUDIT.json'
    assert path.is_file(), 'QUALITY_HELD_UNTIL_BOTH_FINAL2048'
    training=read(path)
    assert training['status']=='PASS_BOTH_2048_MATCHED_TRAINING_INTEGRITY'
    assert training['optimizer_updates']=={'B':2048,'L':2048}
    for arm in ('B','L'):
        binding=read(ROOT/'model_bindings'/f'{arm}2048.json')
        receipt=read(ROOT/'evidence'/arm/'checkpoint-2048.archive_receipt.json')
        assert binding['point']==arm+'2048' and binding['view_type']=='model-only'
        assert receipt['status']=='UPLOADED' and receipt['step']==2048 and receipt['arm']==arm
        assert binding['checkpoint_tree_sha256']==receipt['tree_sha256']
        assert binding['source_revision']==f"{receipt['repo_id']}@{receipt['revision']}/{receipt['prefix']}"
        scope=read(ROOT/'evidence'/f'{arm}2048'/'FROZEN_SCOPE.json')
        assert scope['status']=='PASS_PI0264_SERIALIZED_PRECISION_AND_FROZEN_TENSOR_IDENTITY'
        assert scope['model_binding']==binding and scope['model_queries']==scope['gradient_queries']==0
    assert sha(ROOT/'raw_completed_updates.jsonl')==training['completed_updates_sha256']
    assert sha(ROOT/'raw_training_heads.jsonl')==training['training_heads_sha256']


def run(point):
    assert_release()
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
    binding=card['donor_binding'] if point=='X2000' else read(ROOT/'model_bindings'/f'{point}.json')
    model=None
    assert point in ('X2000','B2048','L2048')
    for name,value in binding['model_files_sha256'].items():
        assert sha(Path(binding['view'])/name)==value
    attempts,headsfile=ROOT/'attempts.jsonl',ROOT/'head_forwards.jsonl'
    heads=count(headsfile)
    cache={}
    valid_fresh=0
    for path in (ROOT/'eval').glob('*/cells/*/record.json'):
        saved=read(path)
        if saved['point'] in card['points'] and not saved.get('donor_reused') and saved.get('forward_reused') is None:
            valid_fresh+=1
    for row in card['rows']:
        donor_reused=point=='X2000' and row['panel']=='A'
        cell=ROOT/'eval'/point/'cells'/row['uid'];recordpath=cell/'record.json'
        if recordpath.exists():
            prior=read(recordpath)
            assert prior['row']==row and prior['point']==point and prior['status']=='VALID'
            assert prior['eval_card_sha256']==sha(ROOT/'UNION_EVAL_CARD.json')
            assert prior['model_binding']==binding and sha(cell/'tensors.npz')==prior['tensor_sha256']
            cache.setdefault((row['identities']['condition_sha256'],row['identities']['noise_sha256']),cell)
            continue
        packet,expert=source(row);policy=batch(proc,packet)['inputs']
        noise=noise_for(row)
        assert tuple_identity(policy,packet,expert,noise)==row['identities']
        cell.mkdir(parents=True,exist_ok=True);tensor=cell/'tensors.npz'
        src=Path(row['donor_cell'])/'tensors.npz' if donor_reused else tensor
        forward_reused=None
        forward_key=(row['identities']['condition_sha256'],row['identities']['noise_sha256'])
        if not donor_reused and not src.exists() and forward_key in cache:
            prior_cell=cache[forward_key];prior=read(prior_cell/'record.json')
            assert prior['model_binding']==binding and prior['status']=='VALID'
            assert sha(prior_cell/'tensors.npz')==prior['tensor_sha256']
            with np.load(prior_cell/'tensors.npz',allow_pickle=False) as z:
                assert np.array_equal(z['state'],packet['state_v1']) and np.array_equal(z['noise'],noise)
                values={k:z[k].copy() for k in ('raw','trace_x','trace_v','trace_t','decoded')}
            values.update(expert=expert,state=packet['state_v1'],noise=noise)
            forward_reused=dict(uid=prior_cell.name,record_sha256=sha(prior_cell/'record.json'),tensor_sha256=prior['tensor_sha256'])
        journal=cell/'FORWARD_RECEIPT.json'
        if donor_reused:
            assert sha(src)==row['donor_tensor_sha256'] and sha(src.parent/'record.json')==row['donor_record_sha256']
        if forward_reused:
            pass
        elif src.exists():
            with np.load(src,allow_pickle=False) as z:
                values={k:z[k].copy() for k in z.files}
            assert np.array_equal(values['expert'],expert) and np.array_equal(values['state'],packet['state_v1'])
            assert np.array_equal(values['noise'],noise)
            if not donor_reused:
                saved=read(journal)
                assert saved['model_binding']==binding and saved['identities']==row['identities']
                forward_reused=saved['forward_reused']
        else:
            if model is None:
                model=AutoModel.from_pretrained(binding['view'],local_files_only=True,trust_remote_code=True).eval().to('cuda:0',dtype=torch.bfloat16)
                assert model.action_head.num_inference_timesteps==4
            ordinal=count(attempts)+1
            assert ordinal<=card['new_attempt_cap']
            assert count(attempts)-valid_fresh<=card['invalid_only_reserve_cap'], 'NO_VALID_RESULT_TECHNICAL_RESERVE'
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
        if not donor_reused and not journal.exists():
            write(journal,dict(model_binding=binding,identities=row['identities'],forward_reused=forward_reused))
        if not tensor.exists():
            tmp=cell/'tensors.partial.npz';np.savez_compressed(tmp,**values)
            with tmp.open('rb') as stream:os.fsync(stream.fileno())
            os.replace(tmp,tensor)
        metrics=stage43._screen_metrics(decoded[0].astype(np.float64),expert.astype(np.float64),None)
        state=packet['state_v1']
        gate=TrajectorySafetyGateV1().evaluate(selector._build_chunk(decoded,chunk_id=f'pi0264-{point}-{row["uid"]}'),now=0.,
            achieved_position=state[:3],achieved_orientation=matrix_to_quat(rot6d_rows_to_matrix(state[None,3:9]))[0])
        write(recordpath,dict(status='VALID',point=point,row=row,metrics=metrics,gate=gate.as_dict(),
            tensor_sha256=sha(tensor),eval_card_sha256=sha(ROOT/'UNION_EVAL_CARD.json'),model_binding=binding,
            forward_reused=forward_reused,donor_reused=donor_reused))
        if not donor_reused and forward_reused is None:
            valid_fresh+=1
        cache.setdefault(forward_key,cell)
        print(json.dumps(dict(status='PI0264_VALID',point=point,uid=row['uid'])),flush=True)
    print(json.dumps(dict(status='PI0264_POINT_COMPLETE',point=point,attempts=count(attempts),heads=count(headsfile))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',choices=('X2000','B2048','L2048'),required=True)
    run(p.parse_args().point)
