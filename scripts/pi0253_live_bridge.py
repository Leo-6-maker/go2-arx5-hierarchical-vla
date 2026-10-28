#!/usr/bin/env python3
"""PI0253 fixed live/screen bridge. Reuse mature inference, decoder and gate."""
from pathlib import Path
from collections.abc import Mapping
import argparse
import hashlib
import inspect
import json
import os
import sys

ROOT = Path('/home/liuyu-nfs/Projects/go2-arx5')
OUT = ROOT / 'PI0253'
OLD = ROOT / 'PI0252'
INPUTS = ROOT / 'PI0250/inputs/frozen_v1'
CODE = ROOT / 'PI0250/runtime/reference-317a'
STAGE10 = Path('/home/liuyu-nfs/Projects/isaac-gr00t')
LIVE = OLD / 'pre_full_sim/G/A/mailbox/requests/model-0000.npz'
CELL = OLD / 'screen/G2000/observations/source-000001/cells/G2000__S4__pi0227_persistent_train_a_j00_nominal_s00_c0_r00_r00_227000__B0'
MODEL = OLD / 'models/G2000/inference_view_step_2000'
LANGUAGE = 'move the end effector to the red target'
REVISION = 'efecea485ae77000599d19171b02491f8d514f42'
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                  HF_HOME=str(ROOT / 'PI0250/runtime/hf_home'))
sys.path[:0] = [str(CODE / 'scripts'), str(CODE / 'src'), str(STAGE10)]
import numpy as np
import torch
import stage21_same_scene_groot_inference as s21
import pi0242_train_infer_bridge as bridge


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def leaves(value, prefix=''):
    # BF16 byte inventory, unlike legacy NumPy conversion which cannot retain BF16.
    if isinstance(value, torch.Tensor):
        v = value.detach().contiguous().cpu()
        return {prefix: dict(shape=list(v.shape), dtype=str(v.dtype), device=str(value.device),
                sha256=hashlib.sha256(v.view(torch.uint8).numpy().tobytes()).hexdigest(),
                finite=bool(torch.isfinite(v).all()))}
    if isinstance(value, np.ndarray):
        return {prefix: dict(shape=list(value.shape), dtype=str(value.dtype), device='cpu',
                sha256=hashlib.sha256(value.tobytes()).hexdigest(), finite=bool(np.isfinite(value).all()))}
    if isinstance(value, Mapping):
        return {k: v for key, item in value.items() for k, v in leaves(item, prefix+'/'+str(key)).items()}
    if isinstance(value, (list, tuple)):
        return {k: v for i, item in enumerate(value) for k, v in leaves(item, prefix+'/'+str(i)).items()}
    return {prefix: dict(type=type(value).__name__, value=value)}


def processor():
    import stage40_b0_model_control as s40
    s40._bind_local_cosmos_processor(STAGE10, INPUTS/'cosmos_snapshot')
    return s21._build_processor(STAGE10, INPUTS/'processor_base', INPUTS/'train_dataset/meta',
                               INPUTS/'modality/go2_arx5_modality.py')


def prepared(proc, emb, path):
    from gr00t.data.types import MessageType
    obs = bridge._screen_observation(path)
    msg = bridge._observation_message(obs, emb, LANGUAGE)
    return proc.collator([proc([{'type': MessageType.EPISODE_STEP.value, 'content': msg}])])


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = read(INPUTS/'source64/manifest.json')
    m = min(manifest['files'], key=lambda x: x['source_episode_id'].encode('utf8'))
    assert m['episode_index'] == 1 and m['path'] == 'source-000001.npz'
    assert sha(LIVE) == '07883fc3422c9df1e3ed6c8afa0e32de47577a75c7bf6df22117ff68535db524'
    assert sha(INPUTS/'source64'/m['path']) == m['sha256']
    old = read(CELL/'record.json')
    assert old['source_episode_id'] == m['source_episode_id'] and old['model_revision'] == REVISION
    assert sha(CELL/'tensors.npz') == old['tensor_sha256']
    with np.load(CELL/'tensors.npz') as d: nm = d['noise_initial'].copy()
    nlive = OLD/'pre_full_sim/G/A/mailbox/responses/model-0000_initial_noise.npz'
    assert sha(nlive) == '12323ebb5a91d4a2696c076f700bd5f6b0252b7d6beb3d90973cc18937b13a44'
    with np.load(nlive) as d: live_bytes, nl = d['raw_bytes'].copy(), d['float_values'].copy()
    noises = {}
    for key, value in [('N_live', nl), ('N_M', nm)]:
        t = torch.from_numpy(value).to(torch.bfloat16)
        assert torch.equal(t.float(), torch.from_numpy(value))
        raw = t.view(torch.uint8).numpy().copy()
        if key == 'N_live': assert np.array_equal(raw, live_bytes)
        noises[key] = dict(dtype=str(t.dtype), shape=list(t.shape),
                           raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest())
    assert noises['N_live']['raw_sha256'] == '585570a70dbb0cfb597bfd49c6ffce1e77b598c744f2a89eb3e890455c9a0852'
    proc, emb = processor()
    inventories, policies = {}, {}
    for key, path in [('L', LIVE), ('M', INPUTS/'source64'/m['path'])]:
        batch = prepared(proc, emb, path)
        policies[key] = bridge._policy_inputs(batch)
        inventories[key] = dict(after_processor=leaves(batch), policy=leaves(policies[key]))
        torch.save(policies[key], OUT/(key+'_policy.pt'))
    original = torch.load(Path(old['condition_source_cell'])/'condition.pt', weights_only=False, map_location='cpu')
    assert bridge._canonical_hash(bridge._leaf_inventory(original)) == old['condition_identity']
    assert leaves(original) == leaves(policies['M']), 'M processor deviates from qualified Q'
    torch.save(original, OUT/'M_policy.pt')
    card = dict(schema='PI0253_FIXED_BRIDGE_CARD_V1', release_commit='82554098fb55b39ceca4b955252f8952c91c0dc0',
        canonical_issue_comment=5870278574, model_revision=REVISION, selected_checkpoint='NONE',
        development_checkpoint='G2000', runner_sha256=sha(__file__), model_queries=0,
        inputs={'L':dict(path=str(LIVE),sha256=sha(LIVE)), 'M':{**m,'path':str(INPUTS/'source64'/m['path'])}},
        noises=noises, inventories=inventories,
        policy_files={k:dict(path=str(OUT/(k+'_policy.pt')),sha256=sha(OUT/(k+'_policy.pt'))) for k in policies},
        reuse={'L__N_live__S':str(OLD/'pre_full_sim/G/A/mailbox/responses/model-0000.npz'),
               'M__N_M__Q':str(CELL/'tensors.npz')}, new_queries=6,
        tolerances=dict(raw_atol=1e-5,raw_rtol=1e-5,xyz_max_m=1e-5,so3_max_rad=1e-4))
    card_path=OUT/'bridge_card.json'
    if card_path.exists(): assert read(card_path)==card
    else: write(card_path,card)
    print(json.dumps(dict(status='PASS_CPU_BRIDGE_PREPARATION', new_queries=6, noises=noises)),flush=True)


def gate(decoded, state, entry, artifact):
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix
    if entry == 'Q':
        import pi0228_train_calib_checkpoint_selection as selector
        chunk=selector._build_chunk(decoded,chunk_id='pi0253-bridge')
    else:
        from stage21_native_model_receding_horizon import _model_chunk
        chunk,_=_model_chunk({'artifact':str(artifact)},chunk_id='pi0253-bridge', observation_timestamp=0.,period_s=.05)
    return TrajectorySafetyGateV1().evaluate(chunk, now=0., achieved_position=state[:3],
        achieved_orientation=matrix_to_quat(rot6d_rows_to_matrix(state[None,3:9]))[0]).as_dict()


def run():
    import pi0245_p1_stock as p1
    from transformers import AutoModel
    card=read(OUT/'bridge_card.json')
    assert card['runner_sha256']==sha(__file__)
    model_files=p1.verify_model_files(MODEL)
    assert model_files['revision']==REVISION
    print(json.dumps(dict(status='LOADING_G', free_bytes=torch.cuda.mem_get_info()[0])),flush=True)
    proc,emb=processor()
    model=AutoModel.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True)
    model.eval().to(device='cuda:0',dtype=torch.bfloat16)
    assert model.action_head.num_inference_timesteps==4
    card_sha=sha(OUT/'bridge_card.json')
    with np.load(CELL/'tensors.npz') as d: nm=d['noise_initial'].copy()
    with np.load(OLD/'pre_full_sim/G/A/mailbox/responses/model-0000_initial_noise.npz') as d: nl=d['float_values'].copy()
    noise_values={'N_live':nl,'N_M':nm}
    for input_key in ('L','M'):
        path=Path(card['inputs'][input_key]['path'])
        assert sha(path)==card['inputs'][input_key]['sha256']
        obs=s21._load_observation(path)
        policy_path=Path(card['policy_files'][input_key]['path'])
        assert sha(policy_path)==card['policy_files'][input_key]['sha256']
        policy=torch.load(policy_path, map_location='cpu',weights_only=False)
        for noise_key in ('N_live','N_M'):
            for entry in ('Q','S'):
                key='__'.join((input_key,noise_key,entry))
                dest=OUT/'bridge'/key; dest.mkdir(parents=True,exist_ok=True)
                receipt_path=dest/'receipt.json'; artifact=dest/'tensors.npz'
                if receipt_path.exists():
                    r=read(receipt_path); assert r['card_sha256']==card_sha and r['artifact_sha256']==sha(artifact)
                    continue
                if (dest/'attempt.json').exists(): raise RuntimeError('incomplete preserved query '+key)
                reused=key in card['reuse']
                write(dest/'attempt.json',dict(key=key,status='REUSED' if reused else 'QUERY_PENDING',card_sha256=card_sha))
                trace=[]; query_inventory=[]; noise_calls=[]
                if reused:
                    with np.load(card['reuse'][key]) as d:
                        if entry=='Q':
                            raw=d['raw_native'].copy(); decoded=d['decoded'].copy()
                            for i in range(4): trace.append(dict(x=d['trace_x'][i].copy(),v=d['trace_v'][i].copy(),t=d['trace_t'][i].copy()))
                        else: raw=d['raw_universal_action'].copy();decoded=d['decoded_physical_action'].copy()
                else:
                    epsilon=torch.from_numpy(noise_values[noise_key]).to(device='cuda:0',dtype=torch.bfloat16)
                    original_randn=torch.randn; original_get=model.get_action
                    def noise_hook(*args,**kw):
                        size=kw.get('size',args[0] if args else None)
                        shape=tuple(size) if isinstance(size,(tuple,list,torch.Size)) else None
                        frame=inspect.currentframe().f_back
                        from_head=frame.f_code.co_name=='get_action_with_features' and frame.f_locals.get('self') is model.action_head
                        if frame.f_code.co_name=='hooked_randn':
                            frame=frame.f_back
                            from_head=frame.f_code.co_name=='get_action_with_features' and frame.f_locals.get('self') is model.action_head
                        if shape==(1,40,132) and from_head:
                            assert not noise_calls and kw.get('dtype')==epsilon.dtype and torch.device(kw['device'])==epsilon.device
                            noise_calls.append(leaves(epsilon)[''])
                            return epsilon.clone()
                        return original_randn(*args,**kw)
                    def get_action(*args,**kw):
                        batch=args[0] if args else kw['inputs']
                        query_inventory.append(leaves(batch))
                        return original_get(*args,**kw)
                    def enc_hook(_m,args): trace.append(dict(x=args[0].detach().float().cpu().numpy().copy(),t=args[1].detach().float().cpu().numpy().copy()))
                    def dec_hook(_m,args,output):
                        v=output[0] if isinstance(output,(tuple,list)) else output
                        trace[-1]['v']=v.detach().float().cpu().numpy()[...,-40:,:].copy()
                    handles=[]
                    if entry=='S':
                        handles=[model.action_head.action_encoder.register_forward_pre_hook(enc_hook),model.action_head.action_decoder.register_forward_hook(dec_hook)]
                    torch.randn=noise_hook;model.get_action=get_action
                    try:
                        if entry=='Q':
                            stock=bridge._stock_query(model,policy,s21,None)
                            raw=stock['raw'];trace=stock['trace']
                            decoded=bridge._decode_physical(proc,emb,raw,obs['state'])
                        else: raw,decoded,_,_=s21._run_one(model,proc,emb,obs,language=LANGUAGE,seed=22301,device='cuda:0')
                    finally:
                        torch.randn=original_randn;model.get_action=original_get
                        for h in handles:h.remove()
                    assert len(noise_calls)==1 and noise_calls[0]['sha256']==card['noises'][noise_key]['raw_sha256']
                    assert len(trace)==4 and len(query_inventory)==1 and not model.training
                payload=dict(raw_universal_action=raw,decoded_physical_action=decoded,state=obs['state'],
                             noise_initial=noise_values[noise_key])
                if trace: payload.update(trace_x=np.stack([x['x'] for x in trace]),trace_v=np.stack([x['v'] for x in trace]),trace_t=np.stack([x['t'] for x in trace]))
                np.savez_compressed(artifact,**payload)
                result=gate(decoded,obs['state'],entry,artifact)
                r=dict(schema='PI0253_BRIDGE_CELL_V1',status='VALID',key=key,card_sha256=card_sha,
                       artifact_sha256=sha(artifact),model_revision=REVISION, reused=reused,queries=0 if reused else 1,
                       actual_noise=card['noises'][noise_key],gate=result,trace_status='OBSERVED' if trace else 'UNOBSERVED',
                       policy_before_get_action=query_inventory if query_inventory else 'UNOBSERVED_OLD_CALL',
                       parameter_dtype=str(next(model.parameters()).dtype),model_mode='eval')
                write(receipt_path,r);print(json.dumps(dict(status='VALID_BRIDGE_CELL',key=key,reused=reused)),flush=True)
    compare()


def compare():
    from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix
    card=read(OUT/'bridge_card.json'); comparisons=[]; receipts=[]
    for inp in ('L','M'):
        for noise in ('N_live','N_M'):
            paths=[OUT/'bridge'/'__'.join((inp,noise,e)) for e in ('Q','S')]
            rs=[read(p/'receipt.json') for p in paths];receipts.extend(rs)
            a,b=[dict(np.load(p/'tensors.npz')) for p in paths]
            raw_ok=bool(np.allclose(a['raw_universal_action'],b['raw_universal_action'],atol=1e-5,rtol=1e-5))
            x,y=a['decoded_physical_action'][0],b['decoded_physical_action'][0]
            xyz=float(np.linalg.norm(x[:,:3]-y[:,:3],axis=1).max())
            so3=float(bridge._geodesic(rot6d_rows_to_matrix(x[:,3:9].astype(np.float64)),rot6d_rows_to_matrix(y[:,3:9].astype(np.float64))).max())
            issues=lambda r:[(i['code'],i['severity']) for i in r['gate']['issues']]
            ga,gb=rs[0]['gate'],rs[1]['gate']
            gate_ok=ga['decision']==gb['decision'] and issues(rs[0])==issues(rs[1])
            metric_delta={}
            for k,v in ga['metrics'].items():
                if k=='chunk_id':continue
                vals=v if isinstance(v,dict) else {'value':v}
                other=gb['metrics'][k] if isinstance(v,dict) else {'value':gb['metrics'][k]}
                for sub,num in vals.items():
                    if num is not None:metric_delta[k+'/'+sub]=abs(num-other[sub])
            gate_ok=gate_ok and all(v<=1e-5 for v in metric_delta.values())
            issue_values_ok=all(abs(i['observed']-j['observed'])<=1e-5 and i['threshold']==j['threshold']
                for i,j in zip(ga['issues'],gb['issues']))
            gate_ok=gate_ok and issue_values_ok
            trace_delta={k:float(np.abs(a[k]-b[k]).max()) for k in ('trace_x','trace_v','trace_t') if k in a and k in b}
            trace_ok=all(np.allclose(a[k],b[k],atol=1e-5,rtol=1e-5) for k in trace_delta)
            observed_policy=[r['policy_before_get_action'] for r in rs]
            policy_ok=None if any(isinstance(p,str) for p in observed_policy) else observed_policy[0]==observed_policy[1]
            comparisons.append(dict(input=inp,noise=noise,raw_allclose=raw_ok,
                raw_max_abs=float(np.abs(a['raw_universal_action']-b['raw_universal_action']).max()),
                xyz_max_m=xyz,so3_max_rad=so3,gate_equal=gate_ok,gate_metric_delta=metric_delta,
                policy_leaves_equal=policy_ok,trace_max_abs=trace_delta,trace_equal=bool(trace_ok),
                trace_comparison='OBSERVED' if trace_delta else 'UNOBSERVED_OLD_L_S',
                pass_=raw_ok and xyz<=1e-5 and so3<=1e-4 and gate_ok and policy_ok is not False and trace_ok))
    report=dict(schema='PI0253_INFERENCE_BRIDGE_REPORT_V1',status='PASS' if all(x['pass_'] for x in comparisons) else 'PI0253_INFERENCE_BRIDGE_UNRESOLVED',
                card_sha256=sha(OUT/'bridge_card.json'),logical_cells=8,new_queries=sum(r['queries'] for r in receipts),
                selected_checkpoint='NONE',comparisons=comparisons,receipts=receipts,
                old_live_intermediate_trace='UNOBSERVED_NO_REQUERY',cpu_recomputed_from_actual_arrays=True,
                readonly_finalizer_sha256=sha(__file__),query_runner_sha256=card['runner_sha256'])
    write(OUT/'bridge_report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='receipts'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','compare'])
    {'prepare':prepare,'run':run,'compare':compare}[p.parse_args().command]()
