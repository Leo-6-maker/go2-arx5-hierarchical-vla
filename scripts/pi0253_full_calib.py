#!/usr/bin/env python3
"""Qualified S4 profile adapter; original CALIB scorer/selector remain frozen."""
import argparse
from collections import defaultdict
import gc
import hashlib
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

from pi0253_full_screen_v2 import ROOT, CODE, read, sha, write, materialize, context
from pi0253_full_live import admission, STEPS
import pi0228_train_calib_checkpoint_selection as selector
import pi0232_train_calib_eval as validation

COUNT = 1088
SELECT_FUNCTION_SHA = '80af848239a5505836d7eeecb75d889403af4ac82c640ccf14e90d0b0703b607'


def complete_profile(profile, rows):
    assert len(rows) == len({r['source_episode_id'] for r in rows}) == COUNT
    assert len(profile['records']) == COUNT
    assert all(math.isfinite(r['reference_min_target_distance_m']) for r in profile['records'])
    return validation.validate_profile_payload(profile, step=profile['checkpoint_step'], expected_rows=rows)


def summarize(records, step, model_path, seal_sha, authority_sha, grid_sha):
    import numpy as np
    accepted = [r for r in records if r['safety_gate']['decision'] != 'REJECT']
    cells = defaultdict(list)
    for r in accepted:
        cells[r['source_rollout_id']].append(r['reference_min_target_distance_m'])
    medians = {k: float(np.median(v)) for k, v in cells.items()}
    counts = {n: sum(r['safety_gate']['decision'] == n for r in records) for n in validation.DECISIONS}
    return dict(schema=validation.PROFILE_SCHEMA, status='PASS' if not counts['REJECT'] else 'SAFETY_GATE_REJECT',
        checkpoint_step=step, checkpoint_path=str(model_path), seed=22301,
        action_contract='official_dense_relative_H40', action_dt_s=.05,
        calib_seal_sha256=seal_sha, target_authority_sha256=authority_sha,
        target_authority_source_grid_manifest_sha256=grid_sha,
        model_input_keys=selector.MODEL_INPUT_KEYS, target_xyz_in_model_input=False, wbc_execution=False,
        forward_count=len(records), expected_forward_count=COUNT, safety_gate_counts=counts,
        reference_min_target_distance_m=selector._stats([r['reference_min_target_distance_m'] for r in accepted]),
        worst_cell_median_reference_min_target_distance_m=max(medians.values()) if medians else None,
        cell_medians_reference_min_target_distance_m=medians, records=records)


def score(raw, decoded, state, row, target_id, target, latency, gate):
    import numpy as np
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix
    assert raw.shape == (1, 40, 132) and decoded.shape == (1, 40, 10)
    assert np.isfinite(raw).all() and np.isfinite(decoded).all()
    i = int(row['episode_index'])
    result = gate.evaluate(selector._build_chunk(decoded, chunk_id=f'pi0228-calib-{i:04d}'), now=0.,
        achieved_position=state[:3], achieved_orientation=matrix_to_quat(rot6d_rows_to_matrix(state[3:9].reshape(1,6)))[0])
    # Preserve the original CALIB statistic exactly; live/physical world-TCP is a separate report.
    distances = np.linalg.norm(decoded[0, :, :3].astype(np.float64) - target[None, :], axis=1)
    return dict(episode_index=i, source_episode_id=row['source_episode_id'],
        source_rollout_id=selector._source_rollout_id(row['source_episode_id']), target_id=target_id,
        target_world_m=target.tolist(), safety_gate=result.as_dict(),
        reference_min_target_distance_m=float(distances.min()), reference_min_target_tick=int(distances.argmin()),
        raw_sha256=selector.sha256_array(raw), decoded_sha256=selector.sha256_array(decoded), latency_s=latency,
        model_input_keys=selector.MODEL_INPUT_KEYS, target_xyz_in_model_input=False, wbc_execution=False)


def setup(path):
    closed = admission()  # Complete FULL and four independent retention replays before any CALIB read.
    for step in STEPS:
        report = read(ROOT / f'live/FULL{step}/cpu_replay.json')
        assert report['status'] == 'PASS_INDEPENDENT_CPU_REPLAY' and report['step'] == step
        receipt = ROOT / f'live/FULL{step}/receipt.json'
        assert report['receipt_sha256'] == sha(receipt) and read(receipt)['admission'] == closed
    b = read(path)
    assert b['schema'] == 'PI0253_FULL_CALIB_BINDINGS_V1' and b['closed_family'] == closed
    assert b['expected_forward_count'] == COUNT and b['selected_checkpoint'] == 'NONE'
    assert b['runner_sha256'] == sha(__file__)
    assert b['profile_validator_sha256'] == sha(validation.__file__) == '67def7619117df454df5043b03a093a7be6ebca83285c0cfc9f6a2c03b3f701d'
    assert b['selector_sha256'] == sha(selector.__file__) == 'e3ff2066d4a1e9a4aed55bb4dfe03775dc59d960b447f7c0bc68d7e3a8309cf6'
    assert hashlib.sha256(inspect.getsource(selector.select).encode()).hexdigest() == SELECT_FUNCTION_SHA
    for ref in (b['seal'], b['authority']):
        selector._reject_locked_path(Path(ref['path'])); assert sha(ref['path']) == ref['sha256']
    seal = read(b['seal']['path'])
    assert seal['schema'] == 'PI0228_TRAIN_CALIB_SEAL_V1' and seal['status'] == 'PASS'
    dataset = Path(b['calib_dataset']['path']).resolve(); selector._reject_locked_path(dataset)
    assert b['calib_dataset']['source_root'] == seal['train_calib']['root']
    import pi0228_h200_large_adapt_v1 as inventory
    tree, entries = inventory.tree_sha256(dataset, b['calib_dataset']['tree_prefix'])
    assert tree == seal['train_calib']['tree_sha256'] == b['calib_dataset']['tree_sha256']
    authority_payload, authority = selector._load_target_authority(Path(b['authority']['path']))
    assert authority_payload['calib_seal_sha256'] == b['seal']['sha256']
    assert authority_payload['train_calib_tree_sha256'] == tree
    rows = selector._load_calib_rows(dataset)
    assert len(rows) == len({r['source_episode_id'] for r in rows}) == COUNT
    assert {selector._source_rollout_id(r['source_episode_id']) for r in rows} == set(authority)
    for r in rows:
        r['source_rollout_id'] = selector._source_rollout_id(r['source_episode_id'])
        r['target_id'] = selector._target_for(r['source_episode_id'], authority)[0]
    return b, dataset, rows, authority_payload, authority, entries


def profile(args):
    import numpy as np
    import torch
    import pandas as pd
    import pi0242_train_infer_bridge as bridge
    import stage21_same_scene_groot_inference as s21
    from stage20s_infer_groot import _read_first_frame
    from gr00t.data.types import MessageType
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    import pi0253_live_bridge as live
    b, dataset, rows, authority_payload, authority, entries = setup(args.bindings)
    _, _, _, _, _, ctx = context()  # Existing hash-bound C_T processor/stats only.
    proc, emb = ctx['processor'], ctx['embodiment']; gate = TrajectorySafetyGateV1()
    if args.step == 1000:
        view = Path(b['baseline']['path']); files = b['baseline']['model_files']
        assert set(r['path'] for r in files) == {'config.json','model.safetensors.index.json',
            'model-00001-of-00002.safetensors','model-00002-of-00002.safetensors'}
        for r in files:
            assert (view/r['path']).stat().st_size == r['bytes'] and sha(view/r['path']) == r['sha256']
        model_identity = dict(role='ORIGINAL_PI0226_ADAPTED1000_COMPARISON_ONLY', files=files)
    else:
        _, _, view, model_identity = materialize(args.step)
    root = ROOT / f'calib/profile-{args.step}'; output = root/'profile.json'; model = None; records = []
    contract = dict(schema='PI0253_FULL_CALIB_PROFILE_CARD_V1', step=args.step, bindings_sha256=sha(args.bindings),
        runner_sha256=sha(__file__), selector_sha256=sha(selector.__file__), model_identity=model_identity,
        original_CALIB_scoring_unchanged=True, runtime='S4', expected_forward_count=COUNT)
    write(root/'card.json', contract); contract_sha = sha(root/'card.json')
    entry_hashes = {r['path']: r['sha256'] for r in entries}
    for row in rows:
        i = row['episode_index']; cell = root/f'cells/source-{i:06d}'; artifact = cell/'tensors.npz'; receipt = cell/'receipt.json'
        target_id, target = selector._target_for(row['source_episode_id'], authority)
        if not receipt.exists():
            assert not args.cpu, 'missing first-valid CALIB cell'
            assert not (cell/'attempt.json').exists(), 'pending preserved attempt requires reconciliation'
            shard = i//1000; stem = f'episode_{i:06d}'
            rel = [f'data/chunk-{shard:03d}/{stem}.parquet'] + [f'videos/chunk-{shard:03d}/{camera}/{stem}.mp4'
                for camera in ('observation.images.external_rgb','observation.images.wrist_rgb')]
            table = pd.read_parquet(dataset/rel[0]); state = np.asarray(table.iloc[0]['observation.state'],dtype=np.float32).reshape(-1)
            assert state.shape == (10,) and np.isfinite(state).all()
            obs = dict(body_rgb=_read_first_frame(dataset/rel[1])[0], wrist_rgb=_read_first_frame(dataset/rel[2])[0], state=state)
            message = bridge._observation_message(obs,emb,selector.LANGUAGE)
            policy = bridge._policy_inputs(proc.collator([proc([{'type':MessageType.EPISODE_STEP.value,'content':message}])]))
            leaves = live.leaves(policy)
            assert not any(t in k.lower() for k in leaves for t in bridge.FORBIDDEN_INPUT_TOKENS)
            noise = None; baseline_receipt = None
            if args.step != 1000:
                parent = ROOT/f'calib/profile-1000/cells/source-{i:06d}'; baseline = read(parent/'receipt.json')
                assert baseline['source_episode_id'] == row['source_episode_id'] and baseline['policy_inventory'] == leaves
                assert sha(parent/'tensors.npz') == baseline['artifact_sha256']
                with np.load(parent/'tensors.npz',allow_pickle=False) as d: noise = d['noise_initial'].copy()
                baseline_receipt = sha(parent/'receipt.json')
            if model is None:
                from transformers import AutoModel
                model = AutoModel.from_pretrained(view,local_files_only=True,trust_remote_code=True).eval().to('cuda:0',dtype=torch.bfloat16)
                assert model.action_head.num_inference_timesteps == 4
            initial = []
            def capture(_module, inputs):
                if not initial: initial.append(live.leaves(inputs[0])[''])
            handle = model.action_head.action_encoder.register_forward_pre_hook(capture)
            write(cell/'attempt.json',dict(status='QUERY_PENDING',card_sha256=contract_sha,source_episode_id=row['source_episode_id'],
                input_files={p:entry_hashes[p] for p in rel},policy_inventory=leaves,baseline_receipt_sha256=baseline_receipt))
            try:
                stock = bridge._stock_query(model,policy,s21,noise)
            finally:
                handle.remove()
            assert len(stock['trace']) == 4 and initial[0]['dtype'] == 'torch.bfloat16' and initial[0]['device'] == 'cuda:0'
            if args.step != 1000:
                assert initial[0] == baseline['initial_noise_tensor']
                assert np.array_equal(stock['trace'][0]['x'],noise)
            try:
                decoded = bridge._decode_physical(proc,emb,stock['raw'],state)
            except Exception as exc:
                ctx['p1'].durable_npz(cell/'invalid_output.npz',raw=stock['raw'],state=state,
                    noise_initial=stock['trace'][0]['x'],trace_x=np.stack([t['x'] for t in stock['trace']]),
                    trace_v=np.stack([t['v'] for t in stock['trace']]))
                write(receipt,dict(schema='PI0253_FULL_CALIB_CELL_V1',status='INVALID_DECODE_STOP',
                    source_episode_id=row['source_episode_id'],card_sha256=contract_sha,error=repr(exc),
                    invalid_artifact_sha256=sha(cell/'invalid_output.npz'),model_queries=1,head_forwards=4))
                raise
            ctx['p1'].durable_npz(artifact,raw=stock['raw'],decoded=decoded,state=state,noise_initial=stock['trace'][0]['x'],
                trace_x=np.stack([t['x'] for t in stock['trace']]),trace_v=np.stack([t['v'] for t in stock['trace']]),
                trace_t=np.stack([t['t'] for t in stock['trace']]))
            write(receipt,dict(schema='PI0253_FULL_CALIB_CELL_V1',status='VALID',source_episode_id=row['source_episode_id'],
                artifact_sha256=sha(artifact),attempt_sha256=sha(cell/'attempt.json'),card_sha256=contract_sha,
                baseline_receipt_sha256=baseline_receipt,policy_inventory=leaves,initial_noise_tensor=initial[0],
                record=score(stock['raw'],decoded,state,row,target_id,target,stock['elapsed_s'],gate),model_queries=1,head_forwards=4))
            del stock
        saved = read(receipt)
        assert saved['status'] == 'VALID' and saved['card_sha256'] == contract_sha
        assert sha(artifact) == saved['artifact_sha256'] and sha(cell/'attempt.json') == saved['attempt_sha256']
        with np.load(artifact,allow_pickle=False) as d:
            raw,state,decoded_saved,noise_initial = [d[k].copy() for k in ('raw','state','decoded','noise_initial')]
            assert d['trace_x'].shape == d['trace_v'].shape == (4,1,40,132)
            assert np.array_equal(d['trace_x'][0],noise_initial)
        assert saved['initial_noise_tensor']['sha256'] == hashlib.sha256(torch.from_numpy(noise_initial).to(torch.bfloat16).view(torch.uint8).numpy().tobytes()).hexdigest()
        if args.step != 1000:
            parent = ROOT/f'calib/profile-1000/cells/source-{i:06d}'
            assert saved['baseline_receipt_sha256'] == sha(parent/'receipt.json')
            with np.load(parent/'tensors.npz',allow_pickle=False) as d: assert np.array_equal(noise_initial,d['noise_initial'])
        decoded = bridge._decode_physical(proc,emb,raw,state)
        assert np.allclose(decoded,decoded_saved,atol=1e-7,rtol=0)
        record = score(raw,decoded,state,row,target_id,target,saved['record']['latency_s'],gate)
        assert record == saved['record']; records.append(record)
    del model; gc.collect(); torch.cuda.empty_cache()
    result = summarize(records,args.step,view,b['seal']['sha256'],b['authority']['sha256'],authority_payload['source_grid_manifest_sha256'])
    result.update(checkpoint_index_sha256=sha(view/'model.safetensors.index.json'),profile_card_sha256=contract_sha)
    complete_profile(result,rows);write(output,result)
    write(root/'cpu_replay.json',dict(schema='PI0253_FULL_CALIB_CPU_REPLAY_V1',status='PASS_INDEPENDENT_CPU_REPLAY',
        checkpoint_step=args.step,profile_sha256=sha(output),count=COUNT,model_queries=0,optimizer_updates=0))
    print(json.dumps(dict(step=args.step,status=result['status'],gate=result['safety_gate_counts'],profile_sha256=sha(output))),flush=True)


def freeze(args):
    _, _, rows, _, _, _ = setup(args.bindings)
    profiles = [ROOT/f'calib/profile-{s}/profile.json' for s in (1000,*STEPS)]
    for s,p in zip((1000,*STEPS),profiles):
        complete_profile(read(p),rows)
        replay = read(p.parent/'cpu_replay.json')
        assert replay['status'] == 'PASS_INDEPENDENT_CPU_REPLAY' and replay['profile_sha256'] == sha(p)
    output = ROOT/'calib/selection.json'
    write(ROOT/'calib/provenance_lock.json',dict(schema='PI0253_CALIB_PROVENANCE_LOCK_V1',bindings_sha256=sha(args.bindings),
        selector_sha256=sha(selector.__file__),select_function_sha256=SELECT_FUNCTION_SHA,
        profiles={str(s):sha(p) for s,p in zip((1000,*STEPS),profiles)},baseline_comparison_only=True))
    if output.exists():
        selected = read(output)
        if selected['status'] == 'NO_ADMISSIBLE_CHECKPOINT':
            assert selected['provenance_lock_sha256'] == sha(ROOT/'calib/provenance_lock.json')
            print('REUSED_FROZEN_NO_ADMISSIBLE_CHECKPOINT');return
        assert [p['sha256'] for p in selected['candidate_profiles']] == [sha(p) for p in profiles[1:]]
    else:
        try:
            selected = selector.select(SimpleNamespace(profiles=profiles[1:],profile_steps=list(STEPS),output=ROOT/'calib/selector_result.staged.json'))
            write(output,selected)
        except RuntimeError as exc:
            if str(exc) != 'no TRAIN-CALIB checkpoint is admissible': raise
            write(output,dict(schema='PI0253_FULL_SELECTION_TERMINAL_V1',status='NO_ADMISSIBLE_CHECKPOINT',selected_checkpoint='NONE',
                selection_frozen_before_wbc=True,provenance_lock_sha256=sha(ROOT/'calib/provenance_lock.json')))
            print('NO_ADMISSIBLE_CHECKPOINT');return
    step = selected['selected_checkpoint_step']; replay = read(ROOT/f'screen/FULL{step}/cpu_replay.json')
    groups = [replay[k][n] for k in ('group_results','old16_group_results') for n in ('B0','B1')]
    passed = all(g['reject_count']==0 and g['invalid_count']==0 and g['median_xyz_rmse_m']<=.020
        and g['p95_xyz_rmse_m']<=.035 and g['median_so3_rmse_rad']<=.050 for g in groups)
    write(ROOT/'calib/winner_retention.json',dict(schema='PI0253_SELECTED_RETENTION_ROUTE_V1',selected_checkpoint=step,
        selection_sha256=sha(output),CPU_replay_sha256=sha(ROOT/f'screen/FULL{step}/cpu_replay.json'),sim_admitted=passed,
        status='ADMIT_ORIGINAL_TRAIN_SIM' if passed else 'SELECTED_RETENTION_FAIL_STOP',runnerup_substitution=False))
    print(json.dumps(dict(selected_checkpoint=step,sim_admitted=passed)),flush=True)


def cpu_check():
    import copy,tempfile
    rows=[dict(episode_index=i,source_episode_id=f'synthetic-{i}:obs-0000',source_rollout_id=f'synthetic-{i}',target_id=f't-{i}') for i in range(COUNT)]
    records=[dict(**r,safety_gate={'decision':'WARN'},reference_min_target_distance_m=.1,raw_sha256='0'*64,
        decoded_sha256='0'*64,target_xyz_in_model_input=False,wbc_execution=False) for r in rows]
    p=summarize(records,2500,Path('synthetic'),'', '', '');complete_profile(p,rows)
    assert hashlib.sha256(inspect.getsource(selector.select).encode()).hexdigest() == SELECT_FUNCTION_SHA
    bad=copy.deepcopy(p);bad['records'][0]['reference_min_target_distance_m']=float('nan')
    for obj,expected in [(bad,rows),(p,rows[:-1]),(p,rows[:-1]+[rows[0]])]:
        try: complete_profile(obj,expected)
        except (AssertionError,validation.EvaluationContractError): pass
        else: raise AssertionError('incomplete/nonfinite/duplicate profile admitted')
    with tempfile.TemporaryDirectory(dir=ROOT/'runtime',prefix='calib-cpu-synthetic-') as folder:
        paths=[]
        for s in STEPS:
            value={**p,'checkpoint_step':s};path=Path(folder)/f'{s}.json';path.write_text(json.dumps(value));paths.append(path)
        winner=selector.select(SimpleNamespace(profiles=paths,profile_steps=list(STEPS),output=Path(folder)/'selection.json'))
        assert winner['selected_checkpoint_step']==2500
        rejected=summarize([{**r,'safety_gate':{'decision':'REJECT'}} for r in records],2500,Path('synthetic'),'','','')
        complete_profile(rejected,rows);paths[0].write_text(json.dumps(rejected))
        winner=selector.select(SimpleNamespace(profiles=paths,profile_steps=list(STEPS),output=Path(folder)/'selection.json'))
        assert winner['selected_checkpoint_step']==5000  # Candidate existence OR, REJECT evidence preserved.
        for path in paths: path.write_text(json.dumps({**rejected,'checkpoint_step':int(path.stem)}))
        try: selector.select(SimpleNamespace(profiles=paths,profile_steps=list(STEPS),output=Path(folder)/'selection.json'))
        except RuntimeError as exc: assert str(exc)=='no TRAIN-CALIB checkpoint is admissible'
        else: raise AssertionError('all-REJECT family admitted')
    print('PASS_CPU_ORIGINAL_SELECTOR_AND_1088_BOUNDARIES_ZERO_PROJECT_QUERIES')


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bindings',type=Path)
    parser.add_argument('--step',type=int,choices=(1000,*STEPS));parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--select',action='store_true');parser.add_argument('--cpu-check',action='store_true');args=parser.parse_args()
    if args.cpu_check: cpu_check()
    else:
        assert args.bindings
        if args.select: freeze(args)
        else:
            assert args.step is not None;profile(args)
