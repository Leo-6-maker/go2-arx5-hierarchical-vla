#!/usr/bin/env python3
"""Four fixed live side reports after FULL closure; reuse qualified Q unchanged."""
import argparse
import json
from pathlib import Path

from pi0253_full_screen_v2 import ROOT, read, sha, write, materialize

STEPS = (2500, 5000, 7500, 10000)


def require_closed(integrity, card_sha, replays):
    assert integrity['schema'] == 'PI0253_FULL_TRAINING_INTEGRITY_V1'
    assert integrity['status'] == 'PASS_PI0253_G_TRAINING_INTEGRITY'
    assert integrity['family'] == 'PI0253_FULL_G_SUPPORT_V1'
    assert integrity['run_card_sha256'] == card_sha
    assert integrity['optimizer_updates'] == 10000 and integrity['checkpoint_steps'] == list(STEPS)
    assert integrity['selected_checkpoint'] == 'NONE'
    assert set(replays) == set(STEPS)
    for step, replay in replays.items():
        assert replay['status'] == 'PASS_INDEPENDENT_CPU_REPLAY' and replay['checkpoint_step'] == step
        cells = replay['cells']
        assert len(cells) == len({(r['source_episode_id'], r['noise_name']) for r in cells}) == 128
        for noise in ('B0', 'B1'):
            assert sum(r['noise_name'] == noise and r['status'] == 'VALID' for r in cells) == 64
        assert {r['source_episode_id'] for r in cells if r['noise_name'] == 'B0'} == {
            r['source_episode_id'] for r in cells if r['noise_name'] == 'B1'}


def admission():
    evidence = ROOT / 'evidence/FULL'
    card_sha = sha(evidence / 'FULL_run_card.json')
    path = evidence / 'pi0253_training_integrity.json'
    integrity = read(path)
    replays = {s: read(ROOT / f'screen/FULL{s}/cpu_replay.json') for s in STEPS}
    require_closed(integrity, card_sha, replays)
    manifest = evidence / 'pi0253_training_manifest.json'
    assert sha(manifest) == integrity['training_manifest_sha256']
    assert read(manifest)['status'] == 'PASS_TRAINING_INTEGRITY'
    for step, replay in replays.items():
        assert sha(ROOT / f'screen/FULL{step}/screen_report.json') == replay['report_sha256']
    return dict(integrity_sha256=sha(path), run_card_sha256=card_sha,
                CPU_replay_sha256={str(s): sha(ROOT / f'screen/FULL{s}/cpu_replay.json') for s in STEPS})


def cpu_check():
    # Synthetic boundary only: intermediate quality FAIL is not a live-report veto.
    integrity = dict(schema='PI0253_FULL_TRAINING_INTEGRITY_V1', status='PASS_PI0253_G_TRAINING_INTEGRITY',
                     family='PI0253_FULL_G_SUPPORT_V1', run_card_sha256='fixture', optimizer_updates=10000,
                     checkpoint_steps=list(STEPS), selected_checkpoint='NONE')
    cells = [dict(source_episode_id=str(i), noise_name=n, status='VALID') for i in range(64) for n in ('B0', 'B1')]
    replays = {s: dict(status='PASS_INDEPENDENT_CPU_REPLAY', checkpoint_step=s, cells=cells) for s in STEPS}
    require_closed(integrity, 'fixture', replays)
    for field, bad in [('optimizer_updates', 9999), ('selected_checkpoint', '2500')]:
        try:
            require_closed({**integrity, field: bad}, 'fixture', replays)
        except AssertionError:
            pass
        else:
            raise AssertionError('boundary accepted ' + field)
    try:
        require_closed(integrity, 'fixture', {s: r for s, r in replays.items() if s != 10000})
    except AssertionError:
        pass
    else:
        raise AssertionError('boundary accepted missing candidate')
    duplicate = {**replays, 10000: {**replays[10000], 'cells': cells[:-1] + [cells[0]]}}
    try:
        require_closed(integrity, 'fixture', duplicate)
    except AssertionError:
        pass
    else:
        raise AssertionError('boundary accepted duplicate source/noise')
    print('PASS_CPU_FULL_LIVE_BOUNDARY_ZERO_PROJECT_QUERIES')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-check', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    if args.cpu_check:
        cpu_check(); return
    closed = admission()  # No model construction before complete family and retention.
    import numpy as np
    import torch
    import pi0253_live_bridge as live
    from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix, quat_to_matrix
    card = read(ROOT / 'bridge_card.json')
    assert sha(live.__file__) == card['runner_sha256']
    policy_path = ROOT / 'L_policy.pt'
    assert sha(policy_path) == card['policy_files']['L']['sha256']
    assert sha(live.LIVE) == card['inputs']['L']['sha256']
    policy = torch.load(policy_path, map_location='cpu', weights_only=False)
    assert live.leaves(policy) == card['inventories']['L']['policy']
    noise_path = live.OLD / 'pre_full_sim/G/A/mailbox/responses/model-0000_initial_noise.npz'
    assert sha(noise_path) == '12323ebb5a91d4a2696c076f700bd5f6b0252b7d6beb3d90973cc18937b13a44'
    with np.load(noise_path, allow_pickle=False) as d:
        noise, raw_bytes = d['float_values'].copy(), d['raw_bytes'].copy()
    assert np.array_equal(torch.from_numpy(noise).to(torch.bfloat16).view(torch.uint8).numpy(), raw_bytes)
    expert_path = ROOT / 'correct_expert_and_tracking.npz'
    assert sha(expert_path) == '1cb7eecbe2044ca5522cd3f4d6934cafea26a938201b67f436dc5b37fee97ebd'
    assert sha(ROOT / 'live_contract_support_audit.json') == 'be7492b7b8f2f9dc5f0a3a95b907e8050308b3fcc29792fe254ca7ca47ead373'
    audit = read(ROOT / 'live_contract_support_audit.json')
    assert sha(ROOT / 'bridge_card.json') == audit['observed']['bridge_card_sha256']
    with np.load(expert_path, allow_pickle=False) as d:
        expert = {k: d[k].copy() for k in ('expert_native_sample_xyz', 'expert_native_sample_quaternion', 'anchor_world', 'target_world')}
    proc, emb = live.processor(); proc.eval()
    obs = live.bridge._screen_observation(live.LIVE)
    baseline = ROOT / 'bridge/L__N_live__Q/receipt.json'
    assert read(baseline)['status'] == 'VALID' and read(baseline)['card_sha256'] == sha(ROOT / 'bridge_card.json')
    assert sha(baseline.parent / 'tensors.npz') == read(baseline)['artifact_sha256']
    for step in STEPS:
        dest = ROOT / f'live/FULL{step}'; artifact = dest / 'tensors.npz'
        receipt = dest / 'receipt.json'
        if not receipt.exists():
            assert not args.cpu, 'CPU replay requires existing first-valid query'
            assert not (dest / 'attempt.json').exists(), 'preserved incomplete query requires reconciliation'
            _, _, view, restore = materialize(step)
            from transformers import AutoModel
            model = AutoModel.from_pretrained(view, local_files_only=True, trust_remote_code=True).eval().to('cuda:0', dtype=torch.bfloat16)
            assert model.action_head.num_inference_timesteps == 4
            inventory, initial = [], []
            original = model.get_action
            def captured(batch):
                inventory.append(live.leaves(batch)); return original(batch)
            def capture_initial(_module, inputs):
                if not initial:
                    initial.append(live.leaves(inputs[0])[''])
            handle = model.action_head.action_encoder.register_forward_pre_hook(capture_initial)
            model.get_action = captured
            write(dest / 'attempt.json', dict(status='QUERY_PENDING', step=step, admission=closed, revision=restore['revision'], runner_sha256=sha(__file__)))
            try:
                stock = live.bridge._stock_query(model, policy, live.s21, noise)
            finally:
                model.get_action = original
                handle.remove()
            assert len(inventory) == 1 and inventory == read(baseline)['policy_before_get_action']
            assert len(initial) == 1 and initial[0]['dtype'] == card['noises']['N_live']['dtype']
            assert initial[0]['device'] == 'cuda:0' and initial[0]['finite']
            assert initial[0]['shape'] == card['noises']['N_live']['shape']
            assert initial[0]['sha256'] == card['noises']['N_live']['raw_sha256']
            trace = stock['trace']; assert len(trace) == 4 and not model.training
            assert np.array_equal(trace[0]['x'], noise)
            decoded = live.bridge._decode_physical(proc, emb, stock['raw'], obs['state'])
            np.savez_compressed(artifact, raw_universal_action=stock['raw'], decoded_physical_action=decoded,
                state=obs['state'], noise_initial=noise, trace_x=np.stack([r['x'] for r in trace]),
                trace_v=np.stack([r['v'] for r in trace]), trace_t=np.stack([r['t'] for r in trace]))
            write(receipt, dict(schema='PI0253_FULL_LIVE_CELL_V1', status='VALID', step=step, admission=closed,
                hf_revision=restore['revision'], artifact_sha256=sha(artifact), expert_sha256=sha(expert_path),
                gate=live.gate(decoded, obs['state'], 'Q', artifact), policy_before_get_action=inventory,
                baseline_receipt_sha256=sha(baseline), runner_sha256=sha(__file__), queries=1, head_forwards=4,
                actual_noise=card['noises']['N_live'], initial_noise_tensor=initial[0], selected_checkpoint='NONE', optimizer_updates=0))
            del stock, model
            torch.cuda.empty_cache()
        record = read(receipt)
        assert record['admission'] == closed and record['artifact_sha256'] == sha(artifact)
        assert record['runner_sha256'] == sha(__file__) and record['actual_noise'] == card['noises']['N_live']
        assert record['expert_sha256'] == sha(expert_path) and record['baseline_receipt_sha256'] == sha(baseline)
        with np.load(artifact, allow_pickle=False) as d:
            raw, saved = d['raw_universal_action'].copy(), d['decoded_physical_action'].copy()
            assert np.array_equal(d['noise_initial'], noise) and np.array_equal(d['trace_x'][0], noise)
            assert d['trace_x'].shape == d['trace_v'].shape == (4, 1, 40, 132)
        decoded = live.bridge._decode_physical(proc, emb, raw, obs['state'])
        assert np.allclose(decoded, saved, atol=1e-7, rtol=0) and np.isfinite(decoded).all()
        gate = live.gate(decoded, obs['state'], 'Q', artifact); assert gate == record['gate']
        p = decoded[0].astype(np.float64)
        distance = np.linalg.norm(p[:, :3] + expert['anchor_world'] - expert['target_world'], axis=1)
        rot = live.bridge._geodesic(rot6d_rows_to_matrix(p[:, 3:9]), quat_to_matrix(expert['expert_native_sample_quaternion']))
        report = dict(schema='PI0253_FULL_LIVE_CPU_SIDE_REPORT_V1', status='PASS_INDEPENDENT_CPU_REPLAY', step=step,
            receipt_sha256=sha(receipt), reference_min_target_m=float(distance.min()), reference_final_target_m=float(distance[-1]),
            expert_xyz_rmse_m=float(np.sqrt(np.mean(np.sum((p[:, :3] - expert['expert_native_sample_xyz'])**2, axis=1)))),
            expert_so3_rmse_rad=float(np.sqrt(np.mean(rot**2))), gate=gate, actual_TCP='UNOBSERVED_NO_PHYSICS_EPISODE',
            model_queries=0, optimizer_updates=0, selected_checkpoint='NONE', affects_training_or_selection=False)
        write(dest / 'cpu_replay.json', report)
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
