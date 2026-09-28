#!/usr/bin/env python3
"""CPU-only PI0252 binding of existing qualified G and original PI0226 runtime."""
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0252')
INPUTS = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0250/inputs/frozen_v1')
CODE = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/reference-317a')
sys.path.insert(0, str(CODE / 'src'))
from go2arx5_vla.safety import TrajectorySafetyGateV1

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def main():
    evidence = ROOT / 'evidence/G2000'
    view = json.loads((evidence / 'inference_view_readback.json').read_text())
    assert view['status'] == 'PASS_G2000_INFERENCE_VIEW_FULL_HASH'
    assert view['revision'] == 'efecea485ae77000599d19171b02491f8d514f42'
    assert view['source_tree_sha256'] == 'cfbfeb54bea2b5ac266047dfd764cfa6b041a8a77b9688737ea70b56935d5226'
    checkpoint = ROOT / 'models/G2000/inference_view_step_2000'
    files = []
    for r in view['files']:
        p = checkpoint / Path(r['path']).name
        observed = sha(p)
        assert p.stat().st_size == r['bytes'] and observed == r['sha256'], p
        files.append(dict(path=str(p), resolved_path=str(p.resolve()), bytes=p.stat().st_size, sha256=observed))
    required = {
        INPUTS / 'train_dataset/meta/stats.json': 'c24d465bc4acd594beb041b42f2e0f4ea955a67c7714e178ffdb313a61adb3f4',
        INPUTS / 'train_dataset/meta/relative_stats.json': '10b52e09d40f3754e05bbecc5d86bbc3e0c60a9dc5d12501442177b15ac12ac9',
        INPUTS / 'processor_base/processor_config.json': '2e6b39e1534028c0bdce801838422c11790dcae76b441ebb4378db1a2924c7f9',
        INPUTS / 'processor_base/statistics.json': '33aa144909ade7b25d42ef962a711d38f58edd50d41937b65d28fcebe9d39763',
        INPUTS / 'modality/go2_arx5_modality.py': '244f37f28e62b7b392a1598b78e78f00e67693f6c08a606b3c5358830ada8b25',
    }
    for p, h in required.items(): assert sha(p) == h, p
    wbc = json.loads((ROOT / 'evidence/current_wbc_readback.json').read_text())
    assert wbc['status'] == 'PASS' and wbc['archive_live_comparison']['identical']
    refs = Path('/home/liuyu-nfs/pi0223_goal_b_20260918/refs_r00')
    reference_manifest = json.loads((refs / 'goal_a_reference_manifest.json').read_text())
    tasks = {}
    for task in 'ABCD':
        r = reference_manifest['cells'][f'TRAIN-{task}_C0']
        p = refs / r['trajectory']
        assert sha(p) == r['trajectory_sha256'], p
        tasks[task] = dict(trajectory=str(p), trajectory_sha256=r['trajectory_sha256'],
                          target_world_m=r['target_world_m'], reset_seed=22301, query_seed=22301,
                          one_chunk_tick_cap=500, receding_prefix_steps=4,
                          receding_query_cap=11 if task == 'A' else 6,
                          receding_tick_cap=44 if task == 'A' else 24)
    result = dict(schema='PI0252_R2_PRE_FULL_SIM_BINDING_V1', status='PASS_CPU_BINDING',
        authority_commit='a5ccfba37a54e53b1ad1818a095edf67319aba9c', canonical=5866382788,
        development_checkpoint='G2000', formal_selected_checkpoint='NONE',
        checkpoint_revision=view['revision'], checkpoint_tree_sha256=view['source_tree_sha256'],
        current_model_files=files, processor_files={str(p):h for p,h in required.items()},
        wbc_current_readback_sha256=sha(ROOT / 'evidence/current_wbc_readback.json'),
        wbc_tree_sha256=wbc['live_tree']['v2_tree_sha256'],
        original_runtime_canonical=5733559806, tasks=tasks,
        source_frame='NATIVE_WBC_RELATIVE_REFERENCE', tcp_semantic_id='x5_link6_gripper_center',
        native_control_dt_s=.02, action_sample_period_s=.05, horizon=40, sampler_steps=4,
        sim_gate_profile=asdict(TrajectorySafetyGateV1().config), success_world_tcp_radius_m=.035,
        target_in_model_input=False, processor_mode='eval', model_dtype='bfloat16',
        full_reference_completion=True, zero_reference_updates_one_chunk=True,
        gpu_index=0, gpu_sharing_user_authorized=True, corelab_training=False,
        source_hashes={str(p):sha(p) for p in sorted((ROOT / 'runtime').glob('*.py'))})
    out = ROOT / 'pre_full_sim/sim_binding.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    assert not out.exists(), 'binding already frozen'
    out.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    out.with_suffix('.sha256').write_text(sha(out)+'\n')
    print(json.dumps({'status': result['status'], 'binding': str(out), 'sha256': sha(out)}), flush=True)

if __name__ == '__main__': main()
