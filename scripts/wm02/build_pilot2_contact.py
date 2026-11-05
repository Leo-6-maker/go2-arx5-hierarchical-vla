"""Seal actual, complete supported-contact episodes without integrating physics."""
import argparse
import gzip
import json
from pathlib import Path


def build(manifest, output):
    from go2arx5_vla.world_model.storage import digest, write_json
    from go2arx5_vla.world_model.pilot2_contact_model import replay
    from go2arx5_vla.world_model.pilot2_supported import contract, PLAN
    if manifest.get('schema') != 'WM02_PILOT2_SUPPORTED_BUNDLE_MANIFEST_V1' or manifest.get('plan_sha') != PLAN:
        raise ValueError('PREDECLARED_PILOT2_ACTUAL_CONTACT_BUNDLE_REQUIRED')
    episodes = []
    for item in manifest['episodes']:
        root = Path(item['run']); spec_path = root/'EXECUTION_SPEC.json'
        native = root/'native-01'; result_path = native/'RESULT.json'
        score_path = native/'SUPPORTED_CLOSE_RESULT.private.json'
        episode_path = root/'EPISODE_MANIFEST.private.json'
        for name, path in [('spec', spec_path), ('result', result_path), ('score', score_path), ('episode', episode_path)]:
            if digest(path) != item[name+'_sha256']:
                raise ValueError('SEALED_CONTACT_EVIDENCE_CHANGED:'+name)
        spec, result, score, episode = [json.loads(p.read_text()) for p in
            (spec_path, result_path, score_path, episode_path)]
        contract(episode)
        if (result['status'] != 'NATIVE_STAGE_PASSED' or result['physics_stops'] or not result['shutdown_clean']
                or result['actual_target_integrations'] != 6703 or result.get('native_outer_guard_rejection_count')
                or not score['same_native_process'] or score['physics_resets'] or score['state_teleports']
                or score['actual_task_atoms'] != 168 or episode['split'] not in ('TRAIN', 'DEV')
                or spec['pilot2_plan_sha'] != PLAN or spec['episode_manifest_sha256'] != digest(episode_path)
                or result['native_binary_sha256'] != digest(spec['native_binary'])
                or result['scene_sha256'] != digest(spec['model_path'])
                or result['source_sha'] != spec['source_sha']):
            raise ValueError('ACTUAL_COMPLETE_SAFE_SUPPORTED_CONTACT_REQUIRED')
        if spec.get('parallel_batch_spec'):
            batch_root = Path(spec['parallel_batch_spec']).parent
            batch = json.loads((batch_root/'batch-01/RESULT.json').read_text())
            if batch['status'] != 'TWO_ENVIRONMENT_NATIVE_PILOT_PASSED':
                raise ValueError('WHOLE_BATCH_CONTACT_ELIGIBILITY_REQUIRED')
        episodes.append(dict(run_id=root.name, parent=episode['parent'], split=episode['split'],
            episode=episode, samples=score['samples'], native_result_sha256=digest(result_path),
            score_sha256=digest(score_path), source_sha=spec['source_sha'],
            native_binary_sha256=result['native_binary_sha256'], scene_sha256=result['scene_sha256'],
            goal_success=score['goal_success'], first_blocker=score['first_blocker']))
    data = replay(episodes)
    output = Path(output); output.mkdir(exist_ok=False)
    path = output/'CONTACT_BUNDLE.private.json.gz'
    value = dict(schema='WM02_PILOT2_SUPPORTED_ACTUAL_BUNDLE_V1', episodes=episodes,
        complete_MPC_WBC_hidden_state=False, free_rollout_qualified=False, plan_sha=PLAN)
    with path.open('xb') as f:
        f.write(gzip.compress(json.dumps(value, allow_nan=False, separators=(',', ':')).encode(), mtime=0))
    result = dict(status='ACTUAL_SUPPORTED_MULTI_HORIZON_CONTACT_BUNDLE_SEALED',
        bundle_sha256=digest(path), episodes=len(episodes), raw_atoms=168*len(episodes),
        raw_native_ticks=6703*len(episodes), forecast_anchors=len(data['history']),
        valid_forecasts=int(data['valid'].sum()), forecast_horizons_atoms=list(data['horizons']),
        TRAIN_parents=sorted({e['parent'] for e in episodes if e['split']=='TRAIN'}),
        DEV_parents=sorted({e['parent'] for e in episodes if e['split']=='DEV'}),
        native_integrations_by_builder=0, model_updates=0, TEST_read=False)
    write_json(output/'RESULT.private.json', result); return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv); print(json.dumps(build(json.loads(a.manifest.read_text()), a.output))); return 0


if __name__ == '__main__': raise SystemExit(main())
