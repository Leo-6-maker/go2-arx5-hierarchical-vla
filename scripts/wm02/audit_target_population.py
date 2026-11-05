"""Audit sealed NIGHT1 production recordings without simulation or training."""
from pathlib import Path
import json


def select(campaign):
    """Include complete production episodes only after the whole peer batch passed."""
    included, excluded = [], []
    for root in sorted((Path(campaign) / 'runs').glob('night1-*')):
        manifest = root / 'EPISODE_MANIFEST.private.json'
        legacy = root.name.startswith('night1-n1-01-slot-')
        if not manifest.exists() and not legacy:
            continue
        episode = json.loads(manifest.read_text()) if manifest.exists() else {}
        if not legacy and episode.get('purpose') not in ('PRODUCTION', 'DECISION_PRODUCTION'):
            continue
        spec = json.loads((root / 'EXECUTION_SPEC.json').read_text())
        result_path = root / 'native-01/RESULT.json'
        reason = None
        if json.loads((root / 'control/LEASE.json').read_text())['status'] != 'CLOSED':
            reason = 'WORKER_NOT_SEALED'
        elif not result_path.exists():
            reason = 'NO_NATIVE_RESULT'
        else:
            result = json.loads(result_path.read_text())
            if result['status'] != 'NATIVE_STAGE_PASSED':
                reason = 'NATIVE_FAILED'
            elif spec.get('parallel_batch_spec'):
                batch = Path(spec['parallel_batch_spec']).parent / 'batch-01/RESULT.json'
                if not batch.exists() or json.loads(batch.read_text())['status'] != 'TWO_ENVIRONMENT_NATIVE_PILOT_PASSED':
                    reason = 'WHOLE_PEER_BATCH_NOT_PASSED'
        if reason:
            excluded.append(dict(episode=root.name, reason=reason))
        else:
            included.append((root, spec, episode))
    return included, excluded


def run(campaign):
    import time
    from scripts.wm02.audit_recorded_target import audit
    from go2arx5_vla.world_model.storage import digest
    wall, cpu = time.monotonic(), time.process_time()
    included, excluded = select(campaign)
    rows = []
    for root, spec, episode in included:
        native = root / 'native-01'
        config = dict(episode_id=root.name, label='TARGET_QUALIFIED_RECORDING_REPLAY', original_native_source_sha=spec['source_sha'])
        for key, path in [('scene', Path(spec['model_path'])), ('journal', native / 'state.bin.journal'), ('native_result', native / 'RESULT.json'), ('commands', native / 'COMMANDS.private.jsonl')]:
            config[key] = dict(path=str(path), sha256=digest(path))
        value = audit(config)
        value['manifest'] = episode
        value['manifest_sha256'] = digest(root / 'EPISODE_MANIFEST.private.json') if (root / 'EPISODE_MANIFEST.private.json').exists() else None
        rows.append(value)
    return dict(schema='WM02_NIGHT1_POPULATION_AUDIT_V1', included_episodes=len(rows), complete_atoms=sum(r['coverage']['complete_atoms'] for r in rows), rows=rows, excluded_production=excluded, new_integrations=0, new_optimizer_updates=0, CPU_s=time.process_time()-cpu, wall_s=time.monotonic()-wall)


def main(argv=None):
    import argparse
    from go2arx5_vla.world_model.storage import write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.campaign)
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
