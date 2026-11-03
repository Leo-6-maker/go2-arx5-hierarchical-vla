"""Bounded native Teacher/perturbation collection; no VLM or rendering."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


def collect_episode(spec, card, folder, *, perturb=False, restore_check=False, save_branches=False):
    import numpy as np
    from go2arx5_vla.world_model.native import NativeAdapter
    from go2arx5_vla.world_model.contracts import command_labels, compile_action, macro_transition
    from go2arx5_vla.world_model.storage import dump_private, write_json, digest
    folder.mkdir()
    write_json(folder/'CARD.json', card)
    began, cpu = time.monotonic(), time.process_time()
    adapter = NativeAdapter(card, spec['asset_root'], executor_id=spec['executor_id'], seed=card['seed'])
    initial = adapter.snapshot()
    atoms, macros, failures, branch_files, checks = [], [], [], [], []
    extra_ticks = 0
    def guard():
        if time.time() >= spec['deadline_epoch'] or (Path(spec['registered_root'])/'STOP.json').exists():
            raise TimeoutError('FINITE_WM01_ENDPOINT_OR_STOP')
    try:
        for macro_index in range(80):
            if time.time() >= spec['deadline_epoch']:
                failures.append('FINITE_WALL_ENDPOINT')
                break
            plan = adapter.teacher_plan()
            raw, representable = command_labels(plan['poses'][:8], plan['opening'][:8], adapter.snapshot()['tcp'])
            if perturb and representable:
                # Independent fixed perturbation law, no resampling after rejection.
                raw[:, :3] += adapter.rng.normal(0, .0005, (8, 3))
                norm = np.linalg.norm(raw[:, :3], axis=1, keepdims=True)
                raw[:, :3] *= np.minimum(1., .01/np.maximum(norm, 1e-12))
                plan = compile_action(raw, adapter.snapshot()['tcp'], adapter.snapshot()['time_s'])
            gate = adapter.gate(plan)
            if gate['decision'] == 'REJECT':
                failures.append(dict(reason='NATIVE_GATE_REJECT', macro_index=macro_index, gate=gate))
                # A rejected command is retained separately, never executed expert data.
                dump_private(folder/f'rejected-{macro_index:03d}.pkl.gz', dict(plan=plan, state=adapter.capture(), gate=gate))
                break
            if save_branches and macro_index in (0, 16, 32, 48):
                name = f'branch-{macro_index:03d}.pkl.gz'
                dump_private(folder/name, dict(restore=adapter.capture(), plan=plan))
                branch_files.append(name)
            if restore_check and macro_index in (0, 16, 32, 48):
                # Two branches from a serialized complete state; all additional
                # physics count even though this is a deterministic restore check.
                saved = adapter.capture()
                dump_private(folder/f'restore-{macro_index:03d}.pkl.gz', saved)
                from go2arx5_vla.world_model.storage import load_private
                serialized = load_private(folder/f'restore-{macro_index:03d}.pkl.gz')
                trajectories = []
                for _ in range(2):
                    adapter.restore(serialized)
                    forecast = adapter.preflight(plan, guard)
                    if forecast['decision'] != 'PASS':
                        failures.append(dict(reason='RESTORE_PROBE_S1_REJECT', forecast=forecast))
                        break
                    branch = []
                    for k in range(8):
                        row = adapter.advance_atom(plan, k, raw=raw[k])
                        extra_ticks += row['physics_ticks']
                        branch.append(row)
                        if row['terminal'] or row['truncated']:
                            break
                    trajectories.append(branch)
                if len(trajectories) != 2:
                    break
                error = max(float(np.max(np.abs(a['state_after'][key]-b['state_after'][key])))
                    for a, b in zip(*trajectories) for key in ('qpos', 'qvel', 'tcp', 'object_pose'))
                contacts_equal = all(a['state_after']['contacts'] == b['state_after']['contacts'] for a, b in zip(*trajectories))
                terminal_equal = all((a['terminal'], a['truncated'], a['safety_reason']) ==
                                     (b['terminal'], b['truncated'], b['safety_reason']) for a, b in zip(*trajectories))
                checks.append(dict(macro_index=macro_index, max_abs_error=error,
                    contacts_equal=contacts_equal, terminal_equal=terminal_equal, tolerance=1e-10,
                    branches=2, physics_ticks=sum(r['physics_ticks'] for b in trajectories for r in b)))
                if error > 1e-10 or not contacts_equal or not terminal_equal:
                    raise RuntimeError('COMPLETE_NATIVE_RESTORE_PARITY_FAILED')
                adapter.restore(saved)
            forecast = adapter.preflight(plan, guard)
            if forecast['decision'] != 'PASS':
                failures.append(dict(reason='ORIGINAL_S1_REJECT', macro_index=macro_index, forecast=forecast))
                dump_private(folder/f'rejected-S1-{macro_index:03d}.pkl.gz', dict(plan=plan, state=adapter.capture(), forecast=forecast))
                break
            chunk = []
            for k in range(8):
                row = adapter.advance_atom(plan, k, raw=raw[k])
                row.update(reference_id=f"{card['id']}:{macro_index:03d}", point_index=k,
                           bc_representable=representable, gate=gate, original_S1=forecast)
                atoms.append(row)
                chunk.append(row)
                if row['terminal'] or row['truncated']:
                    break
            macro = macro_transition(chunk)
            macro['bc_representable'] = representable
            macro['reference_H40'] = plan['rows10']
            macros.append(macro)
            if chunk[-1]['terminal'] or chunk[-1]['truncated']:
                break
        score = adapter.score(initial)
        dump_private(folder/'transitions.pkl.gz', dict(atoms=atoms, macros=macros))
        result = dict(episode_id=card['id'], split=card['split'], perturb=perturb,
            source_sha=spec['source_sha'], executor_id=spec['executor_id'], model_xml_sha256=adapter.model_sha256,
            physics_ticks=adapter.engine.steps, restore_extra_ticks=extra_ticks,
            shadow_ticks=adapter.shadow_ticks, maximum_live_shadow_difference=adapter.max_shadow_error,
            atoms=len(atoms), complete_macros=sum(m['complete_state_target'] for m in macros),
            score=score, failures=failures, restore_checks=checks, branch_files=branch_files,
            transition_sha256=digest(folder/'transitions.pkl.gz'),
            CPU_s=time.process_time()-cpu, wall_s=time.monotonic()-began,
            maximum_RSS_KiB=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss,
            safety_reasons=list(adapter.engine.safety), rendered_frames=0,
            WM_updates=0, actor_updates=0, native_32s=bool(adapter.engine.steps == 16000))
        write_json(folder/'RESULT.json', result)
        return result
    except BaseException:
        # Preserve physically completed intervals even on an engineering failure.
        dump_private(folder/'FAILED_PARTIAL.pkl.gz', dict(atoms=atoms, macros=macros, restore=adapter.capture()))
        write_json(folder/'ENGINEERING_FAILURE.json', dict(error=__import__('traceback').format_exc(),
            physics_ticks=adapter.engine.steps, restore_extra_ticks=extra_ticks,
            shadow_ticks=adapter.shadow_ticks, maximum_live_shadow_difference=adapter.max_shadow_error,
            CPU_s=time.process_time()-cpu, wall_s=time.monotonic()-began))
        raise
    finally:
        adapter.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('TRAIN', 'CALIB'), default='TRAIN')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--attempts', type=int, default=4)
    parser.add_argument('--restore-check', action='store_true')
    args = parser.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    from go2arx5_vla.world_model.storage import write_json
    with admitted_run(args.spec, args.run_id, args.profile, args.output, physics=True) as spec:
        cards = json.loads(Path(spec['splits'][args.split]['path']).read_text())
        if args.start < 0 or args.attempts <= 0 or args.start+args.attempts > len(cards):
            raise ValueError('Finite predeclared attempt range required')
        results = []
        for index in range(args.start, args.start+args.attempts):
            card = cards[index]
            results.append(collect_episode(spec, card, args.output/card['id'],
                perturb=args.split == 'TRAIN' and index >= 32,
                restore_check=args.restore_check, save_branches=args.split == 'CALIB'))
            print(json.dumps(results[-1], default=str), flush=True)
        write_json(args.output/'RESULTS.json', dict(run_id=args.run_id, source_sha=spec['source_sha'], episodes=results))


if __name__ == '__main__':
    main()
