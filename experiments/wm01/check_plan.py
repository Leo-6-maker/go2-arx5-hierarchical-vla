"""Check the WM01 design only; never launch models, simulation or remote jobs."""
from __future__ import annotations
import argparse
import json
import math
import re
import sys
from pathlib import Path


def validate(plan: dict) -> list[str]:
    errors = []
    def require(ok, message):
        if not ok:
            errors.append(message)
    try:
        require(plan['schema_version'] == 1, 'schema_version must be 1')
        require(plan['experiment_id'] == 'WM01', 'wrong experiment_id')
        require(bool(re.fullmatch(r'[0-9a-f]{40}', plan['baseline_commit'])), 'full baseline SHA required')
        require(plan['runtime_status'] == 'NOT_IMPLEMENTED', 'design must not claim runtime qualification')
        s, i, d, p = (plan[k] for k in ('scope', 'interface', 'data', 'policy'))
        require(s['target_wbc_frozen'] is True, 'WBC must stay frozen')
        for key in ('target_dynamic_execution', 'real_robot_execution', 'touch_existing_runs', 'new_scheduler', 'data_deletion'):
            require(s[key] is False, 'out-of-scope effect: '+key)
        require(i['physical_action_dim'] == 10 and i['trajectory_horizon'] == 40, 'EEF10/H40 required')
        require(i['execute_prefix'] == 8, 'prefix8 required')
        require(i['actor_parameter_dim'] == 7*i['execute_prefix'], 'actor shape mismatch')
        require(math.isclose(i['reference_dt_s']*i['execute_prefix'], i['macro_dt_s']), 'macro time mismatch')
        require(i['future_suffix_is_executed_label'] is False, 'unexecuted suffix is not a label')
        require(d['teacher_attempts']+d['perturbed_attempts'] == d['initial_train_attempts'], 'initial attempt count mismatch')
        require(d['normalization_split'] == 'TRAIN' and d['failures_retained'] is True, 'data boundary changed')
        require(p['arms'] == ['ppo', 'ppo_wm'] and p['shared_initialization'] is True, 'paired controls required')
        require(len(set(p['seeds'])) == len(p['seeds']) == 3, 'three distinct training seeds required')
        require(p['pilot_real_macro_steps'] <= p['final_real_macro_steps'], 'pilot exceeds endpoint')
        require(p['checkpoints'] == sorted(set(p['checkpoints'])), 'checkpoints must be unique and ordered')
        require(p['checkpoints'][-1] == p['final_real_macro_steps'], 'missing fixed final checkpoint')
        require(p['final_checkpoint_selection'] == 'FIXED_FINAL_ENDPOINT', 'no TEST checkpoint selection')
        expected = d['test_layouts']*len(p['seeds'])*len(p['arms'])
        require(plan['evaluation']['final_test_episodes'] == expected, 'final evaluation count mismatch')
        require(plan['evaluation']['native_score_unchanged'] is True, 'native score must be preserved')
        require(plan['evaluation']['historical_dev68_unchanged'] is True, 'historical panel must be preserved')
        require(plan['imagination']['freeze_wm_parameters'] is True, 'freeze WM during actor update')
        require(plan['imagination']['retain_action_gradients'] is True, 'do not detach action gradients')
        seen = set()
        for stage in plan['stages']:
            require(stage['id'] not in seen, 'duplicate stage')
            require(set(stage['depends_on']) <= seen, 'unresolved or cyclic stage dependency')
            seen.add(stage['id'])
        require(seen == {'W0','W1','W2','W3','W4','W5'}, 'stage set mismatch')
        require(plan['stages'][-1].get('automatic_execution') is False, 'W5 is a separate design')
        require(all(type(v) is int and v == 0 for v in plan['actual_results_this_publication'].values()), 'publication cannot invent runtime results')
        require(len(plan['required_bindings_before_compute']) > 0, 'native bindings missing')
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append('malformed plan: '+str(exc))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path(__file__).with_name('plan.json'))
    args = parser.parse_args()
    try:
        plan = json.loads(args.plan.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        print('FAIL: '+str(exc), file=sys.stderr)
        return 2
    errors = validate(plan)
    if errors:
        print('\n'.join('FAIL: '+item for item in errors), file=sys.stderr)
        return 1
    print('PASS: WM01 design consistency only; runtime/native/GPU qualification NOT_RUN')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
