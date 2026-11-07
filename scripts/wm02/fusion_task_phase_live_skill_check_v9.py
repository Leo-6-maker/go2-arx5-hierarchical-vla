"""Read-only live composition check; only the original worker may execute."""
import argparse
import json
from pathlib import Path

def check(path):
    from go2arx5_vla.world_model.fusion_task_phase_live_skill_contract_v9 import episode_limit,B,terminal_step
    spec=json.loads(Path(path).read_bytes())
    final=episode_limit(spec,json.loads(Path(spec['previous_stage']).read_bytes()))
    return dict(status='GRADED_SAME_WORLD_FULL_H40_LIVE_COMPOSITION_CHECKED_NOT_EXECUTED',
        initialization_boundary=B,registered_final_step=terminal_step(spec['task_condition']),max_ticks=final,
        old_half_query_ready=False,old_half_full_grade=False,
        source_task_phase_equivalence=False,native_execution_authorized=False,
        GPU_execution_authorized=False,new_integrations=0,new_queries=0,WM_updates=0,Actor_updates=0,
        runtime_models_or_adapters_created=False,existing_worker='scripts.wm02.pilot1_native_worker')

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True)
    group=p.add_mutually_exclusive_group();group.add_argument('--execute',action='store_true');group.add_argument('--check-only',action='store_true')
    args=p.parse_args(argv);value=check(args.spec)
    if not args.execute:
        print(json.dumps(value,sort_keys=True));return 0
    from scripts.wm02.pilot1_native_worker import main as worker
    return worker(['--spec',str(args.spec)])

if __name__=='__main__':raise SystemExit(main())
