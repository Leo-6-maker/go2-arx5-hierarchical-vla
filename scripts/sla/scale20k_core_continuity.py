"""Read a cold Core training segment and derive its next exact continuation."""
import argparse
import json
from pathlib import Path

from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_checkpoint import read_complete
from go2arx5_vla.sla_adapter.scale20k_contract import BATCH, CHECKPOINTS, DOMAINS, PLAN, RECIPE, cursors
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import unit_state
from scripts.pi0292.durable import read
from scripts.sla.scale20k_core_prefix import closed_predecessor


def prepare(previous, destination, worker_files=None):
    previous = previous.resolve(strict=True)
    destination = destination.resolve()
    if destination.exists() or destination.parent != previous.parent:
        raise ValueError('SCALE_CONTINUITY_REQUIRES_A_FRESH_SIBLING_SEGMENT')
    schedule_path = previous/'SCHEDULE.private.json'
    schedule = read(schedule_path)
    arm = schedule['arm']
    expected_unit = 'wm02-'+previous.name
    if schedule['plan'] != PLAN or schedule['unit'] != expected_unit or schedule['root'] != str(previous):
        raise ValueError('SCALE_CONTINUITY_ORIGINAL_CORE_SCHEDULE_REQUIRED')
    state = unit_state(expected_unit+'.service')
    if state['pid'] or state['cgroup_pids'] or state['ActiveState'] not in ('inactive','failed'):
        raise PermissionError('SCALE_CONTINUITY_PREDECESSOR_STILL_OWNS_CORE_RESOURCES')
    parent = read(previous/'RESULT.private.json')
    prior, worker, step = closed_predecessor(dict(schedule,previous_train_result=parent['stages']['train']),destination)
    result = read_pin(prior)
    branch = previous/'train'
    spec_path = branch/'SPEC.private.json'
    card = read(spec_path)
    ancestor = read_pin(card['resume_from_closed_segment'])
    first = ancestor['global_step']
    limit = schedule.get('global_stop_step',200)
    immutable = ('arm','source_sha','GPU_uuid','data','optimizer','parent_manifest','tokenizer','visit_bank_sha256')
    if (not 0 <= first < step <= limit <= CHECKPOINTS[arm][-1]
            or card['plan_sha'] != PLAN or card['recipe'] != RECIPE or limit != card['global_stop_step']
            or card['physics'] is not False or card['hardware_allowed'] is not False
            or any(card[k] != schedule[k] for k in immutable)
            or ancestor['arm'] != arm or ancestor['source_sha'] != schedule['source_sha']
            or parent['status'] != result['status']
            or (result['exit_code'] == 0 and step != limit)):
        raise ValueError('SCALE_CONTINUITY_REQUIRES_THE_ORIGINAL_PLAN_AND_TERMINAL_EFFECTS')
    before, after = cursors(arm,first), cursors(arm,step)
    updates = step-first
    visits = {d:after[d]-before[d] for d in DOMAINS}
    minimum_generations = 4*updates
    extra = 4 if result['exit_code'] == 75 else 0
    if (result['new_updates'] != updates or result['new_visits'] != visits
            or not minimum_generations <= result['new_model_generations'] <= minimum_generations+extra):
        raise ValueError('SCALE_CONTINUITY_MUST_PRESERVE_REAL_OPTIMIZER_AND_GENERATION_COUNTS')
    ledger = branch/'UPDATES.private.jsonl'
    effects = ledger.read_bytes()
    rows = effects.splitlines(keepends=True)
    if len(rows) != updates or any(not line.endswith(b'\n') for line in rows):
        raise ValueError('SCALE_CONTINUITY_REQUIRES_ALL_DURABLE_UPDATES')
    active = {domain for domain,count in BATCH[arm].items() if count}
    for expected_step,line in enumerate(rows,first+1):
        row = json.loads(line)
        if (row['step'] != expected_step or row['cursors'] != cursors(arm,expected_step)
                or set(row['domains']) != active or row['domains']['target']['training_full_generations'] != 4):
            raise ValueError('SCALE_CONTINUITY_DURABLE_UPDATE_CHANGED')
    model = read(branch/'MODEL_AT_START.private.json')
    state_root, complete = read_complete(result['checkpoint'],arm,model['binding'])
    arrays = complete['arrays']
    if (state_root != branch/f'step{step:05d}' or complete['step'] != step or len(arrays) != 140
            or [array['file'] for array in arrays] != [f'{i:06d}.npy' for i in range(140)]):
        raise ValueError('SCALE_CONTINUITY_EXACT_COMPLETE_STATE_DIRECTORY_REQUIRED')
    for array in arrays:
        path = state_root/array['file']
        if path.is_symlink() or sha(path) != array['sha256']:
            raise ValueError('SCALE_CONTINUITY_ORIGINAL_CHECKPOINT_ARRAY_CHANGED:'+array['file'])
    next_step = next((n for n in CHECKPOINTS[arm] if n > step),None)
    draft = None
    member_count = 0
    if next_step is not None:
        members = schedule.get('training_source_files')
        if members is None and worker_files is not None:
            members = read_pin(worker_files)
        if not members or not {'scripts/sla/train_scale20k.py',
                               'src/go2arx5_vla/sla_adapter/scale20k_training.py'}.issubset(members):
            raise ValueError('SCALE_CONTINUITY_ORIGINAL_TRAINING_SOURCE_MANIFEST_REQUIRED')
        for relative,expected in members.items():
            path = (worker/relative).resolve(strict=True)
            if not path.is_relative_to(worker) or sha(path) != expected:
                raise ValueError('SCALE_CONTINUITY_ORIGINAL_TRAINER_SOURCE_CHANGED:'+relative)
        member_count = len(members)
        draft = {k:v for k,v in schedule.items() if k not in ('deadline_epoch','technical_predecessor')}
        draft.update(root=str(destination),unit='wm02-'+destination.name,training_source_files=members,
                     global_stop_step=next_step,previous_train_result=prior)
    return dict(arm=arm,training_source_sha=schedule['source_sha'],worker_source_root=str(worker),
                worker_spec_sha256=sha(spec_path),steps=[first,step],completed_updates=updates,new_visits=visits,
                checkpoint=result['checkpoint'],verified_checkpoint_arrays=len(arrays),update_ledger_sha256=sha(ledger),
                next_formal_step=next_step,draft=draft,verified_source_members=member_count,
                launched_training_units=0,new_optimizer_updates=0,native_requests=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous',type=Path)
    parser.add_argument('destination',type=Path)
    parser.add_argument('--source-files',nargs=2,metavar=('PATH','SHA256'))
    args = parser.parse_args()
    worker_files = dict(zip(('path','sha256'),args.source_files,strict=True)) if args.source_files else None
    print(json.dumps(prepare(args.previous,args.destination,worker_files),allow_nan=False))
