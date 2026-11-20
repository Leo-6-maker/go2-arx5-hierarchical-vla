"""Reconcile a saved SCALE prefix after a dead Core supervisor without changing its receipts."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_contract import BATCH, DOMAINS, PLAN, RECIPE, bank_sha256, canonical_hash, learning_rates
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import unit_state
from scripts.pi0292.durable import OwnerLock, read, write
from scripts.sla.scale20k_scheduled_handoff import verify_seal
from scripts.wm02.multisource_d4_vanda import pin


SCHEMA = 'SLA_SCALE_CORE_CRASH_RECONCILIATION_V1'
IMMUTABLE = ('arm', 'source_sha', 'GPU_uuid', 'neural_owner_lock', 'data', 'optimizer',
             'parent_manifest', 'parent_params', 'tokenizer', 'visit_bank_sha256', 'JAX_memory_fraction')


def stopped(paths):
    return any(Path(path).exists() for path in paths)


def cold_supervisor(previous, schedule, old):
    expected = 'wm02-' + previous.name + '.service'
    cg = Path('/sys/fs/cgroup')/old['ControlGroup'].lstrip('/')
    current = unit_state(expected)
    facts = subprocess.check_output(['systemctl','--user','show',expected,'-p','Result','-p','ExecMainCode',
        '-p','ExecMainStatus','-p','ActiveState','-p','MemoryMax'],text=True,timeout=10)
    fields = dict(row.split('=',1) for row in facts.splitlines())
    lease = read(previous/'control/LEASE.json')
    if (not old['ControlGroup'].endswith('/'+expected) or old['unit'] != expected
            or schedule['unit']+'.service' != expected or current['pid'] or current['cgroup_pids']
            or fields['ActiveState'] != 'failed' or fields['Result'] != 'oom-kill'
            or (fields['ExecMainCode'],fields['ExecMainStatus']) != ('2','9')
            or (cg/'cgroup.procs').exists() and (cg/'cgroup.procs').read_text().strip()
            or (previous/'RESULT.private.json').exists()
            or (lease['status'],lease['unit'],lease['main_pid'],lease['deadline_epoch']) !=
                ('ACTIVE',expected,old['pid'],schedule['deadline_epoch'])
            or schedule['deadline_epoch'] >= time.time()):
        raise PermissionError('SCALE_CRASH_REQUIRES_A_DEAD_RECORDED_OOM_OWNER_AND_UNCHANGED_OLD_LEASE')
    proc = Path('/proc')/str(old['pid'])/'stat'
    if proc.is_file() and proc.read_text().rsplit(')',1)[1].split()[19] == old['proc_start']:
        raise PermissionError('SCALE_CRASH_ORIGINAL_PID_STILL_ALIVE')
    return dict(unit=expected, old_pid=old['pid'], old_cgroup=old['ControlGroup'], fields=fields,
                cgroup_empty=True, outer_terminal_missing=True, historical_lease_status=lease['status'])


def audit(config, resize_with_pad):
    from go2arx5_vla.sla_adapter.scale20k_data import bank_for, readers

    schedule = read_pin(config['schedule'])
    previous = Path(schedule['root']).resolve(strict=True)
    train = previous/'train'
    card_pin = pin(train/'SPEC.private.json')
    card = read_pin(card_pin)
    prior = pin(train/'RESULT.private.json')
    terminal = read_pin(prior)
    if (previous.stat().st_uid != os.getuid() or schedule['uid'] != os.getuid()
            or os.uname().nodename != schedule['host'] or schedule['plan'] != PLAN
            or schedule['arm'] != 'SCALE-MIX20K' or config['schedule'] != pin(previous/'SCHEDULE.private.json')
            or (card['plan_sha'],card['recipe']) != (PLAN,RECIPE)
            or any(card[k] != schedule[k] for k in IMMUTABLE)
            or card['global_stop_step'] != schedule['global_stop_step']
            or card['resume_from_closed_segment'] != schedule['previous_train_result']
            or card['hardware_allowed'] or card['physics'] or card['new_native_requests_cap']
            or card['microbatch'] != dict.fromkeys(DOMAINS,1)
            or stopped(config['stops']) or (train/'STOP.json').exists() or (previous/'STOP.json').exists()):
        raise PermissionError('SCALE_CRASH_REQUIRES_THE_ORIGINAL_FROZEN_RECIPE_AND_NO_STOP')
    old = config['previous_owner']
    owner = cold_supervisor(previous,schedule,old)
    lock_path = Path(schedule['neural_owner_lock'])
    if not lock_path.is_file():
        raise PermissionError('SCALE_CRASH_ORIGINAL_OWNER_LOCK_UNAVAILABLE')
    lock = OwnerLock(lock_path)
    lock.close()
    top_log = subprocess.check_output(['journalctl','--user','-u',owner['unit'],'--no-pager','-n','12',
        '-o','short-iso'],text=True,timeout=12)
    if 'systemd-oomd killed' not in top_log:
        raise ValueError('SCALE_CRASH_USER_JOURNAL_OOMD_EVENT_REQUIRED')
    seal = verify_seal(prior)
    start = read_pin(card['resume_from_closed_segment'])['global_step']
    step, n = seal['step'],terminal['new_updates']
    if (step != config['expected_checkpoint_step'] or (terminal['status'],terminal['exit_code']) != ('BOUNDED_SAVED_PREFIX',75)
            or (terminal['scope'],terminal['arm'],terminal['mode'],terminal['source_sha']) !=
                (RECIPE,schedule['arm'],'train',schedule['source_sha'])
            or n != step-start or any(p.name.startswith('step') and p.name!=f'step{step:05d}' for p in train.iterdir())):
        raise ValueError('SCALE_CRASH_ONLY_EXACT_FINAL_PREFIX_WITHOUT_LOST_UPDATE_TAIL')
    source = Path(config['worker_source']).resolve(strict=True)
    prior_parent = Path(card['resume_from_closed_segment']['path']).resolve(strict=True).parent.parent
    source_members = schedule['training_source_files']
    if (Path(read(prior_parent/'RESULT.private.json')['training_source_root']).resolve(strict=True) != source
            or not {'scripts/sla/train_scale20k.py','src/go2arx5_vla/sla_adapter/scale20k_training.py'}.issubset(source_members)):
        raise ValueError('SCALE_CRASH_ORIGINAL_TRAINING_RUNTIME_UNAVAILABLE')
    for name, expected in source_members.items():
        path = (source/name).resolve(strict=True)
        if not path.is_relative_to(source) or pin(path)['sha256'] != expected:
            raise ValueError(f'SCALE_CRASH_RUNTIME_MEMBER_CHANGED:{name}')
    model = read(train/'MODEL_AT_START.private.json')['binding']
    rates = learning_rates()
    banks, digests = bank_for(readers(card['data'],resize_with_pad))
    if (digests != card['visit_bank_sha256'] or digests != model['visit_bank_sha256']
            or bank_sha256(rates) != card['learning_rates_sha256'] or bank_sha256(rates) != model['learning_rates_sha256']
            or canonical_hash(card['optimizer']) != model['optimizer_sha256']
            or read_pin(card['parent_manifest']) is None or card['parent_manifest']['sha256'] != model['parent_manifest_sha256']
            or pin(Path(card['tokenizer']['path'])) != card['tokenizer'] or card['tokenizer']['sha256'] != model['tokenizer_sha256']
            or {d:card['data'][d]['manifest']['sha256'] for d in DOMAINS} != model['data_manifest_sha256']):
        raise ValueError('SCALE_CRASH_CANNOT_VERIFY_THE_ORIGINAL_DATA_SAMPLER_LR_AND_MODEL_BINDING')
    width = BATCH[schedule['arm']]
    ledger_generations = 0
    for expected, raw in enumerate(Path(seal['ledger']['path']).read_bytes().splitlines(),start+1):
        row = json.loads(raw)
        domains = row['domains']
        if (row['learning_rate'] != rates[expected-1] or not math.isfinite(row['grad_norm'])
                or set(domains) != set(DOMAINS) or domains['target']['training_full_generations'] != 4
                or any(domains[d]['backward_calls'] != width[d] or domains[d]['visit_slice_sha256'] !=
                    bank_sha256(banks[d][(expected-1)*width[d]:expected*width[d]]) for d in DOMAINS)):
            raise ValueError(f'SCALE_CRASH_LEDGER_GRADIENT_BANK_OR_LR_CHANGED:{expected}')
        ledger_generations += domains['target']['training_full_generations']
    visits = {d:n*width[d] for d in DOMAINS}
    extra = {d:terminal['gradient_input_windows'][d]-visits[d] for d in DOMAINS}
    generations = terminal['new_model_generations']
    attempts = terminal['attempted_model_generations']
    if (terminal['new_visits'] != visits or not ledger_generations <= generations <= attempts <= ledger_generations+4
            or any(not 0 <= extra[d] <= width[d] for d in DOMAINS)):
        raise ValueError('SCALE_CRASH_TAIL_EFFECTS_MUST_RECONCILE_WITH_DURABLE_UPDATES')
    cold_supervisor(previous,schedule,old)
    lock=OwnerLock(lock_path)
    lock.close()
    return dict(schema=SCHEMA,observed_epoch=time.time(),owner=owner,owner_lock_clear=True,
        journal_sha256=hashlib.sha256(top_log.encode()).hexdigest(),schedule=config['schedule'],
        old_lease=pin(previous/'control/LEASE.json'),old_spec=card_pin,previous_owner=old,
        stops=config['stops'],training_seal=seal,source_root=str(source),verified_source_members=len(source_members),
        original_source_sha=schedule['source_sha'],previous_step=start,first_successor_update=step+1,
        verified_bank_sha256=digests,verified_LR_sha256=card['learning_rates_sha256'],
        optimizer_updates=n,optimization_visits=visits,completed_generations=generations,generation_attempts=attempts,
        discarded_unapplied_generation_attempts=attempts-ledger_generations,discarded_unapplied_gradient_windows=extra,
        unrecoverable_optimizer_updates=0,native_requests=0)


def resolve(config, root):
    proof = read_pin(config['crash_reconciliation'])
    prior = proof['training_seal']['result']
    old_schedule = read_pin(proof['schedule'])
    previous = Path(old_schedule['root']).resolve(strict=True)
    step = proof['training_seal']['step']
    if (proof['schema'] != SCHEMA or Path(prior['path']) != previous/'train/RESULT.private.json'
            or root == previous or root.parent != previous.parent or stopped(proof['stops']) or
            (previous/'STOP.json').exists() or (previous/'train/STOP.json').exists()
            or proof['old_lease'] != pin(previous/'control/LEASE.json')
            or proof['old_spec'] != pin(previous/'train/SPEC.private.json')
            or proof['training_seal']['ledger'] != pin(previous/'train/UPDATES.private.jsonl')
            or proof['unrecoverable_optimizer_updates'] or proof['native_requests']
            or config['previous_train_result'] != prior or any(config[k] != old_schedule[k] for k in IMMUTABLE)
            or config['training_source_files'] != old_schedule['training_source_files']
            or not step < config['global_stop_step'] <= old_schedule['global_stop_step']
            or not time.time()+1800 < config['deadline_epoch'] < time.time()+14400):
        raise ValueError('SCALE_CRASH_INVALID_COLD_RECEIPT_OR_FINITE_SUCCESSOR')
    cold_supervisor(previous,old_schedule,proof['previous_owner'])
    state = read_pin(prior)
    if (state['global_step'] != step or state['checkpoint'] != proof['training_seal']['checkpoint'] or
            pin(Path(state['checkpoint']['path'])/'COMPLETE.json')['sha256'] != state['checkpoint']['sha256']):
        raise ValueError('SCALE_CRASH_ORIGINAL_SEAL_CHANGED')
    return prior,Path(proof['source_root']).resolve(strict=True),step


if __name__ == '__main__':
    from openpi_client.image_tools import resize_with_pad

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    write(args.output,audit(read_pin(dict(path=args.spec,sha256=args.sha256)),resize_with_pad))
