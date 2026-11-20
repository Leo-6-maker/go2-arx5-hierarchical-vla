"""Train one finite CLOCK branch from the verified ADAPT2k and D2 state."""
import argparse
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

from go2arx5_vla.sla_adapter import adapt5k_contract as adapt, clock500_checkpoint as state
from go2arx5_vla.sla_adapter.clock500 import ARMS, ENDPOINT, PARENT, PLAN
from go2arx5_vla.sla_adapter.clock500_training import ClockEngine
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.site_v2 import validate as validate_site
from scripts.pi0292.durable import OwnerLock, write
from scripts.sla import adapt5k_curriculum as curriculum
from scripts.sla.adapt5k_vanda_handoff import ORIGINAL_RUNTIME, verify_train_result
from scripts.sla.clock500_trigger import SCOPE as TRIGGER_SCOPE, owned_job, pbs
from scripts.sla.freeze_clock500_native_population import validate as validate_population
from scripts.sla.train_adapt5k import inputs, restored_engine
from scripts.wm02.multisource_d4_vanda import pin


SCOPE = 'SLA_CLOCK500_MATCHED_BRANCH_TRAIN_V1'


def comparison_inputs(spec):
    pair = read_pin(spec['comparison'])
    trigger = read_pin(pair['trigger'])
    trigger_spec = read_pin(pair['trigger_spec'])
    trigger_pbs = pbs(trigger['job_id'])
    if (pair['plan']!=PLAN or pair['arms']!=list(ARMS) or pair['updates_per_arm']!=500
            or pair['source_sha']!=spec['source_sha'] or spec['arm'] not in ARMS
            or set(pair['branch_roots'])!=set(ARMS) or len(set(pair['branch_roots'].values()))!=2
            or trigger['scope']!=TRIGGER_SCOPE or trigger['scientific_trigger'] is not True
            or trigger['joint_numerator']!=0 or trigger['eligible_windows']!=24
            or trigger['semantic_audit']['due_experts_verified']!=24 or trigger['semantic_audit']['labels_verified']!=88
            or trigger['optimizer_updates'] or trigger['model_generations'] or trigger['native_requests']
            or trigger_spec['job_id']!=trigger['job_id'] or trigger_spec['source_sha']!=trigger['source_sha']
            or Path(pair['trigger']['path'])!=Path(trigger_spec['root'])/'RESULT.private.json'
            or not owned_job(trigger_pbs,pair,'F') or trigger_pbs.get('Exit_status')!=0):
        raise ValueError('CLOCK500_NO_TRAINING_WITHOUT_THE_ACTUAL_2K_SCIENTIFIC_TRIGGER')
    original = read_pin(trigger['evidence']['train'])
    training_pin = original['completed_roles']['train']
    parent,seal = verify_train_result(training_pin,original['parent_checkpoint'])
    switched = read_pin(pair['D2_decision'])
    if (parent['exit_code'] or seal['step']!=PARENT or seal['checkpoint']!=trigger['checkpoint']
            or switched['schema']!=curriculum.DECISION or switched['mode']!='D2'
            or switched['phase2000_checkpoint']!=seal['checkpoint'] or switched['used_eval_score']
            or switched['ready']!=trigger_spec['D2_ready'] or switched['plan']!=adapt.PLAN):
        raise ValueError('CLOCK500_REQUIRES_THE_ACTUAL_COLD_D2_SWITCH_ON_THE_TRIGGER_PARENT')
    base = read_pin(pair['phase_zero_config'])
    original_data = read_pin(base['data_contract'])
    if (original['data_contract']!=base['data_contract'] or original['parent_checkpoint']!=base['parent_checkpoint']):
        raise ValueError('CLOCK500_ORIGINAL_MIX_ANCESTRY_OR_D1_DATA_CHANGED')
    d1 = read_pin(original_data['train_input'])
    ready,training,proof,_ = curriculum.inputs(switched['ready'],d1,'D2')
    if (switched['training']!=ready['train_input'] or switched['train_proof']!=ready['train_proof']
            or switched['stage2_bank_sha256']!=proof['vanda']['bank_digests']
            or proof['vanda']['target_bank_report']['new_quota_per_update']!=12
            or proof['vanda']['new_target_episodes']!=3):
        raise ValueError('CLOCK500_REQUIRES_MATCHED_D2_12_NEW_AND_52_ORIGINAL_TARGET_VISITS')
    frozen = read_pin(pair['native_population'])
    validate_population(frozen)
    origin = read_pin(pair['population_qualification'])
    if (origin['status']!='CLOCK500_COLD_PAIRED_POPULATION_FROZEN'
            or origin['population']['sha256']!=pair['native_population']['sha256']
            or origin['native_requests'] or origin['native_ticks'] or origin['optimizer_updates']
            or origin['model_generations'] or origin['future_native_cap']!=72
            or origin['compiled']['protection_fingerprints']!=[pair['original_robot_fingerprint']]):
        raise ValueError('CLOCK500_REQUIRES_FRESH_FROZEN_PAIRED_NATIVE_CONDITIONS')

    # These fields verify the unchanged upstream ADAPT state. Branch provenance
    # is separately recorded under its own recipe, executable and arm below.
    data_card = dict(parent_params=base['parent_params'],parent_manifest=base['parent_manifest'],
        tokenizer=base['tokenizer'],source_sha=ORIGINAL_RUNTIME,previous_train_or_qualification=training_pin,
        data=training['data'],train_data_proof=switched['train_proof'],target_D1_sha256=base['D1_freeze_sha256'],
        parent_checkpoint=base['parent_checkpoint'],parent_training_spec=base['parent_training_spec'],
        parent_step=base['parent_step'],stage_start=PARENT,stage_stop=5000,target_seed=training['target_seed'],
        stage_bank_sha256=switched['stage2_bank_sha256'],microbatch=dict.fromkeys(adapt.DOMAINS,1))
    return pair,data_card,seal


def restore_prefix(spec, binding, engine):
    if spec['previous'] is None:
        return PARENT
    before = read_pin(spec['previous'])
    prior_job = pbs(before['job_id'])
    if (before['scope']!=SCOPE or before['binding']!=binding or not before['checkpoint']
            or before['status'] not in ('CLOCK500_SEGMENT_SAVED','CLOCK500_DEADLINE_PREFIX_SAVED')
            or before['exit_code'] not in (0,75) or not owned_job(prior_job,spec,'F')
            or prior_job.get('Exit_status')!=before['exit_code'] or before['native_requests'] or before['native_ticks']
            or not PARENT<before['end_phase']<ENDPOINT):
        raise ValueError('CLOCK500_RESUME_ONLY_OWN_COLD_BOUND_BRANCH_PREFIX')
    validate_ledger(before['ledger'],before['start_phase'],before['end_phase'],banks=engine.banks,
                    parent_step=engine.parent_step)
    arrays,phase = state.restore(before['checkpoint'],binding,parameters=engine.parameters,optimizer=engine.states,rng=engine.rng)
    if phase!=before['end_phase']:
        raise ValueError('CLOCK500_COLD_BRANCH_CURSOR_CHANGED')
    engine.step,engine.parameters,engine.states,engine.rng = phase,arrays['parameters'],arrays['optimizer'],arrays['rng']
    return phase


def validate_ledger(descriptor, start, end, *, banks=None, parent_step=None):
    path = Path(descriptor['path'])
    if pin(path)!=descriptor:
        raise ValueError('CLOCK500_BRANCH_LEDGER_CHANGED')
    data = path.read_bytes()
    if not PARENT<=start<end<=ENDPOINT or not data.endswith(b'\n') or len(data.splitlines())!=end-start:
        raise ValueError('CLOCK500_REQUIRES_COMPLETE_CONTIGUOUS_UPDATE_LEDGER')
    rates = adapt.phase_rates()
    for phase,line in enumerate(data.splitlines(),start+1):
        row = json.loads(line)
        target = row['domains']['target']
        if (row['phase_step']!=phase or row['learning_rate']!=rates[phase-1]
                or not math.isfinite(row['grad_norm']) or row['grad_norm']<=0.
                or target['training_full_generations']!=4 or tuple(target['auxiliary_slots'])!=adapt.auxiliary_slots(phase-1)
                or row['stage_local_cursors']!={domain:n*(phase-PARENT) for domain,n in adapt.BATCH.items()}):
            raise ValueError(f'CLOCK500_UNMATCHED_SAMPLE_CLOCK_OR_AUXILIARY_SLOTS:{phase}')
        if parent_step is not None:
            clocks = adapt.progress(parent_step,phase,PARENT)
            if any(row[key]!=value for key,value in clocks.items()):
                raise ValueError(f'CLOCK500_UNMATCHED_LINEAGE_CLOCK:{phase}')
        if banks is not None:
            for domain,n in adapt.BATCH.items():
                end_cursor = row['stage_local_cursors'][domain]
                actual = row['domains'][domain]['visit_slice_sha256']
                expected = hashlib.sha256(banks[domain][end_cursor-n:end_cursor].tobytes()).hexdigest()
                if actual!=expected:
                    raise ValueError(f'CLOCK500_UNMATCHED_BANK_SLICE:{phase}:{domain}')


def run(spec):
    import jax
    from openpi_client.image_tools import resize_with_pad

    root = Path(spec['registered_root']).resolve(strict=True)
    source = Path(__file__).resolve().parents[2]
    job = os.environ.get('PBS_JOBID')
    if (root.stat().st_uid!=os.getuid() or any(root.iterdir()) or spec['plan']!=PLAN or spec['job_id']!=job
            or not job or spec['physics'] is not False or spec['hardware_allowed'] is not False
            or spec['source_sha']!=(source/'SOURCE_SHA.private').read_text().strip()
            or not PARENT<spec['stop_phase']<=ENDPOINT or spec['save_reserve_s']<300):
        raise PermissionError('CLOCK500_REQUIRES_A_NEW_FINITE_NONPHYSICAL_BRANCH_SCOPE')
    output = dict(scope=SCOPE,arm=spec['arm'],parent_phase=PARENT,source_sha=spec['source_sha'],job_id=job,
                  native_requests=0,native_ticks=0,optimizer_updates=0,optimization_visits=dict.fromkeys(adapt.DOMAINS,0),
                  complete_H40=0,attempted_H40=0,checkpoint=None,exit_code=1)
    stop = False
    created = time.monotonic()
    engine,start,committed,latest = None,None,0,None
    ledger_path = root/'UPDATES.private.jsonl'

    def stop_signal(*_):
        nonlocal stop
        stop = True

    def guard():
        if (stop or Path(spec['STOP_path']).exists() or Path(spec['campaign_stop']).exists()
                or time.time()+spec['save_reserve_s']>=spec['deadline_epoch']):
            raise TimeoutError('CLOCK500_FINITE_STOP_OR_SAVE_RESERVE')

    for signo in (signal.SIGTERM,signal.SIGINT):
        signal.signal(signo,stop_signal)
    lock = OwnerLock(spec['neural_owner_lock'])
    try:
        guard()
        pair,card,seal = comparison_inputs(spec)
        site = validate_site(spec)
        if (jax.default_backend()!='gpu' or len(jax.local_devices())!=1 or pair['project']!=spec['project']
                or str(Path(spec['parent_params']).resolve())!=str(Path(card['parent_params']).resolve())):
            raise PermissionError('CLOCK500_REQUIRES_ONE_MATCHED_A40_AND_ORIGINAL_FROZEN_PARENT_ASSETS')
        runtime = dict(GPU_uuid=site['GPU_uuid'],GPU_name=site['GPU_name'],backend=jax.default_backend(),
                       jax_version=jax.__version__,precision='highest',XLA_FLAGS=os.environ.get('XLA_FLAGS'))
        if pair['runtime']!=runtime:
            raise PermissionError('CLOCK500_BOTH_ARMS_REQUIRE_THE_FROZEN_MATCHED_HARDWARE_AND_BACKEND')
        jax.config.update('jax_compilation_cache_dir',spec['compilation_cache_dir'])
        guard()
        datasets,banks = inputs(card,resize_with_pad)

        def progress(stage, **detail):
            write(root/'PROGRESS.private.json',dict(stage=stage,epoch=time.time(),**detail))

        engine,identity = restored_engine(card,datasets,banks,progress,engine_class=partial(ClockEngine,clock_arm=spec['arm']))
        binding = dict(plan=PLAN,arm=spec['arm'],source_sha=spec['source_sha'],parent_phase=PARENT,
                       parent_state=seal['checkpoint'],D2=pair['D2_decision'],comparison=spec['comparison'],
                       runtime=runtime,original_adapt_contract=identity)
        output['binding'] = binding
        start = restore_prefix(spec,binding,engine)
        if not start<spec['stop_phase']:
            raise ValueError('CLOCK500_BRANCH_MUST_ADVANCE_ITS_OWN_DURABLE_CURSOR')

        def save():
            return state.save(root/f'phase{engine.step:05d}',phase=engine.step,binding=binding,
                              parameters=engine.parameters,optimizer=engine.states,rng=engine.rng)

        interrupted = False
        with ledger_path.open('x') as ledger:
            while engine.step<spec['stop_phase']:
                try:
                    row = engine.update(guard)
                except TimeoutError:
                    interrupted = True
                    break
                ledger.write(json.dumps(row,allow_nan=False)+'\n')
                ledger.flush()
                os.fsync(ledger.fileno())
                committed += 1
                progress('DURABLE_CLOCK_UPDATE',phase_step=engine.step,branch_updates=engine.step-PARENT)
        if committed:
            output['ledger'] = pin(ledger_path)
            validate_ledger(output['ledger'],start,engine.step,banks=engine.banks,parent_step=engine.parent_step)
            latest = save()
        if interrupted:
            output.update(status='CLOCK500_DEADLINE_PREFIX_SAVED' if latest else 'CLOCK500_STOP_BEFORE_DURABLE_UPDATE',
                          exit_code=75,checkpoint=latest)
        else:
            output.update(status='CLOCK500_SEGMENT_SAVED',exit_code=0,checkpoint=latest)
    except TimeoutError as error:
        output.update(status='CLOCK500_STOP_BEFORE_DURABLE_UPDATE',first_failure=str(error),exit_code=75)
    except (Exception,KeyboardInterrupt) as error:
        output.update(status='CLOCK500_TECHNICAL_TERMINAL_RECONCILE_BEFORE_RESUME',
                      first_failure=f'{type(error).__name__}: {error}',last_complete_checkpoint=latest)
    finally:
        end = engine.step if engine else start
        output.update(start_phase=start,end_phase=end,optimizer_updates=committed,
            unsealed_Adam_updates=max(0,end-start-committed) if start is not None else 0,
            optimization_visits={domain:n*committed for domain,n in adapt.BATCH.items()},
            complete_H40=engine.completed_aux if engine else 0,attempted_H40=engine.attempted_aux if engine else 0,
            gradient_windows=engine.gradient_windows if engine else dict.fromkeys(adapt.DOMAINS,0),
            wall_s=time.monotonic()-created)
        try:
            write(root/'RESULT.private.json',output)
        finally:
            lock.close()
    return output['exit_code']


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    args = parser.parse_args()
    raise SystemExit(run(read_pin(dict(path=args.spec,sha256=args.sha256))))
