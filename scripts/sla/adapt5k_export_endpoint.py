"""Archive the training-sealed ADAPT Go2 package after its independent PBS Eval."""
import argparse
import os
from pathlib import Path
import signal
import tempfile
import time

from go2arx5_vla.sla_adapter import adapt5k_contract, adapt5k_package, scale20k_checkpoint, scale20k_package
from go2arx5_vla.sla_adapter.adapt5k_data import STATS_SCHEMA
from go2arx5_vla.sla_adapter.checkpoint_v1 import read_pin, sha
from go2arx5_vla.sla_adapter.multitask_contract_v6 import C20_STATISTICS
from go2arx5_vla.sla_adapter.scale20k_evaluation import RESTORE
from scripts.pi0292.durable import OwnerLock, write
from scripts.sla.adapt5k_vanda import SCHEMA as PBS_SCHEMA, pbs_history
from scripts.sla.adapt5k_vanda_handoff import ORIGINAL_RUNTIME, cold_predecessor
from scripts.sla.evaluate_adapt5k import restore_proof
from scripts.sla.scale20k_vanda_offline import read_pbs_with_auth_retry


GOAL = '1b2a25b7c1720051a2627d30831fb18130462374'
SCHEMA = 'SLA_ADAPT_GO2_ENDPOINT_EXPORT_ACTUAL_V1'
MAX_BYTES = 3 * 1024**3


def pin(path):
    return dict(path=str(path),sha256=sha(path))


def job_status(job, user, project):
    status = read_pbs_with_auth_retry(pbs_history,job)
    if status['Job_Owner'].split('@')[0] != user or status['project'] != project:
        raise PermissionError('ADAPT_EXPORT_PBS_OWNER_OR_PROJECT_CHANGED')
    return status


def evidence(eval_card_pin, base_pin, phase):
    card, base = map(read_pin,(eval_card_pin,base_pin))
    root, previous, lineage = (Path(card[k]).resolve(strict=True) for k in ('root','previous','lineage'))
    if (card['role'] != 'eval' or card['step'] != phase or card['phase_zero_config'] != base_pin
            or root.parent != previous.parent or root.parent != lineage or root.stat().st_uid != os.getuid()):
        raise ValueError('ADAPT_EXPORT_REQUIRES_THE_EXACT_REGISTERED_EVAL_CARD')
    outcome = job_status(card['job_id'],base['user'],base['project'])
    evaluation = pin(root/'RESULT.private.json')
    result = read_pin(evaluation)
    if (outcome['job_state'] != 'F' or outcome['Exit_status'] != 0 or result['schema'] != PBS_SCHEMA
            or result['status'] != 'ADAPT_FORMAL_EVALUATION_COMPLETE' or result['exit_code']
            or result['PBS_JOBID'] != card['job_id'] or result['phase_limit'] != phase
            or result['source_sha'] != card['source_sha'] or result['runtime_revision'] != ORIGINAL_RUNTIME
            or result['previous_job'] != card['previous_job'] or result['parent_checkpoint'] != base['parent_checkpoint']
            or set(result['completed_roles']) != {'formal','formal-restore'}
            or result['new_updates'] or result['native_requests'] or result['native_ticks']
            or any(result['optimization_visits'].values()) or result['model_generations'] != 424):
        raise ValueError('ADAPT_EXPORT_REQUIRES_SUCCESSFUL_EXACT_PBS_EVAL')
    train, trained, seal, _ = cold_predecessor(card,base,previous)
    checkpoint, binding = seal['checkpoint'], trained['binding']
    if seal['step'] != phase or trained['exit_code'] != 0:
        raise ValueError('ADAPT_EXPORT_REQUIRES_EXACT_SUCCESSFUL_TRAINING_ENDPOINT')
    formal_card = read_pin(pin(root/'formal.SPEC.private.json'))
    formal_pin, restored_pin = (result['completed_roles'][k] for k in ('formal','formal-restore'))
    formal, restored, proof = map(read_pin,(formal_pin,restored_pin,result['output_restore_proof']))
    if (formal_card['checkpoint'] != checkpoint or formal_card['sealed_training_actual'] != train
            or formal_card['panel'] != formal['panel'] or formal_card['source_sha'] != ORIGINAL_RUNTIME
            or formal_card['physics'] or formal_card['hardware_allowed'] or formal_card['generation_cap'] != 376
            or formal['checkpoint'] != checkpoint or formal['phase'] != phase
            or any(x['status'] != 'ADAPT_OFFLINE_RESULTS_SEALED' or x['old_TEST_access']
                   or x['source_sha'] != ORIGINAL_RUNTIME or x['optimization_visits']
                   for x in (formal,restored))
            or formal['model_generations'] != formal['generation_attempts'] or formal['model_generations'] != 376
            or restored['generation_attempts'] != 48 or proof != restore_proof(formal,restored)):
        raise ValueError('ADAPT_EXPORT_REQUIRES_376_GENERATIONS_AND_48_EXACT_FRESH_RESTORES')
    original, independent = formal['rows']['target'], restored['rows']['target']
    if tuple(r['ordinal'] for r in independent) != RESTORE:
        raise ValueError('ADAPT_EXPORT_FIXED_TARGET_RGB_ROWS_CHANGED')
    outputs = {}
    for ordinal, fresh in zip(RESTORE,independent,strict=True):
        row = original[ordinal]
        if row['ordinal'] != ordinal or any(row[k] != fresh[k] for k in
                ('identity','index','noise_id','normalized_sha256','decoded_sha256')):
            raise ValueError('ADAPT_EXPORT_ARCHIVED_RGB_AND_RESTORE_DIFFER')
        outputs[str(ordinal)] = row['output']

    parent_root = Path(base['parent_checkpoint']['path']).resolve(strict=True)
    custody = read_pin(base['MIX_parent_custody'])
    transfer = read_pin(custody['transfer'])
    model_name = 'train/MODEL_AT_START.private.json'
    model = read_pin(dict(path=str(parent_root.parent/model_name),sha256=transfer['members'][model_name]['sha256']))
    _, parent = scale20k_checkpoint.read_complete(base['parent_checkpoint'],'SCALE-MIX20K',model['binding'])
    target = formal_card['target_D1']['statistics']
    state = read_pin(target)
    if (parent['step'] != base['parent_step'] or binding['parent_checkpoint_sha256'] != base['parent_checkpoint']['sha256']
            or base['parent_checkpoint'] != custody['checkpoint'] or base['parent_manifest'] != formal_card['parent_manifest']
            or binding['parent_manifest_sha256'] != base['parent_manifest']['sha256']
            or binding['partition_sha256'] != parent['binding']['partition_sha256']
            or binding['tokenizer_sha256'] != base['tokenizer']['sha256']
            or binding['target_state_sha256'] != target['sha256']
            or binding['target_D1_sha256'] != formal_card['D1_freeze_sha256']
            or state['schema'] != STATS_SCHEMA or state['split'] != 'TRAIN'
            or state['frozen_action_statistics_sha256'] != C20_STATISTICS):
        raise ValueError('ADAPT_EXPORT_FROZEN_PARENT_OR_TARGET_STATISTICS_CHANGED')
    return dict(outcome=evaluation,checkpoint=checkpoint,binding=binding,model=model,base=base,
                formal=formal_pin,restore=result['output_restore_proof'],panel=formal['panel'],
                target=target,outputs=outputs,train=train,restore_output=restored_pin)


def archive(root, phase, sealed, guard):
    from go2arx5_vla.sla_adapter.adapt5k_checkpoint import read_complete

    base = sealed['base']
    checkpoint, complete = read_complete(sealed['checkpoint'],sealed['binding'])
    model, parent = sealed['model'], sealed['model']['binding']
    weights = scale20k_package.checkpoint_parameters(complete)
    sources = {f"package/parameters/{r['file']}":dict(path=str(checkpoint/r['file']),sha256=r['sha256'])
               for r in weights}
    sources.update({f'archive/{int(k):03d}.npz':v for k,v in sealed['outputs'].items()})
    sources.update({
        'package/D1_STATE_STATISTICS.json': sealed['target'],
        'package/C20_ACTION_STATISTICS.json': dict(path=str(Path(base['dataset'])/'STATISTICS.json'),sha256=C20_STATISTICS),
        'package/paligemma_tokenizer.model': base['tokenizer'],
        'archive/FORMAL.private.json': sealed['formal'],
        'archive/RESTORE_PROOF.private.json': sealed['restore'],
        'archive/PANEL.private.json': sealed['panel'],
    })
    if (phase not in adapt5k_package.TRAINED_PHASES or phase != complete['phase_step'] or len(weights) != 29
            or sum(Path(v['path']).stat().st_size for v in sources.values()) > MAX_BYTES):
        raise ValueError('ADAPT_EXPORT_TARGET_ONLY_ARCHIVE_BOUND_EXCEEDED')
    manifest = dict(schema=adapt5k_package.TRAINED_SCHEMA,plan=adapt5k_contract.PLAN,phase=phase,
        parent_step=base['parent_step'],checkpoint_sha256=sealed['checkpoint']['sha256'],
        MIX_sha256=base['parent_checkpoint']['sha256'],parent_manifest=base['parent_manifest'],
        D1_freeze_sha256=base['D1_freeze_sha256'],state_statistics_sha256=sealed['target']['sha256'],
        action_statistics_sha256=C20_STATISTICS,tokenizer_sha256=base['tokenizer']['sha256'],
        partition_sha256=parent['partition_sha256'],frozen_fingerprint=model['frozen_fingerprint'],
        initial_state_sha256=parent['initial_state_sha256'],evaluation_sha256=sealed['formal']['sha256'],
        restore_sha256=sealed['restore']['sha256'],panel_sha256=sealed['panel']['sha256'],
        arrays=[dict(r,file=f"parameters/{r['file']}") for r in weights],physics=False,
        hardware_allowed=False,native_admission=False)
    destination = root/'sealed'
    with tempfile.TemporaryDirectory(prefix='EXPORT_PENDING.',dir=root) as tmp:
        stage = Path(tmp)/'sealed'
        (stage/'package/parameters').mkdir(parents=True)
        (stage/'archive').mkdir()
        members = {}
        for rel, source in sources.items():
            guard()
            outfile = stage/rel
            scale20k_package.checked_copy(source['path'],outfile,source['sha256'])
            members[rel] = dict(bytes=outfile.stat().st_size,sha256=source['sha256'])
        package_file = stage/'package/MANIFEST.private.json'
        write(package_file,manifest)
        digest = sha(package_file)
        adapt5k_package.Provider.read_manifest(package_file.parent,digest)
        members['package/MANIFEST.private.json'] = dict(bytes=package_file.stat().st_size,sha256=digest)
        custody = dict(schema=SCHEMA,phase=phase,goal=GOAL,checkpoint=sealed['checkpoint']['sha256'],
            package_sha256=digest,evaluation=sealed['formal']['sha256'],formal_restore=sealed['restore']['sha256'],
            PBS_evaluation=sealed['outcome'],archived_rgb_ordinals=list(RESTORE),members=members,
            optimizer_updates=0,model_generations=0,native_requests=0,native_ticks=0)
        write(stage/'CUSTODY.private.json',custody)
        guard()
        if destination.exists():
            raise FileExistsError(destination)
        for directory in (stage/'package/parameters',stage/'package',stage/'archive',stage):
            sync(directory)
        os.rename(stage,destination)
        sync(root)
    return pin(destination/'CUSTODY.private.json')


def sync(path):
    descriptor = os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run(spec):
    root = Path(spec['root']).resolve(strict=True)
    phase, job, deadline = spec['phase'],spec['job_id'],spec['deadline_epoch']
    revision = (Path(__file__).resolve().parents[2]/'SOURCE_SHA.private').read_text().strip()
    result_file = root/'RESULT.private.json'
    if (spec['goal'] != GOAL or spec['source_sha'] != revision or spec['uid'] != os.getuid()
            or spec['uid'] != root.stat().st_uid or job != os.environ.get('PBS_JOBID')
            or phase not in adapt5k_package.TRAINED_PHASES or result_file.exists() or (root/'sealed').exists()):
        raise PermissionError('ADAPT_EXPORT_REQUIRES_OWNED_FRESH_FINITE_PBS_JOB')
    result = dict(schema=SCHEMA,phase=phase,job=job,source_sha=revision,
                  optimizer_updates=0,optimization_visits=0,model_generations=0,native_requests=0,native_ticks=0,exit_code=1)
    started = time.monotonic()
    lock = OwnerLock(root/'OWNER.lock')
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)

    def guard():
        if stopped or any(Path(p).exists() for p in spec['stops']) or time.time()+30 >= deadline:
            raise TimeoutError('ADAPT_EXPORT_STOP_OR_DEADLINE')

    try:
        guard()
        live = job_status(job,spec['user'],spec['project'])
        if live['job_state'] != 'R' or int(live['Resource_List'].get('ngpus',0)):
            raise PermissionError('ADAPT_EXPORT_REQUIRES_A_REAL_CPU_ONLY_PBS_ALLOCATION')
        sealed = evidence(spec['eval_card'],spec['base'],phase)
        result['custody'] = archive(root,phase,sealed,guard)
        result.update(status='GO2_ADAPT_TARGET_CUSTODY_SEALED',exit_code=0)
    except (Exception, KeyboardInterrupt) as error:
        result.update(status='GO2_ADAPT_TARGET_CUSTODY_BLOCKED',failure=f'{type(error).__name__}: {error}')
    finally:
        write(result_file,dict(result,wall_s=time.monotonic()-started))
        lock.close()
    return result['exit_code']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    args = parser.parse_args()
    raise SystemExit(run(read_pin(dict(path=args.spec,sha256=args.sha256))))
