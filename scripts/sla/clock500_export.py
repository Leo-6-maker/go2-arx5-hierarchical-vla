"""Seal one CLOCK Go2-only package after its exact owned independent Eval."""
import argparse
import os
from pathlib import Path
import signal
import tempfile
import time

from go2arx5_vla.sla_adapter import clock500_checkpoint as checkpoint, clock500_package, scale20k_checkpoint, scale20k_package
from go2arx5_vla.sla_adapter.adapt5k_contract import BATCH
from go2arx5_vla.sla_adapter.adapt5k_data import STATS_SCHEMA
from go2arx5_vla.sla_adapter.clock500 import ARMS, ENDPOINT, PARENT, PLAN
from go2arx5_vla.sla_adapter.multitask_contract_v6 import C20_STATISTICS
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_evaluation import RESTORE
from scripts.pi0292.durable import OwnerLock, write
from scripts.sla.audit_adapt_archived_timing import PANEL_SHA256
from scripts.sla.clock500_activation import activated_comparison
from scripts.sla.clock500_trigger import owned_job, pbs
from scripts.sla.clock500_vanda import WRAPPER, previous_effect
from scripts.sla.evaluate_adapt5k import restore_proof
from scripts.sla.scale20k_vanda_offline import read_pbs_with_auth_retry
from scripts.sla.train_clock500 import comparison_inputs, validate_ledger
from scripts.wm02.multisource_d4_vanda import pin


SCHEMA = 'SLA_CLOCK500_GO2_CUSTODY_ACTUAL_V1'
MAX_BYTES = 3*1024**3
BRANCH_SOURCE = '5ea508b41893cce81b29e4640c032212dbc3e4c2'
POPULATION = 'f80d61a58c2ad20e2d2f59ccba645bf0c5a61cfc8a19d3f53862651e5c76d353'


def verify_eval(spec):
    card = read_pin(spec['eval_card'])
    job = read_pbs_with_auth_retry(pbs,card['job_id'])
    arm = spec['arm']
    root = Path(card['root']).resolve(strict=True)
    terminal = pin(root/'RESULT.private.json')
    result = read_pin(terminal)
    if (not owned_job(job,spec,'F') or job['Exit_status']!=0 or root.stat().st_uid!=os.getuid()
            or card['job_id']!=spec['eval_job'] or result['job_id']!=spec['eval_job']
            or card['arm']!=arm or result['arm']!=arm or card['role']!='eval' or result['role']!='eval'
            or card['source_sha']!=BRANCH_SOURCE or result['source_sha']!=BRANCH_SOURCE
            or card['stop_phase']!=ENDPOINT or result['scope']!=WRAPPER
            or result['status']!='CLOCK500_FIXED_OFFLINE_AND_INDEPENDENT_RESTORE_COMPLETE'
            or result['exit_code'] or result['unresolved_role'] is not None
            or result['model_generations']!=424 or result['optimizer_updates']
            or result['native_requests'] or result['native_ticks']
            or set(result['effects'])!={'ordinary','independent-restore'}):
        raise ValueError('CLOCK500_EXPORT_REQUIRES_OWN_EXACT_COLD_EVAL_424')
    comparison = activated_comparison(card)
    if comparison!=result['comparison']:
        raise ValueError('CLOCK500_EXPORT_REQUIRES_THE_REGISTERED_COMPARISON')
    joined = dict(card,comparison=comparison)
    pair, data, anchor = comparison_inputs(joined)
    train, effect = previous_effect(joined)
    train_card = read_pin(card['previous'])
    first_pin,first = previous_effect(dict(train_card,comparison=comparison))
    for step in (first,effect):
        validate_ledger(step['ledger'],step['start_phase'],step['end_phase'],parent_step=data['parent_step'])
        updates = step['end_phase']-step['start_phase']
        if (step['optimizer_updates']!=updates or step['optimization_visits']!={k:n*updates for k,n in BATCH.items()}
                or not 4*updates<=step['complete_H40']<=4*(updates+1)):
            raise ValueError('CLOCK500_EXPORT_BRANCH_UPDATE_OR_EXPOSURE_DENOMINATORS_CHANGED')
    state, marker = checkpoint.read_complete(effect['checkpoint'],effect['binding'])
    if (pair['native_population']['sha256']!=POPULATION or first['start_phase']!=PARENT
            or first['end_phase']!=effect['start_phase'] or effect['end_phase']!=ENDPOINT
            or marker['branch_updates']!=500 or effect['binding']!=first['binding']
            or marker['binding']['arm']!=arm or marker['binding']['comparison']!=comparison
            or marker['binding']['parent_state']!=anchor['checkpoint'] or marker['binding']['D2']!=pair['D2_decision']):
        raise ValueError('CLOCK500_EXPORT_CLOCK_PARENT_D2_OR_LEDGER_CHANGED')
    readback = read_pin(pin(root/'ordinary.SPEC.private.json'))
    formal_pin, fresh_pin = (result['effects'][key] for key in ('ordinary','independent-restore'))
    formal, fresh = map(read_pin,(formal_pin,fresh_pin))
    proof = read_pin(result['output_restore'])
    expected = dict(restore_proof(formal,fresh),schema='SLA_CLOCK500_BRANCH_OUTPUT_RESTORE_PROOF_V1',
                    arm=arm,comparison=comparison)
    if (readback['sealed_training_actual']!=train or readback['checkpoint']!=effect['checkpoint']
            or readback['comparison']!=comparison or readback['panel']!=formal['panel']
            or formal['checkpoint']!=effect['checkpoint'] or formal['phase']!=ENDPOINT
            or formal['panel']['sha256']!=PANEL_SHA256 or formal['model_generations']!=376
            or formal['generation_attempts']!=376 or fresh['generation_attempts']!=48 or proof!=expected
            or any(r['scope']!='SLA_CLOCK500_BRANCH_OFFLINE_V1' or r['arm']!=arm or r['comparison']!=comparison
                   or r['source_sha']!=BRANCH_SOURCE or r['status']!='ADAPT_OFFLINE_RESULTS_SEALED'
                   or r['optimization_visits'] or r['old_TEST_access'] for r in (formal,fresh))):
        raise ValueError('CLOCK500_EXPORT_RESTORE_OR_D1_PANEL_MISMATCH')
    outputs = {}
    for ordinal, new in zip(RESTORE,fresh['rows']['target'],strict=True):
        old = formal['rows']['target'][ordinal]
        if old['ordinal']!=ordinal or any(old[k]!=new[k] for k in ('ordinal','identity','index','noise_id',
                                                                  'normalized_sha256','decoded_sha256')):
            raise ValueError('CLOCK500_EXPORT_DECODE_DID_NOT_RESTORE_EXACTLY')
        outputs[f'archive/{ordinal:03d}.npz'] = old['output']

    base = read_pin(pair['phase_zero_config'])
    mix = read_pin(base['MIX_parent_custody'])
    transfer = read_pin(mix['transfer'])
    parent = Path(mix['checkpoint']['path']).parent
    name = 'train/MODEL_AT_START.private.json'
    model = read_pin(dict(path=str(parent/name),sha256=transfer['members'][name]['sha256']))
    _, parent_marker = scale20k_checkpoint.read_complete(base['parent_checkpoint'],'SCALE-MIX20K',model['binding'])
    target = readback['target_D1']['statistics']
    stats = read_pin(target)
    bound = marker['binding']['original_adapt_contract']
    if (mix['checkpoint']!=base['parent_checkpoint'] or parent_marker['step']!=base['parent_step']
            or parent_marker['binding']['partition_sha256']!=bound['partition_sha256']
            or bound['parent_checkpoint_sha256']!=base['parent_checkpoint']['sha256']
            or bound['parent_manifest_sha256']!=base['parent_manifest']['sha256']
            or base['parent_manifest']!=readback['parent_manifest'] or base['tokenizer']!=readback['tokenizer']
            or bound['tokenizer_sha256']!=base['tokenizer']['sha256']
            or bound['target_state_sha256']!=target['sha256'] or bound['target_D1_sha256']!=base['D1_freeze_sha256']
            or stats['schema']!=STATS_SCHEMA or stats['split']!='TRAIN'
            or stats['frozen_action_statistics_sha256']!=C20_STATISTICS):
        raise ValueError('CLOCK500_EXPORT_ORIGINAL_PARENT_OR_C20_D1_STATISTICS_CHANGED')
    rows = scale20k_package.checkpoint_parameters(marker)
    files = dict(outputs)
    files.update({f"package/parameters/{r['file']}":dict(path=str(state/r['file']),sha256=r['sha256']) for r in rows})
    files.update({'package/D1_STATE_STATISTICS.json':target,
        'package/C20_ACTION_STATISTICS.json':dict(path=str(Path(base['dataset'])/'STATISTICS.json'),sha256=C20_STATISTICS),
        'package/paligemma_tokenizer.model':base['tokenizer'],'archive/FORMAL.private.json':formal_pin,
        'archive/RESTORE_PROOF.private.json':result['output_restore'],'archive/PANEL.private.json':formal['panel']})
    model_binding = model['binding']
    manifest = dict(schema=clock500_package.SCHEMA,plan=PLAN,arm=arm,phase=ENDPOINT,parent_phase=PARENT,branch_updates=500,
        comparison_sha256=comparison['sha256'],checkpoint_sha256=effect['checkpoint']['sha256'],
        parent_state_sha256=anchor['checkpoint']['sha256'],D2_sha256=pair['D2_decision']['sha256'],
        native_population_sha256=POPULATION,MIX_sha256=base['parent_checkpoint']['sha256'],
        parent_manifest=base['parent_manifest'],parent_step=base['parent_step'],D1_freeze_sha256=base['D1_freeze_sha256'],
        state_statistics_sha256=target['sha256'],action_statistics_sha256=C20_STATISTICS,tokenizer_sha256=base['tokenizer']['sha256'],
        partition_sha256=model_binding['partition_sha256'],initial_state_sha256=model_binding['initial_state_sha256'],
        frozen_fingerprint=model['frozen_fingerprint'],evaluation_sha256=formal_pin['sha256'],
        restore_sha256=result['output_restore']['sha256'],panel_sha256=formal['panel']['sha256'],
        arrays=[dict(r,file=f"parameters/{r['file']}") for r in rows],physics=False,hardware_allowed=False,
        native_admission=False,formal_ADAPT_parent_eligible=False)
    return manifest,files,terminal,(first_pin,train)


def archive(root, manifest, files, terminal, training, guard):
    destination = root/'sealed'
    expected = {f"package/{r['file']}" for r in manifest['arrays']}
    expected.update(f'archive/{i:03d}.npz' for i in RESTORE)
    expected.update(('package/D1_STATE_STATISTICS.json','package/C20_ACTION_STATISTICS.json',
                     'package/paligemma_tokenizer.model','archive/FORMAL.private.json',
                     'archive/RESTORE_PROOF.private.json','archive/PANEL.private.json'))
    if (files.keys()!=expected or sum(Path(r['path']).stat().st_size for r in files.values())>MAX_BYTES
            or destination.exists()):
        raise ValueError('CLOCK500_EXPORT_TARGET_ONLY_CUSTODY_BOUNDS')
    with tempfile.TemporaryDirectory(prefix='.clock-export-',dir=root) as tmp:
        stage = Path(tmp)/'sealed'
        (stage/'package/parameters').mkdir(parents=True)
        (stage/'archive').mkdir()
        members = {}
        for rel,source in files.items():
            guard()
            path = stage/rel
            scale20k_package.checked_copy(source['path'],path,source['sha256'])
            members[rel] = dict(bytes=path.stat().st_size,sha256=source['sha256'])
        record = stage/'package/MANIFEST.private.json'
        write(record,manifest)
        digest = pin(record)['sha256']
        clock500_package.Provider.read_manifest(record.parent,digest)
        members['package/MANIFEST.private.json'] = dict(bytes=record.stat().st_size,sha256=digest)
        write(stage/'CUSTODY.private.json',dict(schema=SCHEMA,goal=PLAN,phase=ENDPOINT,arm=manifest['arm'],
            checkpoint=manifest['checkpoint_sha256'],comparison=manifest['comparison_sha256'],package_sha256=digest,
            PBS_evaluation=terminal,branch_training=list(training),archived_rgb_ordinals=list(RESTORE),members=members,
            optimizer_updates=0,model_generations=0,native_requests=0,native_ticks=0))
        guard()
        if destination.exists():
            raise FileExistsError(destination)
        for folder in (stage/'package/parameters',stage/'package',stage/'archive',stage):
            sync(folder)
        os.rename(stage,destination)
        sync(root)
    return pin(destination/'CUSTODY.private.json')


def sync(folder):
    fd = os.open(folder,os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def run(spec):
    root = Path(spec['root']).resolve(strict=True)
    job = os.environ.get('PBS_JOBID')
    revision = (Path(__file__).resolve().parents[2]/'SOURCE_SHA.private').read_text().strip()
    result = root/'RESULT.private.json'
    if (not job or job!=spec['job_id'] or spec['source_sha']!=revision or spec['goal']!=PLAN or spec['arm'] not in ARMS
            or root.stat().st_uid!=spec['uid'] or spec['uid']!=os.getuid() or result.exists()):
        raise PermissionError('CLOCK500_EXPORT_REQUIRES_FRESH_OWNED_FINITE_PBS')
    actual = dict(schema=SCHEMA,job_id=job,arm=spec['arm'],source_sha=revision,exit_code=1,
                   optimizer_updates=0,optimization_visits=0,model_generations=0,native_requests=0,native_ticks=0)
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    def guard():
        if stopping or any(Path(p).exists() for p in spec['stops']) or time.time()+30>=spec['deadline_epoch']:
            raise TimeoutError('CLOCK500_EXPORT_REACHED_STOP_OR_DEADLINE')

    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    lock = OwnerLock(root/'OWNER.lock')
    try:
        guard()
        scheduler = read_pbs_with_auth_retry(pbs,job)
        if not owned_job(scheduler,spec,'R') or int(scheduler['Resource_List'].get('ngpus',0)):
            raise PermissionError('CLOCK500_EXPORT_ACTUAL_ALLOCATION_MUST_BE_CPU_ONLY')
        manifest,files,terminal,training = verify_eval(spec)
        actual['custody'] = archive(root,manifest,files,terminal,training,guard)
        actual.update(status='CLOCK500_GO2_PACKAGE_SEALED',exit_code=0)
    except Exception as error:
        actual.update(status='CLOCK500_GO2_PACKAGE_BLOCKED',first_failure=f'{type(error).__name__}: {error}')
    finally:
        write(result,actual)
        lock.close()
    return actual['exit_code']


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    args = parser.parse_args()
    raise SystemExit(run(read_pin(dict(path=args.spec,sha256=args.sha256))))
