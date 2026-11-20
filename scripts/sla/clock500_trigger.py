"""Run the conditional CLOCK500 gate on archived data only, after exact ADAPT2k Eval."""
import argparse
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from go2arx5_vla.sla_adapter import adapt5k_checkpoint, adapt5k_contract
from go2arx5_vla.sla_adapter.adapt5k_data import AdaptTargetDataset
from go2arx5_vla.sla_adapter.clock500 import PLAN
from go2arx5_vla.sla_adapter.codec_v1 import pad32, roundtrip_error
from go2arx5_vla.sla_adapter.gripper_latent_v2 import EPSILON
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from scripts.pi0292.durable import OwnerLock, write
from scripts.sla import audit_adapt_archived_timing as audit
from scripts.sla import freeze_adapt5k_panels as panels
from scripts.sla.adapt5k_curriculum import DECISION
from scripts.sla.adapt5k_vanda import SCHEMA, pbs_history
from scripts.sla.adapt5k_vanda_handoff import ORIGINAL_RUNTIME, verify_train_result
from scripts.sla.evaluate_adapt5k import restore_proof
from scripts.sla.scale20k_vanda_offline import read_pbs_with_auth_retry
from scripts.wm02.multisource_d4_vanda import pin


SCOPE = 'SLA_CLOCK500_2K_TRIGGER_DECISION_V1'


def pbs(job):
    return read_pbs_with_auth_retry(pbs_history,job)


def owned_job(row, spec, state):
    return (row['Job_Owner'].split('@')[0]==spec['user'] and row['project']==spec['project']
            and row['job_state']==state)


def scope_result(spec, key, role):
    member = spec[key]
    config = read_pin(member['pbs_spec'])
    row = pbs(member['job'])
    path = Path(config['root'])/'RESULT.private.json'
    if (config['job_id']!=member['job'] or config['role']!=role or config['step']!=2000
            or not owned_job(row,spec,'F') or row.get('Exit_status')!=0 or not path.is_file()):
        raise ValueError(f'CLOCK500_NEEDS_COLD_EXACT_{role.upper()}_2K:{row.get("job_state")}:{row.get("Exit_status")}')
    result_pin = pin(path)
    result = read_pin(result_pin)
    if (result['schema']!=SCHEMA or result['PBS_JOBID']!=member['job'] or result['exit_code']!=0
            or result['phase_limit']!=2000 or result['role']!=role or result['runtime_revision']!=ORIGINAL_RUNTIME
            or result['native_requests'] or result['native_ticks']):
        raise ValueError('CLOCK500_NEEDS_ORIGINAL_BOUNDARY_EXECUTION_RECEIPT:'+role)
    return config,result_pin,result


def exact_panel(spec):
    cfg,train_pin,train = scope_result(spec,'ADAPT2000','train')
    eval_cfg,eval_pin,evaluation = scope_result(spec,'ADAPT2000_eval','eval')
    readback,seal = verify_train_result(train['completed_roles']['train'],train['parent_checkpoint'])
    if (seal['step']!=2000 or train['sealed']!=seal or readback['exit_code']!=0
            or readback['binding']['stage_start']!=0 or readback['binding']['stage_stop']!=2000
            or eval_cfg['previous_job']!=train['PBS_JOBID'] or eval_cfg['previous']!=cfg['root']
            or evaluation['previous_job']!=train['PBS_JOBID'] or evaluation['new_updates']
            or evaluation['model_generations']!=424 or evaluation['status']!='ADAPT_FORMAL_EVALUATION_COMPLETE'):
        raise ValueError('CLOCK500_REQUIRES_EXACT_2K_AND_COMPLETE_INDEPENDENT_EVALUATION')
    first = read_pin(evaluation['completed_roles']['formal'])
    restored = read_pin(evaluation['completed_roles']['formal-restore'])
    expected = read_pin(evaluation['output_restore_proof'])
    if (first['checkpoint']!=seal['checkpoint'] or first['phase']!=2000 or first['model_generations']!=376
            or first['panel']['sha256']!=audit.PANEL_SHA256 or first['old_TEST_access'] or restored['old_TEST_access']
            or restore_proof(first,restored)!=expected or expected['independent_restore_exact']!=48):
        raise ValueError('CLOCK500_EXPECTS_FROZEN_88_PHASE_ROWS_AND_EXACT_48_INDEPENDENT_RESTORES')
    panel = read_pin(first['panel'])
    rows = audit.evaluate_phase(first,panel)
    due = [row for row in rows if row['bucket']=='P10_DUE_CLOSE']
    if len(rows)!=88 or len(due)!=24 or len({r['identity']['episode_id'] for r in due})!=24:
        raise ValueError('CLOCK500_FROZEN_24_DUE_P10_DENOMINATOR_CHANGED')
    return seal,panel,rows,due,dict(train=train_pin,eval=eval_pin,restore=evaluation['output_restore_proof'])


def label_proof(spec, panel, rows):
    archive = read_pin(spec['G3_audit'])
    expert_spec = read_pin(spec['expert_reader'])
    if (archive['panel_sha256']!=audit.PANEL_SHA256 or expert_spec['panel']['sha256']!=audit.PANEL_SHA256
            or archive['expert_spec_sha256']!=spec['expert_reader']['sha256']
            or any(archive[k] for k in ('new_updates','new_visits','new_generations','native_requests','native_ticks'))
            or [v['phase'] for v in archive['phases']]!=[0,500]):
        raise ValueError('CLOCK500_REQUIRES_ORIGINAL_ZERO_GENERATION_ADAPT0_500_G3_AUDIT')
    labels = audit.audit_labels(expert_spec,panel,[archive['phases'][0],dict(rows=rows)])
    if labels!=archive['expert_labels'] or len(labels)!=88 or not all(label['gates']['both'] for label in labels):
        raise ValueError('CLOCK500_EXPERT_LABEL_P10_CLOCK_OR_ORIGINAL_FEASIBILITY_CHANGED')

    def no_frames(*_):
        raise RuntimeError('CLOCK500_READ_ONLY_LABEL_PROOF_CANNOT_READ_IMAGES')

    data = expert_spec['target_D1']
    ds = AdaptTargetDataset(data['legacy'],data['captures'],resize_with_pad=no_frames)
    ds.bind_state(data['statistics'])
    if panels.phase_rows(ds)[0]!=panel['phase_train']:
        raise ValueError('CLOCK500_PHASE_CLASSIFIER_CHANGED_FIXED_ACTUAL_PANEL')
    metrics = []
    for entry in labels:
        index = entry['index']
        episode,anchor = ds.rows[index]
        member = ds.labels[episode['id']]
        label = member['nominal_policy_H40'][anchor]
        state = member['state_eef10'][anchor]
        if not member['supervision_mask'][anchor].all():
            raise ValueError('CLOCK500_ORIGINAL_COMPLETE_40_POINT_LABEL_REQUIRED')
        codec = ds.codecs[episode['id']]
        _,latent = codec.encode(state,label)
        if not np.allclose(latent,member['latent7'][anchor],rtol=1e-5,atol=1e-6):
            raise ValueError('CLOCK500_ACTION_CODEC_DISAGREES_WITH_ARCHIVED_TARGET_LATENTS')
        stats = ds.statistics['actions']
        decoded = codec.decode(state,pad32(stats.unnormalize(stats.normalize(latent))))
        previous = ds.reference(index)['width_m']
        if (not np.allclose(decoded[:,9],np.clip(label[:,9],EPSILON,1-EPSILON),rtol=1e-6,atol=1e-8)
                or audit.first(audit.transitions(previous,decoded)[0])!=entry['expert_close']):
            raise ValueError('CLOCK500_ACTION_CODEC_CHANGED_THE_ORIGINAL_EXPERT_CLOCK')
        metrics.append(roundtrip_error(label,decoded))
    return dict(labels_verified=len(labels),due_experts_verified=24,original_H40_and_seam_verified=88,
                codec_maximum={k:max(r[k] for r in metrics) for k in metrics[0]},G3_audit=spec['G3_audit'])


def decision(spec, result):
    seal,panel,rows,due,evidence = exact_panel(spec)
    numerator = sum(row['timely_and_gates'] for row in due)
    result.update(checkpoint=seal['checkpoint'],eligible_windows=24,joint_numerator=numerator,evidence=evidence,
                  due_summary=audit.distribution(due),due_rows=due)
    if numerator:
        result['scientific_trigger'] = False
        return 'NOT_TRIGGERED'
    try:
        result['semantic_audit'] = label_proof(spec,panel,rows)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        result['first_failure'] = f'{type(error).__name__}: {error}'
        return 'BLOCKED_DIAGNOSIS_OR_SEMANTICS'
    result['scientific_trigger'] = True
    stage = Path(spec['curriculum_decision'])
    if not stage.is_file():
        result['first_failure'] = 'MISSING_COLD_D2_CURRICULUM_DECISION'
        return 'BLOCKED_ARTIFACT'
    fix = pin(stage)
    curriculum = read_pin(fix)
    if (curriculum['schema']!=DECISION or curriculum['plan']!=adapt5k_contract.PLAN or curriculum['mode']!='D2'
            or curriculum['phase2000_checkpoint']!=seal['checkpoint'] or curriculum['ready']!=spec['D2_ready']):
        result['first_failure'] = 'COLD_D2_CURRICULUM_INELIGIBLE_OR_MISMATCHED'
        return 'BLOCKED_ARTIFACT'
    result['curriculum'] = fix
    result['first_failure'] = 'CLOCK500_BRANCH_ALLOCATION_AND_NEW_NATIVE_POPULATION_NOT_PART_OF_CPU_TRIGGER'
    return 'TRIGGERED_AWAITING_EXECUTION_READINESS'


def run(spec):
    root = Path(spec['root']).resolve(strict=True)
    source = Path(__file__).resolve().parents[2]
    job = os.environ.get('PBS_JOBID')
    if (spec['plan']!=PLAN or spec['job_id']!=job or root.stat().st_uid!=os.getuid()
            or spec['source_sha']!=(source/'SOURCE_SHA.private').read_text().strip()
            or (root/'RESULT.private.json').exists() or not job):
        raise PermissionError('CLOCK500_CPU_OBSERVER_REQUIRES_EXACT_NEW_OWNED_CAPSULE')
    result = dict(scope=SCOPE,source_sha=spec['source_sha'],job_id=job,scientific_trigger=None,
                  optimizer_updates=0,optimization_visits=0,model_generations=0,native_requests=0,native_ticks=0)
    lock = OwnerLock(root/'OWNER.lock')
    try:
        own = pbs(job)
        if not owned_job(own,spec,'R') or int(own['Resource_List'].get('ngpus',0))!=0:
            raise PermissionError('CLOCK500_TRIGGER_REQUIRES_ACTUAL_CPU_ONLY_ALLOCATION')
        if (root/'STOP.json').exists() or time.time()+90>=spec['deadline_epoch']:
            result['status'] = 'BLOCKED_DEADLINE_OR_STOP'
        else:
            result['status'] = decision(spec,result)
    except subprocess.CalledProcessError as error:
        auth = 'errno=15019' in (error.stderr or '')
        result.update(status='BLOCKED_AUTH' if auth else 'BLOCKED_ADAPT2K_OR_EVALUATION_EVIDENCE',
                      first_failure='PBS_AUTHENTICATION_UNAVAILABLE' if auth else f'PBS_READ_EXIT_{error.returncode}')
    except (ValueError, KeyError, OSError, subprocess.TimeoutExpired) as error:
        result.update(status='BLOCKED_ADAPT2K_OR_EVALUATION_EVIDENCE',first_failure=f'{type(error).__name__}: {error}')
    finally:
        try:
            write(root/'RESULT.private.json',result)
        finally:
            lock.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    args = parser.parse_args()
    run(read_pin(dict(path=args.spec,sha256=args.sha256)))
