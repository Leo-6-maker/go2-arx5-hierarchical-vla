"""Bind one cold ADAPT2K curriculum decision to independently read TRAIN data."""
import json
from pathlib import Path
import time

from go2arx5_vla.sla_adapter import adapt5k_contract as contract
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_contract import canonical_hash
from scripts.pi0292.durable import write
from scripts.wm02.multisource_d4_vanda import pin


READY = 'SLA_ADAPT5K_STAGE2_READY_V1'
DECISION = 'SLA_ADAPT5K_COLD_2K_CURRICULUM_ACTUAL_V1'
STAGE2_SEED = 2026112026


def inputs(candidate_pin, d1, expected_mode):
    ready = read_pin(candidate_pin)
    training, proof = map(read_pin, (ready['train_input'], ready['train_proof']))
    r = proof['vanda']
    start, stop = contract.SWITCH_PHASE, contract.UPDATES
    immutable = ('plan','parent_step','parent_checkpoint_sha256')
    baseline, target = d1['data']['target'], training['data']['target']
    old, captures = baseline['captures'], target['captures']
    tail = captures[len(old):]
    if (ready['schema'] != READY or ready['plan'] != contract.PLAN or ready['mode'] != expected_mode
            or any(ready[k] for k in ('native_requests','optimizer_updates','model_generations'))
            or ready['ready_epoch'] > time.time()
            or (ready['stage_start'],ready['stage_stop']) != (start,stop)
            or (training['stage_start'],training['stage_stop']) != (start,stop)
            or (r['stage_start'],r['stage_stop']) != (start,stop)
            or training['target_seed'] != STAGE2_SEED or r['target_bank_report']['seed'] != STAGE2_SEED
            or any(training[k] != d1[k] for k in immutable)
            or (r['parent_step'],r['parent_checkpoint_sha256']) != (d1['parent_step'],d1['parent_checkpoint_sha256'])
            or target['legacy'] != baseline['legacy'] or target['statistics'] != baseline['statistics']
            or target['statistics']['sha256'] != r['target_state_sha256'] or captures[:len(old)] != old
            or any(training['data'][d] != d1['data'][d] for d in ('robodojo','libero'))
            or proof['schema'] != 'SLA_ADAPT5K_MATCHED_TRAIN_READERS_V1' or proof['core'] != r
            or r['schema'] != 'SLA_ADAPT5K_ACTUAL_TRAIN_READERS_V1'
            or any(r[k] for k in ('optimizer_updates','optimization_visits','model_generations','native_requests','TEST_access'))
            or set(r['inputs']) != set(contract.DOMAINS) or any(not r['inputs'][d] for d in contract.DOMAINS)
            or r['input_sha256'] != canonical_hash(r['inputs'])
            or r['banks_windows'] != {d:(stop-start)*n for d,n in contract.BATCH.items()}
            or bool(tail) != (expected_mode == 'D2')):
        raise ValueError('ADAPT5K_STAGE2_REQUIRES_MATCHED_TRAIN_BOTH_SITES_WITH_FROZEN_D1_REPRESENTATION')
    known = set()
    recent = []
    for i, item in enumerate(captures):
        admission = read_pin(item['admission'])
        manifest = read_pin(item['manifest'])
        ids = [row['id'] for row in manifest['episodes']]
        verified = {row['id'] for row in admission['cases'] if row['admitted']}
        if (manifest['schema'] != 'SLA_TARGET_ADAPTATION_NEW_EXPERTS_V1' or not ids
                or any(row['split'] != 'TRAIN' or row['individual_pass'] is not True for row in manifest['episodes'])
                or admission['dataset_sha256'] != item['manifest']['sha256']
                or admission['admitted_episodes'] != len(ids) or set(ids) != verified
                or len(set(ids)) != len(ids) or known.intersection(ids)):
            raise ValueError('ADAPT5K_STAGE2_CAPTURES_MUST_BE_INDEPENDENT_AND_ACTUALLY_ADMITTED')
        known.update(ids)
        if i >= len(old):
            recent.extend(ids)
    if r['new_target_episodes'] != len(known):
        raise ValueError('ADAPT5K_STAGE2_READER_DID_NOT_INCLUDE_THE_SEALED_EPISODES')
    if expected_mode == 'D2':
        audit = r['d2_audit']
        required = {'first','before_closure','P10_due_closure','last'}
        rows = audit['episodes']
        if (audit['manifest_sha256'] != tail[-1]['manifest']['sha256']
                or audit['tokenizer_sha256'] != training['d2_audit']['tokenizer']['sha256']
                or [entry['episode'] for entry in rows] != recent
                or audit['verified_semantic_windows'] != sum(entry['windows'] for entry in rows)
                or audit['confirmed_P10_atoms'] != 10*audit['verified_semantic_windows']
                or any({tag for sample in entry['samples'] for tag in sample['anchors']} != required
                       or any(sample['identity']['episode_id'] != entry['episode'] for sample in entry['samples'])
                       or not entry['due_close_anchors'] for entry in rows)):
            raise ValueError('ADAPT5K_D2_REQUIRES_MATCHED_READS_AT_BOTH_CLOSURE_TRANSITIONS')
    return ready, training, proof, recent


def select(spec, lineage, d1, predecessor, seal):
    """Call under the one training lineage lock. Evaluation reads the decision."""
    path = lineage/'STAGE2_DECISION.private.json'
    if path.is_file():
        decision = read_pin(pin(path))
        require_boundary = seal['step'] == contract.SWITCH_PHASE
        if (decision['schema'] != DECISION or decision['plan'] != contract.PLAN
                or decision['mode'] not in ('D1','D2') or seal['step'] < contract.SWITCH_PHASE
                or (require_boundary and (decision['phase2000_checkpoint'] != seal['checkpoint']
                                           or spec['role'] != 'train' or predecessor['exit_code']))):
            raise ValueError('ADAPT5K_UNCHANGED_COLD_CURRICULUM_DECISION_REQUIRED')
        ready, training, proof, recent = inputs(decision['ready'],d1,decision['mode'])
        if (decision['stage2_bank_sha256'] != proof['vanda']['bank_digests']
                or decision['new_independent_episodes'] != recent
                or decision['training'] != ready['train_input'] or decision['train_proof'] != ready['train_proof']):
            raise ValueError('ADAPT5K_STAGE2_DECISION_BINDING_CHANGED')
        return pin(path), training, proof
    if (spec['role'] != 'train' or seal['step'] != contract.SWITCH_PHASE or predecessor['exit_code']
            or predecessor['binding']['stage_start'] != 0):
        raise ValueError('ADAPT5K_CURRICULUM_MUST_BE_CHOSEN_FROM_ACTUAL_COLD_PHASE2000')
    pointer = lineage/'STAGE2_D2_READY.private.json'
    rejected = None
    available = None
    if pointer.is_file():
        try:
            available = json.loads(pointer.read_bytes())
            ready, training, proof, recent = inputs(available,d1,'D2')
        except (ValueError, OSError, KeyError, TypeError) as error:
            rejected = str(type(error).__name__) + ': ' + str(error)
            available = None
    if available is None:
        available = spec['stage2_d1_fallback']
        ready, training, proof, recent = inputs(available,d1,'D1')
    decision = dict(schema=DECISION, plan=contract.PLAN, mode=ready['mode'], chosen_epoch=time.time(),
        phase2000_checkpoint=seal['checkpoint'], ready=available, training=ready['train_input'], train_proof=ready['train_proof'],
        stage2_bank_sha256=proof['vanda']['bank_digests'], new_independent_episodes=recent,
        rejected_D2_receipt_reason=rejected, optimizer_updates=0, native_requests=0, used_eval_score=False)
    write(path,decision)
    return pin(path),training,proof
