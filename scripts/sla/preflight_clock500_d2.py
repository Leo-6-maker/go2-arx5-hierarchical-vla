"""Read the real D2 bank and audit CLOCK supervision without model execution."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np

from go2arx5_vla.sla_adapter import adapt5k_contract as adapt
from go2arx5_vla.sla_adapter.adapt5k_data import AdaptTargetDataset
from go2arx5_vla.sla_adapter.clock500 import ENDPOINT, PARENT, PLAN, width_weights
from go2arx5_vla.sla_adapter.dense_g7_v6 import previous_command_context
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_contract import bank_sha256
from scripts.pi0292.durable import write
from scripts.sla.adapt5k_curriculum import READY


def check(ready_pin):
    ready = read_pin(ready_pin)
    if ready['schema']!=READY or ready['plan']!=adapt.PLAN or ready['mode']!='D2':
        raise ValueError('CLOCK500_PREFLIGHT_REQUIRES_REAL_D2_READY')
    proof = read_pin(ready['train_proof'])
    site = proof['vanda']
    prepared = read_pin(ready['train_input'])
    if (proof['core']!=site or site['TEST_access'] or site['optimizer_updates'] or site['model_generations']
            or site['native_requests'] or site['target_bank_report']['new_quota_per_update']!=12
            or (prepared['stage_start'],prepared['stage_stop'])!=(2000,5000)):
        raise ValueError('CLOCK500_PREFLIGHT_REQUIRES_INDEPENDENT_D2_TRAIN_READS')

    def forbid_frames(*args,**kwargs):
        raise RuntimeError('CLOCK500_LABEL_AUDIT_DOES_NOT_READ_IMAGES')

    target = prepared['data']['target']
    reader = AdaptTargetDataset(target['legacy'],target['captures'],resize_with_pad=forbid_frames)
    reader.bind_state(target['statistics'])
    bank,report = adapt.target_bank(reader,stage_start=2000,phase_stop=5000,seed=prepared['target_seed'])
    if bank_sha256(bank)!=site['bank_digests']['target'] or report!=site['target_bank_report']:
        raise ValueError('CLOCK500_PREFLIGHT_D2_VISIT_BANK_BYTES_CHANGED')
    walks = bank.reshape(-1,64)[:ENDPOINT-PARENT]
    new = np.count_nonzero(walks>=reader.original_windows,axis=1)
    if np.any(new!=12) or reader.new_episodes!=3:
        raise ValueError('CLOCK500_PREFLIGHT_EXPECTS_12_NEW_AND_52_ORIGINAL_PER_UPDATE')
    bucket_counts,weight_counts,episodes = Counter(),Counter(),set()
    digest = hashlib.sha256()
    for row,indices in enumerate(walks):
        for slot in adapt.auxiliary_slots(PARENT+row):
            index = int(indices[slot])
            episode,anchor = reader.rows[index]
            window = reader.labels[episode['id']]
            previous = previous_command_context(reader.reference(index))
            label = window['nominal_policy_H40'][anchor]
            consumed = window['consumed_mask'][anchor]
            phase = str(window['phase'][anchor])
            if not window['supervision_mask'][anchor].all():
                raise ValueError('CLOCK500_PREFLIGHT_PARTIAL_LABEL_IN_MATCHED_D2_BANK')
            weight = width_weights(label,consumed,previous,phase)
            bucket_counts[adapt.phase_bucket(label,consumed,previous,phase)] += 1
            weight_counts[int(np.count_nonzero(weight==4.))] += 1
            episodes.add(episode['id'])
            digest.update(np.asarray((PARENT+row,slot,index),dtype='<i8').tobytes())
            digest.update(weight.astype('<f4',copy=False).tobytes())
    return dict(schema='SLA_CLOCK500_ACTUAL_D2_WIDTH_SUPPORT_PREFLIGHT_V1',plan=PLAN,D2_ready=ready_pin,
        stage_bank_sha256=site['bank_digests'],first_500_target_visits_audited=walks.size,
        target_new_visits_audited=int(new.sum()),target_original_visits_audited=int(walks.size-new.sum()),
        auxiliary_label_reads=sum(bucket_counts.values()),distinct_auxiliary_episodes=len(episodes),
        label_buckets=dict(bucket_counts),weighted_points_per_auxiliary=dict(weight_counts),
        ordered_auxiliary_weight_sha256=digest.hexdigest(),all_40_points_supervised=True,
        optimizer_updates=0,optimization_visits=0,model_generations=0,native_requests=0,native_ticks=0,old_TEST_access=False)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ready')
    parser.add_argument('sha256')
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    write(args.output,check(dict(path=args.ready,sha256=args.sha256)))
