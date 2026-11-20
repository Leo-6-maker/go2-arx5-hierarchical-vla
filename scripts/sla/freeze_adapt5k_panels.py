"""Freeze ADAPT0 and all later endpoints against one D1 diagnostic population."""
import argparse
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from go2arx5_vla.sla_adapter import adapt5k_contract as contract
from go2arx5_vla.sla_adapter.adapt5k_data import AdaptTargetDataset
from go2arx5_vla.sla_adapter.codec_v1 import rigid
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_contract import DOMAINS
from go2arx5_vla.sla_adapter.scale20k_data import readers
from go2arx5_vla.sla_adapter.scale20k_evaluation import RESTORE, data_sha, panel_rows, target_contexts
from scripts.pi0292.durable import write
from scripts.sla.export_mfree_c20_v1 import _reference


SCHEMA = 'SLA_ADAPT5K_FIXED_D1_OFFLINE_PANELS_V1'
BUCKETS = dict(P10_DUE_CLOSE=32, NOT_DUE_CLOSE=32, APPROACH=16, RELEASE_OR_RECOVERY=16)
PHASE25_ORDINALS = RESTORE[::2]


def datasets(spec, resize_with_pad):
    previous = readers(spec['original_source_dev'],resize_with_pad,source_split='SOURCE_DEV')
    data = spec['target_D1']
    previous['target'] = AdaptTargetDataset(data['legacy'],data['captures'],resize_with_pad=resize_with_pad)
    previous['target'].bind_state(data['statistics'])
    return previous


def bindings(spec):
    d = spec['target_D1']
    return dict(target_D1_proof_sha256=spec['D1_freeze_sha256'],
                target_new=[c['manifest']['sha256'] for c in d['captures']],
                target_state_sha256=d['statistics']['sha256'], source_dev=data_sha(spec['original_source_dev']))


def phase_rows(target):
    groups = {bucket: defaultdict(list) for bucket in BUCKETS}
    for index, (episode, anchor) in enumerate(target.rows):
        eid = episode['id']
        data = target.labels[eid]
        phase = str(data['phase'][anchor])
        previous = target.reference(index)['width_m']
        label = data['nominal_policy_H40'][anchor]
        bucket = contract.phase_bucket(label,data['consumed_mask'][anchor],previous,phase)
        if bucket in groups:
            groups[bucket][eid].append(index)

    rng = np.random.default_rng(2026112007)
    def take(bucket):
        episodes = groups[bucket]
        pending = deque(list(rng.permutation(episodes[eid])) for eid in sorted(episodes))
        for _ in range(BUCKETS[bucket]):
            if not pending:
                return
            sequence = pending.popleft()
            yield int(sequence.pop())
            if sequence:
                pending.append(sequence)

    out, counts = [], {}
    for bucket in BUCKETS:
        rows = []
        for index in take(bucket):
            episode, _ = target.rows[index]
            window = target.identity(index)
            rows.append(dict(index=index,identity=window,window_id=window,
                group=bucket,phase_bucket=bucket,label_kind=episode.get('label_kind','expert_query_plan_H40'),
                noise_id=int(rng.integers(2**32))))
        out.extend(rows)
        counts[bucket] = dict(requested=BUCKETS[bucket],available_windows=sum(map(len,groups[bucket].values())),
                         selected=len(rows),independent_episodes=len({r['window_id']['episode_id'] for r in rows}))
    if len({r['index'] for r in out}) != len(out):
        raise ValueError('ADAPT5K_PHASE_PANEL_MUST_NEVER_DUPLICATE_WINDOWS')
    return out,counts


def query_contexts(target, rows, pins):
    old = [row for row in rows if row['index'] < target.original_windows]
    context = target_contexts(target.members[0],old)
    captured = {}
    for row in rows:
        index = row['index']
        if index < target.original_windows:
            continue
        episode, anchor = target.rows[index]
        eid = episode['id']
        if eid not in captured:
            identity = pins[eid]
            card = read_pin(identity['spec'])
            actual = read_pin(identity['result'])
            original = actual['detail']['capture']
            manifest = read_pin(original)
            if (actual['condition']['collection_episode'] != eid or not actual['detail']['task_success']
                    or episode['native_result_sha256'] != identity['result']['sha256']
                    or manifest['queries'] != episode['windows']
                    or actual['source_sha'] != card['source_sha']):
                raise ValueError('ADAPT5K_PHASE_PANEL_REQUIRES_ACTUAL_ADMITTED_QUERY_PROVENANCE')
            captured[eid] = (rigid(card['policy_from_world']), read_pin(manifest['labels']),
                             identity['spec']['sha256'],manifest['labels']['sha256'])
        frame, queries, spec_sha, label_sha = captured[eid]
        packet = queries[anchor]
        tick = row['identity']['native_step']
        if (packet['query'] != anchor or packet['native_step'] != tick
                or packet['recent_reference']['native_step'] != tick
                or packet['previous_confirmed_command'] != target.reference(index)):
            raise ValueError('ADAPT5K_QUERY_FRAME_AND_ACTUAL_PREVIOUS_COMMAND_MUST_JOIN')
        context[str(index)] = dict(policy_from_world=frame.tolist(),reference=rigid(_reference(packet)).tolist(),
            previous_command=target.reference(index),capture_labels_sha256=label_sha,native_spec_sha256=spec_sha)
    return context


def freeze(spec, resize_with_pad):
    if spec['plan'] != contract.PLAN:
        raise ValueError('ADAPT5K_FROZEN_PLAN_REQUIRED')
    ds = datasets(spec,resize_with_pad)
    result = panel_rows(ds)
    phase,denominators = phase_rows(ds['target'])
    unique = {row['index']:row for row in (*result['target'], *phase)}
    contexts = query_contexts(ds['target'],list(unique.values()),spec['new_native_results'])
    return dict(schema=SCHEMA,plan=contract.PLAN,bindings=bindings(spec),formal=result,phase_train=phase,
        phase_train_denominators=denominators,restore_ordinals=RESTORE,phase25_ordinals=PHASE25_ORDINALS,
        target_contexts=contexts,source_splits={'robodojo':'SOURCE_DEV','libero':'SOURCE_DEV'},
        same_window_noise_all_endpoints=True,old_TEST_access=False,optimizer_updates=0,model_generations=0,
        native_requests=0)


def main():
    from openpi_client.image_tools import resize_with_pad

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    write(args.output,freeze(read_pin(dict(path=args.spec,sha256=args.sha256)),resize_with_pad))


if __name__ == '__main__':
    main()
