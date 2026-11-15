"""Run the V4 logical sampler seam on explicitly simulated catalogs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from go2arx5_vla.sla_adapter.mixed_visits_v4 import Catalog, MixedVisits, visit_record


def run():
    began = time.perf_counter()
    rows = [dict(split='TRAIN', task=f'simulated-task-{t}', episode=f'simulated-episode-{e}',
                 phase=phase, anchor=anchor)
            for t in range(2) for e in range(4)
            for phase, anchor in (('approach', 0), ('close', 10), ('release', 20))]
    catalogs = {domain: Catalog(domain, digit*64, rows)
                for domain, digit in (('T', 'a'), ('S_cmd', 'b'))}
    expected = {'T100_3K': (32000, 0), 'M75_3K': (24000, 8000),
                'M50_3K': (16000, 16000), 'SEQ_3K': (8000, 24000), 'S_ONLY_3K': (0, 32000)}
    actual = {}
    for arm, counts in expected.items():
        selected = {domain: value for domain, value in catalogs.items()
                    if not (arm == 'T100_3K' and domain == 'S_cmd')
                    and not (arm == 'S_ONLY_3K' and domain == 'T')}
        sampler = MixedVisits(arm, selected, seed=20261113)
        seen = Counter(); digest = hashlib.sha256()
        for _ in range(1000):
            batch = sampler.next_batch()
            assert len(batch) == 32
            seen.update(row.domain for row in batch)
            for row in batch:
                digest.update(json.dumps(visit_record(row), sort_keys=True).encode())
        assert (seen['T'], seen['S_cmd']) == counts
        assert sampler.state_dict()['domain_visits'] == dict(T=counts[0], S_cmd=counts[1])
        fresh = MixedVisits(arm, selected, seed=20261113)
        fresh.restore(json.loads(json.dumps(sampler.state_dict())))
        assert fresh.next_batch() == sampler.next_batch()
        actual[arm] = dict(simulated_batches=1000, T_references=counts[0], S_cmd_references=counts[1],
                          reference_stream_sha256=digest.hexdigest(), fresh_restore_next_batch_exact=True)
    return dict(schema='SLA_V4_SIMULATED_MIXED_VISIT_AUDIT_ACTUAL', catalogs='SIMULATED_IDS_ONLY',
                arms=actual, wall_s=time.perf_counter()-began, actual_dataset_visits=0,
                model_calls=0, optimizer_updates=0, physical_requests=0,
                CUDA_qualification=False, production_reader_or_dataset_admission=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = run()
    with a.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))
