"""CPU-only verification of the 504 old M1 rows eligible for crossed-card reuse."""
import json
from pathlib import Path

import numpy as np
import pi0260_r3_eval as r3
from pi0260_r3_inputs import batch, build, identity
from pi0261_eval import plan, source
from pi0262_old_reorg import ROOT, OLD, wording


def run():
    import stage40_b0_model_control as stage40
    stage40._bind_local_cosmos_processor(r3.GROOT, r3.COSMOS)
    processor = build(r3.read(r3.ROOT / 'dataset/statistics.json'), r3.COSMOS)
    card, pairs = plan()
    motion = r3.read(r3.R3 / 'motion2_eval_cases.json')['cases']
    rows, conditions = [], {}
    for row in card['rows']:
        if row['cohort'] == 'TRAIN_ENTRY':
            if row['task'] != 'M1':
                continue
            cell_id = f"TRAIN_ENTRY-{row['pair_index']:02d}-c{row['command']}-{row['noise']}"
        else:
            case = motion[row['case_index']] if row['cohort'] == 'Motion2' else None
            if not case or case['task'] != 'M1' or case['kind'] not in ('SEMANTIC', 'PARAPHRASE'):
                continue
            cell_id = f"Motion2-{row['case_index']:03d}-{row['noise']}"
        packet, expert, case = source(row, pairs, motion, [])
        assert case['index'] == 0
        template, color = wording(packet['language'])
        point = row['point']
        donor = point == 'Donor' and row['cohort'] != 'TRAIN_ENTRY'
        base = r3.R3 / 'eval/Base_6000/cells' if donor else OLD / 'eval' / point / 'cells'
        cell = base / cell_id
        record = r3.read(cell / 'record.json')
        assert record['status'] == 'VALID' and record['source'] == case
        binding = r3.R3 / 'model_bindings/Base_6000.json' if point == 'Donor' else OLD / 'model_bindings' / (point + '.json')
        assert record['model_binding_sha256'] == r3.sha(binding)
        assert r3.sha(cell / 'tensors.npz') == record['tensor_sha256']
        key = (case['frame_sha256'], packet['state_v1'].tobytes(), packet['language'])
        if key not in conditions:
            conditions[key] = identity(batch(processor, packet)['inputs'])
        assert conditions[key] == record['condition_sha256']
        with np.load(cell / 'tensors.npz', allow_pickle=False) as z:
            assert np.array_equal(z['expert'], expert) and np.array_equal(z['state'], packet['state_v1'])
            with np.load(r3.R3 / 'noise' / (row['noise'] + '.npz'), allow_pickle=False) as noise:
                assert np.array_equal(z['noise'], noise['float32'])
        rows.append(dict(point=point, sequence=case['sequence'], template=template, color=color,
            noise=row['noise'], cell_id=cell_id, cell=str(cell),
            condition_sha256=record['condition_sha256'], tensor_sha256=record['tensor_sha256'],
            record_sha256=r3.sha(cell / 'record.json'), model_binding_sha256=r3.sha(binding)))
    assert len(rows) == 504 and all(sum(row['point'] == point for row in rows) == 168
                                  for point in ('Donor', 'C2000', 'P2000'))
    result = dict(status='PASS_504_ACTUAL_PROCESSED_CONDITION_AND_NOISE_CACHE_ROWS',
        rows=rows, unique_processed_conditions=len(conditions),
        statistics_sha256=r3.sha(r3.ROOT / 'dataset/statistics.json'),
        old_card_sha256=r3.sha(OLD / 'FIXED_EVAL_CARD.json'), model_queries=0, optimizer_updates=0)
    r3.write(ROOT / 'CACHE_SOURCE_INDEX.json', result)
    print(json.dumps(dict(status=result['status'], rows=len(rows),
                         unique_processed_conditions=len(conditions))))


if __name__ == '__main__':
    run()
