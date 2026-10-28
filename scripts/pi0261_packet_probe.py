"""Inspect the actual frozen training processor boundary without model queries."""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0260/R2')
CODE = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/reference-317a')
COSMOS = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0250/inputs/frozen_v1/cosmos_snapshot')
GROOT = Path('/home/liuyu-nfs/Projects/isaac-gr00t')
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                  HF_HOME='/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/hf_home')
sys.path[:0] = [str(ROOT / 'R3'), str(ROOT / 'runtime'), str(ROOT / 'qa'),
                str(CODE / 'scripts'), str(CODE / 'src'), str(GROOT)]

from pi0260_r3_inputs import batch, build, identity, inventory


def main():
    plan = json.loads(Path('/home/liuyu-nfs/Projects/go2-arx5/PI0261/PAIRING_PLAN.json').read_text())
    processor = build(json.loads((ROOT / 'dataset/statistics.json').read_text()), COSMOS)
    nonlanguage = ('state', 'embodiment_id', 'pixel_values', 'image_grid_thw')
    language = ('input_ids', 'attention_mask')
    receipt = []
    for pair in plan['common_start_bank']:
        rows = []
        for item in pair['commands']:
            path = ROOT / 'dataset' / item['sequence']
            rec = json.loads(path.with_suffix('.json').read_text())
            j = next(j for j, row in enumerate(rec['rows']) if row['index'] == 0)
            with np.load(path.with_suffix('.npz'), allow_pickle=False) as z:
                state, label = z['state_v1'][j].copy(), z['raw_action'][j].copy()
            with np.load(ROOT / item['frame'], allow_pickle=False) as z:
                packet = dict(body_rgb=z['body_rgb'].copy(), wrist_rgb=z['wrist_rgb'].copy(),
                              state_v1=state, language=rec['sequence']['instruction'])
            encoded = batch(processor, packet, label)['inputs']
            assert set(encoded) == set(nonlanguage + language + ('action', 'action_mask'))
            rows.append(encoded)
        for key in nonlanguage:
            assert inventory(rows[0][key]) == inventory(rows[1][key]), (pair['task'], pair['family'], key)
        assert inventory(rows[0]['input_ids']) != inventory(rows[1]['input_ids'])
        assert identity({key: rows[0][key] for key in nonlanguage + language}) != identity(
            {key: rows[1][key] for key in nonlanguage + language})
        receipt.append(dict(task=pair['task'], family=pair['family'],
                            nonlanguage_equal=True, token_distinct=True,
                            condition_digest=[identity({key: row[key] for key in nonlanguage + language}) for row in rows]))
    result = dict(schema='PI0261_PROCESSED_PAIR_BANK_AUDIT_V1', status='PASS_ALL_60_PROCESSED_PAIRS',
                  pairs=len(receipt), conditioning_keys=list(nonlanguage + language),
                  excluded_keys=['action', 'action_mask'], rows=receipt, model_queries=0, optimizer_updates=0)
    output = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0261/PROCESSED_PAIR_BANK_AUDIT.json')
    rendered = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if output.exists():
        assert output.read_text() == rendered
    else:
        output.write_text(rendered)
    print(json.dumps(dict(status=result['status'], pairs=len(receipt), output=str(output))))


if __name__ == '__main__':
    main()
