"""Recompute paired attribution from the sealed 32-row V3.1 receipt; no model."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def quantiles(values):
    return dict(zip(('p50', 'p95', 'p99', 'max'), map(float, np.percentile(values, [50,95,99,100]))))


def analyze(receipt):
    rows = receipt['records']['value']
    if len(rows) != 32:
        raise ValueError('SEALED32_SCOPE_REQUIRED')
    ordinary = [r for r in rows if r['kind'] == 'normal']
    hot = ordinary[1:]
    if len(ordinary) != 16 or [r['index'] for r in ordinary] != list(range(16)):
        raise ValueError('FIXED16_ORDINARY_REQUIRED')
    summaries = {}
    for name, selected in (('all32', rows), ('ordinary16', ordinary), ('hot_ordinary15', hot)):
        result = dict(n=len(selected), archive_s=quantiles([r['post_ACK_archive_s'] for r in selected]))
        if name != 'all32':
            model = [sum(r['model_branch_s']) for r in selected]
            ack = [r['phases']['capture_to_ACK_s'] for r in selected]
            serial = [r['phases']['capture_to_ACK_s']+r['post_ACK_archive_s'] for r in selected]
            result.update(ACK_s=quantiles(ack), model_sum_s=quantiles(model),
                nonmodel_ACK_s=quantiles([a-b for a,b in zip(ack, model)]),
                serial_ACK_plus_archive_s=quantiles(serial),
                serial_over_500ms=sum(v > .5 for v in serial),
                ACK_over_350ms=sum(v > .35 for v in ack), ACK_at_least_400ms=sum(v >= .4 for v in ack),
                EEF_branch_s=quantiles([r['model_branch_s'][0] for r in selected]),
                grip_branch_s=quantiles([r['model_branch_s'][1] for r in selected]),
                phases={k:quantiles([r['phases'][k] for r in selected]) for k in
                        ('input_read_pack_s','IPC_to_response_s','compiler_s','wire_s','install_total_s')})
        summaries[name] = result
    return dict(schema='R0_V31_SEALED_PAIRED_LATENCY_RECOMPUTE_V1', source_records_sha256=receipt['records']['sha256'],
        percentile_method='numpy linear', summaries=summaries,
        compiler_wire_install_overlap=True, IPC_includes_model=True,
        serial_sum_is_lower_bound_cycle_excluding_consumption_and_other_controller_work=True,
        live_camera_measured=False, real_time_qualified=False, new_generations=0, native_requests=0,
        rows=[dict(index=r['index'],kind=r['kind'],family=r['family'],noise=r['noise'],
                   phases=r['phases'],model_branch_s=r.get('model_branch_s'),archive_s=r['post_ACK_archive_s']) for r in rows])


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('receipt');p.add_argument('output');a=p.parse_args()
    raw=Path(a.receipt).read_bytes();result=analyze(json.loads(raw));result['local_receipt_sha256']=hashlib.sha256(raw).hexdigest()
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
