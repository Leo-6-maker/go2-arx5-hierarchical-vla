"""Actual staged AGG inputs on Hopper CPU; no learned model or optimizer."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', NO_ALBUMENTATIONS_UPDATE='1')
import json
from collections import Counter
import numpy as np
import torch
from train import ROOT, paired, read, sha, Pairs, seed_for
from pi0260_r3_inputs import build, identity


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (16*2**30, 16*2**30)); os.nice(19)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    if hasattr(os, 'sched_getaffinity'): os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-4:])
    original_call = torch.nn.Module.__call__
    def guard(self, *a, **kw):
        if type(self).__module__.startswith('torchvision.transforms'): return original_call(self, *a, **kw)
        raise RuntimeError('PI0270_CPU_NO_LEARNED_QUERY')
    def forbidden(*a, **kw): raise RuntimeError('PI0270_CPU_NO_OPTIMIZER')
    torch.nn.Module.__call__ = guard; torch.optim.Optimizer.__init__ = forbidden
    assert not torch.cuda.is_available()
    assert read(ROOT/'evidence/DATA_COVERAGE_ADMISSION.json')['status'] == 'PASS_ALL96_FIXED_NEW_SUPPORT_CELLS'
    assert read(ROOT/'evidence/NEWFIT_ADMISSION.json')['status'] == 'PASS_FIXED_NEWFIT192_AVAILABLE'
    staged = read(ROOT/'evidence/TRAINING_DATA_LOCAL_READBACK.json')
    assert staged['status'] == 'PASS_NEW_FIXED_DATA_PACKAGE_AND_ALL_MEMBERS_STAGED'
    stats = ROOT/'contracts/ROT/source/dataset/statistics.json'
    assert sha(stats) == '0ea96eaf9ba6f2bfb5c16681542723e9ee0e82281dd07c4e33ed9189799580e9'
    processor = build(read(stats), paired.COSMOS)
    output = ROOT/'evidence/input-audit/AGG'; output.mkdir(parents=True, exist_ok=True)
    audit = output/'sample_audit.jsonl'; target = ROOT/'evidence/HOPPER_CPU_ADMISSION.json'
    if target.exists():
        prior = read(target); assert prior['sample_audit_sha256'] == sha(audit)
        assert prior['mapping_sha256'] == sha(ROOT/'RECOVERY_MAPPING.json')
        print(prior['status'], flush=True); return
    assert not audit.exists(), 'PRESERVE_PARTIAL_CPU_AUDIT_BEFORE_SAME_PACKET_RECOVERY'
    dataset = Pairs(processor, output, 'AGG', 0); slots = Counter(); ranges = {}; fit = read(ROOT/'design/NEW_FIT192.json')['rows']
    wanted = {(r['source_mapping']['unit'], r['source_mapping']['micro']): r for r in fit}
    assert len(fit) == len(wanted) == 192
    consumed = []; count = 0
    for count, value in enumerate(dataset, 1):
        inputs = value['inputs']; meta = value['_pi0264_meta']; unit, micro = (count-1)//2+1, (count-1)%2
        assert (meta['unit'], meta['micro']) == (unit, micro)
        u = dataset.plan['units'][unit-1]; assert u['seed'] == seed_for(unit)
        condition = {k: v for k, v in inputs.items() if k not in ('action', 'action_mask')}
        assert identity(condition) == meta['condition_sha256']
        for name, shape in (('state', (1,1,132)), ('action', (1,40,132))):
            x = inputs[name].detach().cpu().numpy()
            assert x.dtype == np.float32 and x.shape == shape and np.isfinite(x).all() and not x[...,10:].any()
            # Finite out-of-range affine values remain unmodified, as in the fixed collector.
            ranges[name] = max(ranges.get(name, 0.), float(np.abs(x).max()))
        mask = inputs['action_mask']; assert torch.all(mask[...,:10] == 1) and not torch.any(mask[...,10:])
        slots[u['slot']] += 1
        if (unit, micro) in wanted:
            row = wanted[unit, micro]
            labels = identity({k: inputs[k] for k in ('action', 'action_mask')})
            assert meta['condition_sha256'] == row['condition_sha256'] and labels == row['labels_masks_sha256']
            consumed.append(dict(input_id=row['input_id'], unit=unit, micro=micro,
                                 condition_sha256=meta['condition_sha256'], labels_masks_sha256=labels))
        if count % 512 == 0: print('ACTUAL_AGG_CPU_INPUTS', count, flush=True)
    assert count == 8192 and slots == dict(COMMON_NOMINAL=4096, RECOVERY_INTERVENTION=4096) and len(consumed) == 192
    result = dict(status='PASS_PI0270_ACTUAL_FROZEN_MAPPING_AND_CONTROL_INPUTS', microbatches=count, slots=dict(slots),
                  common_actual_control_bit_identity=True, actual_recovery_condition_label_binding=True,
                  NEW_FIT192_CPU_plan_bindings=consumed, NEW_FIT192_neural_training_consumption_not_yet_proved=True,
                  normalized_abs_max=ranges, mapping_sha256=sha(ROOT/'RECOVERY_MAPPING.json'),
                  statistics_sha256=sha(stats), sample_audit_sha256=sha(audit), source_sha256=sha(__file__),
                  model_queries=0, optimizer_updates=0, real_canary_still_required=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n'); print(result['status'], flush=True)


if __name__ == '__main__': main()
