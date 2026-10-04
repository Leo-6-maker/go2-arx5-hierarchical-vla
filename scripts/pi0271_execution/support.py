"""Freeze the registered consumed-state panel and all eight Q3 noises, CPU only."""
import argparse
import hashlib
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                  OPENBLAS_NUM_THREADS='4', OPENBLAS_CORETYPE='Prescott')
BASE = Path('/home/liuyu-nfs/Projects/go2-arx5')
OLD = BASE/'PI0270'
ROOT = BASE/'PI0271'
RAW = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0271')
sys.path.insert(0, str(OLD/'code'))
from admission import read, sha, write
from plan import noise_seed


def select(mapping, families):
    ordered = sorted(mapping, key=lambda r: (r['unit'], r['micro']))
    assert len(ordered) == 4096 and len({(r['unit'], r['micro']) for r in ordered}) == 4096
    late = []
    for family in families:
        for task in ('M1', 'M3'):
            for command in (0, 1):
                for bucket in ('ORIGINAL_Q2', 'ORIGINAL_Q3'):
                    candidates = [r for r in ordered if (r['family'], r['task'], r['command'], r['bucket']) ==
                                  (family, task, command, bucket)]
                    assert candidates, ('MISSING_REGISTERED_BUCKET', family, task, command, bucket)
                    late.append(candidates[0])
    components = [r for r in ordered if r['bucket'] in ('M1_TERMINAL', 'M3_ENTRY', 'M3_MIDARC', 'M3_TERMINAL')]
    assert len(late) == 192 and len(components) == 104
    assert Counter(r['bucket'] for r in components) == dict(M1_TERMINAL=32, M3_ENTRY=24, M3_MIDARC=24, M3_TERMINAL=24)
    rows = late + components
    assert len({(r['unit'], r['micro']) for r in rows}) == len(rows) == 296
    return rows


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (16*2**30, 16*2**30)); os.nice(19)
    import numpy as np
    import torch
    from quality_worker import build, batch, identity
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    original = torch.nn.Module.__call__
    def guard(self, *a, **kw):
        if type(self).__module__.startswith('torchvision.transforms'): return original(self, *a, **kw)
        raise RuntimeError('NO_LEARNED_CPU_QUERY')
    def forbidden(*a, **kw): raise RuntimeError('NO_CPU_OPTIMIZER')
    torch.nn.Module.__call__ = guard; torch.optim.Optimizer.__init__ = forbidden
    assert not torch.cuda.is_available()
    mapping = OLD/'design/RECOVERY_MAPPING.json'
    assert sha(mapping) == '53fe049e406f079e079d03af71e25c0ffd339b587fab7d8e5dc80bebe35fd633'
    audit = read(OLD/'evidence/MAIN_TRAINING_AUDIT.json')
    assert audit['status'] == 'PASS_AGG4096_REUSED_CONTROL_TRAINING_INTEGRITY'
    samples = OLD/'evidence/training_hopper/AGG/main/training/sample_audit.jsonl'
    assert sha(samples) == audit['sample_audit_sha256']
    import json
    actual = [json.loads(line) for line in samples.open()]; assert len(actual) == 8192
    control = BASE/'PI0269/evidence/training_hopper/PREFIX/main/training/sample_audit.jsonl'
    # Absence of a comparator consumption log is explicit, never "all unseen".
    comparator = set()
    if control.exists():
        assert sha(control) == '1bc54a47f82bff833609f12c74e377ee8a22aaa0a9292021ce57cb9f44efa0c7'
        comparator = {(r['condition_sha256'], r['labels_masks_sha256']) for r in
                      (json.loads(line) for line in control.open())}
    families = [f['family_id'] for f in read(BASE/'PI0267/design/SCENE_CARD.json')['families'] if f['split'] == 'FEEDBACK_TRAIN']
    assert len(families) == 24
    chosen = select(read(mapping)['rows'], families)
    stats = OLD/'contracts/ROT/source/dataset/statistics.json'
    assert sha(stats) == '0ea96eaf9ba6f2bfb5c16681542723e9ee0e82281dd07c4e33ed9189799580e9'
    proc = build(read(stats), BASE/'PI0250/inputs/frozen_v1/cosmos_snapshot')
    rows = []; noises = []
    for source in chosen:
        consumed = actual[2*(source['unit']-1)+source['micro']]
        assert consumed['actual_source_uid'] == source['source_uid'] and consumed['source_sample_sha256'] == source['sample_sha256']
        assert all(consumed[k] == source[k] for k in ('unit', 'micro', 'language', 'condition_sha256', 'labels_masks_sha256'))
        assert sha(source['sample']) == source['sample_sha256'] and sha(source['source_codec_path']) == source['source_codec_sha256']
        codec = read(source['source_codec_path']); assert codec['eligible'] and codec['roundtrip']['PASS']
        with np.load(source['sample'], allow_pickle=False) as z:
            packet = dict(body_rgb=z['body_rgb'].copy(), wrist_rgb=z['wrist_rgb'].copy(),
                          state_v1=z['state_v1'].copy(), language=source['language'])
            raw = z['raw_action'].copy()
        pure = batch(proc, packet)['inputs']; supervised = batch(proc, packet, raw)['inputs']
        assert set(pure).isdisjoint(('action', 'action_mask'))
        assert identity(pure) == identity({k:v for k,v in supervised.items() if k not in ('action', 'action_mask')}) == source['condition_sha256']
        assert identity({k:supervised[k] for k in ('action', 'action_mask')}) == source['labels_masks_sha256']
        input_id = f"u{source['unit']:04d}-m{source['micro']}"
        bank = []
        for j in range(8 if source['bucket'] == 'ORIGINAL_Q3' else 2):
            namespace = f'PI0271:SUPPORT:27171:{input_id}:{j}'; seed = noise_seed(namespace)
            tensor = torch.randn((1,40,132), generator=torch.Generator(device='cpu').manual_seed(seed), dtype=torch.float32)
            fp32 = tensor.numpy(); bf16 = tensor.to(torch.bfloat16).float().numpy()
            path = RAW/'noise-support'/f'{input_id}-j{j}.npz'; path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                with np.load(path, allow_pickle=False) as z: assert np.array_equal(z['float32'], fp32)
            else:
                temp = path.with_suffix('.partial.npz'); np.savez_compressed(temp, float32=fp32); temp.replace(path)
            entry = dict(file=str(path), sha256=sha(path), input_id=input_id, j=j, namespace=namespace, seed=seed,
                         keys=[dict(name='float32', noise_sha256=hashlib.sha256(fp32.tobytes()).hexdigest(),
                                    actual_bf16_initial_float32_sha256=hashlib.sha256(bf16.tobytes()).hexdigest())])
            bank.append(entry); noises.append(entry)
        rows.append(dict(input_id=input_id, source_mapping=source, input=source['sample'], input_sha256=source['sample_sha256'],
                         language=source['language'], condition_sha256=source['condition_sha256'], labels_masks_sha256=source['labels_masks_sha256'],
                         actual_AGG_consumption=dict(unit=source['unit'], micro=source['micro'], actual_seen=True),
                         PREFIX_consumption='SEEN' if (source['condition_sha256'], source['labels_masks_sha256']) in comparator else
                                            ('NOT_SEEN' if control.exists() else 'UNOBSERVED_LOG_MISSING'),
                         source_timestamp_s=codec['source']['actual_timestamp_s'], source_lane=codec['source']['source_lane'], noise=bank))
    assert len(noises) == 1168 and sum(r['source_mapping']['bucket'] == 'ORIGINAL_Q3' for r in rows) == 96
    provenance = [dict(path=str(p), sha256=sha(p)) for p in (mapping, stats, samples, OLD/'evidence/MAIN_TRAINING_AUDIT.json',
                  OLD/'code/quality_worker.py', OLD/'code/plan.py', Path(__file__))]
    if control.exists(): provenance.append(dict(path=str(control), sha256=sha(control)))
    card = dict(schema='PI0271_FIXED_LATE_SUPPORT_V1', rows=rows, noise=noises, policies=['PREFIX4096','AGG4096'],
                source_mapping_sha256=sha(mapping), statistics_sha256=sha(stats), source_files=provenance,
                A1_queries=1184, A2_additional_queries=1152, new_queries_by_preparation=0,
                model_queries_allowed_only_after_A0=False, selected_checkpoint='NONE', deployment_selected_checkpoint='NONE')
    write(ROOT/'design/LATE_SUPPORT_CARD.json', card)
    write(ROOT/'evidence/SUPPORT_CPU_PREPARATION.json', dict(status='PASS_296_ACTUALLY_CONSUMED_SUPPORT_AND_1168_NOISES_CPU_ONLY',
          card_sha256=sha(ROOT/'design/LATE_SUPPORT_CARD.json'), rows=296, Q3_rows=96, noise_files=1168, source_files=provenance,
          PREFIX_consumption_counts=dict(Counter(r['PREFIX_consumption'] for r in rows)), A0_still_required=True,
          model_queries=0, teacher_calls=0, optimizer_updates=0, renders=0, new_episodes=0))
    print('PASS_296_CONSUMED_SUPPORT_1168_FROZEN_NOISES_NO_MODEL_QUERY', flush=True)


def self_check():
    families = [f'f{i}' for i in range(24)]; rows = []; unit = 1
    for f in families:
        for t in ('M1','M3'):
            for c in (0,1):
                for b in ('ORIGINAL_Q2','ORIGINAL_Q3'):
                    rows.append(dict(unit=unit,micro=0,family=f,task=t,command=c,bucket=b)); unit += 1
    for b,n in [('M1_TERMINAL',32),('M3_ENTRY',24),('M3_MIDARC',24),('M3_TERMINAL',24)]:
        for _ in range(n): rows.append(dict(unit=unit,micro=0,family='f0',task='M1',command=0,bucket=b)); unit += 1
    while len(rows)<4096: rows.append(dict(unit=unit,micro=0,family='f0',task='M1',command=0,bucket='ORIGINAL_Q0')); unit += 1
    selected=select(list(reversed(rows)), families); assert selected[:2]==rows[:2] and len(selected)==296
    broken=[dict(r) for r in rows];broken[0]['bucket']='ORIGINAL_Q0'
    try: select(broken,families)
    except AssertionError: pass
    else: raise AssertionError('Missing Q2 must not select a replacement')
    assert noise_seed('PI0271:SUPPORT:27171:u0001-m0:0') == int.from_bytes(hashlib.sha256(b'PI0271:SUPPORT:27171:u0001-m0:0').digest()[:8],'big')%(2**63-1)
    print('PASS_FIXED_SELECTION_ORDER_MISSING_BUCKET_AND_NAMESPACE_NO_QUERY')


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-check',action='store_true')
    if parser.parse_args().self_check:self_check()
    else:main()
