"""Actual new-support processor/noise identities; no learned CPU or render query."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
import hashlib
import subprocess
import numpy as np
import torch
from quality_worker import BASE, ROOT, read, sha, write, build, batch, identity


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (16*2**30, 16*2**30)); os.nice(19)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    native_call = torch.nn.Module.__call__
    def guard(self, *a, **kw):
        if type(self).__module__.startswith('torchvision.transforms'): return native_call(self, *a, **kw)
        raise RuntimeError('NO_LEARNED_CPU_QUERY')
    def forbidden(*a, **kw): raise RuntimeError('NO_CPU_OPTIMIZER')
    torch.nn.Module.__call__ = guard; torch.optim.Optimizer.__init__ = forbidden
    assert not torch.cuda.is_available()
    fit_path = ROOT/'design/NEW_FIT192.json'; fit = read(fit_path)
    assert len(fit['rows']) == 192 and fit['source_mapping_sha256'] == sha(ROOT/'design/RECOVERY_MAPPING.json')
    stats = ROOT/'contracts/ROT/source/dataset/statistics.json'
    assert sha(stats) == fit['statistics_sha256'] == '0ea96eaf9ba6f2bfb5c16681542723e9ee0e82281dd07c4e33ed9189799580e9'
    proc = build(read(stats), BASE/'PI0250/inputs/frozen_v1/cosmos_snapshot')
    checks = []; noise = []
    for row in fit['rows']:
        assert sha(row['input']) == row['input_sha256'] and sha(row['noise_file']) == row['noise_file_sha256']
        with np.load(row['input']) as z:
            packet = dict(body_rgb=z['body_rgb'].copy(), wrist_rgb=z['wrist_rgb'].copy(), state_v1=z['state_v1'].copy(), language=row['language'])
            raw = z['raw_action'].copy()
        train = batch(proc, packet, raw)['inputs']; pure = batch(proc, packet)['inputs']
        assert 'action' not in pure and 'action_mask' not in pure
        assert identity(pure) == identity({k:v for k,v in train.items() if k not in ('action', 'action_mask')}) == row['condition_sha256']
        assert identity({k:train[k] for k in ('action', 'action_mask')}) == row['labels_masks_sha256']
        with np.load(row['noise_file']) as z: tensor = z['float32'].copy()
        assert tensor.shape == (1,40,132) and tensor.dtype == np.float32 and np.isfinite(tensor).all()
        assert hashlib.sha256(tensor.tobytes()).hexdigest() == row['noise_sha256']
        actual = torch.from_numpy(tensor).to(torch.bfloat16).float().numpy()
        noise.append(dict(file=row['noise_file'], sha256=row['noise_file_sha256'], keys=[dict(name='float32',
            noise_sha256=row['noise_sha256'], actual_bf16_initial_float32_sha256=hashlib.sha256(actual.tobytes()).hexdigest())]))
        checks.append(dict(input_id=row['input_id'], condition_sha256=row['condition_sha256'], labels_masks_sha256=row['labels_masks_sha256']))
    assert len({r['input_id'] for r in checks}) == len(checks) == 192
    from collect import native_environment
    py, env = native_environment(); env['CUDA_VISIBLE_DEVICES'] = ''
    run = subprocess.run([str(py), str(ROOT/'code/quality_virtual.py'), '--self-check'], env=env, capture_output=True, text=True)
    assert run.returncode == 0, (run.stdout, run.stderr)
    assert 'PASS_E0_E1_NONIDENTITY_PREFIX_AND_TIMESTAMPED_ORIGINAL_GATE_ZERO_QUERY_ZERO_PHYSICS' in run.stdout
    write(ROOT/'design/NEWFIT_QUERY_CARD.json', dict(schema='PI0270_FIXED_NEWFIT_NOISE_REGISTRY_V1', noise=noise,
          source_fit_card_sha256=sha(fit_path), fixed_conditions=192, points=['PREFIX4096','AGG4096'], fixed_queries=384))
    write(ROOT/'evidence/EVALUATION_CPU_CHECK.json', dict(status='PASS_PI0270_ACTUAL_NEWFIT192_AND_ORIGINAL_EXECUTOR_GATE', checks=checks,
          fit_card_sha256=sha(fit_path), source_virtual_sha256=sha(ROOT/'code/quality_virtual.py'), original_executor_cpu_stdout=run.stdout,
          source_sha256=sha(__file__), model_queries=0, optimizer_updates=0, renderer_calls=0, physics_calls=0,
          old416_condition_checks_reused=True, actual_neural_consumption_still_required=True))
    print('PASS_ACTUAL_NEWFIT192_AND_ORIGINAL_EXECUTOR_NO_QUERY', flush=True)


if __name__ == '__main__': main()
