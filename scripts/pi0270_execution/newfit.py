"""Only the fixed192 actually consumed conditions, paired PREFIX/AGG final queries."""
import argparse
import json
import os
import subprocess
from pathlib import Path
import numpy as np
from quality_worker import BASE, ROOT, read, sha, write, admitted
from quality_offline import message

RAW = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0270')


def evaluate(point, worker):
    import pi0243_staged_learning_to_sim as stage43
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix
    from go2arx5_vla.providers import GR00TProviderConfig, GR00TTrajectoryProvider
    path = ROOT/'design/NEW_FIT192.json'; card = read(path); assert len(card['rows']) == 192
    consumption = {r['input_id']: r for r in read(ROOT/'evidence/MAIN_TRAINING_AUDIT.json')['NEW_FIT192_actual_consumption']}
    for row in card['rows']:
        actual = consumption[row['input_id']]
        assert actual['actual_seen'] and actual['condition_sha256'] == row['condition_sha256']
        assert actual['labels_masks_sha256'] == row['labels_masks_sha256']
        folder = RAW/'newfit'/point/row['input_id']; folder.mkdir(parents=True, exist_ok=True)
        record = folder/'record.json'
        if record.exists():
            prior = read(record)
            assert prior['status'] == 'VALID' and prior['row'] == row and prior['point'] == point and prior['card_sha256'] == sha(path)
            assert sha(prior['tensor']) == prior['tensor_sha256']
            continue
        assert sha(row['input']) == row['input_sha256'] and sha(row['noise_file']) == row['noise_file_sha256']
        with np.load(row['input']) as z: expert = z['raw_action'].copy(); sv = z['state_v1'].copy()
        tensor = folder/'prediction.npz'
        request = dict(input=row['input'], output=str(tensor), language=row['language'],
                       noise_file=row['noise_file'], noise_name='float32', noise_registry='NEWFIT_QUERY_CARD.json',
                       expected_condition_sha256=row['condition_sha256'])
        worker.stdin.write(json.dumps(request)+'\n'); worker.stdin.flush(); answer = message(worker)
        assert answer['status'] in ('VALID', 'VALID_REUSED') and answer['output'] == str(tensor)
        with np.load(tensor) as z: decoded = z['decoded'].copy()
        if np.isfinite(decoded).all():
            metrics = stage43._screen_metrics(decoded[0].astype(float), expert.astype(float), None)
            provider = GR00TTrajectoryProvider(decoded, config=GR00TProviderConfig(
                source_frame='PI0267_STATIC_POLICY_ANCHOR_WORLD_AXES', tcp_semantic_id='PI0267_VIRTUAL_TOOL_TCP_V1',
                action_sample_period_s=.05, timestamp_provenance='PI0267_FIXED_OFFLINE'),
                observation_timestamp=0., chunk_id='PI0270-NEWFIT-'+point+'-'+row['input_id'])
            gate = TrajectorySafetyGateV1().evaluate(provider.next_chunk(0.), now=0., achieved_position=sv[:3],
                achieved_orientation=matrix_to_quat(rot6d_rows_to_matrix(sv[None, 3:9]))[0]).as_dict()
        else: metrics = None; gate = dict(decision='REJECT', reason='NONFINITE_POLICY_OUTPUT')
        write(record, dict(status='VALID', point=point, domain='newfit', row=row, metrics=metrics, gate=gate,
              tensor=str(tensor), tensor_sha256=sha(tensor), card_sha256=sha(path), actual_AGG_seen=True,
              reference_semantics='ARCLENGTH_EXP_CORRECTION_V1', TRAIN_support_diagnosis=True, blind_TEST=False))
    print('FIXED_NEWFIT192_COMPLETE', point, flush=True)


def main(point):
    admitted(); env = dict(os.environ); env.pop('LD_PRELOAD', None); env.pop('LD_LIBRARY_PATH', None)
    env['PYTHONPATH'] = str(ROOT/'code')
    with (ROOT/'evidence'/('newfit-worker-'+point+'.log')).open('ab') as log:
        worker = subprocess.Popen([str(BASE/'PI0250/runtime/venv/bin/python'), str(ROOT/'code/quality_worker.py'), '--point', point],
                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, env=env)
        try:
            assert message(worker) == dict(ready=True, point=point)
            evaluate(point, worker); worker.stdin.close(); assert worker.wait() == 0
        finally:
            if worker.poll() is None: worker.terminate(); worker.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--point', choices=('PREFIX4096', 'AGG4096'), required=True)
    main(parser.parse_args().point)
