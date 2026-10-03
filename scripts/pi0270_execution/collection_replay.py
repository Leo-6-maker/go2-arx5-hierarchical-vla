"""Replay completed fixed collection families only, using saved S4 arrays."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
import ast
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from admission import BASE, ROOT, read, sha, write

RAW = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0270')


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (16*2**30, 16*2**30)); os.nice(19)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    if hasattr(os, 'sched_getaffinity'): os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    def forbidden(*a, **kw): raise RuntimeError('COLLECTION_REPLAY_NO_MODEL_OR_OPTIMIZER')
    torch.nn.Module.__call__ = forbidden; torch.optim.Optimizer.__init__ = forbidden
    assert not torch.cuda.is_available()
    sys.path[:0] = [str(RAW.parent/'PI0260/R2/qa'), '/home/liuyu-nfs/Projects/isaac-gr00t']
    import pi0260_r2_codec as codec
    helper = BASE/'PI0268/R1/code/report.py'
    assert sha(helper) == '1db82278af12f085144990a644f43c183b24bb072c149662ff04c5758c6b3e73'
    node = next(n for n in ast.parse(helper.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'replay')
    scope = dict(np=np, sha=sha)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(helper), 'exec'), scope)
    replay = scope['replay']
    stats = ROOT/'contracts/ROT/source/dataset/statistics.json'
    assert sha(stats) == '0ea96eaf9ba6f2bfb5c16681542723e9ee0e82281dd07c4e33ed9189799580e9'
    processor = codec.processor(read(stats)); binding = read(ROOT/'model_bindings/PREFIX4096.json')
    assert binding['checkpoint_tree_sha256'] == '12aff760d05dd05e0e11a024be3353fb45a5cb98a8b110ae7c7a0da7199609a5'
    card_path = ROOT/'design/COLLECTION_CARD.json'; card = read(card_path)
    noise = {r['file']: r for r in card['noise']}
    families = [f for f in dict.fromkeys(r['family'] for r in card['cases'])
                if (ROOT/'evidence'/('SUPPORT_CPU_'+f+'.json')).exists()]
    assert families
    receipts = []; new_predictions = 0
    for case in [r for r in card['cases'] if r['family'] in families]:
        path = RAW/'collection/PREFIX4096'/case['case']/'result.json'; result = read(path)
        assert result['validity'] == 'VALID' and result['point'] == 'PREFIX4096'
        assert result['collection_card_sha256'] == sha(card_path)
        destination = ROOT/'evidence/replay/collection'/(case['case']+'.json')
        if destination.exists():
            prior = read(destination)
            assert prior['status'] == 'PASS_SAVED_COLLECTION_NATIVE_S4_NO_MODEL_QUERY'
            assert prior['source_result_sha256'] == sha(path) and prior['statistics_sha256'] == sha(stats)
            receipts.append(dict(path=str(destination), sha256=sha(destination), predictions=len(prior['predictions'])))
            continue
        predictions = []; checked_noise = set()
        for query in result['queries']:
            request = query['query']; prediction = Path(request['output']); observation = Path(request['input'])
            assert sha(prediction) == query['prediction_sha256'] and request['noise_registry'] == 'COLLECTION_CARD.json'
            observation_sha = sha(observation)
            if query.get('observation_sha256'): assert observation_sha == query['observation_sha256']
            registered = noise[request['noise_file']]
            if request['noise_file'] not in checked_noise:
                assert sha(request['noise_file']) == registered['sha256']; checked_noise.add(request['noise_file'])
            expected = next(r for r in registered['keys'] if r['name'] == request['noise_name'])
            with np.load(prediction) as z, np.load(observation) as obs:
                receipt = json.loads(str(z['receipt_json']))
                assert receipt['status'] == 'VALID' and receipt['binding'] == binding and receipt['point'] == 'PREFIX4096'
                assert receipt['heads'] == 4 and receipt['native_S4'] and receipt['no_teacher_or_privileged_condition']
                assert hashlib.sha256(z['noise'].tobytes()).hexdigest() == expected['noise_sha256'] == receipt['noise_sha256']
                assert hashlib.sha256(z['actual_initial_noise'].tobytes()).hexdigest() == expected['actual_bf16_initial_float32_sha256']
                assert np.array_equal(z['state'], obs['state_v1']) and abs(float(obs['timestamp'])-.4*query['replan']) < 1e-10
                observed_timesteps = z['trace_t'].tolist()
            check = replay(prediction, processor, torch, codec)
            check.update(observation_sha256=observation_sha, state_matches_saved_observation=True,
                         registered_FP32_and_actual_BF16_noise_identity=True, observed_native_timesteps=observed_timesteps,
                         actual_attempt=receipt['attempt'], finite_output=receipt['finite_output'])
            predictions.append(check); new_predictions += 1
        write(destination, dict(status='PASS_SAVED_COLLECTION_NATIVE_S4_NO_MODEL_QUERY', case=case['case'], predictions=predictions,
              source_result_sha256=sha(path), statistics_sha256=sha(stats), replay_helper_sha256=sha(helper),
              codec_module_path=codec.__file__, codec_module_sha256=sha(codec.__file__), source_sha256=sha(__file__),
              model_queries=0, gradient_queries=0, optimizer_updates=0, teacher_calls=0, renderer_calls=0,
              input_RGB_processing_not_repeated=True, original_V3_and_full_trajectory_replay_still_required=True))
        receipts.append(dict(path=str(destination), sha256=sha(destination), predictions=len(predictions)))
        print('SAVED_COLLECTION_S4_REPLAYED', case['case'], len(predictions), flush=True)
    assert len(receipts) == 8*len(families)
    destination = ROOT/'evidence'/('COLLECTION_S4_REPLAY_FAMILIES_'+str(len(families))+'.json')
    # The batch identity is the closed case set, independent of whether this invocation reused checks.
    write(destination, dict(status='PASS_COMPLETED_FIXED_COLLECTION_FAMILY_S4_REPLAY', families=families, cases=len(receipts),
          predictions=sum(r['predictions'] for r in receipts), receipts=receipts, total_fixed_cases=192,
          all_collection_complete=len(receipts)==192, model_queries=0, gradient_queries=0, optimizer_updates=0,
          teacher_calls=0, renderer_calls=0, source_sha256=sha(__file__)))
    print('COMPLETED_COLLECTION_S4_CPU_BATCH', len(receipts), sum(r['predictions'] for r in receipts), 'NEW_REPLAY', new_predictions, flush=True)


if __name__ == '__main__': main()
