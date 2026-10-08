"""Describe the frozen natural pairs from saved outputs; never load a model."""
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(predicted, expert):
    mean = np.mean(predicted, axis=0)
    norm = float(np.linalg.norm(expert))
    assert norm > 0
    return dict(expert_difference=expert.tolist(), predicted_differences=predicted.tolist(),
                mean_predicted_norm=float(np.mean(np.linalg.norm(predicted, axis=1))),
                expert_norm=norm, response_ratio=float(np.mean(np.linalg.norm(predicted, axis=1)) / norm),
                mean_residual_norm=float(np.linalg.norm(mean-expert)),
                mean_cosine=float(np.dot(mean, expert)/(np.linalg.norm(mean)*norm)) if np.linalg.norm(mean) else None,
                noise_spread=float(np.linalg.norm(predicted[0]-predicted[1])))


def run(root):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    root = Path(root)
    pairs = json.loads((root/'design/NATURAL_PAIRS.json').read_text())['pairs']
    points = {}
    for point in ('CTRL2048', 'LR5X2048'):
        source = root/'endpoint_diagnostics'/point/'NATURAL_RESULTS.json'
        records = json.loads(source.read_text())['records']
        rows = []
        for i, pair in enumerate(pairs):
            generated = {7: [], 39: []}
            checks = []
            expert = {}
            for end in (7, 39):
                targets = []
                for member in ('a', 'b'):
                    with np.load(pair[member]['target']) as f:target = f['commands'].copy()
                    targets.append(target[end, :3] - np.asarray(pair[member]['state'][:3]))
                expert[end] = targets[1]-targets[0]
            for packet in (0, 1):
                outputs = []
                for member in (0, 1):
                    record = next(v for v in records if (v['pair'],v['packet'],v['member']) == (i,packet,member))
                    assert sha(record['path']) == record['sha256']
                    with np.load(record['path']) as f:
                        assert int(f['K']) == 4 and not str(f['decode_error'])
                        outputs.append({k:f[k].copy() for k in ('decoded','noise_uint16','actual_state','condition_fields_json','feature_hashes_json')})
                assert np.array_equal(outputs[0]['noise_uint16'], outputs[1]['noise_uint16'])
                for end in (7, 39):
                    generated[end].append((outputs[1]['decoded'][end,:3]-outputs[1]['actual_state'][:3]) - (outputs[0]['decoded'][end,:3]-outputs[0]['actual_state'][:3]))
                fields = [json.loads(str(v['condition_fields_json'])) for v in outputs]
                features = [json.loads(str(v['feature_hashes_json'])) for v in outputs]
                checks.append(dict(packet=packet,changed_condition_fields=[k for k in fields[0] if fields[0][k]!=fields[1][k]],backbone_features_differ=features[0]!=features[1],full_noise_bytes_equal=True))
            rows.append(dict(pair=i,waypoint7=compare(np.asarray(generated[7]),expert[7]),endpoint39=compare(np.asarray(generated[39]),expert[39]),conditioning_checks=checks))
        points[point]=dict(source_sha256=sha(source),pairs=rows)
    result=dict(status='PASS_CPU_SAVED_ENDPOINT_NATURAL_RESPONSE',pair_count=len(pairs),weak_response='INCONCLUSIVE' if len(pairs)<6 else 'CHECK_PREDECLARED_RULE',points=points,new_model_queries=0,new_physics=0,causal_visual_ablation=False)
    (root/'audit/ENDPOINT_NATURAL_CPU.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--self-check',action='store_true');a=p.parse_args()
    if a.self_check:
        v=compare(np.asarray([[1.,0,0],[1.,0,0]]),np.asarray([1.,0,0]));assert v['response_ratio']==1 and v['mean_residual_norm']==0 and v['mean_cosine']==1
        print('PASS_CPU_ARITHMETIC_NO_MODEL')
    else:run(a.root)
