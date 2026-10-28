"""Register only the already-frozen support and saved-visited queries; no model load."""
import argparse
from pathlib import Path
from support import ROOT, OLD, RAW, read, sha, write


def requests(support, visited):
    rows = []
    for point in ('PREFIX4096', 'AGG4096'):
        for source in support['rows']:
            for noise in source['noise']:
                j = noise['j']; uid = source['input_id']
                assert j < (8 if source['source_mapping']['bucket'] == 'ORIGINAL_Q3' else 2)
                request = dict(input=source['input'], language=source['language'], noise_file=noise['file'],
                    noise_name='float32', noise_registry='LATE_SUPPORT_CARD.json',
                    expected_condition_sha256=source['condition_sha256'],
                    output=str(RAW/'support'/point/uid/f'j{j}'/'prediction.npz'))
                rows.append(dict(point=point, panel='A1' if j < 2 else 'A2', input_id=uid, j=j,
                    input_file_sha256=source['input_sha256'], request=request))
    for source in visited['rows']:
        if source['status'] == 'MISSING_AFTER_STOP': continue
        assert source['status'] == 'AVAILABLE_EXACT_SAVED_AGG_INPUT_NOISE'
        prior = source['request']; uid = source['input_id']
        request = dict(input=prior['input'], language=prior['language'], noise_file=prior['noise_file'],
            noise_name=prior['noise_name'], noise_registry='VISITED_QUERY_CARD.json',
            expected_condition_sha256=source['condition_sha256'],
            output=str(RAW/'visited/PREFIX4096'/uid/'prediction.npz'))
        rows.append(dict(point='PREFIX4096', panel='A3', input_id=uid, j='EXACT_AGG_REUSE',
            input_file_sha256=source['observation_sha256'], request=request))
    assert sum(r['panel'] == 'A1' for r in rows) == 1184
    assert sum(r['panel'] == 'A2' for r in rows) == 1152
    assert sum(r['panel'] == 'A3' for r in rows) <= 64
    assert len(rows) <= 2400 and len({r['request']['output'] for r in rows}) == len(rows)
    return rows


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (16*2**30, 16*2**30))
    support_path = ROOT/'design/LATE_SUPPORT_CARD.json'
    visited_path = ROOT/'design/VISITED_STATE_CARD_V2.json'
    support, visited = read(support_path), read(visited_path)
    assert len(support['rows']) == 296 and len(visited['rows']) == 64
    original = OLD/'design/EVALUATION_CARD.json'; original_card = read(original)
    filenames = {r['request']['noise_file'] for r in visited['rows'] if 'request' in r}
    noise = [r for r in original_card['noise'] if r['file'] in filenames]
    assert {r['file'] for r in noise} == filenames
    for row in noise: assert sha(row['file']) == row['sha256']
    for row in visited['rows']:
        if 'request' not in row: continue
        entry = next(r for r in noise if r['file'] == row['request']['noise_file'])
        key = next(k for k in entry['keys'] if k['name'] == row['request']['noise_name'])
        assert key['noise_sha256'] == row['source_noise_sha256']
    rows = requests(support, visited)
    for row in rows: assert sha(row['request']['input']) == row['input_file_sha256']
    sources = [dict(path=str(p), sha256=sha(p)) for p in (support_path, visited_path, original, Path(__file__))]
    query_card = ROOT/'design/VISITED_QUERY_CARD.json'
    write(query_card, dict(schema='PI0271_EXACT_SAVED_AGG_NOISE_REGISTRY_V1', noise=noise,
        proposals=visited['rows'], source_files=sources, DIAGNOSIS_ONLY=True))
    path = ROOT/'design/QUERY_PLAN.json'
    write(path, dict(schema='PI0271_FINITE_QUERY_PLAN_V1', rows=rows, fixed_scientific_queries=len(rows),
        planned_heads=4*len(rows), hard_attempt_cap=2560, hard_head_cap=10240,
        A0_required=True, launch_allowed=False, source_files=sources,
        visited_query_card_sha256=sha(query_card), teacher_proposals=64, teacher_hard_cap=80,
        optimizer_updates=0, episodes=0, renders=0, selected_checkpoint='NONE', deployment_selected_checkpoint='NONE'))
    write(ROOT/'evidence/QUERY_PLAN_CPU_CHECK.json', dict(status='PASS_FIXED_REQUESTS_AND_SOURCE_HASHES_NO_QUERY',
        query_plan_sha256=sha(path), planned_queries=len(rows), planned_heads=4*len(rows),
        A1=1184, A2=1152, A3=sum(r['panel']=='A3' for r in rows),
        model_queries=0, teacher_calls=0, launch_allowed=False, A0_required=True, source_files=sources))
    print('PASS_FIXED_QUERY_REGISTRATION_NO_MODEL_OR_TEACHER', len(rows), flush=True)


if __name__ == '__main__': main()
