"""Pin unchanged scientific helpers and check only the new routing boundary."""
import ast
import difflib
import os
import subprocess
import sys
from pathlib import Path
from admission import BASE, ROOT, OLD, read, sha, write


def function(source, name):
    return ast.dump(next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name), include_attributes=False)


def main():
    assert not (ROOT / 'evidence/model_attempts.jsonl').exists()
    assert read(ROOT / 'evidence/A0_PREDECESSOR_CLOSEOUT.json')['status'] == 'PASS_CONTINUE_PI0270_A1'
    card = read(ROOT / 'design/COLLECTION_CARD.json')
    assert len(card['cases']) == len(card['noise']) == 192
    assert sum(len(r['keys']) for r in card['noise']) == 19200
    assert len(card['reused_components']) == 32
    for row in card['noise']:
        assert sha(row['file']) == row['sha256']
    for row in card['reused_components']:
        assert sha(row['result_path']) == row['result_sha256']
    old = read(OLD / 'design/QUALITY_EXECUTION_FILESET.json')
    paths = []
    for row in old['files']:
        assert sha(row['path']) == row['sha256'], row['path']
        paths.append(Path(row['path']))
    old_virtual = (OLD / 'code/virtual_episode.py').read_text()
    new_virtual = (ROOT / 'code/collect_episode.py').read_text()
    assert function(old_virtual, 'advance') == function(new_virtual, 'advance')
    assert function(old_virtual, 'language') == function(new_virtual, 'language')
    old_worker = (OLD / 'code/infer_worker.py').read_text()
    new_worker = (ROOT / 'code/infer_worker.py').read_text()
    begin = "            with torch.inference_mode():stock=bridge._stock_query"
    end = "        receipt=dict(status='VALID'"
    expected = old_worker[old_worker.index(begin):old_worker.index(end)]
    actual = new_worker[new_worker.index(begin):new_worker.index(end)]
    added = "        assert hashlib.sha256(actual_noise.tobytes()).hexdigest()==row['actual_bf16_initial_float32_sha256']\n"
    assert actual.replace(added, '') == expected, 'NATIVE_S4_OR_DECODE_CHANGED'
    for text in (new_virtual, new_worker):
        compile(text, '<PI0270>', 'exec')
    assert "noise_registry='COLLECTION_CARD.json'" in new_virtual
    assert "NO_NEW_COMPONENT_COLLECTION" in new_virtual
    assert "ordinal<=30720" in new_worker and "heads<=122880" in new_worker
    assert "input_path.resolve().is_relative_to(raw_root/'collection/PREFIX4096')" in new_worker
    from collect import native_environment
    py, env = native_environment()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    result = subprocess.run([str(py), str(ROOT / 'code/collect_episode.py'), '--self-check'], env=env, capture_output=True, text=True)
    (ROOT / 'evidence/COLLECTION_NATIVE_CPU_CHECK.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0 and 'PASS_E0_E1_NONIDENTITY_PREFIX_AND_TIMESTAMPED_ORIGINAL_GATE_ZERO_QUERY_ZERO_PHYSICS' in result.stdout, (result.returncode, result.stderr[-4000:])
    cpu = dict(status='PASS_PI0270_ROUTING_NATIVE_E0_E1_GATE_AND_ACTUAL_FROZEN_PACKETS',
               original_advance_ast_identical=True, original_language_ast_identical=True,
               native_S4_and_decoder_code_identical=True, native_cpu_exit=result.returncode,
               source_counts=dict(cases=192, actual_packets=19200, component_reuses=32),
               partial_head_attempts_accounted_by_journal=True, native_python=str(py),
               cpu_log_sha256=sha(ROOT / 'evidence/COLLECTION_NATIVE_CPU_CHECK.log'),
               model_queries=0, teacher_calls=0, render_calls=0, physics_steps=0)
    write(ROOT / 'evidence/COLLECTION_CPU_ADMISSION.json', cpu)
    derivation = []
    for before, after in ((OLD / 'code/infer_worker.py', ROOT / 'code/infer_worker.py'),
                          (OLD / 'code/virtual_episode.py', ROOT / 'code/collect_episode.py')):
        derivation.append(dict(source=str(before), source_sha256=sha(before), derived=str(after), derived_sha256=sha(after),
                               diff=''.join(difflib.unified_diff(before.read_text().splitlines(True), after.read_text().splitlines(True), fromfile=str(before), tofile=str(after)))))
    write(ROOT / 'design/COLLECTION_SOURCE_DERIVATION.json', dict(scientific_algorithm_change=False, derivation=derivation))
    paths.extend((ROOT / 'code').glob('*.py'))
    paths.extend(ROOT / p for p in ('design/COLLECTION_CARD.json', 'design/COLLECTION_SOURCE_DERIVATION.json',
                  'design/PAIRING_PLAN.json', 'design/PROTOCOL.md', 'evidence/A0_PREDECESSOR_CLOSEOUT.json',
                  'evidence/COLLECTION_CPU_ADMISSION.json', 'model_bindings/PREFIX4096.json',
                  'contracts/ROT/source/dataset/statistics.json'))
    write(ROOT / 'design/COLLECTION_FILESET.json', dict(status='FROZEN_BEFORE_FIRST_PI0270_QUERY',
          files=[dict(path=str(p), sha256=sha(p)) for p in sorted(set(paths))],
          old_quality_seal_sha256=sha(OLD / 'design/QUALITY_EXECUTION_FILESET.json'),
          new_model_teacher_render_optimizer_calls=0, selected_checkpoint='NONE', deployment_selected_checkpoint='NONE'))
    print(cpu['status'], len(set(paths)), flush=True)


if __name__ == '__main__':
    main()
