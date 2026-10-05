"""Prepare a future-only patch; CPU file checks, no Trainer/model imports."""
import ast
import difflib
import hashlib
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/object_transfer/preparation'


def main():
    path = ROOT / 'scripts/pi0273_execution/train.py'
    source = path.read_text(encoding='utf-8')
    candidate = source
    replacements = [
        (
            '    from gr00t.experiment import experiment\n    class Trainer',
            "    from gr00t.experiment import experiment\n"
            "    closed_packets=None\n"
            "    if args.arm=='TARGET':\n"
            "        other=ROOT/'BROAD'/args.phase/'training/step_rng.jsonl'\n"
            "        closed_stat=other.stat()\n"
            "        closed_packets=read_lines(other)\n"
            "    class Trainer",
        ),
        (
            '            result=super().training_step(model,inputs,*pos,**kw)\n',
            "            rng_path=output/'step_rng.jsonl'\n"
            "            position=rng_path.stat().st_size if rng_path.exists() else 0\n"
            '            result=super().training_step(model,inputs,*pos,**kw)\n',
        ),
        (
            "            row=read_lines(output/'step_rng.jsonl')[-1];assert np.isfinite(row['loss'])\n",
            "            with rng_path.open('rb') as stream:\n"
            "                stream.seek(position);line=stream.readline()\n"
            "                assert line.endswith(b'\\n') and not stream.read(1)\n"
            "                row=json.loads(line)\n"
            "            assert np.isfinite(row['loss'])\n",
        ),
        (
            "                other=ROOT/'BROAD'/args.phase/'training/step_rng.jsonl'\n"
            "                expected=read_lines(other)[2*(row['unit']-1)+row['micro']]\n",
            "                current=other.stat()\n"
            "                for name in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'):\n"
            "                    assert getattr(current,name)==getattr(closed_stat,name),'CLOSED_RANDOM_PACKETS_CHANGED'\n"
            "                expected=closed_packets[2*(row['unit']-1)+row['micro']]\n",
        ),
    ]
    for before, after in replacements:
        assert candidate.count(before) == 1, 'SOURCE_DRIFT'
        candidate = candidate.replace(before, after)
    tree = ast.parse(candidate)
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    # PI0274's versioned source adapter still requires exactly one `other` binding.
    assert sum(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'other' for t in n.targets)
               for n in ast.walk(run)) == 1
    compile(tree, str(path), 'exec')
    trainer = next(n for n in run.body if isinstance(n, ast.ClassDef))
    step = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == 'training_step')
    tail = next(n for n in step.body if isinstance(n, ast.With))
    code = compile(ast.Module(body=[tail], type_ignores=[]), 'draft-tail-only', 'exec')
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=OUT) as folder:
        p = Path(folder) / 'step_rng.jsonl'
        old = b'{"unit":1,"micro":0}\n'
        new = b'{"unit":1,"micro":1}\n'
        for prefix in (b'', old):
            p.write_bytes(prefix + new)
            scope = dict(rng_path=p, position=len(prefix), json=json)
            exec(code, scope)
            assert scope['row'] == json.loads(new)
        for invalid in (b'', new.rstrip(b'\n'), new + new):
            p.write_bytes(old + invalid)
            try:
                exec(code, dict(rng_path=p, position=len(old), json=json))
            except (AssertionError, ValueError):
                pass
            else:
                raise AssertionError('INVALID_APPEND_ACCEPTED')
    patch = ''.join(difflib.unified_diff(source.splitlines(True), candidate.splitlines(True),
                     fromfile='a/scripts/pi0273_execution/train.py', tofile='b/scripts/pi0273_execution/train.py'))
    patch_path = OUT / 'FUTURE_LOG_IO.patch'
    patch_path.write_text(patch, encoding='utf-8', newline='\n')
    receipt = dict(status='FUTURE_PATCH_CPU_SOURCE_AND_APPEND_BOUNDARY_PASS', source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   patch_sha256=hashlib.sha256(patch_path.read_bytes()).hexdigest(), positive_cases=2, negative_cases=3,
                   current_runtime_changed=False, launch_allowed=False, new_model_queries=0, gradients=0, optimizer_updates=0,
                   qualified_training_throughput='UNOBSERVED', future_canary_required=True,
                   read_scope='Only the one appended actual micro packet; immutable closed packets once; all comparison fields unchanged')
    (OUT / 'CPU_LOG_IO_DRAFT_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
