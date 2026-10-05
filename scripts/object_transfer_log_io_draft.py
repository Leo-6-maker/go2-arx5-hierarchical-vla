"""Prepare a future-only patch; CPU file checks, no Trainer/model imports."""
import ast
import difflib
import hashlib
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/object_transfer/preparation'


def raw_cache_draft():
    import numpy as np
    sources = [ROOT / 'scripts/pi0270_execution/train.py', ROOT / 'scripts/pi0273_execution/train.py']
    originals = [p.read_text(encoding='utf-8') for p in sources]
    candidates = originals.copy()
    before = "        return np.load(folder/'proposals.npz'),np.load(folder/'policy_frames.npz'),read(folder/'rows.json')"
    after = """        with np.load(folder/'proposals.npz',allow_pickle=False) as z, np.load(folder/'policy_frames.npz',allow_pickle=False) as fr:
            states={k:z[k] for k in ('state_v1','raw_action')}
            frames={k:fr[k] for k in ('body_rgb','wrist_rgb')}
        return states,frames,read(folder/'rows.json')"""
    assert candidates[0].count(before) == 1
    candidates[0] = candidates[0].replace(before, after)
    # Preserve the old fresh-array isolation at every inherited family consumer.
    for i in (0, 1):
        for expression in ("fr['body_rgb'][j]", "fr['wrist_rgb'][j]", "z['state_v1'][i]", "z['raw_action'][i]"):
            assert candidates[i].count(expression) == 1, 'FAMILY_CONSUMER_SOURCE_DRIFT'
            candidates[i] = candidates[i].replace(expression, expression + '.copy()')
        compile(ast.parse(candidates[i]), str(sources[i]), 'exec')
    family = next(n for n in ast.walk(ast.parse(candidates[0])) if isinstance(n, ast.FunctionDef) and n.name == 'family')
    family.decorator_list = []
    encode = next(n for n in ast.walk(ast.parse(candidates[1])) if isinstance(n, ast.FunctionDef) and n.name == 'encode')
    branch = next(n for n in encode.body if isinstance(n, ast.If))
    selected = [n for n in branch.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('packet', 'action') for t in n.targets)]
    assert len(selected) == 2
    select = compile(ast.Module(body=selected, type_ignores=[]), 'future-isolated-slices', 'exec')
    with tempfile.TemporaryDirectory(dir=OUT) as folder:
        root = Path(folder)
        data = root / 'PI0267/source/S2/fixture'
        data.mkdir(parents=True)
        values = dict(state_v1=np.zeros((2,10),np.float32), raw_action=np.zeros((2,40,10),np.float32),
                      body_rgb=np.zeros((2,96,96,3),np.uint8), wrist_rgb=np.zeros((2,96,96,3),np.uint8))
        np.savez_compressed(data/'proposals.npz', **{k:values[k] for k in ('state_v1','raw_action')})
        np.savez_compressed(data/'policy_frames.npz', **{k:values[k] for k in ('body_rgb','wrist_rgb')})
        (data/'rows.json').write_text('[]')
        (data/'codec_admission.json').write_text('{"pass_admission":true}')
        read = lambda p:json.loads(p.read_text())
        sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        complete = dict(hashes={n:sha(data/n) for n in ('proposals.npz','policy_frames.npz','rows.json')})
        (data/'COMPLETE.json').write_text(json.dumps(complete))
        scope = dict(ROOT=root/'PI0270', np=np, read=read, sha=sha)
        exec(compile(ast.Module(body=[family],type_ignores=[]),'future-materialized-family','exec'),scope)
        z,fr,rows = scope['family'](None,'fixture')
        assert isinstance(z,dict) and isinstance(fr,dict) and rows == []
        assert all(x.dtype==values[k].dtype and np.array_equal(x,values[k]) for group in (z,fr) for k,x in group.items())
        selection = dict(z=z,fr=fr,i=0,j=0,text='SYNTHETIC_CPU_ARRAY_FIXTURE')
        exec(select,selection)
        for key in ('body_rgb','wrist_rgb','state_v1'):selection['packet'][key][...]=1
        selection['action'][...]=1
        assert all(not x.any() for group in (z,fr) for x in group.values()), 'CACHED_SOURCE_MUTATED'
        complete['hashes']['policy_frames.npz'] = 'wrong'
        (data/'COMPLETE.json').write_text(json.dumps(complete))
        try:scope['family'](None,'fixture')
        except AssertionError:pass
        else:raise AssertionError('SOURCE_HASH_DRIFT_ACCEPTED')
    patch = ''.join(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),
             fromfile='a/'+p.relative_to(ROOT).as_posix(),tofile='b/'+p.relative_to(ROOT).as_posix()))
             for p,a,b in zip(sources,originals,candidates))
    path = OUT/'FUTURE_RAW_ARRAY_CACHE.patch'
    path.write_text(patch,encoding='utf-8',newline='\n')
    receipt = dict(status='FUTURE_RAW_CACHE_CPU_SOURCE_BYTE_ISOLATION_AND_HASH_BOUNDARY_PASS',
        patch_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), synthetic_fixture_only=True,
        sources=[dict(path=p.relative_to(ROOT).as_posix(),normalized_source_sha256=hashlib.sha256(s.encode()).hexdigest()) for p,s in zip(sources,originals)],
        qualified_processor_and_real_canary='NOT_RUN', existing_sealed_helpers_must_remain_unchanged=True,
        future_private_versioned_source_copy_required=True, current_runtime_changed=False, launch_allowed=False,
        new_model_queries=0,new_teacher_calls=0,new_renders=0,new_physics=0,optimizer_updates=0)
    (OUT/'CPU_RAW_ARRAY_CACHE_DRAFT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))


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
    raw_cache_draft()


if __name__ == '__main__':
    main()
