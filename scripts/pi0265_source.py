"""Finite source/readback inventory; no simulator, policy or optimizer calls."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0265')
BASE = ROOT.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def main():
    for name in ('code', 'design', 'evidence', 'logs', 'private_source'):
        (ROOT/name).mkdir(parents=True, exist_ok=True)
    old = json.loads((BASE/'PI0256/evidence/runtime_readback.json').read_text())
    rows = []
    extracts = []
    for row in old['files']:
        path = Path(row['path']); digest = sha(path)
        assert digest == row['sha256'], row['name']
        rows.append(dict(row, actual_sha256=digest, matches=True))
        if row['name'] in ('wbc_env', 'wbc_wrapper', 'trajectory_module', 'native_config'):
            text = path.read_text(); lines = text.splitlines(); nodes = ast.walk(ast.parse(text))
            names = []
            for node in nodes:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    segment = ast.get_source_segment(text, node)
                    if isinstance(node, ast.FunctionDef) and (any(s in segment for s in
                        ('matrix_to_rotation_6d', 'rotation_6d_to_matrix', 'end_effector_state', 'preview'))
                        or node.name in ('reset_idx', '_reset_dofs', '_reset_root_states', 'step', 'check_termination')):
                        extracts.append('\n# '+row['name']+' :: '+node.name+'\n'+segment)
                        names.append(node.name)
            if row['name']=='native_config':
                extracts.append('\n# native_config full\n'+text)
            rows[-1]['inspected_functions'] = names
    shim = BASE/'PI0252/runtime/PI0226_pytorch3d_shim/pytorch3d/transforms.py'
    if not shim.exists():
        candidates = list((BASE/'PI0252/runtime/PI0226_pytorch3d_shim').rglob('transforms.py'))
        assert len(candidates)==1; shim=candidates[0]
    rows.append(dict(name='compatibility_shim',path=str(shim),sha256=sha(shim),matches=True))
    extracts.append('\n# actual compatibility shim\n'+shim.read_text())
    c0 = BASE/'PI0252/pre_full_sim/controls/A_attempt2'
    control = json.loads((c0/'control.json').read_text())
    first = json.loads((c0/'rows.jsonl').read_text().splitlines()[0])
    write(ROOT/'evidence/C0_SOURCE_INDEX.json',dict(control=control,first_record=first,
        files=[dict(path=str(c0/n),sha256=sha(c0/n)) for n in
               ('control.json','rows.jsonl','actual_expert_reference.npz')]))
    (ROOT/'private_source/conditioning_and_reset.txt').write_text('\n'.join(extracts))
    result=dict(status='PASS_PINNED_SOURCE_FULL_FILE_READBACK',files=rows,
        checkpoint_training_import_convention='UNKNOWN_PENDING_AVAILABLE_METADATA',
        new_physics_starts=0,GR00T_queries=0,WBC_actor_calls=0)
    write(ROOT/'evidence/SOURCE_IDENTITY.json',result)
    print(json.dumps(dict(status=result['status'],files=len(rows),C0_control=control,
                         C0_first_keys=list(first),private_excerpt_bytes=(ROOT/'private_source/conditioning_and_reset.txt').stat().st_size)))


if __name__=='__main__':
    main()
