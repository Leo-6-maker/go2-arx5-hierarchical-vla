"""Stdlib source check only; does not claim runtime/input/canary admission."""
import ast
import hashlib
import json
from pathlib import Path


def check(new_path, control_path):
    source = control_path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == 'c268bebfc32c90073b1a711e7666625b2bb981b0846cdc20ecaffd34c3274e26'
    old = ast.parse(source); new = ast.parse(new_path.read_bytes())
    compile(new, str(new_path), 'exec')
    def functions(tree):
        return {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}
    before, after = functions(old), functions(new)
    for name in ('seed_for', 'language', 'read', 'sha', 'read_lines'):
        assert before[name] == after[name], name
    def calls(tree, name):
        return [ast.dump(n, include_attributes=False) for n in ast.walk(tree)
                if isinstance(n, ast.Call) and ast.unparse(n.func) == name]
    for name in ('old.build_config', 'old.run_official', '_bind_local_cosmos_processor',
                 'torch.use_deterministic_algorithms', 'native_clip'):
        assert calls(old, name) == calls(new, name), name
    def config(tree):
        return [ast.dump(n, include_attributes=False) for n in ast.walk(tree)
                if isinstance(n, ast.Assign) and any(ast.unparse(t).startswith('config.') for t in n.targets)]
    assert config(old) == config(new), 'OPTIMIZATION_CONFIG_CHANGED'
    install = next(n for n in ast.walk(new) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'paired.install')
    assert {k.arg: ast.literal_eval(k.value) for k in install.keywords} == dict(raw_head_cap=80000, raw_update_cap=4368)
    temporal = next(n for n in ast.walk(new) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'install_temporal')
    assert ast.literal_eval(temporal.args[0]) == 'PREFIX'
    return dict(status='PASS_SOURCE_ONLY_NO_RUNTIME_ADMISSION', inherited_source_sha256=hashlib.sha256(source).hexdigest(),
                prepared_source_sha256=hashlib.sha256(new_path.read_bytes()).hexdigest(),
                unchanged_language_unit_seed_and_optimization_config=True, inherited_temporal_recipe='PREFIX',
                raw_training_heads_cap=80000, raw_completed_updates_cap=4368,
                model_queries=0, optimizer_updates=0, launch_allowed=False,
                actual_data_processor_and_canary_still_required=True)


if __name__ == '__main__':
    folder = Path(__file__).resolve().parent
    print(json.dumps(check(folder/'train.py', folder.parent/'pi0269_execution/train.py'), indent=2, sort_keys=True))
