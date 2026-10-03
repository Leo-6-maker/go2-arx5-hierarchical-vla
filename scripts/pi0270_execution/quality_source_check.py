"""Stdlib check of the actual inherited S4/decoder/gate/executor; no model load."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

EXPECTED = {
    'infer_worker.py': 'a2b2679497f23920cd8034bbddd7c60b4ddf8600cec1a8807d32012c34046b11',
    'virtual_episode.py': '639e9ebadc9b92c52fc1581ab38098e8d61cf1495aed4e4df9093fddba15bd34',
    'component_episode.py': '69f08493992274db375f0753f3a303bfe9892d677a916a7a22491a5dbd08959e',
    'offline.py': '213aa88252546657233e072b9b5023dd52dfa8a7da517b658d3a54d3c6820058',
    'materialize.py': '6751f218670f7409783f9fbec2b8cba48b5a4580febbcd4ef988d22540b465a9',
}


def check(folder, inherited):
    def tree(path): return ast.parse(path.read_bytes())
    def functions(value): return {n.name: n for n in value.body if isinstance(n, ast.FunctionDef)}
    def same(a, b): assert ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)
    for name, expected in EXPECTED.items():
        assert hashlib.sha256((inherited/name).read_bytes()).hexdigest() == expected, name
    old = functions(tree(inherited/'virtual_episode.py')); new = functions(tree(folder/'quality_virtual.py'))
    for name in ('advance', 'message', 'language', 'selected_families'): same(old[name], new[name])
    class Routes(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == 240: node.value = 304
            elif node.value == 17000: node.value = 28672
            return node
        def visit_BinOp(self, node):
            if ast.unparse(node) == "RAW / 'noise'": node.left = ast.Name(id='CONTROL_RAW', ctx=ast.Load())
            return self.generic_visit(node)
    same(Routes().visit(old['run_episode']), new['run_episode'])
    old_worker = functions(tree(inherited/'infer_worker.py'))['main']
    new_worker = functions(tree(folder/'quality_worker.py'))['main']
    def tail(function):
        start = next(i for i, n in enumerate(function.body[-1].body)
                     if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'handle' for t in n.targets))
        return ast.Module(body=function.body[-1].body[start:], type_ignores=[])
    # Exact complete registered-hook -> query -> decode -> S4 arrays -> first-valid save path.
    same(tail(old_worker), tail(new_worker))
    old_offline = (inherited/'offline.py').read_text().replace('PI0269-', 'PI0270-')
    same(functions(ast.parse(old_offline))['evaluate'], functions(tree(folder/'quality_offline.py'))['evaluate'])
    for name in ('materialize.py', 'quality_gate.py', 'quality_worker.py', 'quality_virtual.py',
                 'quality_components.py', 'quality_offline.py', 'newfit.py'):
        compile(tree(folder/name), str(folder/name), 'exec')
    worker = (folder/'quality_worker.py').read_text()
    assert "point=='AGG4096' or (point=='PREFIX4096' and registry=='NEWFIT_QUERY_CARD.json')" in worker
    assert 'ordinal<=30720' in worker and 'heads<=122880' in worker
    assert "choices=('AGG',)" in (folder/'materialize.py').read_text()
    assert "choices=('AGG4096',)" in (folder/'quality_virtual.py').read_text()
    return dict(status='PASS_PREPARED_QUALITY_SOURCE_NO_QUERY', inherited_actual_sha256=EXPECTED,
                unchanged_native_S4_decoder_save_path=True, unchanged_full_virtual_gate_executor_scorer=True,
                unchanged_original_offline_evaluation=True, PREFIX_only_NEWFIT=True, AGG_final_only=True,
                model_queries=0, gradient_queries=0, optimizer_updates=0, launch_allowed=False,
                actual_NEWFIT_CPU_and_final_training_and_quality_seal_still_required=True,
                prepared_files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [folder/n for n in ('materialize.py', 'quality_gate.py', 'quality_worker.py', 'quality_virtual.py',
                                       'quality_components.py', 'quality_offline.py', 'newfit.py', 'quality_source_check.py')]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--inherited', type=Path)
    args = parser.parse_args(); folder = Path(__file__).resolve().parent
    print(json.dumps(check(folder, args.inherited or folder.parent/'pi0269_execution'), indent=2, sort_keys=True))
