import ast,importlib.util,types
import pytest
from pathlib import Path
from scripts.sla.mfree_c20_vectorfield_v3 import extracted_fields
from go2arx5_vla.sla_adapter import batch_v2

def test_extraction_returns_fields_without_rng_or_euler_update(tmp_path):
    # Exercise extraction against the exact source copy fetched from the pinned
    # official checkout. No model or JAX is constructed in this local check.
    source=Path('/home/liuyu/ISSAC-GR00T/work/private/mfree-c20-recovery-diagnostics-20261116/src_openpi_models_pi0.py')
    if not source.exists():pytest.skip('Pinned runtime source is not present on this host')
    tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Pi0');methods=[n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name in ('compute_loss','sample_actions')]
    lines=source.read_text().splitlines()
    stub=tmp_path/'pinned_methods.py';stub.write_text('from __future__ import annotations\nclass Pi0:\n'+''.join('\n'.join(lines[m.lineno-1:m.end_lineno])+'\n'for m in methods))
    spec=importlib.util.spec_from_file_location('pinned_methods_for_audit',stub);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module._model=types.SimpleNamespace()
    _,generated=extracted_fields(batch_v2,module);node=ast.parse(generated)
    assert len(node.body)==4
    assert 'return predicted'in generated and 'return v_t'in generated
    assert 'return x_t +'not in generated and 'random.'not in generated
    assert 'kv_cache=kv_cache'in generated
