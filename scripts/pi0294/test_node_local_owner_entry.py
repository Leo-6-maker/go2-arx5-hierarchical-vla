"""Exercise the actual saved private methods; no model or simulator imports."""
import base64,json,os
from pathlib import Path
import pytest
from scripts.pi0294.pv1 import node_local_owner_entry as adapter


def test_actual_methods_change_only_the_two_immutable_cold_entry_routes():
    directory=os.environ.get('D25_NODE_LOCAL05_PRIVATE_CAPTURE_DIR')
    if not directory:pytest.skip('exact original private methods capture required')
    value=json.loads((Path(directory)/'D25_FINITE_LOCAL04_OWNER_PREPARED.private.json').read_bytes())
    original=base64.b64decode(value['files']['v5_core_resident_methods.py'])
    derived,proof=adapter.derive(original)
    restored=derived.decode()
    for old,new in reversed(adapter.CHANGES):
        assert restored.count(new)==1
        restored=restored.replace(new,old)
    assert restored.encode()==original
    assert proof['changed_entry_locations']==2 and not proof['scientific_variables_changed']
    assert b"info['python'],'-u','-c',bootstrap"in derived
    assert b"resource.setrlimit(resource.RLIMIT_CPU,(3600,3600))"in derived
    with pytest.raises(PermissionError):adapter.derive(original+b'\n')


def test_separate_owner_entry_sets_local_cache_for_both_subprocess_bootstraps():
    import ast
    directory=os.environ.get('D25_NODE_LOCAL05_PRIVATE_CAPTURE_DIR')
    if not directory:pytest.skip('exact original private finite owner capture required')
    value=json.loads((Path(directory)/'D25_FINITE_LOCAL04_OWNER_PREPARED.private.json').read_bytes())
    original=base64.b64decode(value['files']['v5_core_resident_owner.py'])
    derived,proof=adapter.derive_owner_environment(original)
    tree=ast.parse(derived);ns={}
    # Exercise actual header assembly without importing or starting the owner.
    selected=[]
    for node in tree.body:
        if isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id in('LOCAL_HEADER','REMOTE_HEADER','_NODE_LOCAL_ENVIRONMENT')for t in node.targets):selected.append(node)
        elif isinstance(node,ast.FunctionDef)and node.name=='_node_local_environment':selected.append(node)
        elif isinstance(node,ast.AugAssign)and isinstance(node.target,ast.Name)and node.target.id=='LOCAL_HEADER':selected.append(node)
    exec(compile(ast.Module(body=selected,type_ignores=[]),'actual-cold-header','exec'),ns)
    compile(ns['LOCAL_HEADER'],'actual-role-local-header','exec')
    environment=ns['_node_local_environment'](Path('/node-local/role'))
    assert set(environment)==set(adapter.LOCAL_ENVIRONMENT)and 'HOME'not in environment
    assert all(v.startswith('/node-local/role/control/')for v in environment.values())
    assert b"repr(_node_local_environment(root))"in derived
    assert b"resource.setrlimit(resource.RLIMIT_CPU,(soft,6900))"in derived
    assert proof['original_entry_retained']and not proof['resource_or_scientific_variables_changed']
    with pytest.raises(PermissionError):adapter.derive_owner_environment(original+b'\n')
