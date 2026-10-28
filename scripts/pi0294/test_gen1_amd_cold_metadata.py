"""Prove the management bootstrap loads with Python stdlib only."""
import ast,json,subprocess,sys
from pathlib import Path
import pytest
from scripts.pi0294.generalization import amd_environment_cold as m


def test_actual_site_packages_disabled_import():
    p=Path(m.__file__).resolve()
    result=subprocess.run([sys.executable,'-S','-c',
        'import runpy,sys,json; runpy.run_path('+repr(str(p))+'); print(json.dumps(dict(native=any(n.split(".")[0]in("numpy","PIL","mujoco","torch")for n in sys.modules))))'],
        capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)=={'native':False}


def test_static_metadata_child_does_not_import_native_packages():
    tree=ast.parse(Path(m.__file__).read_text());text=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant)and isinstance(node.value,str)and node.value.startswith('import importlib.metadata'):text.append(node.value)
    assert len(text)==1
    imports=[n for n in ast.walk(ast.parse(text[0]))if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert all(not isinstance(n,ast.ImportFrom)and all(x.name in('importlib.metadata','json')for x in n.names)for n in imports)


def test_wrong_host_rejected_before_subprocess(tmp_path,monkeypatch):
    called=[];monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:called.append(1))
    with pytest.raises(PermissionError):m.run(tmp_path)
    assert called==[] and not list(tmp_path.iterdir())
