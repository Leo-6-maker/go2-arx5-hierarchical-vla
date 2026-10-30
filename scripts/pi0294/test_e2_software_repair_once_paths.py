"""Execute the actual embedded repair helper against the original onceIO."""
import ast,hashlib,json
from pathlib import Path
import pytest
from scripts.pi0294.generalization.e2_owner_software_repair_management import PROGRAM
from scripts.pi0294.dec1.source_cpu_once_io import commit


@pytest.mark.parametrize('relative',['role/RUN_MANIFEST.json','authority/leases/OWN.json'])
def test_actual_repair_helper_keeps_all_auxiliaries_beside_exact_target(tmp_path,relative):
    tree=ast.parse(PROGRAM)
    functions=[n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='once']
    assert len(functions)==1
    sha=lambda b:hashlib.sha256(b).hexdigest()
    ns={'commit':commit,'sha':sha}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),'actual-repair-once','exec'),ns)
    path=tmp_path/relative;path.parent.mkdir(parents=True);before=b'{"status":"before"}\n';after=b'{"status":"after"}\n';path.write_bytes(before)
    value=ns['once'](path,before,after,'ACTUAL-REPAIR-ONCE')
    assert path.read_bytes()==after and value['after_sha256']==sha(after)
    intent=json.loads(path.with_name('ACTUAL-REPAIR-ONCE.INTENT.json').read_bytes())
    assert Path(intent['target'])==path and Path(intent['temporary']).parent==path.parent
    assert path.with_name('ACTUAL-REPAIR-ONCE.RECEIPT.json').is_file()
    with pytest.raises(FileExistsError):ns['once'](path,after,b'{"status":"replayed"}\n','ACTUAL-REPAIR-ONCE')
    assert path.read_bytes()==after
