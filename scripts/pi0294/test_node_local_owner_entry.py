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
