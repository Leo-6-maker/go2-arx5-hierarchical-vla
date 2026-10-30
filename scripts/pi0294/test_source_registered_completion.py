"""Actual saved LOCAL schema and the exact cold adoption fault."""
import base64,gzip,hashlib,json,os
from pathlib import Path
import pytest
from scripts.pi0294.multitask import night_source_stage as stage
from scripts.pi0294.multitask.source_registered_completion import program,manifest_program


def test_actual_local_additive_adoption_preserves_all_existing_fields():
    directory=os.environ.get('NIGHT_SOURCE_MAINTENANCE_CAPTURE_DIR')
    if not directory:pytest.skip('Existing exact original LOCAL capture required')
    p=Path(directory)/'NODE_LOCAL07FIX_RUNTIME_ASSEMBLY_FINAL_PREPARED.private.json'
    b=gzip.decompress(base64.b64decode(json.loads(p.read_bytes())['proposed_LOCAL_gzip_base64']))
    assert hashlib.sha256(b).hexdigest()=='d98efaf9ab92234719e5db2c23635f122c87f39e9d862777980c11696484a89e'
    local=json.loads(b);assert 'revision'not in local
    broken=json.loads(b)
    with pytest.raises(KeyError):broken['revision']+=1
    line=next(x for x in stage.PROGRAM.splitlines()if x.startswith('new_local=json.loads(lb);'))
    marker={'test_only':'exact additive metadata'};ns=dict(json=json,lb=b,marker=marker)
    exec(line,ns);after=ns['new_local'];assert after.pop('NIGHT_SOURCE30_BOUNDED_AUTONOMY')==marker
    assert after==local


def test_registered_completion_contains_no_registry_replay_or_capsule_write():
    body=program();compile(body,'complete','exec')
    assert 'CONFIRMED_CURRENT_PROPOSED_BYTES_WITH_RECEIPT'in body
    assert 'UNWRITTEN_TAIL_REQUIRED'in body
    assert 'cas(registry'not in body and 'root.mkdir'not in body
    assert "new_local['revision']"not in body
    assert 'subprocess.run(args'not in body and 'systemd-run'not in body


def test_complete_entry_programs_have_no_unbound_top_level_names():
    import builtins,symtable
    for body in(program(),manifest_program()):
        table=symtable.symtable(body,'complete','exec')
        declared={s.get_name()for s in table.get_symbols()if s.is_assigned()or s.is_imported()}
        referenced={s.get_name()for s in table.get_symbols()if s.is_referenced()}
        assert referenced-declared-set(dir(builtins))-{'P'}==set()
    assert 'cas('not in manifest_program()and 'commit('not in manifest_program()
