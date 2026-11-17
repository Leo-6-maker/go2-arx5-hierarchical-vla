import json
from pathlib import Path
import pytest
from go2arx5_vla.world_model.hybrid_ipc_storage_v31 import archive_request,check_size
from scripts.wm02.hybrid_ipc_storage_control_v31 import panel


def test_post_ACK_archive_preserves_bytes_and_never_replaces_evidence(tmp_path):
    live=tmp_path/'live';live.mkdir();(live/'receiver').mkdir()
    (live/'INPUT.npz').write_bytes(b'captured ordinary RGB bytes')
    (live/'receiver'/'live.json').write_text('{"emitted_atoms":7}')
    destination=tmp_path/'archive';result=archive_request(live,destination)
    assert (destination/'INPUT.npz').read_bytes()==b'captured ordinary RGB bytes'
    assert json.loads(Path(result['path']).read_text())['complete']
    (live/'INPUT.npz').write_bytes(b'later source change')
    assert (destination/'INPUT.npz').read_bytes()==b'captured ordinary RGB bytes'
    with pytest.raises(FileExistsError):archive_request(live,destination)


def test_private_IPC_rejects_aliased_external_file(tmp_path):
    live=tmp_path/'live';live.mkdir();outside=tmp_path/'other';outside.write_bytes(b'x')
    (live/'alias').symlink_to(outside)
    with pytest.raises(RuntimeError,match='REGULAR'):check_size(live)
    with pytest.raises(RuntimeError,match='SYMLINK'):archive_request(live,tmp_path/'archive')


def test_fixed_panel_covers_both_families_and_known_new_guards():
    records=[dict(episode=f+'-one',family=f,phase=phase)for f in('F1','F2')for phase in('PREGRASP','DESCEND','CLOSE','LIFT','HOLD')]
    rows=panel(records);normal=rows[:16]
    assert len(rows)==32 and all(r['kind']=='normal'for r in normal)
    assert sum(records[r['entry']]['family']=='F1'for r in normal)==8
    assert sum(records[r['entry']]['family']=='F2'for r in normal)==8
    assert sum(r['partial']for r in normal)==8
    faults={r['kind']:r for r in rows[16:]}
    assert faults['service_restart']['cached']is False
    for name in('compile_expire','wire_stop','after_ACK_stop','renew_ACK_late','strict_50ms_margin'):
        assert faults[name]['cached']
    assert faults['compile_expire']['clock_mode']=='WALL_CLOCK_STRICT'
