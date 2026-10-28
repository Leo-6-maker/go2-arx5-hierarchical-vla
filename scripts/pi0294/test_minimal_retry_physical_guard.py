import os
from pathlib import Path
import pytest
from scripts.pi0294.generalization import minimal_retry_physical_guard as guard


@pytest.fixture
def roots(tmp_path):
    old=[tmp_path/'old1',tmp_path/'old2'];new=tmp_path/'new'
    for r in [*old,new]:r.mkdir();(r/'metadata').write_bytes(b'data')
    return old,new


def test_actual_allocated_blocks_and_headroom(roots):
    old,new=roots;proof=guard.check(old,new,old_metadata_ceiling=10**7,paid_output_ceiling=10**8,next_output_upper_bytes=10**6)
    assert proof['old_allocated_bytes']>0 and proof['new_allocated_bytes']>0
    assert not proof['ledger_changed'] and not proof['native_permission']
    with pytest.raises(PermissionError):guard.check(old,new,old_metadata_ceiling=1,paid_output_ceiling=10**8)
    with pytest.raises(PermissionError):guard.check(old,new,old_metadata_ceiling=10**7,paid_output_ceiling=10**8,next_output_upper_bytes=10**8)


def test_unexpected_symlink_or_retained_weights_cannot_hide(roots,tmp_path):
    old,new=roots;(new/'output-link').symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(PermissionError):guard.check(old,new,old_metadata_ceiling=10**7,paid_output_ceiling=10**8)
    (new/'output-link').unlink();(old[0]/'unexpected.safetensors').write_bytes(b'x')
    with pytest.raises(PermissionError):guard.check(old,new,old_metadata_ceiling=10**7,paid_output_ceiling=10**8)


def test_cross_root_hardlink_cannot_hide_billed_output(roots):
    old,new=roots;os.link(old[0]/'metadata',new/'hidden')
    with pytest.raises(PermissionError):guard.check(old,new,old_metadata_ceiling=10**7,paid_output_ceiling=10**8)
