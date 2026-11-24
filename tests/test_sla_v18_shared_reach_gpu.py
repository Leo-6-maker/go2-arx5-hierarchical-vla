import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/sla_go2_v18"))
import reach_forward


@pytest.mark.parametrize(("free", "accepted"), [(8600, True), (8597, False)])
def test_shared_board_uses_measured_headroom_with_foreign_compute(
    tmp_path, monkeypatch, free, accepted
):
    lock = tmp_path / "card.lock"
    lock.touch()
    minimum = {"board_peak_mib": 8598, "resident_peak_kib": 1}

    def config(p):
        if p == reach_forward.RECEIVER:
            return {"gpu": reach_forward.UUID, "gpu_lock": str(lock)}
        return {"O2": minimum, "P2_EXTRA": minimum}

    monkeypatch.setenv("REACH_SHARED_GPU", "1")
    monkeypatch.setattr(reach_forward, "read", config)
    monkeypatch.setattr(reach_forward.socket, "gethostname", lambda: "corelab-4080s")
    xml = f"<nvidia_smi_log><gpu><uuid>{reach_forward.UUID}</uuid><fb_memory_usage><free>{free} MiB</free></fb_memory_usage><processes><process_info><type>C</type></process_info></processes></gpu></nvidia_smi_log>"
    monkeypatch.setattr(
        reach_forward.subprocess, "check_output", lambda args, timeout: xml.encode()
    )
    if accepted:
        lease, info = reach_forward.site(tmp_path)
        assert lease == lock and info["compute_processes"] == 1
        assert (
            info["board_free_mib"] == free and info["measured_board_peak_mib"] == 8598
        )
    else:
        with pytest.raises(ValueError, match="headroom"):
            reach_forward.site(tmp_path)


def test_original_unshared_admission_is_preserved(tmp_path, monkeypatch):
    monkeypatch.delenv("REACH_SHARED_GPU", raising=False)

    def exclusive(root):
        assert root == tmp_path
        raise ValueError("GPU0 has additional users")

    monkeypatch.setattr(reach_forward, "dedicated_site", exclusive)
    with pytest.raises(ValueError, match="additional users"):
        reach_forward.site(tmp_path)
