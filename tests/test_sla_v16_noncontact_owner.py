from pathlib import Path
import sys

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts/sla_hierarchical_v16")
)
from noncontact_owner import NoncontactOwner

GEOM = [
    dict(id=i, name=n, body=b)
    for i, (n, b) in enumerate(
        (
            ("task_table", "world"),
            ("task_cube_geom", "task_cube"),
            ("task_right_pad", "x5_link8"),
        )
    )
]


def packet(i, other=0, force=2.0):
    return dict(
        step=i,
        time_s=i * 0.002,
        contact_time_s=(i - 1) * 0.002,
        contacts=[dict(geom0=1, geom1=other, wrench=[force])],
    )


def test_transient_in_middle_of_grant_is_terminal_even_after_contact_clears():
    stop = []
    g = NoncontactOwner("e", 10, GEOM, stop.append)
    g.after_pause("e", 10, [packet(10)])
    g.grant("e", 10, 35, lambda: None)
    with pytest.raises(ValueError, match="verified"):
        g.dispatch("e", 10, lambda: pytest.fail("sent while integrating"))
    g.after_pause("e", 35, [packet(i, 2 if i == 11 else 0, 0.0) for i in range(11, 36)])
    assert (
        stop == [g.terminal]
        and stop[0]["observed_tick"] == 11
        and stop[0]["pause_tick"] == 35
    )
    with pytest.raises(ValueError, match="terminal"):
        g.grant("e", 35, 60, lambda: pytest.fail("late grant"))
    assert len(stop) == 1


def test_stale_epoch_does_not_interrupt_current_owner_but_missing_evidence_does():
    stop = []
    g = NoncontactOwner("e", 10, GEOM, stop.append)
    with pytest.raises(ValueError, match="wrong"):
        g.after_pause("past", 10, [packet(10)])
    g.after_pause("e", 10, [packet(10)])
    assert g.dispatch("e", 10, lambda: 4) == 4
    g.grant("e", 10, 12, lambda: None)
    g.after_pause("e", 12, [packet(12)])
    assert g.terminal["reason"] == "CONTACT_STREAM_INVALID" and len(stop) == 1


def test_contact_clock_and_delivery_uncertainty_seal_before_stop_callback():
    for kind in ("clock", "send", "grant"):
        stop = []

        def failed_stop(value):
            assert g.terminal == value
            stop.append(value)
            raise RuntimeError("transport unavailable")

        def failed_send():
            raise RuntimeError("unconfirmed")

        g = NoncontactOwner("e", 10, GEOM, failed_stop)
        p = packet(10)
        with pytest.raises(RuntimeError, match="unavailable"):
            if kind == "clock":
                p["contact_time_s"] = p["time_s"]
                g.after_pause("e", 10, [p])
            else:
                g.after_pause("e", 10, [p])
                if kind == "grant":
                    g.grant("e", 10, 35, failed_send)
                else:
                    g.dispatch("e", 10, failed_send)
        with pytest.raises(ValueError, match="terminal"):
            g.dispatch("e", 10, lambda: pytest.fail("resent"))
        assert len(stop) == 1
