import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts/sla_hierarchical_v16")
)
import moving_stop_owner as owner
from verify_grip_card import states


def allocation():
    return dict(
        card=owner.CARD,
        campaign=owner.CAMPAIGN,
        max_native_starts=1,
        maximum_native_ticks=6503,
        candidate_sha256=owner.sha(owner.CANDIDATE),
    )


@pytest.mark.parametrize("field", list(allocation()))
def test_changed_scientific_condition_rejected(field):
    plan = allocation()
    owner.validate_allocation(plan)
    plan[field] = 2
    with pytest.raises(ValueError):
        owner.validate_allocation(plan)


def fake_state(raw):
    (tick,) = struct.unpack_from("<Q", raw)
    return {"step": tick, "time_s": tick * 0.002}


def frame(tick):
    return struct.pack("<IQ32x", 40, tick)


def test_counter_reaches_eof_beyond_old_gripper_cap(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "scripts.pi0294.task_state",
        SimpleNamespace(parse_state=fake_state),
    )
    file = tmp_path / "state.bin.journal"
    file.write_bytes(b"".join(frame(i) for i in range(1, 6507)) + b"\x08")
    it = states(tmp_path)
    for i in range(1, 6507):
        assert next(it)["step"] == i
    with pytest.raises(ValueError, match="incomplete state frame length"):
        next(it)


def write(path, obj):
    path.write_text(json.dumps(obj))


def closed(tmp_path):
    case = tmp_path / owner.CARD
    (case / "control").mkdir(parents=True)
    ep = case / "native-test"
    ep.mkdir()
    sp = case / "EXECUTION_SPEC.json"
    write(sp, dict(owner="test", registered_root=str(case), max_ticks=6503))
    write(
        case / "control/LEASE.json",
        dict(owner="test", spec_sha256=owner.sha(sp), status="CLOSED"),
    )
    write(case / "NATIVE_REQUEST.json", dict(episode=ep.name))
    write(case / "RESULT.private.json", dict(episode=ep.name, outcome="TECHNICAL_STOP"))
    return case, ep


def test_partial_state_keeps_prefix_and_unverified_total(tmp_path, monkeypatch):
    case, ep = closed(tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "scripts.pi0294.task_state",
        SimpleNamespace(parse_state=fake_state),
    )
    monkeypatch.setitem(
        sys.modules, "go2arx5_vla.world_model.init_trace_reader", SimpleNamespace()
    )
    monkeypatch.setattr(
        sys.modules["go2arx5_vla.world_model.init_trace_reader"],
        "audit_trace",
        lambda *a, **kw: dict(
            native_step_records=1, status="RECORDED_TRACE_CHECKS_PASS"
        ),
        raising=False,
    )
    (ep / "init.trace").touch()
    (ep / "state.bin.journal").write_bytes(frame(1) + frame(2)[:-1])
    row = owner.closed_report(tmp_path, case)
    assert (
        row["issued_requests"]
        == row["simulator_starts"]
        == row["state_prefix_ticks"]
        == 1
    )
    assert row["actual_ticks"] is None and row["qualified"] is False
    assert "incomplete state frame" in row["errors"]


def test_launch_attempt_without_evidence_is_unknown(tmp_path, monkeypatch):
    case, ep = closed(tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "go2arx5_vla.world_model.init_trace_reader",
        SimpleNamespace(audit_trace=None),
    )
    assert owner.native_counts(case)["actual_ticks"] == 0
    (ep / "sim.log").touch()
    counts = owner.native_counts(case)
    assert counts["actual_ticks"] is counts["simulator_starts"] is None
    assert counts["issued_requests"] == 1


def test_owner_closes_lease_on_interrupt_and_cannot_replace_case(tmp_path, monkeypatch):
    (tmp_path / "v16").mkdir()
    registry = tmp_path / "registry.json"
    write(registry, {"entries": []})
    receiver = tmp_path / "receiver"
    receiver.write_bytes(b"test")
    plan = tmp_path / "plan.json"
    write(plan, allocation())
    base = dict(
        cpus=[0],
        frozen_files={},
        campaign_root=str(tmp_path),
        mobile_world_lock=str(tmp_path / "world.lock"),
        registry=str(registry),
        source_sha="test",
        host="test",
        scope="test",
        ROS_DOMAIN_ID=193,
        wall_limit_s=2,
        native_binary="test-sim",
        controller_binary="test-controller",
        solver_binary="test-solver",
    )
    monkeypatch.setattr(owner, "inputs", lambda root: (base, {}, receiver, [], {}))
    monkeypatch.setattr(owner.os, "sched_setaffinity", lambda *args: None)
    monkeypatch.setattr(owner.os, "umask", lambda mode: None)
    monkeypatch.setitem(
        sys.modules,
        "go2arx5_vla.world_model.init_trace_reader",
        SimpleNamespace(audit_trace=None),
    )

    def interrupt(root, spec, sp, prepared):
        assert prepared[0] is base
        assert owner.read(sp.parent / "control/LEASE.json")["status"] == "ACTIVE"
        assert owner.read(registry)["entries"][-1]["run_uuid"] == spec["owner"]
        raise RuntimeError("interrupted before request")

    monkeypatch.setattr(owner, "run_case", interrupt)
    row = owner.run(tmp_path, plan)
    case = tmp_path / "v16" / owner.CAMPAIGN / owner.CARD
    assert row["outcome"] == "TECHNICAL_STOP"
    assert row["registered_requests"] == 1 and row["issued_requests"] == 0
    assert owner.read(case / "control/LEASE.json")["status"] == "CLOSED"
    with pytest.raises(FileExistsError):
        owner.run(tmp_path, plan)
    assert len(owner.read(registry)["entries"]) == 1
    assert (
        owner.read(case.parent / "OWNER_INTERRUPTION.json")["error"]
        == "RuntimeError: interrupted before request"
    )
    with pytest.raises(ValueError, match="unidentified pre-request interruption"):
        owner.run(tmp_path, plan, recover_before_request=True)

    write(
        case.parent / "OWNER_INTERRUPTION.json",
        {
            "error": "FileNotFoundError: [Errno 2] No such file or directory: '/tmp/tmpfixture/planner.jsonl'"
        },
    )
    (case / "logs" / "unexpected").touch()
    with pytest.raises(ValueError, match="unexpected prior native"):
        owner.run(tmp_path, plan, recover_before_request=True)
    (case / "logs" / "unexpected").unlink()
    second = owner.run(tmp_path, plan, recover_before_request=True)
    assert second["issued_requests"] == 0
    assert len(owner.read(registry)["entries"]) == 2
    with pytest.raises(FileExistsError):
        owner.run(tmp_path, plan, recover_before_request=True)
