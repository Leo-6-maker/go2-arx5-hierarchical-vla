"""CPU-only invariants for the time law, geometry, event anchors and future revisions."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(BASE / "src"),
    str(BASE / "scripts/sla_hierarchical_v16"),
    str(BASE / "scripts/sla_transfer_v15"),
]
from compiler import Compiler, SourceCurve, vectors
from governor import PersistentReference
from prefix_replay import WireProgram

LIMITS = {
    "linear_velocity": 0.05,
    "linear_acceleration": 0.25,
    "angular_velocity": 0.25,
    "angular_acceleration": 1.0,
}


def source():
    u = np.linspace(0, 1, 50)
    a = np.zeros((50, 16))
    a[:, 8:11] = np.stack(
        [0.015 * u, 0.002 * np.sin(2 * np.pi * u), 0.004 * u**2], axis=1
    )
    a[:, 11:15] = Rotation.from_rotvec(
        np.stack([0.04 * np.sin(4 * u), -0.03 * u, 0.05 * u**2], axis=1)
    ).as_quat()[:, [3, 0, 1, 2]]
    a[:, 15] = 0.65 + 0.25 * np.cos(2 * np.pi * u)
    return a


def plan(bounded=False, held=False):
    s = SourceCurve(source(), np.eye(4), np.eye(4))
    anchor = np.eye(4)
    anchor[0, 3] = -0.003
    return Compiler(
        s,
        anchor,
        np.array([[0.001, 0, 0], [0, 0.003, 0]]),
        LIMITS,
        bounded=bounded,
        protect_all=held,
    )


@pytest.mark.parametrize("bounded", [False, True])
def test_join_matches_consumed_twist_and_two_derivatives(bounded):
    c = plan(bounded)
    h = 1e-5
    first = c.at([0, h])[0]
    assert np.allclose(first[0], c.anchor)
    v, w = vectors(first, h)
    assert np.allclose(v[0], c.v, atol=1e-7) and np.allclose(w[0], c.w, atol=1e-7)
    for t in [c.brake, *c.boundaries]:
        p = c.at(t + h * np.arange(-2, 3))[0]
        for derivative in vectors(p, h):
            assert np.max(abs(derivative[1] - derivative[2])) < 1e-5
            assert (
                np.max(
                    abs(
                        (derivative[1] - derivative[0]) / h
                        - (derivative[3] - derivative[2]) / h
                    )
                )
                < 0.012
            )
    _, g, s = c.at([0, c.lead, c.end, c.end + 50])
    assert np.allclose(s, [0, 0, 49 / 30, 49 / 30])
    assert np.allclose(g, c.source.arm[[0, 0, -1, -1], 7])


def test_budget_and_event_protection():
    c = plan(True)
    assert len(c.source.events) == 2
    for part in c.parts:
        dx, da = part.distortion()
        assert dx <= 0.0005 and da <= np.deg2rad(0.25)
        if set(range(part.start, part.end)) & c.source.protected:
            assert not part.spline
    assert not any(b.spline for b in plan(True, True).parts)


@pytest.mark.parametrize("edit", ["width", "quaternion", "frame", "shape"])
def test_unsafe_inputs(edit):
    a, tool = source(), np.eye(4)
    if edit == "width":
        a[3, 15] = 1.2
    if edit == "quaternion":
        a[0, 11] = 0
    if edit == "frame":
        tool[0, 0] = -1
    if edit == "shape":
        a = a[:-1]
    with pytest.raises(ValueError):
        SourceCurve(a, np.eye(4), tool)


def manager():
    n = 140
    poses = np.repeat(np.eye(4)[None], n, axis=0)
    grip = np.full(n, 0.5)
    phase = np.linspace(0, 1, n)

    def program():
        return WireProgram("one", poses, grip, phase)

    m = PersistentReference(program(), "episode", 30)

    def accept(t):
        key, _ = m.window(t)
        raw = json.dumps(
            {
                "_pi0294": {
                    "episode": "episode",
                    "sequence": key[1],
                    "source_mpc_s": t,
                    "ttl_s": 0.4,
                    "operation": "install",
                }
            }
        )
        h = hashlib.sha256(raw.encode()).hexdigest()
        m.acknowledge(
            key,
            h,
            {"kind": "accept", "raw": raw, "ref_id": h},
            {"kind": "installed", "ref_id": h},
        )

    for i in range(31):
        accept(30 + 0.1 * i)
    return m, poses, grip, phase, program


@pytest.mark.parametrize("component", ["pose", "width", "phase"])
def test_already_installed_prefix_beyond_50_samples_is_protected(component):
    m, p, g, s, program = manager()
    if component == "pose":
        p[88, 0, 3] += 0.001
    if component == "width":
        g[88] += 0.01
    if component == "phase":
        s[88] += 0.001
    with pytest.raises(ValueError, match="prefix"):
        m.revise(program(), 33, "episode")
    assert m.revision == 0


def test_future_revision_and_stale_handoff():
    m, p, g, s, program = manager()
    p[110:, 0, 3] = 0.001
    m.revise(program(), 33, "episode")
    assert m.revision == 1 and m.intent.at(np.array([5.5]))[0][0, 0, 3] == 0.001
    with pytest.raises(ValueError):
        m.revise(program(), 33.4, "episode")
    assert m.terminal


def test_pending_ack_blocks_revision():
    m, p, g, s, program = manager()
    m.window(33.1)
    with pytest.raises(ValueError):
        m.revise(program(), 33.1, "episode")
    assert m.terminal
