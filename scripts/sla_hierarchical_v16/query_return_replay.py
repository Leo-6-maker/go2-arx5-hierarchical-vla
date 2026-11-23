"""Exercise policy return delivery with sealed real calls and originally measured camera frames."""

import argparse
import json
from pathlib import Path

import numpy as np

from demonstrated_grip import pin, read, require
from sealed_panel import configure_site


def rejected(action, reason):
    try:
        action()
    except ValueError as exc:
        require(str(exc) == reason, f"unexpected rejection {exc}")
        return True
    raise ValueError("accepted invalid return: " + reason)


def replay(root, repo):
    from compiler import SourceCurve, relative_angle
    from observation_visibility import run
    from policy_query import policy_response

    reports = repo / "reports/SLA_HIERARCHICAL_V16"
    old = read(
        pin(
            reports / "H1_REVISION_ACTUAL.json",
            "855860708527c0a2b0d86b5ad710ea94da7201ba89e02bca5c41e08518ca206c",
        )
    )
    calls = old["calls"]
    pins = read(
        pin(
            root / "v13/run01/S0_ACTUAL.json",
            "6917c8ce81dd237c2e54dca491bec3347ae3c5730546aadae69b12ee36ce21b9",
        )
    )["inputs"]
    abi = root / "v12/A4_P2_ABI.json"
    chains = {r["window"]: r for r in read(pin(abi, pins[str(abi)]))["windows"]}
    by_hash = {v: Path(k) for k, v in pins.items() if k.endswith(".npy")}
    last = {
        ep: max(r["tick"] for r in calls.values() if r["episode"] == ep)
        for ep in {r["episode"] for r in calls.values()}
    }
    pending = {}
    negative = {}
    outcomes = []
    checks = []

    def delivery(stream, window, data):
        saved, response, reference = data
        key = saved["id"]
        k = window["tick"]
        stream.finish_query(*response)
        proposal = stream.take_proposal()
        curve = proposal["curve"]
        require(
            proposal["query_tick"] == saved["tick"]
            and proposal["latest_sensor_tick"] == k
            and proposal["episode"] == saved["episode"]
            and proposal["input_sha256"] == response[1]
            and stream.take_proposal() is None,
            "incorrect query or treatment boundary",
        )
        require(
            np.allclose(curve.knots, reference, atol=2e-7, rtol=0)
            and np.allclose(
                curve.origin, saved["T_W_source_origin"], atol=2e-9, rtol=0
            ),
            "world conversion differs from sealed query",
        )
        wrong = SourceCurve(
            response[2], window["T_W_source_origin"], window["T_link6_TCP"]
        )
        dx = (
            np.linalg.norm(curve.knots[:, :3, 3] - wrong.knots[:, :3, 3], axis=1) * 1000
        )
        angle = np.rad2deg(
            relative_angle(curve.knots[:, :3, :3], wrong.knots[:, :3, :3])
        )
        outcomes.append(
            dict(
                id=key,
                arrival_window=window["id"],
                source_tick=saved["tick"],
                arrival_sensor_tick=k,
                injected_lag_ticks=k - saved["tick"],
                input_sha256=response[1],
                archive_input_sha256=calls[key]["input_sha256"],
                original_output_sha256=calls[key]["file_sha256"],
                source_knots=len(reference),
                max_abs_matrix_residual=float(abs(curve.knots - reference).max()),
                latest_frame_substitution_max_mm=float(dx.max()),
                latest_frame_substitution_max_deg=float(angle.max()),
            )
        )
        checks.append(
            dict(
                id=key,
                control="duplicate",
                rejected=rejected(
                    lambda: stream.finish_query(*response),
                    "unknown or already closed query",
                ),
            )
        )

    def query(stream, window):
        ep = window["episode"]
        name = window["id"]
        record = calls[name]
        require(
            record["episode"] == ep
            and record["tick"] == window["tick"]
            and record["input_sha256"] == window["input_sha256"],
            "inference receipt mismatch",
        )
        if ep in pending:
            delivery(stream, window, pending.pop(ep))
        raw = np.load(
            pin(by_hash[record["file_sha256"]], record["file_sha256"]),
            allow_pickle=False,
        )
        abi = chains[name]
        require(
            abi["raw_sha256"] == record["file_sha256"]
            and abi["tick"] == window["tick"],
            "chain mismatch",
        )
        with np.load(
            pin(root / "v12/A4_ABI" / (name + ".npz"), abi["chain_sha256"]),
            allow_pickle=False,
        ) as f:
            require(np.array_equal(f["raw16"], raw), "original output differs")
            ref = f["world_knots"]

        def archived(obs):
            with np.load(
                pin(
                    root / "v12/STUDENT_INPUT" / (name + ".npz"), record["input_sha256"]
                ),
                allow_pickle=False,
            ) as z:
                require(
                    obs["prompt"] == str(z["prompt"])
                    and np.array_equal(obs["state"], z["state"])
                    and np.array_equal(obs["images"]["cam_high"], z["head"])
                    and np.array_equal(
                        obs["images"]["cam_right_wrist"], z["right_wrist"]
                    )
                    and not obs["images"]["cam_left_wrist"].any(),
                    "wrong captured call input",
                )
            return raw

        if ep not in negative:
            key, obs = stream.begin_query()
            negative[ep] = policy_response(key, obs, archived)
        key, obs = stream.begin_query()
        response = policy_response(key, obs, archived)
        data = (window, response, ref)
        if window["tick"] == last[ep]:
            delivery(stream, window, data)
            checks.append(
                dict(
                    id=name,
                    control="out_of_order",
                    rejected=rejected(
                        lambda: stream.finish_query(*negative.pop(ep)),
                        "newer proposal already accepted",
                    ),
                )
            )
            key, obs = stream.begin_query()
            obs["state"][15] = np.float32(0 if obs["state"][15] else 1)
            bad = policy_response(key, obs, lambda unused: raw)
            checks.append(
                dict(
                    id=name,
                    control="wrong_input",
                    rejected=rejected(
                        lambda: stream.finish_query(*bad),
                        "worker processed a different observation",
                    ),
                )
            )
            key, obs = stream.begin_query()
            late = policy_response(key, obs, archived)
            stream.stop(ep)
            checks.append(
                dict(
                    id=name,
                    control="stop_during_pending",
                    rejected=rejected(
                        lambda: stream.finish_query(*late), "episode stopped"
                    ),
                )
            )
            checks.append(
                dict(
                    id=name,
                    control="stop_before_take",
                    rejected=rejected(stream.take_proposal, "episode stopped"),
                )
            )
        else:
            pending[ep] = data

    visibility = run(root, repo, on_query=query)
    expected = read(reports / "OBSERVATION_VISIBILITY_ACTUAL.json")
    require(
        not pending and not negative and visibility["target"] == expected["target"],
        "different original sensor replay",
    )
    require(
        len(outcomes) == 16
        and sum(x["injected_lag_ticks"] == 250 for x in outcomes) == 10
        and sum(x["injected_lag_ticks"] == 0 for x in outcomes) == 6
        and len(checks) == 40,
        "incomplete population",
    )
    return dict(
        proposals=outcomes,
        negative_controls=checks,
        closed_teacher_episodes=6,
        new_model_forwards=0,
        native_student_commands=0,
        native_integrations=0,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    repo = Path(__file__).resolve().parents[2]
    configure_site(root, repo)
    rv = replay(root, repo)
    with a.out.open("x") as f:
        json.dump(rv, f, indent=2, allow_nan=False)
        f.write("\n")
