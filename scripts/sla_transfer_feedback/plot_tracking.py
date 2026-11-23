"""Plot journal-derived task errors for all native paired cards (or all eligible A0)."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("feedback", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    root = args.feedback
    cfg = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "reports/SLA_FROZEN_FEEDBACK/F1_F2_REGISTRATION.json"
        ).read_text()
    )

    def rows(stage):
        return json.loads((root / (stage + "_VERIFIED.json")).read_text())["records"]

    a = {r["card"]: r for r in rows("F1") if r["outcome"] == "A0_COMPLETE"}
    f2 = rows("F2") if (root / "F2_VERIFIED.json").is_file() else []
    pairs = [r for r in f2 if "paired_control" in r]

    def draw(stage, record, cutoff=None, *, label=None):
        folder = root / cfg["campaigns"][stage] / record["card"]
        p = folder / "TRACK.private.npz"
        assert hashlib.sha256(p.read_bytes()).hexdigest() == record["curve_sha256"]
        with np.load(p, allow_pickle=False) as data:
            t = data["elapsed_ms"] * 0.001
            n = len(t) if cutoff is None else min(len(t), 1 + int(cutoff))
            ids = np.unique(np.r_[np.arange(0, n, 10), n - 1])
            style = {
                "lw": 1.2,
                "alpha": 0.70,
                "color": "#2c7190" if stage == "F2" else "#805a40",
                "label": label,
            }
            for axis, k in zip(ax, ["abs_mm", "abs_deg"], strict=True):
                axis.plot(t[ids], data[k][ids], **style)
                if stage == "F2":
                    axis.plot(
                        t[n - 1],
                        data[k][n - 1],
                        marker="o",
                        color=style["color"],
                        markersize=4,
                    )

    fig, ax = plt.subplots(
        2, 1, figsize=(7.5, 5.5), sharex=True, constrained_layout=True
    )
    for axis, limit, unit in zip(
        ax, (4, 2), ("Position error (mm)", "Orientation error (deg)"), strict=True
    ):
        axis.axvspan(1, 2.9, color="#c7cdd1", alpha=0.24, lw=0)
        axis.axhline(limit, color="#22683f", ls="--", lw=1, label="Task tolerance")
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    if pairs:
        for i, r in enumerate(pairs):
            draw(
                "F1",
                a[r["paired_control"]["card"]],
                r["post_init_ticks"],
                label=None if i else "A0 matched control",
            )
            draw("F2", r, label=None if i else "A1 feedback")
        fig.suptitle(f"Known target tracking across {len(pairs)} matched pairs")
    else:
        for i, r in enumerate(a.values()):
            draw("F1", r, label=None if i else "A0 reference continuity")
        fig.suptitle(f"Known target tracking across {len(a)} eligible A0 starts")
    for axis in ax:
        axis.set_ylim(bottom=0)
    ax[0].legend(frameon=False, loc="upper right")
    ax[1].set_xlabel("Time from reference handoff (s)")
    fig.savefig(args.output, dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()
