"""Plot red-object distance gain from achieved native TCP on a common entry-relative clock."""

import argparse
import json
from pathlib import Path

import numpy as np
from reach_pair_readout import lines, read


def collect(root):
    output = []
    for prefix, layout in (("Z1_REACH_483", "G00"), ("Z1_REACH_483_G01", "G01")):
        group = root / "v18" / prefix
        read(group / "PANEL_CLOSE.json")
        for arm in ("H0", "A"):
            case = group / (layout + "-" + arm)
            if not (case / "REACH_ENTRY.json").exists():
                continue
            entry = read(case / "REACH_ENTRY.json")
            result = read(case / "RESULT.private.json")
            native = case / result["episode"]
            data = [
                x
                for x in lines(native / "MOBILE_SAMPLES.private.jsonl")
                if x["native_end_step"] >= entry["tick"]
            ]
            times = []
            gains = []
            for x in data:
                red = np.asarray(x["object_pose"])[:3, 3]
                tcp = np.asarray(x["actual_TCP"])[:3, 3]
                times.append(round(0.002 * (x["native_end_step"] - entry["tick"]), 4))
                gains.append(
                    round(
                        float(
                            1000
                            * (
                                np.linalg.norm(np.asarray(entry["actual_xyz"]) - red)
                                - np.linalg.norm(tcp - red)
                            )
                        ),
                        4,
                    )
                )
            calls = []
            for q in sorted(case.glob("query-*")):
                meta = read(q / "REQUEST.json")
                calls.append(
                    {
                        "t": round(0.002 * (meta["tick"] - entry["tick"]), 4),
                        "returned": (q / "RETURN.json").exists(),
                    }
                )
            output.append(
                {
                    "layout": layout,
                    "arm": arm,
                    "outcome": result["outcome"],
                    "elapsed_s": times,
                    "red_gain_mm": gains,
                    "queries": calls,
                }
            )
    return output


def plot(data, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    styles = {"H0": "#28644c", "A": "#245ba7"}
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), sharex=True, sharey=True)
    for ax, layout in zip(axes, ("G00", "G01")):
        for r in data:
            if r["layout"] != layout:
                continue
            t = r["elapsed_s"]
            y = r["red_gain_mm"]
            ax.plot(t, y, color=styles[r["arm"]], lw=2)
            if t:
                ax.scatter(
                    [t[-1]],
                    [y[-1]],
                    marker="x" if r["outcome"] == "TECHNICAL_STOP" else "o",
                    s=32,
                    color=styles[r["arm"]],
                    zorder=4,
                )
            for q in r["queries"]:
                if q["returned"]:
                    ax.scatter(
                        [q["t"]],
                        [np.interp(q["t"], t, y)],
                        marker="D",
                        color="white",
                        edgecolor=styles["A"],
                        s=42,
                        zorder=5,
                    )
        ax.axhline(10, color="#54575b", lw=0.9, ls="--", zorder=0)
        ax.axhline(0, color="#8e9394", lw=0.7, zorder=0)
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(layout, loc="left", fontweight="bold")
        ax.set_xlabel("Seconds from stable entry")
        ax.set_xlim(-0.12, 9.15)
    axes[0].set_ylabel("TCP–red cube distance reduction (mm)")
    handles = [
        Line2D([0], [0], color=styles["H0"], label="Teacher"),
        Line2D([0], [0], color=styles["A"], label="Frozen SLA-PT"),
        Line2D(
            [0],
            [0],
            ls="",
            marker="D",
            mfc="white",
            mec=styles["A"],
            label="Model return",
        ),
        Line2D(
            [0],
            [0],
            ls="",
            marker="x",
            color=styles["A"],
            label="Technical interruption",
        ),
        Line2D([0], [0], ls="--", color="#54575b", label="10 mm task threshold"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.155, 1, 1))
    fig.savefig(path, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("collect", "plot"))
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    if a.mode == "collect":
        with a.output.open("x") as f:
            json.dump(collect(a.source.resolve()), f, separators=(",", ":"))
    else:
        plot(read(a.source), a.output)
