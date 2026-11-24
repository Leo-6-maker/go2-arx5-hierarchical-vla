"""Plot native movement relative to entry together with the original absolute error."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from known_curve_scoring import wire
from known_dynamic_panel import CARDS, RATES
from scipy.spatial.transform import Rotation


def plot(folder, dest):
    cases = [c for c in CARDS if (folder / (c + ".npz")).is_file()]
    if not cases:
        raise ValueError("no completed curves to plot")
    fig, axes = plt.subplots(
        len(cases), 4, figsize=(13, 2.8 * len(cases) + 0.55), squeeze=False, sharex=True
    )
    times = np.arange(3501) * 0.002
    for name, row in zip(cases, axes):
        with np.load(folder / (name + ".npz"), allow_pickle=False) as z:
            measured, nom = z["actual"], z["nominal"]
        nom[500:2501] = wire(nom[0], *RATES[name.rsplit("_", 1)[0]])
        for line, color, style, label in (
            (nom, "#41546a", "--", "Nominal"),
            (measured, "#007f77", "-", "Measured"),
        ):
            position = 1000 * (line[:, :3, 3] - line[0, :3, 3])
            tilt = np.degrees(
                Rotation.from_matrix(line[:, :3, :3] @ line[0, :3, :3].T).as_rotvec()[
                    :, 1
                ]
            )
            for ax, values in zip(row[:3], (position[:, 0], position[:, 2], tilt)):
                ax.plot(times, values, style, color=color, label=label, lw=1.6)
        row[-1].plot(
            times,
            1000 * np.linalg.norm(measured[:, :3, 3] - nom[:, :3, 3], axis=1),
            color="#a64521",
            lw=1.65,
        )
        row[-1].axhline(4, color="#6d4854", ls="--", lw=1.1)
        row[-1].set_ylim(bottom=0)
        for ax in row:
            ax.axvspan(1, 5, color="#f0f3f5", zorder=-3)
            ax.grid(axis="y", alpha=0.25)
            ax.spines[["right", "top"]].set_visible(False)
            ax.margins(x=0)
        row[0].set_ylabel(name.replace("_", " ") + "\nmm")
    for ax, label in zip(
        axes[0],
        (
            "World X change (mm)",
            "World Z change (mm)",
            "World Y rotation (deg)",
            "Absolute position error (mm)",
        ),
    ):
        ax.set_title(label, fontsize=10, loc="left")
    axes[0, 0].legend(frameon=False, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Time from settled entry (s)")
    fig.tight_layout()
    fig.savefig(dest, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    plot(a.folder, a.out)
