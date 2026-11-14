"""Render the sealed read-only joint diagnostic; no experiment execution."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT/'JOINT_SUFFIX_ACTUAL_20261114.json').read_text())
rows = data['rows']
labels = [f"{r['layout']}/r{r['restart']} · q{r['query']}" for r in rows]
y = np.arange(len(rows))[::-1]
fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
for ax, prefix, suffix, title in [
    (axes[0], 'prefix_max_abs_acceleration_rad_s2', 'suffix_max_abs_acceleration_rad_s2', 'All 6 arm joints'),
    (axes[1], 'joint3_prefix_max_abs_acceleration_rad_s2', 'joint3_suffix_max_abs_acceleration_rad_s2', 'Joint 3'),
]:
    left = np.array([r[prefix] for r in rows])
    right = np.array([r[suffix] for r in rows])
    ax.hlines(y, np.minimum(left, right), np.maximum(left, right), color='#b8c1cc', linewidth=1.2, zorder=1)
    ax.scatter(left, y, s=43, color='#2369ad', label='Prefix: k < 10', zorder=2)
    ax.scatter(right, y, s=43, color='#bc4637', marker='s', label='Suffix: 10 ≤ k < 40', zorder=3)
    ax.axvline(1.0, color='#38424f', linestyle='--', linewidth=1.2, label='Original bound: 1.0')
    ax.set_xlim(.45, 1.2)
    ax.set_xticks(np.arange(.5, 1.21, .1))
    ax.set_xlabel('Maximum absolute joint acceleration (rad/s²)')
    ax.set_title(title, loc='left', weight='bold')
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#e3e8ed', linewidth=.8)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
axes[0].set_yticks(y, labels)
handles, legend_labels = axes[1].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc='lower center', bbox_to_anchor=(.60,.08), ncol=3, frameon=False, fontsize=10)
fig.suptitle('Original 16 joint rejections: prefix versus suffix', x=.02, ha='left', weight='bold', fontsize=16)
fig.text(.02, .925, 'Any-joint exceedance: prefix 6/16, suffix 12/16, both 2/16. First exceedance: prefix 6, suffix 10.', fontsize=11)
fig.text(.02, .025, 'k is the rightmost H40 knot in each second difference; k=0/1 includes the saved two-point reference history.\nRead-only clone diagnostic. Prefix values within the bound do not grant native execution. Full H40 checks remain unchanged.', fontsize=9, color='#3b4652')
fig.subplots_adjust(left=.225, right=.98, top=.875, bottom=.20, wspace=.13)
for extension in ('png', 'svg'):
    fig.savefig(ROOT/f'JOINT_SUFFIX_20261114.{extension}', dpi=180, facecolor='white')
