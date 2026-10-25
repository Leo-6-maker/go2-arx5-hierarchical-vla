"""Reproduce first-panel summaries from the published paired categorical CSV.

This does not rescore trajectories or validate private raw evidence. The
original paired_development reader and independent native closure joins supply
those qualifications. No model, simulator, network or native operations occur.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def reproduce(directory):
    result = json.loads((directory/'V4A_A24_B24_COMPLETE_PAIRED_RESULT_AND_CLOSURE_ACTUAL_20261025.json').read_text())
    raw = (directory/'V4A_A24_B24_N294101_PAIRED_RESULTS_20261025.csv').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == result['paired_csv_sha256']
    rows = list(csv.DictReader(raw.decode().splitlines()))
    assert len(rows) == len({r['condition_id'] for r in rows}) == 24
    groups = sorted({r['scene_group'] for r in rows})
    assert len(groups) == 12 and all(r['noise_bank'] == '294101' for r in rows)
    indexed = {(r['scene_group'], r['task']): r for r in rows}
    assert set(indexed) == {(g, t) for g in groups for t in ('T1', 'T2')}
    blocks = np.asarray([[int(indexed[g,t]['B_success'])-int(indexed[g,t]['A_success']) for t in ('T1','T2')] for g in groups], dtype=float)
    blocks = np.column_stack([blocks, blocks.mean(axis=1)])
    samples = np.random.default_rng(20261024).integers(0,12,size=(2000,12))
    means = blocks[samples].mean(axis=1)
    names = ('T1_B_minus_A','T2_B_minus_A','macro_B_minus_A')
    effects = dict(zip(names, blocks.mean(axis=0).tolist()))
    intervals = {n: np.quantile(means[:,i],[.025,.975]).tolist() for i,n in enumerate(names)}
    assert effects == result['comparison']['effect_estimates']
    assert intervals == result['comparison']['intervals']
    counts = {a:{t:sum(int(r[a+'_success']) for r in rows if r['task']==t) for t in ('T1','T2')} for a in ('A','B')}
    for label,arm in [('A','A_CONTROL'),('B','B_COVERAGE')]:
        for t in ('T1','T2'):
            assert counts[label][t] == result['comparison']['by_candidate_task'][arm][t]['successes']
    return counts,effects,intervals


def figure(counts,effects,intervals,directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.size':10,'svg.hashsalt':'V4A-N294101-20261025','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.4),gridspec_kw={'width_ratios':[1,1.2]})
    x=np.arange(3);width=.34
    for offset,label,color in [(-width/2,'A','#277DA1'),(width/2,'B','#D45D32')]:
        nums=[counts[label]['T1'],counts[label]['T2'],sum(counts[label].values())]
        totals=[12,12,24]
        bars=axes[0].bar(x+offset,[v/n for v,n in zip(nums,totals)],width,label=label+'4096',color=color)
        for bar,v,n in zip(bars,nums,totals):axes[0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+.025,str(v)+'/'+str(n),ha='center',fontsize=9)
    axes[0].set_xticks(x,['T1 pick/place','T2 push','Overall'])
    axes[0].set_ylim(0,.77);axes[0].yaxis.set_major_formatter(PercentFormatter(1));axes[0].set_ylabel('Complete task success');axes[0].legend(frameon=False,loc='upper right')
    names=['T1_B_minus_A','T2_B_minus_A','macro_B_minus_A']
    for y,name in enumerate(names):
        point=100*effects[name];low,high=[100*v for v in intervals[name]]
        axes[1].errorbar(point,y,xerr=[[point-low],[high-point]],fmt='o',color='#34495E',capsize=4)
        axes[1].text(18,y,f'{point:.1f} [{low:.1f}, {high:.1f}]',ha='right',va='bottom',fontsize=8)
    axes[1].axvline(0,color='#777777',linestyle='--',linewidth=1)
    axes[1].set_yticks(range(3),['T1','T2','Macro']);axes[1].invert_yaxis();axes[1].set_xlim(-68,20);axes[1].set_ylim(2.55,-.55)
    axes[1].set_xlabel('B minus A success (percentage points)')
    fig.suptitle('A4096 vs B4096: original first DEV noise panel N294101',fontsize=12)
    fig.text(.5,.025,'12 paired geometry blocks; 95% percentile intervals, 2,000 draws, seed 20261024.\nOne development panel: not full DEV48, independent confirmation, or model selection.',ha='center',fontsize=8,color='#444444')
    fig.tight_layout(rect=[0,.12,1,.94])
    directory.mkdir(parents=True,exist_ok=True)
    stem=directory/'V4A_A24_B24_N294101_RESULTS_20261025'
    fig.savefig(str(stem)+'.png',dpi=180)
    fig.savefig(str(stem)+'.svg',metadata={'Date':None})
    fig.savefig(str(stem)+'.pdf',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).parent)
    parser.add_argument('--figure-dir',type=Path)
    args=parser.parse_args()
    counts,effects,intervals=reproduce(args.data_dir)
    if args.figure_dir:figure(counts,effects,intervals,args.figure_dir)
    print(json.dumps(dict(status='PASS_CATEGORICAL_CSV_REPRODUCES_ORIGINAL_PAIRED_MATH',counts=counts,effects=effects,intervals=intervals,new_native_operations=0)))
