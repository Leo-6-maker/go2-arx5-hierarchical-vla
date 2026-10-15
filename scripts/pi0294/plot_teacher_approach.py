"""Offline scientific figures from a saved, bounded progress-audit export."""
import argparse
import json
from pathlib import Path
import numpy as np


def run(source,target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=json.loads(Path(source).read_text(encoding='utf-8'));target=Path(target);target.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})
    cases=data['plots'];fig,axes=plt.subplots(3,3,figsize=(12,9),sharex=True,layout='constrained')
    for column,case in enumerate(cases):
        rows=case['rows'];t=np.asarray([r['time_s'] for r in rows]);actual=np.asarray([r['actual_xyz'] for r in rows]);goal=np.asarray([r['target_xyz'] for r in rows])
        reference=np.asarray([[np.nan]*3 if r['same_time_reference_xyz'] is None else r['same_time_reference_xyz'] for r in rows])
        for component in range(3):
            ax=axes[component,column]
            ax.plot(t,goal[:,component],color='#20252b',ls='--',lw=1.5,label='Persistent stage target')
            ax.plot(t,reference[:,component],color='#df8423',lw=1.6,label='Same-time previous installed reference')
            ax.plot(t,actual[:,component],color='#176baf',lw=1.1,label='Actual TCP')
            ax.grid(alpha=.18);ax.set_ylabel('XYZ'[component]+' world (m)');ax.set_xlim(0,32)
            if component==0:ax.set_title(case['case'].replace('TASK-PILOT-','').replace('-engineering04',''))
            if component==2:ax.set_xlabel('Actual task time (s)')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=9)
    fig.suptitle('Saved full-body pilots: stage goal, same-time reference and actual TCP\nNo new rollout; reference uses the previous transaction, never a future H40 endpoint',fontsize=12)
    fig.savefig(target/'APPROACH_WORLD_XYZ.png');fig.savefig(target/'APPROACH_WORLD_XYZ.pdf');plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(12,9),sharex=True,layout='constrained')
    for column,case in enumerate(cases):
        rows=case['rows'];t=np.asarray([r['time_s'] for r in rows])
        for row,key in enumerate(('stage_position_error_m','stage_angle_error_rad','station_xy_error_m')):
            ax=axes[row,column];ax.plot(t,[r[key] for r in rows],color='#176baf',lw=1.5);ax.grid(alpha=.18);ax.set_xlim(0,32)
            if row==0:ax.axhline(.004,color='#a32732',ls='--',lw=1,label='Original 4 mm stage predicate');ax.set_ylabel('Stage position distance (m)');ax.set_ylim(bottom=0);ax.set_title(case['case'].replace('TASK-PILOT-','').replace('-engineering04',''))
            if row==1:ax.axhline(.04,color='#a32732',ls='--',lw=1);ax.set_ylabel('Stage orientation distance (rad)');ax.set_ylim(bottom=0)
            if row==2:ax.set_ylabel('BASE station XY distance (m)');ax.set_ylim(bottom=0);ax.set_xlabel('Actual task time (s)')
    fig.suptitle('Original approach predicates and actual BASE station progress\n0.04 rad is a Push transition predicate; Pick uses position only. No success threshold changed.',fontsize=12)
    fig.savefig(target/'APPROACH_PREDICATES_BASE.png');fig.savefig(target/'APPROACH_PREDICATES_BASE.pdf');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--target',required=True);a=p.parse_args();run(a.source,a.target)
