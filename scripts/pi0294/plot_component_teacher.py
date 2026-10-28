"""V2 saved-array scientific figure, no simulation or model."""
import json
from pathlib import Path
import numpy as np


def run(source,target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=json.loads(Path(source).read_text());target=Path(target);target.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})
    fig,axes=plt.subplots(2,len(data),figsize=(12,7),layout='constrained')
    for col,c in enumerate(data):
        t=np.asarray(c['time_s']);q=np.asarray(c['query_time_s']);cursor=np.asarray(c['cursor_xyz']);goal=np.asarray(c['goal_xyz'])
        ax=axes[0,col];ax.plot(t,c['goal_distance_m'],color='#176baf',label='Actual TCP to stage goal')
        ax.plot(q,np.linalg.norm(cursor-goal,axis=1),color='#d17c19',marker='o',ms=2,label='Planned cursor to stage goal')
        ax.set_title(c['id'].replace('V2_DEV_','').replace('_',' / '));ax.set_ylabel('Position distance (m)');ax.set_ylim(bottom=0)
        ax=axes[1,col];ax.plot(t,c['goal_angle_rad'],color='#176baf');ax.set_ylabel('Actual goal orientation error (rad)');ax.set_ylim(bottom=0)
        for row in range(2):axes[row,col].grid(alpha=.2);axes[row,col].set_xlim(0,12);axes[row,col].set_xlabel('Actual task time (s)')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2)
    fig.suptitle('V2: observed orientation progress, no precontact or object contact\nThree valid 12s short attempts; W0/T2 stopped before Teacher and has no task arrays')
    fig.savefig(target/'V2_ACTUAL_POSITION_AND_ORIENTATION.png');fig.savefig(target/'V2_ACTUAL_POSITION_AND_ORIENTATION.pdf');plt.close(fig)
