"""Scientific figures from frozen actual development arrays; no simulation."""
import argparse,json
from pathlib import Path
import numpy as np


def run(source,target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=json.loads(Path(source).read_text());target=Path(target);target.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ax,c in zip(axes.ravel(),data):
        t=np.array(c['time_s']);qt=np.array(c['query_time_s']);cursor=np.array(c['cursor_xyz']);goal=np.array(c['goal_xyz'])
        ax.plot(t,c['goal_distance_m'],color='#176baf',lw=1.8,label='Actual TCP to stage goal')
        ax.plot(qt,np.linalg.norm(cursor-goal,axis=1),color='#d17c19',marker='o',ms=3,label='Persistent cursor to stage goal')
        ax.set_title(c['id'].replace('NEW_TEACHER_DEV_','').replace('_',' / ')+'\n'+('Engineering incomplete' if 'ENGINEERING' in c['status'] else 'Valid no-progress FAIL'))
        ax.set_xlabel('Actual recorded task time (s)');ax.set_ylabel('Position distance (m)');ax.grid(alpha=.2)
        ax.set_xlim(0,5.3);ax.set_ylim(bottom=0)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2)
    fig.suptitle('Adopted persistent Teacher: four short development attempts\nNo safe precontact, object contact, or manipulation observed; no full-task claim')
    fig.savefig(target/'ACTUAL_PROGRESS_AND_CURSOR.png');fig.savefig(target/'ACTUAL_PROGRESS_AND_CURSOR.pdf');plt.close(fig)
    fig,axes=plt.subplots(3,4,figsize=(14,8),layout='constrained')
    for col,c in enumerate(data):
        t=np.array(c['time_s']);qt=np.array(c['query_time_s']);actual=np.array(c['actual_TCP_xyz']);cursor=np.array(c['cursor_xyz'])
        for k in range(3):
            ax=axes[k,col];ax.axhline(c['goal_xyz'][k],color='#333333',ls='--',label='Stage goal')
            ax.plot(qt,cursor[:,k],color='#d17c19',marker='o',ms=2,label='Persistent cursor')
            ax.plot(t,actual[:,k],color='#176baf',label='Actual TCP');ax.grid(alpha=.2)
            ax.set_ylabel('XYZ'[k]+' world (m)');ax.set_xlim(0,5.3)
            if k==0:ax.set_title(c['id'].replace('NEW_TEACHER_DEV_',''))
            if k==2:ax.set_xlabel('Actual task time (s)')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3)
    fig.suptitle('Persistent stage goal, planned cursor and actual TCP\nCursor has zero translational advance in all saved query snapshots; solver consumption identity UNKNOWN')
    fig.savefig(target/'WORLD_XYZ_CURSOR_ACTUAL.png');fig.savefig(target/'WORLD_XYZ_CURSOR_ACTUAL.pdf');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--target',required=True)
    a=p.parse_args();run(a.source,a.target)
