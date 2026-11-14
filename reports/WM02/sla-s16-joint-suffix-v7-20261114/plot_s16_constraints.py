"""Plot fixed S16 endpoint errors and original EEF limits from sealed JSON."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
data=json.loads((ROOT/'S16_ACTUAL_20261114.json').read_text())
rows=data['predictions'];limits=data['EEF_geometric_limits']
fig,axes=plt.subplots(1,2,figsize=(11,5.5))
for ax,xkey,ykey,xscale,xlabel,title in [
    (axes[0],'xyz_RMSE_m','maximum_discrete_translation_acceleration_m_s2',1000,'XYZ RMSE (mm)','Translation acceleration'),
    (axes[1],'nominal_grip_MAE','maximum_finger_speed_m_s',1,'Nominal gripper MAE','Mechanical finger speed'),
]:
    for passed,color,label in [(True,'#2369ad','Full EEF H40 passes'),(False,'#bc4637','Full EEF H40 rejected')]:
        selected=[r for r in rows if r['EEF_full_H40_pass']==passed]
        ax.scatter([r[xkey]*xscale for r in selected],
            [r['geometric_metrics'][ykey]/limits[ykey] for r in selected],
            s=50,color=color,label=label,alpha=.85,edgecolor='white',linewidth=.5,zorder=3)
    ax.axhline(1,color='#38424f',linestyle='--',linewidth=1.2)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(xlabel);ax.set_ylabel('Peak value / original limit')
    ax.set_title(title,loc='left',weight='bold');ax.set_axisbelow(True)
    ax.grid(color='#e3e8ed',linewidth=.8);ax.spines[['top','right']].set_visible(False)
axes[1].axvline(.02,color='#76818c',linestyle=':',linewidth=1.2)
axes[1].text(.0205,.04,'MAE target: 0.02',fontsize=9,color='#56616d',rotation=90,va='bottom',transform=axes[1].get_xaxis_transform())
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.105),ncol=2,frameon=False)
fig.suptitle('S16 fixed500: position fit and H40 constraint failures',x=.02,ha='left',fontsize=15,weight='bold')
fig.text(.02,.91,'32 fixed predictions · full EEF H40 passes 24/32 · all fit targets met 21/32 · XYZ RMSE max 0.765 mm',fontsize=10)
fig.text(.02,.025,'Original limits: translation acceleration 0.25 m/s²; mechanical finger speed 0.04 m/s. Failures overlap.\nOffline EEF geometry only; no native joint certificate or new physical requests. No smoothing, clipping, or threshold changes.',fontsize=9,color='#3b4652')
fig.subplots_adjust(left=.085,right=.98,top=.83,bottom=.28,wspace=.28)
for extension in ('png','svg'):
    target=ROOT/f'S16_CONSTRAINTS_20261114.{extension}'
    fig.savefig(target,dpi=180,facecolor='white')
    if extension=='svg':
        target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
