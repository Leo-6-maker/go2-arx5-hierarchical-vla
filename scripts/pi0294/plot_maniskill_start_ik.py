"""Figure from sealed candidate reconciliation; no planner or physics calls."""
import json
from pathlib import Path


def run(source,target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    target=Path(target);cases=json.loads(Path(source).read_text(encoding='utf-8'))['cases']
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True,layout='constrained')
    for ax,c in zip(axes,cases):
        values=[max((v['violation_rad_or_m'] for v in r['violations']),default=0.) for r in c['candidates']]
        colors=['#176baf' if r['numerical_IK_converged'] else '#929aa3' for r in c['candidates']]
        ax.bar(range(1,9),values,color=colors)
        for i,v in enumerate(values):ax.text(i+1,v+.025,f'{v:.3f}',ha='center',fontsize=8)
        converged=sum(r['numerical_IK_converged'] for r in c['candidates'])
        ax.set_title(f"{c['task']}: {converged}/8 CLIK converged, 0/8 admissible")
        ax.set_xlabel('Registered IK candidate');ax.set_xticks(range(1,9));ax.set_ylim(0,1.65);ax.grid(axis='y',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Maximum native joint-limit violation (rad)')
    fig.suptitle('Fixed start snapshots: saved IK candidates only — NO PHYSICS EXECUTED')
    fig.legend(handles=[Patch(color='#176baf',label='CLIK converged; rejected by original joint limits'),
        Patch(color='#929aa3',label='CLIK did not converge; also violates joint limits')],loc='outside lower center',ncol=1,fontsize=9)
    fig.savefig(target/'FIXED_START_IK_LIMIT_VIOLATIONS.png',dpi=160)
    fig.savefig(target/'FIXED_START_IK_LIMIT_VIOLATIONS.pdf');plt.close(fig)
