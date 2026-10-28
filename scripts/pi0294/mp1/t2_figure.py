"""Static event/diagnostic figure from sanitized, fixed scientific receipts."""
import argparse
import json
from pathlib import Path


def chart_rows(attribution,progress):
    if attribution['status']!='ACTUAL_EXACT_HISTORY_SHADOW_ATTRIBUTION_COMPLETE':raise ValueError('ACTUAL_ATTRIBUTION_REQUIRED')
    records={r['attempt_id']:r for r in progress['rows']};rows=[]
    for row in attribution['rows']:
        if row['maximum_actual_highrate_difference']!=0:raise ValueError('EXACT_HISTORY_REQUIRED')
        saved=records[row['attempt_id']];checks={r['candidate']:r for r in row['collision_checks']}
        expected=['MODEL','SAME_STATE_EXPERT','EXPERT_XYZ_ONLY','EXPERT_SO3_ONLY','EXPERT_GRIPPER_ONLY']
        if set(checks)!=set(expected):raise ValueError('ALL_FIVE_DIAGNOSTIC_PREFIXES_REQUIRED')
        rows.append(dict(layout=row['attempt_id'].split('-')[-3],contact=saved['progress']['first_saved_robot_contact_s'],
            zone=saved['progress']['first_goal_zone_all_corners_s'],stop=saved['original_duration_s'],
            depth_mm=-checks['MODEL']['minimum_reported_illegal_arm_surface_contact_distance_m']*1000,
            decisions=[checks[k]['decision'] for k in expected]))
    if [r['layout'] for r in rows]!=['F0','F1']:raise ValueError('EXACT_TWO_PREDECLARED_LAYOUTS')
    return rows


def render(attribution,progress,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    rows=chart_rows(attribution,progress)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(14,8.5),facecolor='white');grid=GridSpec(2,2,height_ratios=[1,1.15],figure=fig)
    fig.subplots_adjust(left=.09,right=.97,top=.84,bottom=.16,hspace=.55,wspace=.55)
    fig.suptitle('M1@1024 T2: Saved Progress and Exact-History Contact Diagnosis',x=.09,y=.97,ha='left',fontsize=18)
    fig.text(.09,.917,'Original EEF10 / D0 / S1 | Two fixed TRAIN layouts | No new autonomous task evaluation',fontsize=11,color='#555555')
    ax=fig.add_subplot(grid[0,:]);ax.set_title('A  Saved live events',loc='left',fontsize=13,pad=14)
    for i,row in enumerate(rows):
        y=1-i;ax.plot([0,row['stop']],[y,y],color='#444444',linewidth=2,zorder=2)
        ax.broken_barh([(row['stop'],32-row['stop'])],(y-.15,.3),facecolors='#eeeeee',edgecolors='#dddddd',hatch='///')
        for x,marker,color,label,dy in [(row['contact'],'o','#267d95',f"first sampled contact\n{row['contact']:.2f}s",-.32),
                (row['zone'],'D','#a57a16',f"in zone\n{row['zone']:.2f}s",.25),
                (row['stop'],'X','#933c4c',f"S1 reject\n{row['stop']:.2f}s",-.32)]:
            ax.scatter([x],[y],marker=marker,c=color,s=55,zorder=3)
            ax.text(x,y+dy,label,ha='center',va='center',fontsize=10,color=color)
        ax.text(22,y,'no later live observations',ha='center',va='center',color='#555555',fontsize=10)
    ax.axvspan(28,30,color='#cccccc',alpha=.4,zorder=0)
    ax.text(29,1.4,'final hold\nnot observed',ha='center',fontsize=9,color='#555555')
    ax.set(xlim=(0,32),ylim=(-.55,1.7),yticks=[1,0],yticklabels=['F0','F1'],xlabel='Original simulation time (s)',xticks=[0,4,8,12,16,20,24,28,32])
    ax.grid(axis='x',color='#e9e9e9');ax.set_axisbelow(True)

    bx=fig.add_subplot(grid[1,0]);bx.set_title('B  Diagnostic prefix8 results (0.4s maximum)',loc='left',fontsize=12,pad=18);bx.axis('off')
    names=['Model','Same-state expert','Expert XYZ only','Expert SO3 only','Expert gripper only']
    values=[[row['decisions'][i] for row in rows] for i in range(5)]
    colors=[['#267d95' if v=='PASS' else '#e4e5e8' for v in line] for line in values]
    table=bx.table(cellText=values,rowLabels=names,colLabels=['F0','F1'],cellColours=colors,loc='center',colWidths=[.27,.27])
    table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.65)
    for (i,j),cell in table.get_celld().items():
        cell.set_edgecolor('white')
        if i>0 and j>=0 and values[i-1][j]=='PASS':cell.get_text().set_color('white')

    cx=fig.add_subplot(grid[1,1]);cx.set_title('C  Forbidden pad-table overlap at first violation',loc='left',fontsize=12,pad=18)
    for i,row in enumerate(rows):
        y=1-i;cx.barh([y],[row['depth_mm']],height=.45,color='#267d95')
        cx.text(row['depth_mm']+.002,y,f"{row['depth_mm']:.4f} mm",va='center',fontsize=11)
    cx.set(xlim=(0,.06),ylim=(-.6,1.6),yticks=[1,0],yticklabels=['F0: left + right pad','F1: right pad'],xlabel='Maximum contact overlap (mm)')
    cx.grid(axis='x',color='#eeeeee');cx.set_axisbelow(True)
    fig.text(.09,.085,'Original ARM_TABLE_OR_FLOOR forbids any arm-table contact. The separate 4mm penetration rule was not triggered.',fontsize=10)
    fig.text(.09,.055,'Shadow diagnosis: 18,000 historical steps matched exactly; 10 prefix checks were not installed in the original live episodes.',fontsize=10)
    fig.text(.09,.025,'Expert / XYZ prefix PASS is not full recovery, task success, a Z-bias fix, or permission to relax S1.',fontsize=10,color='#555555')
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer();bounds=fig.bbox
    for text in fig.findobj(match=matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            b=text.get_window_extent(renderer)
            if b.width and b.height and (b.x0<0 or b.y0<0 or b.x1>bounds.width or b.y1>bounds.height):
                raise ValueError('FIGURE_TEXT_OUTSIDE_CANVAS:'+text.get_text())
    for header in fig.texts:
        for axis in fig.axes:
            for title in [axis.title,axis._left_title,axis._right_title]:
                if title.get_text() and header.get_window_extent(renderer).overlaps(title.get_window_extent(renderer)):
                    raise ValueError('FIGURE_HEADER_OR_FOOTNOTE_OVERLAPS_PANEL_TITLE')
    fig.savefig(output,dpi=180,facecolor='white');plt.close(fig)
    return dict(paired_TRAIN_parents=2,event_timelines_not_continuous_kinematics=True,rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attribution',required=True);p.add_argument('--progress',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    value=render(json.loads(Path(a.attribution).read_text()),json.loads(Path(a.progress).read_text()),a.output)
    print(json.dumps(value),flush=True)
