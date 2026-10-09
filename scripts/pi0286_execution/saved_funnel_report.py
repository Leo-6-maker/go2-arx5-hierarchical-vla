"""Aggregate existing independent scores; no model or simulator is loaded."""
import hashlib,json,statistics
from collections import Counter
from pathlib import Path

REPORT=Path('reports/PI0286/final')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def summarize(rows):
    out=Counter(requested=len(rows))
    for row in rows:
        s=row['score'];h=s['history'];p=s['subpredicates'];full=lambda k:p[k]['hold_pass']==41
        grasp=h['grasp'];lift=grasp and h['lift'];transfer=lift and h['horizontal_transfer']
        place=transfer and all(full(k) for k in ('zone','table','static'))
        release=place and full('not_grasped');withdraw=release and full('TCP_clear')
        for name,passed in [('grasp_history',grasp),('grasp_and_lift_history',lift),('acquisition_history',transfer),('acquisition_plus_stable_place',place),('previous_plus_release',release),('previous_plus_withdraw',withdraw),('complete_task',s['persistent_success'])]:
            out[name]+=int(passed)
        assert row['persistent_success']==s['persistent_success']
    chain=[out[k] for k in ('requested','grasp_history','grasp_and_lift_history','acquisition_history','acquisition_plus_stable_place','previous_plus_release','previous_plus_withdraw','complete_task')]
    assert all(a>=b for a,b in zip(chain,chain[1:])),chain
    return dict(out)


def main():
    source=REPORT/'ACTUAL_SCORE_SOURCE_READBACK.json';v=json.loads(source.read_text())
    old=Path('reports/PI0285/r3/WP2/REPAIR_A2048-2048/EPISODE_FUNNEL.json');f=json.loads(old.read_text())
    correction=Path('reports/PI0286/correction/ACTUAL_CORRECTION_ADMISSION_AND_FAILURES.json');c=json.loads(correction.read_text())
    native=Path('reports/PI0286/correction/BANK_NATIVE_CONTRACT_V2_COMPACT.json');n=json.loads(native.read_text())
    assert c['receipts']['source/evidence/CORRECTIONS32.json']['sha256']==v['CORRECTIONS32_SOURCE_SHA']
    panels={key:summarize(value['rows']) for key,value in v.items() if isinstance(value,dict) and 'rows' in value}
    teacher=[dict(layout=r['layout'],persistent_success=r['physical_success'],score=r['full_score'],slot=r['slot']) for r in c['slots']]
    for name,tick in [('CORRECTION3S',80),('CORRECTION12S',260)]:
        panels[name]=summarize([r for r,t in zip(teacher,c['slots']) if t['takeover_tick']==tick])
    failed=[r for r in f['rows'] if r['panel']=='CONFIRM_A' and not r['persistent_success']]
    assert f['proofs']['CONFIRM_A']=={k:v['CONFIRM_A'][k] for k in ('source_sha256','CPU_sha256')}
    categories=Counter('ACQUISITION_HISTORY_INCOMPLETE' if not all(r['history'].values()) else 'LATE_TASK_HOLD_FAILURE' for r in failed)
    assert categories=={'ACQUISITION_HISTORY_INCOMPLETE':6,'LATE_TASK_HOLD_FAILURE':6}
    late=[r for r in v['CONFIRM_A']['rows'] if not r['persistent_success'] and all(r['score']['history'].values())]
    hold_failures={k:sum(r['score']['subpredicates'][k]['hold_pass']!=41 for r in late) for k in ('zone','table','static','not_grasped','TCP_clear','green_unmoved')}
    associations=[]
    for key,passed in [('CONFIRM_SUCCESS',True),('CONFIRM_FAIL',False)]:
        rows=[r for r in f['rows'] if r['panel']=='CONFIRM_A' and r['persistent_success']==passed]
        associations.append(dict(group=key,n=len(rows),with_IK_holds=sum(r['IK_holds']>0 for r in rows),zero_IK_holds=sum(r['IK_holds']==0 for r in rows),gate_rejects=sum(r['gate_rejects'] for r in rows),reference_to_installed_XYZ_RMS_m_median=statistics.median(r['reference_to_installed_XYZ_RMS_m'] for r in rows),installed_to_next_50ms_achieved_XYZ_RMS_m_median=statistics.median(r['installed_to_same_issued_achieved_XYZ_RMS_m'] for r in rows)))
    parent={r['layout']:r['persistent_success'] for r in v['EXPLORATORY_TRAIN16']['rows']};admitted=set(n['valid_slots']);paired=[]
    for passed in (True,False):
        rows=[r for r in teacher if parent[r['layout']]==passed]
        paired.append(dict(parent_32s_task_success=passed,requested_branches=len(rows),teacher_32s_task_success=sum(r['persistent_success'] for r in rows),native_codec_valid=sum(r['slot'] in admitted for r in rows)))
    assert panels['CONFIRM_A']['complete_task']==20 and len(teacher)==32 and n['valid_branches']==27
    result=dict(status='PASS_SAVED_SCORE_FUNNEL_AND_PAIRED_TEACHER_DESCRIPTIVE_REPORT',panels=panels,old_confirmation_failure_categories=dict(categories),late_failure_predicates=dict(episodes=len(late),counts=hold_failures,overlapping_counts_not_independent_causes=True),IK_and_tracking_association=associations,parent_outcome_conditioned_teacher_branches=paired,method=dict(history='ORIGINAL_SUSTAINED_CONTACT_LIFT_AND_HORIZONTAL_HISTORY',cumulative='PREFIX_HISTORY_THEN_ALL41_FINAL_HOLD_SAMPLES_ZONE_TABLE_STATIC_RELEASE_TCP_CLEAR',release_and_withdraw_not_counted_by_first_true_at_reset=True,correction_history_includes_faithful_replayed_model_prefix=True,tracking_time='INSTALLED_COMMAND_TO_ACHIEVED_AFTER_SAME_50MS_CONTROL_INTERVAL',causal_learning_comparison='UNOBSERVED_C0_C1_NOT_EXECUTED',heldout_confirmation='OLD_EXPOSED_CONFIRM_A_DESCRIPTIVE_ONLY'),sources={str(p):sha(p) for p in (source,old,correction,native)},new_neural_queries=0,new_physics=0,new_optimizer_updates=0,selected='NONE',deployment='NONE')
    (REPORT/'SAVED_FUNNEL_REPORT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
