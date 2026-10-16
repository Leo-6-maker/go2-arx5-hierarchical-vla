"""Pure evidence aggregation: episode-macro probe errors and separate outcome classes."""
from collections import Counter
import statistics


def errors(rows):
    output={}
    for view,count in [('prefix8',8),('H40',40)]:
        output[view]={}
        for metric in ['XYZ_euclidean_m','SO3_rad','gripper_absolute']:
            episodes={}
            for row in rows:
                values=row['errors']['pointwise'][metric][:count]
                episodes.setdefault(row['episode_id'],[]).extend(values)
            valid={key:statistics.mean(values) for key,values in episodes.items() if values and all(v is not None for v in values)}
            output[view][metric]=dict(episodes=len(episodes),finite_complete_episodes=len(valid),
                episode_macro_mean=statistics.mean(valid.values()) if len(valid)==len(episodes) and valid else None,
                worst_episode_mean=max(valid.values()) if valid else None,
                expected_points=len(rows)*count,not_independent_samples=True)
    return output


def probe_summary(records):
    return {task:dict(windows=len(chosen),episodes=len({r['episode_id'] for r in chosen}),
        errors=errors(chosen),reference_gate_rejections=sum(r['gate']['decision']=='REJECT' for r in chosen),
        missing_close_events=sum(r['errors']['close_events']['missing'] for r in chosen),
        missing_open_events=sum(r['errors']['open_events']['missing'] for r in chosen))
        for task in ['T1','T2'] for chosen in [[r for r in records if r['task']==task]]}


def autonomous_summary(rows):
    result={}
    for label in sorted({r['model'] for r in rows}):
        result[label]={}
        for task in ['T1','T2']:
            chosen=[r for r in rows if r['model']==label and r['task']==task and r['model_autonomous_evaluation'] and not r['excluded_scope_deviation']]
            layouts={r['scene_group'] for r in chosen};complete=bool(len(chosen)==2 and layouts=={'PI0294/TRAIN/F0/000','PI0294/TRAIN/F1/000'})
            successes=sum(r['pipeline_trustworthy'] and r['persistent_success'] and r['native_safety_flag_rows']==0 and r['native_duration_s']==32. for r in chosen)
            result[label][task]=dict(conditions_attempted=len(chosen),planned_distinct_conditions=2,
                valid_policy_outcomes=sum(r['pipeline_trustworthy'] for r in chosen),task_successes=successes,
                failure_stages=dict(Counter(r['failure_stage'] for r in chosen)),
                complete_predeclared_panel=complete,
                source_task_competence_observed=complete and successes==2,
                DEV_generalization_proven=False,deployment_qualified=False)
    return result
