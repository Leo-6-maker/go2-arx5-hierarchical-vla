"""Fixed real decision probes; phase metadata selects rows, never model inputs."""
from . import task_learning_grasp_v1 as task
from .task_learning_export_v1 import PROFILE

PHASES=('PREGRASP','DESCEND','CLOSE','LIFT','HOLD')


def contract():
    return dict(schema='TASK_LEARNING_FIXED_REAL_PHASE_PROBES_V1',
        phases=list(PHASES),first_real_supervised_anchor_in_each_phase=True,
        TRAIN='FIRST_FOUR_NORMAL_AND_FIRST_FOUR_GENUINE_RECOVERY_EPISODES_ON_DISTINCT_LAYOUTS_PER_GROUP',
        DEV='FOUR_FIXED_REFERENCE_EPISODES_ON_DISTINCT_LAYOUTS_NOT_BC',
        diagnostic_noise_seed_base=2026110900,metadata_in_student_inputs=False,
        repeated_optimizer_visits_are_not_new_queries=True)


def selections(data):
    if data.manifest.get('profile')!=PROFILE:
        seen=set();result=[]
        for index,(row,anchor)in enumerate(data.rows):
            if row['id']in seen:continue
            seen.add(row['id']);result.append((index,None))
            if len(result)==4:break
        return result
    # Optimization repeats do not add observations, and a masked intervention
    # cannot be a probe. Each TRAIN group retains its own layout coverage.
    episodes={}
    for recovery in ((False,True)if data.split=='TRAIN'else(False,)):
        layouts=set();group={}
        for row,anchor in data.rows:
            if (row.get('genuine_recovery_success')is True)!=recovery:continue
            if row['id']in group or row['parent']in layouts:continue
            layouts.add(row['parent']);group[row['id']]=row
            if len(group)==4:break
        task.need(len(group)==4,'FOUR_FIXED_DISTINCT_LAYOUTS_PER_REAL_PHASE_PROBE_GROUP')
        episodes.update(group)
    indexes={}
    for index,(row,anchor)in enumerate(data.rows):
        if row['id']in episodes:indexes.setdefault((row['id'],anchor),index)
    result=[]
    for name,row in episodes.items():
        stages=row['expert_stage_audit_only'];eligible=row['supervised_anchors']
        for phase in PHASES:
            anchors=[a for a in eligible if stages[a]==phase]
            task.need(bool(anchors),'REAL_SUPERVISED_DECISION_REQUIRED_FOR_PHASE_'+phase)
            anchor=min(anchors)
            task.need((name,anchor)in indexes,'DECLARED_PHASE_ANCHOR_PRESENT_IN_DATASET')
            result.append((indexes[name,anchor],phase))
    return result
