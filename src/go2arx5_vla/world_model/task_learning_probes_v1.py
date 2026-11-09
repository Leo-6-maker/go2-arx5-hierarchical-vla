"""Fixed real decision probes; phase metadata selects rows, never model inputs."""
from . import task_learning_grasp_v1 as task
from .task_learning_export_v1 import PROFILE
from pathlib import Path

PHASES=('PREGRASP','DESCEND','CLOSE','LIFT','HOLD')


def contract():
    return dict(schema='TASK_LEARNING_FIXED_REAL_PHASE_PROBES_V1',
        phases=list(PHASES),first_real_supervised_anchor_in_each_phase=True,
        TRAIN='FIRST_FOUR_NORMAL_AND_FIRST_FOUR_GENUINE_RECOVERY_EPISODES_ON_DISTINCT_LAYOUTS_PER_GROUP',
        DEV='FOUR_FIXED_REFERENCE_EPISODES_ON_DISTINCT_LAYOUTS_NOT_BC',
        diagnostic_noise_seed_base=2026110900,metadata_in_student_inputs=False,
        repeated_optimizer_visits_are_not_new_queries=True)


def artifact(folder,row):
    name=row.get('sample_file',row['episode']+'.npz')
    expected=row['episode']+(f"-a{row['anchor']:02d}"if 'phase_audit_only'in row else'')+'.npz'
    task.need(name==expected and Path(name).name==name,'OWN_NUMBERED_REAL_PROBE_ARTIFACT')
    return Path(folder)/name


def matched_count(first,second):
    one,two=first['records'],second['records']
    if 'selection_contract'not in first and 'selection_contract'not in second:
        task.need(len(one)==len(two)==4,'ORIGINAL_FOUR_MATCHED_PROBES')
        return 4
    task.need(first.get('selection_contract')==second.get('selection_contract')==contract()
        and bool(one)and len(one)==len(two),'SAME_FIXED_REAL_PHASE_SELECTION_CONTRACT')
    split=one[0]['split'];count=40 if split=='TRAIN'else 20
    task.need(split in('TRAIN','DEV')and len(one)==count
        and all(v['split']==split for v in one+two)
        and len({(v['episode'],v['anchor'])for v in one})==count
        and [v['phase_audit_only']for v in one]==list(PHASES)*(count//5)
        and sum(v['genuine_recovery_audit_only']is True for v in one)==(20 if split=='TRAIN'else 0),
        'ALL_UNIQUE_PHASE_AND_RECOVERY_PROBES_RETAINED')
    task.need(all(all(a[k]==b[k]for k in('phase_audit_only','genuine_recovery_audit_only','sample_file'))
        for a,b in zip(one,two)),'SAME_REAL_PHASE_AND_ARTIFACT_ACROSS_RECONSTRUCTION')
    return count


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
