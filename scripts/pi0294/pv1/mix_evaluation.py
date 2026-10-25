"""C/D's one original DEV noise panel; pure plan, no runtime admission."""
from copy import deepcopy
from .mix1024 import ARMS
from .dataset import POPULATION
from .plan import evaluation_cards
from .recipe import PARENT
from scripts.pi0294.e1.scope import STATS
from .successor_cd import TASK,TASK_SOURCE,ORDER


def panel(groups,arm):
    if arm not in ARMS:raise PermissionError('V5_ONLY_C50_D25_EVAL')
    label=arm+'-DEV-N294101';cards=[]
    for base in evaluation_cards(groups,'DEV'):
        if base['noise_bank']!=294101:continue
        c=deepcopy(base);c.update(id=arm+'-'+base['id'],policy_candidate=arm,
            pv1_model_label=arm,pv1_session_label=label)
        cards.append(c)
    if len(cards)!=24 or len({c['scene_group']for c in cards})!=12:
        raise PermissionError('V5_ORIGINAL_TWELVE_GEOMETRIES_TWO_TASKS')
    return dict(candidate=arm,checkpoint_step=1024,split='DEV',noise_bank=294101,
        noise_namespace='PI0294/PV1/DEV/N294101',cards=cards,Teacher_takeover=False,
        complete_task_duration_s=32)


def plan(groups):
    return dict(schema='V5_CD_FROZEN_PAIRED_FIRST_DEV_PANEL_V1',task=TASK,
        task_source=TASK_SOURCE,order=list(ORDER),population_plan_sha256=POPULATION,
        parent_checkpoint_manifest_sha256=PARENT,statistics_sha256=STATS,
        panels={arm:panel(groups,arm)for arm in ARMS},regular_conditions=48,
        engineering_replacements_per_condition_max=1,engineering_replacements_total_max=4,
        valid_policy_outcomes_repeated=False,original_S1_and_score=True,
        Teacher_takeover=False,evaluation_data_for_training=False,
        analysis=dict(primary='T2_D_MINUS_C',retention='T1_D_MINUS_C',
            bootstrap_unit='JOINT_T1_T2_GEOMETRY_BLOCK',blocks=12,replicates=2000,
            seed=20261024,evidence='EXPOSED_DEVELOPMENT_NOT_INDEPENDENT_GENERALIZATION'),
        new_training_candidates=0,new_admission=False)


def session_spec(root,label):
    from pathlib import Path
    from scripts.pi0292.durable import read
    from scripts.pi0294.native_dataset import sha
    r=Path(root);m=read(r/'RUN_MANIFEST.json');name='plans/V5_CD_PAIRED_EVALUATION_PLAN.json'
    if (sha(r/'plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json')!=POPULATION
        or m['execution']['sealed_inputs'].get(name)!=sha(r/name)):
        raise PermissionError('V5_FROZEN_REAL_POPULATION_AND_PAIRED_PLAN')
    expected=plan(read(r/'plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json')['groups'])
    if read(r/name)!=expected:raise PermissionError('V5_NO_ADAPTIVE_CD_PANEL')
    matches=[v for v in expected['panels'].values()if v['cards'][0]['pv1_session_label']==label]
    if len(matches)!=1:raise PermissionError('V5_ONLY_ONE_ORIGINAL_NOISE_BANK_PER_ARM')
    return matches[0]
