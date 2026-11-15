"""Fixed new-geometry PICK_HOLD development inputs, separate from old DEV/TEST.

This early one-object population is for complete3K policies. It does not
replace the later two-task/five-object evaluation or C24 mechanism panel.
"""
import json
from pathlib import Path
import subprocess
import numpy as np

from .storage import digest
from .stable_task_scene_v1 import pinned
from . import task_learning_grasp_v1 as task

OFFSETS={f'SLAV5DEV{i+1:02d}':(x,y)for i,(x,y)in enumerate(
    (x,y)for x in(-.0031,-.0011,.0013,.0031)for y in(-.0028,.0028))}
SPLIT='SLA_V5_DEV'
POPULATION='PICK_HOLD_DEV48'
DECLARATION='experiments/sla_go2_posttrain/raw3k_pick_hold_dev48_inputs_v5.json'
DECLARATION_SCHEMA='SLA_RAW3K_PICK_HOLD_NEW_GEOMETRY_DEV48_INPUTS_V5'
SCHEMA='SLA_RAW3K_PICK_HOLD_NEW_GEOMETRY_DEV48_PANEL_V5'
INPUT_SCHEMA='SLA_RAW3K_PICK_HOLD_NEW_GEOMETRY_DEV48_GROUP_INPUT_V5'
EXPERIMENT='FIXED_NEW_GEOMETRY_PICK_HOLD_DEV48_NOT_TWO_TASK_TEST_OR_C24'
STATIC_OPERATION='SLA_V5_DEV48_INPUT_STATIC_PREPARATION'
STATIC_SCHEMA='SLA_V5_DEV48_STATIC_INPUT_PREPARATION_ACTUAL'
DISJOINT_SCHEMA='SLA_V5_DEV48_ACTUAL_SCENE_DISJOINTNESS'


def declaration():
    p=Path(__file__).resolve().parents[3]/DECLARATION
    return json.loads(p.read_bytes()),digest(p)


def inputs():
    d,_=declaration()
    task.need(d['schema']==DECLARATION_SCHEMA and d['population']==POPULATION
        and d['layouts']=={k:list(v)for k,v in OFFSETS.items()}
        and d['restart_ids']==[1,2,3,4,5,6] and d['maximum_queries_per_condition']==60
        and d['conditions_per_method']==48 and d['prefix_atoms']==10 and d['final_hold_atoms']==40
        and d['raw_H40_preview_atoms']==40 and d['physics_dt_s']==.002
        and d['seed_rule']=='202611150000 + layout_index0*10000 + restart_id*100 + query_index0'
        and d['old_DEV_TEST_relabelled'] is False and d['training_admitted'] is False,
        'NEW_DEV48_FIXED_INPUTS_NOT_OLD_EVALUATION_OR_TRAINING')
    return [dict(layout=l,restart=r,query_seeds=list(range(202611150000+i*10000+r*100,
                  202611150000+i*10000+r*100+60)))for i,l in enumerate(OFFSETS)for r in range(1,7)]


def disjointness(pin):
    proof=pinned(pin);d,p=declaration()
    from .sla_population_v1 import geometry_sha,initial_geometry
    from .task_learning_scene_v1 import geometry
    task.need(proof['schema']==DISJOINT_SCHEMA and proof['passes']
        and proof['preregistration']['sha256']==p and pinned(proof['preregistration'])==d
        and len(proof['new_initial_inputs'])==8
        and {r['layout']for r in proof['new_initial_inputs']}==set(OFFSETS)
        and proof['old_TEST_initial_inputs']>0 and proof['original_physical_scenes_unchanged']
        and proof['old_TEST_scores_read'] is proof['old_TEST_images_actions_trajectories_read'] is False
        and proof['physics_integrations']==proof['model_calls']==proof['optimizer_updates']==0,
        'ACTUAL_NEW_DEV48_INITIAL_GEOMETRY_DISJOINTNESS_WITHOUT_TEST_OUTPUTS')
    task.need(all(r['initial_geometry_sha256']==geometry_sha(geometry(r['layout']))
        and len(r['model_sha256'])==64 and all(np.allclose(
            r['actual'][k],v,rtol=0,atol=1e-12)for k,v in initial_geometry(geometry(r['layout'])).items())
        for r in proof['new_initial_inputs']),'DEV48_ACTUAL_GEOMETRY_BINDS_THE_DECLARED_EIGHT_LAYOUTS')
    return proof


def validate_panel(panel):
    from . import sla_dense_native_v5 as dense
    d,p=declaration();rows=panel.get('assignments')or[];method=panel.get('method')
    task.need(panel['schema']==SCHEMA and panel['task_learning_role']==dense.PROTOCOL
        and panel['protocol']==dense.protocol() and panel['B1_global_step']==3000
        and panel['population']==POPULATION and method in dense.METHODS+(dense.NAMED_METHOD,dense.reference.METHOD)
        and set(panel['policies'])=={method} and panel['policies'][method]['arm']==method
        and panel['preregistration']['sha256']==p and pinned(panel['preregistration'])==d
        and len(rows)==len({r['native_run']for r in rows})==48,
        'COMPLETE_FIXED_NEW_DEV48_RAW3K_POPULATION')
    identities={r['layout']:r for r in disjointness(panel['scene_disjointness'])['new_initial_inputs']}
    for row,expected in zip(rows,inputs(),strict=True):
        task.need(all(row[k]==expected[k]for k in ('layout','restart','query_seeds'))
            and row['method']==method and row['split']==SPLIT
            and row['parent']=='TASK_LEARNING_UNDER_HAND_'+row['layout']
            and row['command']==task.COMMAND and row['purpose']==EXPERIMENT
            and row['model_sha256']==identities[row['layout']]['model_sha256']
            and Path(row['native_run']).name==row['native_run'],
            'DEV48_NO_LAYOUT_RESTART_NOISE_SCORE_OR_SPLIT_SELECTION')
    return rows


def retained_prefix(inp,panel):
    """Use actual closed earlier groups of this same panel, never replay a row."""
    proof=inp.get('population_predecessor')
    if proof is None:return []
    result=pinned(proof);root=Path(proof['path']).parent
    old=json.loads((root/'INPUT.private.json').read_bytes());lease=json.loads((root/'control/LEASE.json').read_bytes())
    rows=json.loads((root/'OUTCOMES.private.json').read_bytes())
    task.need(proof['path']==str(root/'RESULT.private.json')
        and old['schema']==INPUT_SCHEMA and old['panel']==inp['panel']
        and old['panel_sha256']==inp['panel_sha256']==digest(inp['panel'])
        and old['source_sha']==inp['source_sha']==panel['source_sha']==result['source_sha']
        and old['assignment_slice'][1]==inp['assignment_slice'][0]==len(rows)
        and lease['input_sha256']==digest(root/'INPUT.private.json')
        and lease['status']=='CLOSED'and lease['worker_exit_code']==result['exit_code']==0
        and result['unresolved_pair']is None and result['cumulative_completed']==len(rows)
        and result['summary']['requested']==48 and not result['summary']['complete']
        and all(r['technical_valid']and not r['hard_safety_stop']for r in rows),
        'DEV48_ACTUAL_CLOSED_CONTIGUOUS_GROUPS_NOT_FAILURE_RETRY')
    expected=validate_panel(panel)
    task.need(all(all(r[k]==expected[i][k]for k in
        ('native_run','layout','restart','method','query_seeds','split','parent'))for i,r in enumerate(rows)),
        'DEV48_RETAIN_EVERY_ORIGINAL_CONDITION_AND_NEGATIVE')
    task.need(subprocess.check_output(['systemctl','--user','show','wm02-'+root.name,
        '--property=MainPID','--value'],text=True,timeout=10).strip()=='0','DEV48_PRIOR_GROUP_ACTUALLY_COLD')
    # Older rows may belong to still-earlier groups. Validate their closed
    # custody recursively rather than inventing run paths in this last group.
    earlier=retained_prefix(old,panel)
    task.need(len(earlier)==old['assignment_slice'][0]and rows[:len(earlier)]==earlier,
              'DEV48_UNCHANGED_PRIOR_ACTUAL_PREFIX')
    for row in rows[len(earlier):]:
        for tail in('','-gpu'):
            r=root/'runs'/(row['native_run']+tail)
            end=json.loads((r/'control/LEASE.json').read_bytes());worker=json.loads((r/'WORKER_EXIT.private.json').read_bytes())
            task.need(end['status']=='CLOSED'and end['worker_exit_code']==worker['exit_code']
                and isinstance(worker['exit_code'],int)and worker['exit_code']>=0
                and end['spec_sha256']==digest(r/'EXECUTION_SPEC.json'),'DEV48_BOTH_PREFIX_PARTNERS_TERMINAL')
    task.need(all(not(root/'runs'/(r['native_run']+tail)).exists()
        for r in expected[len(rows):]for tail in('','-gpu')),'DEV48_LATER_ROWS_NEVER_STARTED_IN_PREDECESSOR')
    return rows


def selection(inp,panel):
    rows=validate_panel(panel);span=inp['assignment_slice']
    task.need(inp['schema']==INPUT_SCHEMA and inp['source_sha']==panel['source_sha']
        and len(span)==2 and all(type(v)is int for v in span)
        and 0<=span[0]<span[1]<=48 and span[1]-span[0]<=8,
        'DEV48_FINITE_CONTIGUOUS_AT_MOST_EIGHT_CONDITIONS_PER_GROUP')
    prior=retained_prefix(inp,panel)
    task.need(len(prior)==span[0] and (bool(prior)==bool(inp.get('population_predecessor'))),
        'DEV48_EVIDENCED_FULL_PREFIX_BEFORE_NEXT_UNSTARTED_SLICE')
    return rows[span[0]:span[1]]


def scene_custody(producer,reference,model):
    from .sla_native_v1 import closed
    from .task_learning_scene_v1 import MEMBER
    source=closed(producer);root=Path(producer['path']).parent;proof=pinned(reference)
    actual=json.loads((root/'RESULT.private.json').read_bytes());d,p=declaration()
    task.need(proof['layout']in OFFSETS and proof['split']==SPLIT
        and source['operation']==STATIC_OPERATION and source['preregistration']['sha256']==p
        and pinned(source['preregistration'])==d
        and source['source_files'][MEMBER]==proof['builder_sha256']==digest(Path(source['source_root'])/MEMBER)
        and actual['schema']==STATIC_SCHEMA and actual['passes']
        and actual['physics_integrations']==actual['model_calls']==actual['optimizer_updates']==0
        and actual['scenes'][proof['layout']]==reference
        and digest(model)==proof['compiled']['scene_sha256'],'DEV48_CLOSED_ACTUAL_STATIC_PRODUCER_AND_MODEL_BYTES')
    disjointness(actual['scene_disjointness'])
    return proof['builder_sha256']
