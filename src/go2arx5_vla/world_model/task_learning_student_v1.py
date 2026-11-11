"""Fixed new DEV conditions and unmodified decoded student H40 commands."""
from pathlib import Path
import numpy as np
from . import task_learning_grasp_v1 as task
from .task_learning_dev_v1 import DEV_OFFSETS
from .storage import digest

SCHEMA = 'TASK_LEARNING_NATIVE_STUDENT_DEV_PANEL_V1'
NATIVE_SCHEMA = 'TASK_LEARNING_NATIVE_STUDENT_DEV_ACTUAL_V1'
PURPOSE = 'NATIVE_STUDENT_DEV_NOT_BC'
EXPERIMENT = 'TASK_LEARNING_ORIGINAL_B0_VERSUS_SFT_NATIVE_DEV'


class StudentProposalRejected(ValueError):
    """An actual generated proposal outside the original reference domain."""


def validate(panel):
    from .task_learning_scene_v1 import scene_id
    rows = panel.get('assignments') or []
    policies=panel.get('policies')or{}
    task.need(panel.get('schema') == SCHEMA
        and panel.get('task_learning_role') == task.STUDENT_DEV_ROLE
        and panel.get('protocol') == task.protocol(task.STUDENT_DEV_ROLE)
        and len(panel.get('source_sha', '')) == 40
        and type(panel.get('B1_global_step'))is int and panel['B1_global_step']in(2000,4000,8000)
        and len(rows) == 80 and len({r['native_run'] for r in rows}) == 80
        and set(policies)=={'B0','B1'}
        and all(policies[m].get('arm')==m for m in policies),
        'FIXED_NEW_DEV20_TWO_RESTARTS_FOR_EACH_OF_TWO_FROZEN_METHODS')
    for method in ('B0', 'B1'):
        for layout in DEV_OFFSETS:
            matched = [r for r in rows if r['method'] == method and r['layout'] == layout]
            task.need(len(matched) == 2 and {r['restart'] for r in matched} == {1, 2},
                      'ALL_FORTY_REQUESTED_CONDITIONS_PER_METHOD')
    order=[]
    for i,layout in enumerate(DEV_OFFSETS):
        for restart in(1,2):
            first=('B0','B1')if(2*i+restart-1)%2==0 else('B1','B0')
            order.extend((layout,restart,method)for method in first)
    task.need([(r['layout'],r['restart'],r['method'])for r in rows]==order,
        'PREREGISTERED_INTERLEAVED_MATCHED_METHOD_ORDER')
    from .target_gr00t_profile_v1 import PARENT
    task.need(all(p.get('parent_manifest_sha256')==PARENT for p in policies.values())
        and policies['B0'].get('package')is None and policies['B0'].get('Core_qualification')is None
        and isinstance(policies['B1'].get('package'),dict)
        and isinstance(policies['B1'].get('Core_qualification'),dict),
        'ONE_FROZEN_ORIGINAL_B0_AND_ONE_CORE_QUALIFIED_SFT_PACKAGE_PER_PANEL')
    for row in rows:
        seeds = row.get('query_seeds')
        task.need(row['parent'] == scene_id(row['layout']) and row['split'] == 'DEV'
            and row['purpose'] == PURPOSE and row['command'] == task.COMMAND
            and Path(row['native_run']).name == row['native_run']
            and not row['native_run'].startswith('.') and len(row['model_sha256']) == 64
            and type(seeds) is list and len(seeds) == len(set(seeds)) == task.QUERIES
            and all(type(v) is int and 0 <= v < 2**63 for v in seeds)
            and 'Teacher_mode' not in row and not row.get('diagnostic_control'),
            'NEW_STUDENT_CONDITION_CURRENT_OBSERVATIONS_NO_TEACHER_INPUT_OR_RETRY')
        other = next(r for r in rows if r['layout'] == row['layout']
                     and r['restart'] == row['restart'] and r['method'] != row['method'])
        task.need(other['query_seeds'] == seeds and other['model_sha256'] == row['model_sha256'],
                  'MATCHED_SCENE_AND_PREDECLARED_NOISE_ACROSS_FROZEN_METHODS')
    return rows


def summarize(panel,outcomes):
    rows=validate(panel);wanted={r['native_run']:r for r in rows}
    task.need(len({v['native_run']for v in outcomes})==len(outcomes)
        and all(v['native_run']in wanted and all(v[k]==wanted[v['native_run']][k]
            for k in('method','layout','restart','parent','split','query_seeds'))for v in outcomes),
        'NO_DEV_RETRY_OR_FOREIGN_OUTCOME')
    methods={}
    for method in('B0','B1'):
        values=[v for v in outcomes if v['method']==method]
        attempted=sum(v['attempted']is True for v in values)
        ready=sum(v['reset_ready']is True for v in values)
        success=sum(v['task_success']is True and v['reset_ready']is True
            and v['technical_valid']is True for v in values)
        methods[method]=dict(requested=40,attempted=attempted,unrun=40-attempted,
            reset_ready=ready,task_successes=success,
            hard_safety_stop_conditions=sum(v['hard_safety_stop']is True for v in values),
            technical_failure_conditions=sum(v['technical_valid']is not True for v in values),
            P_reset_ready=ready/40,P_success_per_requested_episode=success/40,
            P_success_given_reset_ready=success/ready if ready else None,
            actual_GR00T_queries=sum(v.get('actual_GR00T_queries',0)for v in values))
    complete=len(outcomes)==80 and all(v['attempted']==40 for v in methods.values())
    return dict(schema='TASK_LEARNING_NATIVE_STUDENT_DEV_SUMMARY_V1',methods=methods,
        complete=complete,qualified_training_demonstrations=0,GR00T_updates=0,
        WBC_changes=False,TEST_results_read=False)


def classify_actual(row,root,expected_source,*,core_local=False,conic=False,endpoint=False,matched=False,native_body=False):
    """Read both actual terminal workers; valid policy rejection remains a failure."""
    import json
    from .task_learning_teacher_panel_v1 import classify as original
    role=task.CORE_LOCAL_ROLE if core_local else task.STUDENT_DEV_ROLE
    schema=NATIVE_SCHEMA
    if native_body:
        from .native_arx5_diagnostic_v1 import NATIVE_SCHEMA as schema,PROTOCOL as role
    if core_local:
        from .task_learning_local_policy_v1 import NATIVE_SCHEMA as schema
    if conic:
        from .task_learning_conic_native_v3 import NATIVE_SCHEMA as schema,PROTOCOL as role
    if endpoint:
        from .task_learning_endpoint_native_v3 import NATIVE_SCHEMA as schema,PROTOCOL as role
    if matched:
        from .task_learning_matched_native_v3_1 import NATIVE_SCHEMA as schema,PROTOCOL as role
    root=Path(root)
    def read(path):return json.loads(path.read_bytes())if path.exists()else{}
    spec=read(root/'EXECUTION_SPEC.json');native=root/'native-01'
    n=read(native/'RESULT.json');worker=read(root/'WORKER_EXIT.private.json')
    lease=read(root/'control/LEASE.json')
    out=original(row,native=n,worker=worker,lease=lease,
        readiness=read(native/'TASK_LEARNING_START_READINESS.private.json'),
        score=read(native/'TASK_LEARNING_SCORE.private.json'),spec=spec,
        expected_source=expected_source,attempted=native.exists(),
        expected_role=role,expected_native_schema=schema)
    gpu=Path(spec['target_gr00t_gpu_root']);gs=read(gpu/'EXECUTION_SPEC.json')
    gl=read(gpu/'control/LEASE.json');gw=read(gpu/'WORKER_EXIT.private.json')
    gr=read(gpu/'fit-01/RESULT.private.json');records=gr.get('records')or[]
    server=(gl.get('status')=='CLOSED'and gl.get('worker_exit_code')==gw.get('exit_code')==0
        and lease.get('spec_sha256')==digest(root/'EXECUTION_SPEC.json')
        and lease.get('run_uuid')==root.name
        and gw.get('failure')is None and gl.get('spec_sha256')==digest(gpu/'EXECUTION_SPEC.json')
        and gs.get('producer_native_spec')==dict(path=str(root/'EXECUTION_SPEC.json'),sha256=digest(root/'EXECUTION_SPEC.json'))
        and gs.get('target_gr00t_policy')==spec.get('target_gr00t_policy')
        and gs.get('target_gr00t_query_seeds')==spec.get('target_gr00t_query_seeds')
        and gr.get('status')=='ACTUAL_CONTINUOUS_LIVE_QUERIES_TERMINAL'
        and gr.get('actual_model_loads')==1
        and gr.get('actual_VLA_inferences')==gr.get('Processor_decode_calls')==len(records)<=15
        and gr.get('GR00T_updates')==gr.get('WM_updates')==gr.get('Actor_updates')==gr.get('native_integrations')==0)
    rejected=read(native/'TASK_LEARNING_STUDENT_REJECTION.private.json')
    children=n.get('owned_process_exits')or[]
    controlled=(server and rejected and spec.get('source_sha')==expected_source
        and spec.get('task_learning_role')==role
        and 'TASK_LEARNING_STUDENT_FULL_PROPOSAL_REJECTED:'in(n.get('failure')or'')
        and n.get('shutdown_clean')is True and len(children)==4
        and {p.get('name')for p in children}=={'sim','solver','controller','planner'}
        and all(p.get('exit_code')==0 for p in children)
        and lease.get('status')=='CLOSED'and lease.get('worker_exit_code')==worker.get('exit_code')
        and rejected.get('proposal_executed')is False and rejected.get('command_clips')==0
        and 0<=rejected.get('index',-1)<len(records)
        and rejected['index']==len(records)-1
        and rejected.get('action_sha256')==records[-1].get('action_sha256')
        and rejected.get('native_step')==task.B+1000*rejected['index'])
    out['technical_valid']=bool(server and(out['technical_valid']or controlled))
    out['task_success']=bool(out['task_success']and out['technical_valid'])
    failed=read(gpu/'fit-01/FAILED.private.json')
    neural_attempted=bool(gw or(gpu/'SUPERVISOR.private.json').exists())
    out['native_world_attempted']=native.exists()
    out['neural_worker_attempted']=neural_attempted
    # A serving process may fail before the native partner starts. That is an
    # attempted deployment condition, while its physical integrations remain0.
    out['attempted']=bool(out['attempted']or neural_attempted)
    if neural_attempted and not server and not out.get('first_failure'):
        out['first_failure']=failed.get('failure')or gw.get('failure')or'GPU_WORKER_INCOMPLETE_BEFORE_VALID_NATIVE_OUTCOME'
    out['CPU_s']+=gw.get('child_CPU_s',0)
    out.update(actual_GR00T_queries=len(records),policy_proposal_rejected=bool(controlled),
        actual_GR00T_query_attempts=failed.get('generation_invocations',len(records)),
        query_completion_unknown_after_intent=failed.get('query_completion_unknown_after_intent',False),
        GPU_worker_wall_s=gw.get('wall_s'),GPU_child_CPU_s=gw.get('child_CPU_s',0),
        result_sha256=digest(native/'RESULT.json')if(native/'RESULT.json').exists()else None,
        GPU_result_sha256=digest(gpu/'fit-01/RESULT.private.json')if(gpu/'fit-01/RESULT.private.json').exists()else None)
    return out


def compile_proposal(action, pin, *, source_from_target, query_state10, actual_TCP,
                     consumed_reference_TCP, base_hold):
    from scripts.pi0292.eval_contract import matrix10
    from scripts.pi0291_arx5_data.native_dataset import pose10
    task.need(digest(action) == pin, 'ACTUAL_STUDENT_RESPONSE_ACTION_BYTES')
    with np.load(action, allow_pickle=False) as packet:
        raw, normalized = packet['physical_H40'].copy(), packet['normalized']
        task.need(raw.shape == (40, 10) and normalized.shape == (1, 40, 132)
            and np.isfinite(raw).all() and np.isfinite(normalized).all(),
            'ONE_ORIGINAL_DECODE_FINITE_FULL_H40')
    frame = np.asarray(source_from_target, float)
    from .fusion_observation import rigid
    rigid(frame)
    task.need(np.allclose(matrix10(query_state10), frame @ np.asarray(actual_TCP),
        rtol=0, atol=1e-9), 'SAME_CAPTURED_POLICY_STATE_AND_CURRENT_NATIVE_TCP')
    # The original Processor already decoded absolute poses in the policy frame.
    # Map them once; preserve their first-knot error, gripper and full trajectory.
    try:
        target = pose10(np.linalg.inv(frame) @ matrix10(raw), raw[:, 9])
        program = task.FiftyMsProgram(target, actual_TCP, consumed_reference_TCP, base_hold)
    except ValueError as error:
        raise StudentProposalRejected('TASK_LEARNING_STUDENT_FULL_PROPOSAL_REJECTED: '+str(error))from error
    return program, dict(raw_policy_H40=raw.tolist(), mapped_target_H40=target.tolist(),
        original_Processor_decode_calls=1, action_sha256=pin,
        policy_frame_mapping_count=1, command_clips=0, teacher_fallback=False,
        external_inference_adapter=False, expert_training_eligible=False)


class StudentPolicy:
    """Current observation -> actual resident GR00T query; scene state scores only."""
    stage = 'STUDENT'

    def __init__(self, adapter, initial, *, consumed_reference=None):
        self.adapter = adapter
        self.cube = np.asarray(initial['object_pose']).copy()
        self.index = 0
        self.observation = None
        self.consumed_reference=consumed_reference

    def decide(self, sample, reference, base):
        from scripts.pi0292.relay_protocol import unseal
        from .task_learning_live_pair_v1 import query_from_native
        import json
        task.need(self.observation is not None and sample['native_step'] ==
                  task.B + 1000 * self.index, 'FRESH_STUDENT_QUERY_AT_ORIGINAL_NATIVE_CLOCK')
        action, pin, actual = query_from_native(self.adapter, self.observation, self.index)
        body = unseal(json.loads(Path(self.observation['path']).read_bytes()))
        shadow = unseal(body['original_shadow_ABI'])
        from .fusion_observation import shadow_inference_payload
        payload = shadow_inference_payload(body['original_shadow_ABI'])
        try:
            compiler=compile_proposal
            consumption=None
            if self.adapter.spec.get('task_learning_role')in(task.ENDPOINT_ROLE,task.MATCHED_ROLE,task.NATIVE_BODY_ROLE):
                from functools import partial
                from .task_learning_endpoint_policy_v3 import compile_proposal as endpoint_compile,consumed_grip
                task.need(callable(self.consumed_reference),'ENDPOINT_ACTUAL_CONSUMPTION_CALLBACK')
                cycle,previous_program=self.consumed_reference()
                grip,consumption=consumed_grip(cycle,previous_program,sample,reference)
                compiler=partial(endpoint_compile,consumed_nominal_grip=grip)
                if self.adapter.spec['task_learning_role']==task.MATCHED_ROLE:
                    from .task_learning_matched_policy_v3_1 import compile_proposal as matched_compile
                    compiler=partial(matched_compile,method=self.adapter.spec['target_gr00t_policy']['arm'],consumed_nominal_grip=grip)
            if self.adapter.spec.get('task_learning_role')==task.CONIC_ROLE:
                from functools import partial
                from .task_learning_conic_policy_v3 import compile_proposal as conic_compile
                compiler=partial(conic_compile,method=self.adapter.spec['target_gr00t_policy']['arm'])
            program, proposal = compiler(action, pin,
                source_from_target=shadow['source_policy_view']['T_source_from_target'],
                query_state10=payload['state_eef10'],
                actual_TCP=sample['tcp'], consumed_reference_TCP=reference, base_hold=base)
        except StudentProposalRejected as error:
            from .storage import write_json
            write_json(self.adapter.output/'TASK_LEARNING_STUDENT_REJECTION.private.json',
                dict(index=self.index,native_step=sample['native_step'],action_sha256=pin,
                    observation=self.observation,actual_model_query=actual,failure=str(error),
                    proposal_executed=False,teacher_fallback=False,command_clips=0))
            raise
        proposal.update(actual_model_query=actual, label_role='STUDENT_NOT_EXPERT', index=self.index)
        if consumption is not None:proposal['previous_consumed_grip_witness']=consumption
        if self.adapter.spec.get('task_learning_role')in(task.CORE_LOCAL_ROLE,task.CONIC_ROLE,task.ENDPOINT_ROLE,task.MATCHED_ROLE,task.NATIVE_BODY_ROLE):
            from .task_learning_local_policy_v1 import execution_stage,EXECUTION_STAGE_REVISION
            self.stage=execution_stage(program.decoded_H40,sample)
            proposal.update(execution_stage=self.stage,execution_stage_revision=EXECUTION_STAGE_REVISION,
                execution_stage_is_learner_input=False,original_contact_predicate_changed=False)
        self.index += 1
        self.observation = None
        return program, proposal
