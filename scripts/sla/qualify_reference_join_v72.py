"""Verify the changed native compiler against saved real queries, no effects."""
import argparse
import json
from pathlib import Path
import numpy as np

from go2arx5_vla.world_model.storage import digest


def read(path):
    return json.loads(Path(path).read_bytes())


def qualify(group, offline):
    from scripts.pi0292.relay_protocol import unseal
    from scripts.pi0292.eval_contract import matrix10
    from go2arx5_vla.world_model.fusion_observation import shadow_inference_payload
    from go2arx5_vla.world_model.sla_reference_join_v72 import compile_proposal, VERSION
    from go2arx5_vla.world_model.sla_dense_policy_v5 import compile_proposal as raw_compile
    from go2arx5_vla.world_model.task_learning_student_v1 import StudentProposalRejected
    group,offline=Path(group),Path(offline)
    assert digest(group/'RESULT.private.json')=='d9185b7dff07e364368efccd04a1e2d26c382900c6b02000c7a8904d6e6a2401'
    assert digest(offline/'analysis/RESULT.private.json')=='24aa9ca894fb2ed0644f81ea99f69bc22bcc054afb68f2971f0ac61641faf4ab'
    prior=read(offline/'analysis/RESULT.private.json')
    saved={(r['native_run'],r['query']):r for r in prior['rows']}
    rows=[]
    for outcome in read(group/'RESULT.private.json')['outcomes']:
        root=group/'runs'/outcome['native_run'];native=root/'native-01'
        gpu=root.with_name(root.name+'-gpu')/'fit-01'
        assert digest(native/'RESULT.json')==outcome['result_sha256']
        assert digest(gpu/'RESULT.private.json')==outcome['GPU_result_sha256']
        captures=read(native/'TASK_LEARNING_CAPTURES.private.json')['records']
        past=list(read(native/'STABLE_TASK_CYCLES/-1/RESULT.private.json')['windows'])
        measured_base_hold=np.asarray(past[-1]['actual_installed_consumed_EEF10_BASE7'][10:17])
        assert measured_base_hold.shape==(7,)and np.isfinite(measured_base_hold).all()
        commands=native/'TASK_LEARNING_CONFIRMED_COMMANDS.private.json'
        if commands.exists():
            past.extend(dict(c,actual_installed_consumed_EEF10_BASE7=c['installed_consumed_EEF10'])
                        for c in read(commands)['records'])
        for index,(query,capture)in enumerate(zip(read(gpu/'RESULT.private.json')['records'],captures,strict=True)):
            old=saved[(outcome['native_run'],index)]
            assert digest(capture['path'])==query['observation_sha256']
            body=unseal(read(capture['path']));shadow=unseal(body['original_shadow_ABI'])
            payload=shadow_inference_payload(body['original_shadow_ABI'])
            previous=[c for c in past if c['native_end_step']<=capture['native_step']][-2:]
            history=[dict(end_native_step=c['native_end_step'],pose=matrix10(c['actual_installed_consumed_EEF10_BASE7'][:10]))for c in previous]
            action=gpu/'queries'/f'{index:02d}'/'ACTION.npz'
            assert digest(action)==query['action_sha256']
            kwargs=dict(source_from_target=shadow['source_policy_view']['T_source_from_target'],
                query_state10=payload['state_eef10'],actual_TCP=matrix10(shadow['target_observation']['actual_state_eef10']),
                consumed_reference_TCP=history[-1]['pose'],base_hold=measured_base_hold)
            error=None;program=baseline=None
            try:
                baseline,_=raw_compile(action,query['action_sha256'],**kwargs)
                program,proposal=compile_proposal(action,query['action_sha256'],reference_history=history,
                    query_native_step=capture['native_step'],**kwargs)
            except StudentProposalRejected as exc:error=str(exc)
            file=offline/'analysis'/old['file'];assert digest(file)==old['file_sha256']
            delta=None
            if old['original_classification']=='EEF_REJECT':
                assert program is None and 'SLA_DENSE_ORIGINAL_FULL_H40_GATE'in error
            else:
                assert error is None
                with np.load(file,allow_pickle=False)as z:
                    delta=float(np.max(np.abs(program.reference_path-z['candidate_reference_H40'])))
                assert delta<=1e-12
                assert np.array_equal(program.reference_path[9:],baseline.reference_path[9:])
                assert np.array_equal(program.reference_path[:,9],baseline.reference_path[:,9])
                assert np.array_equal(program.decoded_H40,baseline.decoded_H40)
                assert np.array_equal(program.base_pose_hold,baseline.base_pose_hold)
                assert not proposal['teacher_fallback']and not proposal['native_execution_authorized']
                assert np.array_equal(program.preview(7)[:33],program.reference_path[7:])
            rows.append(dict(native_run=outcome['native_run'],query=index,
                raw_action_sha256=query['action_sha256'],original_classification=old['original_classification'],
                candidate_compiled=program is not None,retained_original_rejection=error,
                maximum_difference_from_saved_offline_candidate=delta))
    assert len(rows)==15 and sum(r['candidate_compiled']for r in rows)==14
    return dict(schema=VERSION+'_SAVED_QUERY_INTERFACE_QUALIFICATION',rows=rows,
        source_sha=read(Path(__file__).resolve().parents[2]/'SOURCE_STAGED.private.json')['source_sha'],
        original_raw_result_sha256=digest(group/'RESULT.private.json'),
        offline_result_sha256=digest(offline/'analysis/RESULT.private.json'),
        passed=True,source_interface_only=True,native_execution_authorized=False,
        optimizer_updates=0,training_visits=0,model_loads=0,model_generations=0,
        certifier_calls=0,physical_requests=0,physics_integrations=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group',required=True);p.add_argument('--offline',required=True)
    p.add_argument('--output',required=True);args=p.parse_args()
    value=qualify(args.group,args.offline)
    with Path(args.output).open('x')as stream:json.dump(value,stream,indent=2)
    print(json.dumps({k:v for k,v in value.items()if k!='rows'}))


if __name__=='__main__':main()
