"""Audit and export only successful sealed target-adaptation PLACE demonstrations."""
import argparse
import json
from pathlib import Path

import numpy as np

from go2arx5_vla.sla_adapter.codec_v1 import Quantiles
from go2arx5_vla.sla_adapter.dense_g7_v6 import VERSION
from go2arx5_vla.sla_adapter.multitask_data_v6 import C20_STATISTICS
from go2arx5_vla.world_model import target_adaptation_teacher as adaptation
from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
from go2arx5_vla.world_model.storage import digest
from scripts.pi0292.durable import read, write
from scripts.sla.export_multitask_g7_v6 import fp32_audit
from scripts.sla.multitask_g7_capture_v6 import copy_shards, read_capture


def audit(config):
    parent = read_pin(config['campaign_spec'])
    population = read_pin(parent['compiled_population'])
    order = adaptation.conditions(population)
    scenes = {row['id']:row['scene'] for row in population['clusters']}
    root = Path(parent['registered_root'])
    result_pin = config['campaign_result']
    lease = read(root/'control/LEASE.json')
    result = read_pin(result_pin)
    if ((parent['scope'],parent['plan_commit']) != (adaptation.SCOPE,adaptation.PLAN)
            or lease['status'] != 'CLOSED' or lease['spec_sha256'] != config['campaign_spec']['sha256']
            or Path(result_pin['path']) != root/'RESULT.private.json'
            or result['source_sha'] != parent['source_sha'] or result['unresolved_case_launches']
            or result['new_updates'] or result['new_optimization_visits'] or result['real_robot_requests']
            or len(parent['cases']) != len(order)):
        raise ValueError('Sealed reconciled first-six expert campaign required')
    reports = [row for row in result['records'] if row['role']=='EXPERT']
    if ([row['id'] for row in reports] != [r['id'] for r in order[:len(reports)]]
            or len(reports) != result['new_verified_native_requests']
            or result['attempted_case_launches'] != len(reports)):
        raise ValueError('Actual expert receipts must preserve the registered condition order')
    raw = read_pin(dict(path=config['C20_statistics']['path'],sha256=C20_STATISTICS))
    stats = {name:Quantiles(raw[name]['q01'],raw[name]['q99']) for name in ('state','actions')}
    target = Path(config['output'])
    target.mkdir(mode=0o700)
    episodes, audits = [], []
    for card_path, condition, row in zip(parent['cases'][:len(reports)],order[:len(reports)],reports,strict=True):
        path = Path(card_path)
        case = read(path)
        file = path.parent/'RESULT.private.json'
        closed = read(path.parent/'control/LEASE.json')
        native_pin = dict(path=str(file),sha256=row['result_sha256'])
        native = read_pin(native_pin)
        scene = scenes[condition['cluster']]
        if (path.parent.parent != root/'cases' or path.parent.name != condition['id']
                or closed['status'] != 'CLOSED' or closed['spec_sha256'] != digest(path)
                or case['scope'] != adaptation.SCOPE or case['source_sha'] != parent['source_sha']
                or case['sim_only_role'] != 'EXPERT' or case['panel_entry_id'] != condition['id']
                or case['task_instruction_id'] != condition['instruction']
                or case['mobile_condition']['reset'] != condition['reset']
                or case['mobile_condition']['seed'] != condition['noise_seed']
                or case['model_path'] != scene['path'] or row['scene_sha256'] != scene['sha256']
                or case['collection_contract'] != 'SLA_V6_MULTITASK_TEACHER_CURRENT_FULL_H40_AND_CONSUMED_MASK'
                or native['source_sha'] != case['source_sha']
                or native['physical_requests'] != 1 or native['model_generations'] or native['optimizer_updates']):
            raise ValueError('Exact sealed native expert evidence required: '+condition['id'])
        summary = dict(id=condition['id'],instruction=condition['instruction'],native_result_sha256=row['result_sha256'],
                       scene_sha256=row['scene_sha256'],reset=condition['reset'],
                       native_ticks=native['actual_native_integrations'],first_failure=row['first_failure'],
                       admitted=False)
        if row['completed'] and row['task_and_mobile_success'] and not row['hard_STOP'] and not row['technical_failure']:
            try:
                folder,manifest,frame,codec,data = read_capture(condition,case,native)
                qualification = fp32_audit([(condition['id'],codec,data,data['previous_actual_command'])],stats)
                if any(g['over_original_gate'] or (g['codec_path']=='fp32_codec' and g['over_margin'])
                       for g in qualification['label_margin_by_domain_codec_phase']):
                    raise ValueError('ORIGINAL_C20_FP32_TRAINING_MARGIN_FAILED')
                copy_shards(folder,manifest,data)
            except (ValueError,KeyError,OSError) as error:
                summary['first_failure'] = str(error)
            else:
                episode_root = target/condition['id']
                episode_root.mkdir()
                rgb = copy_shards(folder,manifest,data,episode_root)
                labels = episode_root/'LABELS.npz'
                with labels.open('xb') as stream:
                    np.savez(stream,**data)
                episodes.append(dict(id=condition['id'],split='TRAIN',task=condition['instruction'],
                    family=condition['family'],language=case['language'],windows=manifest['queries'],
                    mount=dict(T_policy_from_mount=frame.tolist()),scene_sha256=row['scene_sha256'],
                    reset=condition['reset'],label_kind='expert_query_plan_H40',
                    expert_stage_audit_only=data['phase'].tolist(),
                    labels=dict(file=str(labels.relative_to(target)),sha256=digest(labels)),
                    frame_shards=rgb,native_result_sha256=row['result_sha256'],individual_pass=True))
                summary.update(admitted=True,windows=manifest['queries'],confirmed_P10_atoms=10*manifest['queries'])
        audits.append(summary)
    manifest = target/'MANIFEST.private.json'
    write(manifest,dict(schema='SLA_TARGET_ADAPTATION_NEW_EXPERTS_V1',plan=adaptation.PLAN,
        representation=VERSION,scope=adaptation.SCOPE,statistics_sha256=C20_STATISTICS,
        original_campaign_result_sha256=result_pin['sha256'],episodes=episodes,
        windows=sum(row['windows'] for row in episodes),live_SCALE_training_data_changed=False))
    output = dict(verified_expert_requests=len(reports),planned=len(order),admitted_episodes=len(episodes),
        admitted_windows=sum(row['windows'] for row in episodes),cases=audits,dataset_sha256=digest(manifest),
        optimizer_updates=0,new_native_requests=0,new_native_ticks=0)
    write(target/'ACTUAL.private.json',output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('sha256')
    args = parser.parse_args()
    print(json.dumps(audit(read_pin(dict(path=args.config,sha256=args.sha256))),allow_nan=False))


if __name__ == '__main__':
    main()
