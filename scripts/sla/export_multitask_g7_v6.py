"""Seal complete native M Teacher captures for the shared G7 target learner."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np

from scripts.pi0291_arx5_data.native_dataset import matrix_from10, pose10
from scripts.pi0292.durable import read, write
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.codec_v1 import Quantiles, TOLERANCES, eef, pad32, rigid, roundtrip_error
from go2arx5_vla.sla_adapter.dense_g7_v6 import VERSION, label_margin_report
from go2arx5_vla.sla_adapter.gripper_v2 import EPSILON, GripLogitCodec
from go2arx5_vla.sla_adapter.mfree_training_v1 import MobileDataset
from go2arx5_vla.sla_adapter.multitask_data_v6 import C20_MANIFEST, CLOCK_FIELDS, SCHEMA, legacy_contexts, reference
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.world_model.multitask_scope_v6 import PLAN, SCOPE
from go2arx5_vla.world_model.multitask_score_v6 import INSTRUCTIONS


def teacher_results(spec):
    population = read_pin(spec['compiled_population'])['conditions']['TEACHER']
    if len(population) != 48 or len({row['id'] for row in population}) != 48:
        raise ValueError('SEALED_UNIQUE_TEACHER48_REQUIRED')
    entries, terminal = {}, []
    for pins in spec['teacher_campaigns']:
        stage, result = read_pin(pins['spec']), read_pin(pins['result'])
        root = Path(stage['registered_root'])
        lease = read(root/'control/LEASE.json')
        if (Path(pins['result']['path']) != root/'RESULT.private.json'
                or lease['status'] != 'CLOSED' or lease['spec_sha256'] != pins['spec']['sha256']
                or stage['plan_commit'] != PLAN or stage['scope'] != SCOPE
                or stage['compiled_population'] != spec['compiled_population']
                or result['source_sha'] != stage['source_sha'] or result['unresolved_case_launches']):
            raise ValueError('SEALED_TEACHER_CAMPAIGN_AND_ORIGINAL_POPULATION_REQUIRED')
        cases = {Path(path).parent.name: Path(path) for path in stage['cases']}
        for row in result['records']:
            if row['role'] != 'EXPERT':
                continue
            path = cases[row['id']]
            if path.parent.parent != root/'cases' or row['id'] in entries:
                raise ValueError('UNIQUE_ORIGINAL_TEACHER_CASE_ID_REQUIRED')
            entries[row['id']] = path
            terminal.append(row)
    if [row['id'] for row in terminal] != [row['id'] for row in population[:len(terminal)]]:
        raise ValueError('COMPLETE_ORDERED_VERIFIED_TEACHER_PREFIX_REQUIRED')
    accepted, rejected = [], []
    for row, condition in zip(terminal, population, strict=False):
        path = entries[row['id']]
        case = read(path)
        lease = read(path.parent/'control/LEASE.json')
        result_path = path.parent/'RESULT.private.json'
        pin = dict(path=str(result_path), sha256=row['result_sha256'])
        result = read_pin(pin)
        if (lease['status'] != 'CLOSED' or lease['spec_sha256'] != sha(path)
                or case['registered_root'] != str(path.parent) or row['actual_native_requests'] != 1
                or result['physical_requests'] != 1 or result['model_generations'] or result['optimizer_updates']
                or case['panel_entry_id'] != condition['id'] or case['sim_only_role'] != 'EXPERT'
                or case['task_instruction_id'] != condition['instruction']
                or case['mobile_condition']['family'] != condition['family']
                or case['mobile_condition']['layout'] != condition['cluster']
                or case['mobile_condition']['reset'] != condition['reset']
                or case['mobile_condition']['seed'] != condition['noise_seed']
                or result['source_sha'] != case['source_sha'] or row['native_ticks'] != result['actual_native_integrations']
                or row.get('integrations_after_STOP')
                or row['native_ticks'] > case['max_ticks']):
            raise ValueError('RECONCILE_NATIVE_TEACHER_EVIDENCE_BEFORE_DATA_PREPARATION')
        if row['completed'] and not row['hard_STOP'] and not row['technical_failure'] and row['task_and_mobile_success']:
            accepted.append((condition, case, pin, result))
        else:
            rejected.append(dict(id=row['id'], first_failure=row['first_failure'], native_ticks=row['native_ticks'],
                                 hard_STOP=row['hard_STOP'], technical_failure=row['technical_failure'], result=pin))
    return accepted, rejected, sum(row['native_ticks'] for row in terminal)


def read_capture(condition, case, native):
    detail = native['detail']
    score = detail['score']
    end = native['actual_native_integrations']
    clock = native['trace_audit']
    if (not native['completed'] or native['unsafe'] or native['technical_failure']
            or not native['metrics']['task_and_required_motion_success'] or not score['task_success']
            or score['instruction'] != condition['instruction'] or score['native_step'] != end
            or score['states'] != end-5703+1 or score['missing_native_intervals'] or score['unsafe_states']
            or clock['status'] != 'RECORDED_TRACE_CHECKS_PASS' or clock['native_step_records'] != end):
        raise ValueError('ACTUAL_COMPLETE_2MS_TEACHER_SUCCESS_REQUIRED')
    path = Path(case['registered_root'])/'native-01'
    capture = path/'LIVE_DATA'
    if Path(detail['capture']['path']) != capture/'MANIFEST.private.json':
        raise ValueError('EXACT_NATIVE_RESULT_CAPTURE_REQUIRED')
    manifest = read_pin(detail['capture'])
    count = manifest['queries']
    if (manifest['schema'] != 'MFREE_LIVE_FINAL_PLAN_CAPTURE_V1' or manifest['writer_error'] is not None
            or manifest['final_native_step'] != end or count != detail['expert_queries']
            or not 1 <= count <= case['policy_queries'] or end != 5703+250*count
            or detail['consumed_task_atoms'] != 10*count or manifest['planned_suffix_is_executed'] is not False
            or manifest['frozen_scene_sha256'] != case['frozen_files'][case['model_path']]
            or manifest['language'] != case['language'] or manifest['data_contract'] != case['collection_contract']
            or Path(manifest['labels']['path']) != capture/'QUERIES.private.json'):
        raise ValueError('COMPLETE_LIVE_H40_RESULT_CLOCK_AND_CAPTURE_IDENTITY_REQUIRED')
    rows = read_pin(manifest['labels'])
    if len(rows) != count:
        raise ValueError('EXACT_TEACHER_QUERY_DENOMINATOR_REQUIRED')
    commands = [json.loads(line) for line in (path/'MOBILE_SAMPLES.private.jsonl').read_text().splitlines()]
    joined = {row['native_start_step']: row for row in commands}
    if len(joined) != len(commands):
        raise ValueError('UNIQUE_REAL_COMMAND_CLOCK_REQUIRED')
    frame = rigid(rows[0]['episode_fixed_T_policy_from_mount'])
    transform = rigid(case['policy_from_world'])
    codec = GripLogitCodec(frame)
    state10, nominal, state7, latent7, contexts = [], [], [], [], []
    for index, row in enumerate(rows):
        step = 5703+250*index
        context = row['previous_confirmed_command']
        prior = joined[step-25]
        if (row['query'] != index or row['native_step'] != step or abs(row['native_time_s']-step*.002) > 1e-8
                or row['instruction'] != condition['instruction'] or row['actual_elapsed_prefix_ticks'] != 250
                or row['partial_interval_ticks'] or row['full_EEF_H40_planner_check'] is not True
                or row['rgb_provenance'] != 'CURRENT_OWNED_PAUSED_NATIVE_PACKET'
                or row['object_phase_memory_in_student_inputs'] is not False
                or row['supervision_mask'] != [True]*40 or row['consumed_mask'] != [True]*10+[False]*30
                or not np.array_equal(row['episode_fixed_T_policy_from_mount'], frame)
                or prior['native_end_step'] != step or prior['ticks'] != 25
                or context['width_m'] != .044*prior['installed_EEF10'][9]
                or context['source_native_s'] != prior['gripper_command']['source_native_s']
                or context['query_native_s'] != row['native_time_s'] or len(row['intervals']) != 10):
            raise ValueError(f'FULL_H40_REAL_PREVIOUS_COMMAND_AND_P10_REQUIRED:{index}')
        for atom, sample in enumerate(row['intervals']):
            start = step + 25*atom
            observed = joined[start]
            if (sample['atom'] != atom or sample['start_native_step'] != start
                    or sample['end_native_step'] != start+25 or sample['confirmed_ticks'] != 25
                    or observed['ticks'] != 25 or observed['native_end_step'] != start+25
                    or observed['gripper_command']['sequence'] != 128+index*10+atom
                    or sample['installed_reference_sha256'] != observed['installed_reference_sha256']
                    or sample['boundary']['native_step'] != start+25 or not sample['boundary']['exact_receiver_knot']
                    or not observed['safety_valid']):
                raise ValueError(f'P10_INTERVAL_MUST_MATCH_NATIVE_CONSUMPTION:{index}:{atom}')
        world = eef(row['nominal_world_H40'], (40, 10))
        margin = label_margin_report(world, context)
        if margin != row['original_label_G7_margin'] or margin['original_execution_speed_exceeded']:
            raise ValueError(f'EXACT_RAW_G7_LABEL_MARGIN_AND_NATIVE_SPEED_REQUIRED:{index}')
        state = eef(row['state_eef10'], (10,))
        measured = row['actual_state']
        expected_state = pose10(transform @ np.asarray(measured['tcp']), measured['gripper_fraction'])
        # Native capture and offline export can use different floating-point
        # matrix kernels. Preserve the captured bytes and allow only roundoff
        # when independently recomputing the pose.
        if state[9] != expected_state[9] or not np.allclose(state[:9], expected_state[:9], rtol=0, atol=1e-12):
            raise ValueError('SAME_PAUSED_MEASURED_STATE_AND_POLICY_FRAME_REQUIRED')
        target = pose10(transform @ matrix_from10(world), world[:, 9])
        s, a = codec.encode(state, target)
        state10.append(state); nominal.append(target); state7.append(s); latent7.append(a)
        contexts.append([context[key] for key in CLOCK_FIELDS])
    arrays = dict(state7=np.asarray(state7), latent7=np.asarray(latent7), state_eef10=np.asarray(state10),
        nominal_policy_H40=np.asarray(nominal), native_step=np.asarray([r['native_step'] for r in rows]),
        frame_index=np.arange(count), phase=np.asarray([r['phase'] for r in rows]),
        previous_actual_command=np.asarray(contexts), supervision_mask=np.ones((count,40), dtype=bool),
        consumed_mask=np.asarray([r['consumed_mask'] for r in rows], dtype=bool))
    return capture, manifest, frame, codec, arrays


def copy_shards(source, manifest, data, output=None):
    shards = []
    start = 0
    for raw in manifest['frame_shards']:
        path = Path(raw['path'])
        count = raw['count']
        if (path.parent != source or path.name != f'frames-{len(shards):03d}.npz'
                or raw['first_query'] != start or not 1 <= count <= 16):
            raise ValueError('IMMUTABLE_CONTIGUOUS_SOURCE_RGB_SHARDS_REQUIRED')
        observed = path
        if output is not None:
            observed = output/path.name
            shutil.copyfile(path, observed)
        if sha(observed) != raw['sha256']:
            raise ValueError('BYTE_EXACT_ORIGINAL_PAUSED_RGB_TRANSFER_REQUIRED')
        with np.load(observed, allow_pickle=False) as rgb:
            span = slice(start, start+count)
            if (rgb['rgb'].shape != (count, 2, 256, 256, 3) or rgb['rgb'].dtype != np.uint8
                    or not np.array_equal(rgb['native_step'], data['native_step'][span])
                    or not np.array_equal(rgb['state_eef10'], data['state_eef10'][span])):
                raise ValueError('EXACT_CURRENT_RGB_STATE_AND_TEACHER_QUERY_JOIN_REQUIRED')
        file = str(observed.relative_to(output.parent)) if output is not None else observed.name
        shards.append(dict(file=file, sha256=raw['sha256'],
                           first_frame=start, frames=count))
        start += count
    if start != len(data['native_step']):
        raise ValueError('ALL_TEACHER_QUERIES_HAVE_MATCHING_RGB_SHARDS')
    return shards


def fp32_audit(episodes, statistics):
    maxima = dict.fromkeys(TOLERANCES, 0.)
    groups = {}
    for domain, codec, arrays, contexts in episodes:
        for state, state7, latent, target, values, phase in zip(
                arrays['state_eef10'], arrays['state7'], arrays['latent7'], arrays['nominal_policy_H40'],
                contexts, arrays['phase'], strict=True):
            q = statistics['state']; decoded = q.unnormalize(q.normalize(state7).astype(np.float32))
            q = statistics['actions']; denormalized = q.unnormalize(q.normalize(latent).astype(np.float32))
            physical = codec.decode(state, pad32(denormalized))
            expected = target.copy()
            expected[:,9] = np.clip(expected[:,9], EPSILON, 1-EPSILON)
            error = roundtrip_error(expected, physical)
            if (np.max(abs(decoded-state7)) > 1e-6 or any(error[key] > cap for key, cap in TOLERANCES.items())):
                raise ValueError('COMMON_G7_STATISTICS_FULL_FP32_CODEC_ROUNDTRIP_FAILED')
            for key, value in error.items():
                maxima[key] = max(maxima[key], value)
            context = reference(values)
            for origin, h40 in (('raw', target), ('fp32_codec', physical)):
                result = label_margin_report(h40, context)
                row = groups.setdefault((domain, origin, str(phase)), dict(windows=0, over_margin=0,
                                                                         over_original_gate=0, maximum_m_s=0.))
                row['windows'] += 1
                row['over_margin'] += int(result['training_margin_exceeded'])
                row['over_original_gate'] += int(result['original_execution_speed_exceeded'])
                row['maximum_m_s'] = max(row['maximum_m_s'], result['maximum_speed_m_s'])
    return dict(maximum_FP32_roundtrip_error=maxima, label_margin_by_domain_codec_phase=[
        dict(domain=d, codec_path=c, phase=p, **counts) for (d,c,p), counts in sorted(groups.items())])


def export(spec, destination):
    start, cpu = time.monotonic(), time.process_time()
    if spec['plan_commit'] != PLAN:
        raise ValueError('FIXED_SLA_V6_PLAN_REQUIRED')
    accepted, rejected, ticks = teacher_results(spec)
    cells = Counter((row['instruction'], row['family']) for row, *_ in accepted)
    coverage = {task: {f: cells[task,f] for f in ('F1','F2')} for task in INSTRUCTIONS}
    root = Path(destination)
    root.mkdir(exist_ok=False)
    effects = dict(teacher_native_requests=len(accepted)+len(rejected), teacher_native_ticks=ticks,
        teacher_successes=len(accepted), coverage=coverage, failed_teacher_cases=rejected,
        new_optimizer_updates=0, new_model_generations=0, new_native_requests=0, new_native_ticks=0)
    if any(n < 2 for values in coverage.values() for n in values.values()):
        result = dict(status='TEACHER_COVERAGE_NOT_MET', **effects)
        write(root/'ACTUAL.private.json', result)
        return result
    legacy = spec['legacy_C20']
    c20 = MobileDataset(legacy['dataset'], C20_MANIFEST, resize_with_pad=None, max_cache_bytes=0)
    old_contexts = legacy_contexts(c20, legacy['time_audit'])
    audit_inputs = []
    offset = 0
    for episode in c20.manifest['episodes']:
        eid, count = episode['id'], episode['windows']
        audit_inputs.append(('C20', c20.codecs[eid], c20.labels[eid], old_contexts[offset:offset+count]))
        offset += count
    episodes = []
    for condition, case, pin, native in accepted:
        capture, manifest, frame, codec, data = read_capture(condition, case, native)
        folder = root/condition['id']; folder.mkdir()
        shards = copy_shards(capture, manifest, data, folder)
        labels = folder/'LABELS.npz'
        with labels.open('xb') as stream:
            np.savez(stream, **data)
        episodes.append(dict(id=condition['id'], split='TRAIN', task=condition['instruction'],
            family=condition['family'], cluster=condition['cluster'], language=case['language'],
            windows=manifest['queries'], mount=dict(T_policy_from_mount=frame.tolist()),
            expert_stage_audit_only=data['phase'].tolist(), frame_shards=shards,
            labels=dict(file=str(labels.relative_to(root)), sha256=sha(labels)), native_result=pin,
            original_live_capture=native['detail']['capture']))
        audit_inputs.append(('M_TEACHER', codec, data, data['previous_actual_command']))
    values = dict(state=np.concatenate([d['state7'] for _,_,d,_ in audit_inputs]),
                  actions=np.concatenate([d['latent7'].reshape(-1,7) for _,_,d,_ in audit_inputs]))
    stats = {key: Quantiles.fit(value) for key,value in values.items()}
    audit = fp32_audit(audit_inputs, stats)
    count = sum(ep['windows'] for ep in episodes)
    stats_path = root/'STATISTICS.json'
    write(stats_path, dict(schema=SCHEMA, representation=VERSION, split='TRAIN',
        fit_episodes=len(episodes)+len(c20.manifest['episodes']), legacy_windows=len(c20), teacher_windows=count,
        legacy_manifest_sha256=C20_MANIFEST, **{key: q.summary(values[key]) for key,q in stats.items()}))
    bad_speed = sum(row['over_original_gate'] for row in audit['label_margin_by_domain_codec_phase'])
    margin_conflicts = sum(row['over_margin'] for row in audit['label_margin_by_domain_codec_phase']
                           if row['domain'] == 'M_TEACHER' and row['codec_path'] == 'fp32_codec')
    result = dict(status='CHECK_TEACHER_G7_TRAINING_MARGIN' if margin_conflicts else 'G7_TARGET_TRAIN_READY',
        teacher_windows=count, legacy_windows=len(c20), **audit, **effects,
        wall_s=time.monotonic()-start, CPU_s=time.process_time()-cpu)
    if bad_speed:
        result['status'] = 'ORIGINAL_G7_SPEED_GATE_CONFLICT'
    write(root/'ACTUAL.private.json', result)
    if result['status'] == 'G7_TARGET_TRAIN_READY':
        write(root/'MANIFEST.private.json', dict(schema=SCHEMA, state='committed', representation=VERSION,
            split='TRAIN', episodes=episodes, windows=count, legacy_C20=legacy,
            compiled_population_sha256=spec['compiled_population']['sha256'], teacher_campaigns=spec['teacher_campaigns'],
            statistics=dict(file=stats_path.name, sha256=sha(stats_path)), audit_sha256=sha(root/'ACTUAL.private.json'),
            phase_input_to_student=False, teacher_memory_input=False, unconsumed_suffix_is_executed=False))
    return result


def main(spec):
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close

    root = Path(spec['registered_root'])
    try:
        if (spec['physics'] or spec['new_physical_requests_budget'] or spec['new_model_generations_budget']
                or spec['new_optimizer_updates_budget'] or time.time() >= spec['deadline_epoch']):
            raise PermissionError('FINITE_DATA_ONLY_PREPARATION_REQUIRED')
        actual = export(spec, root/'data'/'G7_TRAIN')
        if actual['status'] == 'G7_TARGET_TRAIN_READY':
            path = root/'data'/'G7_TRAIN'/'MANIFEST.private.json'
            actual['dataset'] = dict(path=str(path), sha256=sha(path))
        actual['exit_code'] = int(actual['status'] != 'G7_TARGET_TRAIN_READY')
    except BaseException:
        actual = dict(exit_code=1,status='PRESERVED_M_DATA_PREPARATION_FAILURE',failure=traceback.format_exc(),
                      new_optimizer_updates=0,new_model_generations=0,new_native_requests=0,new_native_ticks=0)
    actual['source_sha'] = spec['source_sha']
    write(root/'RESULT.private.json',actual)
    close(spec,actual['exit_code'])
    return actual['exit_code']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('spec')
    parser.add_argument('destination')
    args = parser.parse_args()
    print(json.dumps(export(read(args.spec), args.destination), allow_nan=False))
