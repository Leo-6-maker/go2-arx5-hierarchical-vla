"""Complete the required measured/command gripper seam metrics, zero model calls."""
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np

from scripts.pi0292.durable import write
from go2arx5_vla.world_model.mfree_unattended_ledger_v1 import read,sha
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import unit_state


def summarize(records):
    result={}
    for phase in ('ALL','CLOSE','NON_CLOSE'):
        rows=[r for r in records if phase=='ALL'or(phase=='CLOSE')==(r['phase']=='CLOSE')]
        result[phase]=dict(predictions=len(rows),
            model_first_to_previous_applied_command_peak_abs_m=max(abs(r['model_first_minus_previous_command_m'])for r in rows),
            model_first_to_previous_applied_command_MAE_m=float(np.mean([abs(r['model_first_minus_previous_command_m'])for r in rows])),
            expert_first_to_previous_applied_command_peak_abs_m=max(abs(r['expert_first_minus_previous_command_m'])for r in rows),
            model_command_seam_over40mm_s=sum(abs(r['model_command_seam_speed_m_s'])>.04*(1+1e-10)for r in rows),
            expert_command_seam_over40mm_s=sum(abs(r['expert_command_seam_speed_m_s'])>.04*(1+1e-10)for r in rows),
            model_command_seam_peak_abs_m_s=max(abs(r['model_command_seam_speed_m_s'])for r in rows),
            model_first_to_measured_width_peak_abs_m=max(abs(r['model_first_minus_measured_width_m'])for r in rows),
            model_first_to_current_expert_plan_MAE_m=float(np.mean([abs(r['model_first_minus_expert_first_m'])for r in rows])))
    return result


def run(spec):
    root=Path(spec['registered_root']);goal=Path(spec['goal_root']);training=Path(spec['training_root']);started=time.time()
    if sha(training/'EXECUTION_SPEC.json')!=spec['training_spec_sha256']:
        raise ValueError('PINNED_MANAGED_MATCHED_PAIR_SPEC_CHANGED')
    old=Path(spec['C20_root']);manifest=read(old/'dataset/MANIFEST.private.json');episodes={e['id']:e for e in manifest['episodes']}
    if sha(old/'dataset/STATISTICS.json')!=spec['statistics_sha256']:raise ValueError('FROZEN_C20_STATS')
    bank=read(Path(spec['previous_campaign'])/'diagnostic/WINDOW48_PLAN.private.json')['windows']
    windows={w['index']:w for w in bank};contexts={};originals={}
    for episode in manifest['episodes']:
        capture=read(episode['original_live_capture']['path']);labels=read(capture['labels']['path'])
        if sha(capture['labels']['path'])!=capture['labels']['sha256']:raise ValueError('ORIGINAL_FINAL_PLAN_LABELS_CHANGED')
        case=Path(episode['native_result']['path']).parent
        samples=[json.loads(line)for line in(case/'native-01/MOBILE_SAMPLES.private.jsonl').read_text().splitlines()]
        previous={r['native_end_step']:r for r in samples}
        with np.load(old/'dataset'/episode['labels']['file'],allow_pickle=False)as data:
            measured=data['state_eef10'][:,9]*.044
            for index,w in windows.items():
                if w['episode']!=episode['id']:continue
                label=labels[w['anchor']];sample=previous[label['native_step']]
                source=sample['gripper_command']['source_native_s'];dt=label['native_time_s']-source
                if not np.isclose(dt,.05,atol=1e-8,rtol=0):raise ValueError('EXACT_PREVIOUS_REAL_GRIPPER_COMMAND_CLOCK_REQUIRED')
                last=.044*float(sample['installed_EEF10'][9])
                expert=.002+.040*float(label['nominal_world_H40'][0][9])
                reference=.044*float(label['installed_reference_H40'][0][9])
                if abs(expert-reference)>1e-12:raise ValueError('EXPERT_NOMINAL_AND_REFERENCE_WIDTH_MAP_DISAGREE')
                contexts[index]=dict(episode=episode['id'],phase=w['phase'],index=index,
                    previous_command_width_m=last,previous_command_source_s=source,
                    query_native_time_s=label['native_time_s'],command_interval_s=dt,
                    measured_width_m=float(measured[w['anchor']]),expert_first_width_m=expert)
        originals[episode['id']]=dict(live_labels_sha256=capture['labels']['sha256'],metric_sha256=sha(case/'native-01/MOBILE_SAMPLES.private.jsonl'))
    assert len(contexts)==48
    write(root/'CONTEXT48.private.json',dict(records=contexts,source_pins=originals,
        measured_width='.044*original_measured_mechanical_fraction',
        command_width='.044*previous_actually_emitted_installed_EEF10_mechanical_fraction',
        model_width='.002+.040*single_sigmoid_decoded_nominal_fraction',
        seam_metric_is_not_a_new_native_gate=True))
    completed={}

    def analyze(label,rows):
        records=[]
        for row in rows:
            index=row['window']['index']if'window'in row else row['index'];ctx=contexts[index]
            output=row['output']
            if sha(output['path'])!=output['sha256']:raise ValueError('SAVED_MODEL_PREDICTION_CHANGED')
            with np.load(output['path'],allow_pickle=False)as z:physical=z['physical_H40']
            first=.002+.040*float(physical[0,9]);previous=ctx['previous_command_width_m']
            records.append(dict(**ctx,noise=row['noise'],model_first_width_m=first,
                model_first_minus_previous_command_m=first-previous,
                expert_first_minus_previous_command_m=ctx['expert_first_width_m']-previous,
                model_command_seam_speed_m_s=(first-previous)/ctx['command_interval_s'],
                expert_command_seam_speed_m_s=(ctx['expert_first_width_m']-previous)/ctx['command_interval_s'],
                model_first_minus_measured_width_m=first-ctx['measured_width_m'],
                model_first_minus_expert_first_m=first-ctx['expert_first_width_m']))
        if len(records)!=96 or len({(r['index'],r['noise'])for r in records})!=96:raise ValueError('FIXED48_NOISE2_FULL_DENOMINATOR')
        target=root/(label+'.private.json');write(target,dict(records=records,summary=summarize(records),
            new_generations=0,new_updates=0,physical_requests=0,original_scores_or_gates_changed=False))
        completed[label]=dict(summary=summarize(records),artifact=dict(path=str(target),sha256=sha(target)))
        write(root/'PROGRESS.private.json',dict(epoch=time.time(),completed=completed,new_model_generations=0,new_physical_requests=0))

    baseline=[r for r in read(goal/'ALL_SAMPLER_RECORDS.private.json')if r['route']=='cached_actual']
    analyze('C20_1000',baseline)
    for arm in ('fm-control','grip-temporal'):
        endpoint=training/arm/'RESULT.private.json';replay=training/(arm+'-fresh-rgb')/'RESULT.private.json'
        while time.time()<spec['deadline_epoch']-60:
            if(root/'STOP.json').exists():raise TimeoutError('BOUNDED_SEAM_REPORT_STOP')
            if endpoint.exists():
                result=read(endpoint)
                if result['exit_code']!=0:raise RuntimeError('PRESERVED_LEARNER_FAILURE_SEAM_REPORT_NOT_RELAUNCHING')
                if result['new_updates']!=500 or result['new_training_visits']!=16000:raise ValueError('TRUE_NEW500_ENDPOINT')
                if replay.exists():
                    if read(replay)['exit_code']!=0:raise RuntimeError('FRESH_RGB_FAILED_NO_SEAM_SUCCESS_CLAIM')
                    break
            current=unit_state(spec['training_unit'])
            if not current['pid']:
                raise RuntimeError('TRAINING_HANDLE_TERMINAL_WITHOUT_REQUIRED_ENDPOINT_OR_RGB')
            if current['pid']!=spec['training_main_PID']or current.get('proc_start')!=spec['training_proc_start']:
                raise RuntimeError('TRAINING_OWNER_IDENTITY_CHANGED_NO_REPLAY')
            write(root/'WAIT.private.json',dict(epoch=time.time(),arm=arm,process=current,new_generations=0,physical_requests=0))
            time.sleep(30)
        else:raise TimeoutError('FINITE_MATCHED_ENDPOINT_SEAM_WAIT')
        analyze(arm,read(training/arm/'TRAIN48_PREDICTIONS.private.json'))
    output=dict(schema='SLA_CLOSE_PARENT_AND_MATCHED_ENDPOINT_GRIPPER_SEAMS_V1',exit_code=0,
        completed=completed,new_model_generations=0,new_optimizer_updates=0,new_physical_requests=0,
        original_scores_or_gates_changed=False,physical_admission=False,goal_met=False,
        elapsed_wall_s=time.time()-started,CPU_s=time.process_time())
    write(root/'RESULT.private.json',output);return output


if __name__=='__main__':
    s=read(sys.argv[2]);root=Path(s['registered_root'])
    try:run(s);code=0
    except BaseException:
        write(root/'RESULT.private.json',dict(exit_code=1,failure=traceback.format_exc(),new_generations=0,
                                             new_physical_requests=0,original_scores_or_gates_changed=False));code=1
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
    close(s,code);raise SystemExit(code)
