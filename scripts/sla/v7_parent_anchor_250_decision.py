"""Apply the V3 no-new-harm TRAIN screen before any B6 250-to-500 extension."""
import argparse
from pathlib import Path

from m38_gpu_preflight_v1 import read, require, sha
from scripts.pi0292.durable import write

ARMS = ('LAMBDA_ZERO', 'SHARED_L2SP')


def decide(root, panel, output):
    require(not output.exists(), 'Preserve the original continuation decision')
    proof = read(root/'CORE-PROVIDERS/ACTUAL.json')
    fixed = read(panel)
    summary = read(root/'TRAIN-PANEL-OUTPUT/SUMMARY.json')
    actual = read(root/'TRAIN-PANEL-OUTPUT/ACTUAL.json')
    parameters = read(root/'PARAMETER-DISTANCE.json')
    require(proof['complete_training_endpoint'] and proof['step'] == parameters['complete_step'] == 250
            and fixed['split'] == 'TRAIN_DIAGNOSTIC' and fixed['protected_conditions'] == 0
            and fixed['teacher_scene_count'] == 4 and len(fixed['frames']) == 37
            and summary['panel_sha256'] == actual['panel_sha256'] == sha(panel)
            and actual['complete'] and actual['actual_forwards'] + actual.get('reused_old_forwards', 0) == 222
            and summary['actual_sha256'] == sha(root/'TRAIN-PANEL-OUTPUT/ACTUAL.json'),
            'Require the same complete TRAIN panel and independently restored matched endpoint')
    rows = {(r['model'], r['frame']): r for r in summary['rows']}
    require(len(rows) == 111 and all(r['repeat_max_abs'] == 0 for r in rows.values()),
            'Incomplete or nonrepeatable fixed-condition measurements')
    new_illegal, lifts = [], []
    for frame in fixed['frames']:
        control, treatment = (rows[arm, frame['id']] for arm in ARMS)
        if treatment['g7']['original_execution_speed_exceeded'] and not control['g7']['original_execution_speed_exceeded']:
            new_illegal.append(dict(frame=frame['id'], control=control['g7'], treatment=treatment['g7']))
        if frame['phase'] in ('LIFT', 'target_lift20mm'):
            lifts.append(dict(frame=frame['id'], control_up_mm=control['p10_up_request_mm'],
                              treatment_up_mm=treatment['p10_up_request_mm']))
    suppressed = [r for r in lifts if r['treatment_up_mm'] < r['control_up_mm']]
    safe = {frame: {model: rows[model, frame] for model in ('R4', *ARMS)}
            for frame in ('R4_SAFE_GRASP', 'R4_SAFE_LIFT')}
    train_pass = parameters['shared_squared_distance_reduction'] >= .05 and not new_illegal and not suppressed
    source_paths = [root/(arm+'-SOURCE/RESULT.json') for arm in ARMS]
    source_complete = all(p.is_file() for p in source_paths)
    source_alerts = {}
    if source_complete:
        for arm, path in zip(ARMS, source_paths, strict=True):
            source = read(path)
            require(source['step'] == 250 and source['model_forwards'] == 42
                    and source['pair_sha256'] == sha(root/'CORE-PROVIDERS/ACTUAL.json'),
                    'A different endpoint supplied the source result')
            source_alerts[arm] = source['retention_alert_domains']
    if not train_pass:
        decision = 'NO_EXTENSION_TRAIN_GATE_FAILED'
    elif not source_complete:
        decision = 'WAIT_FOR_SOURCE_RETENTION'
    else:
        decision = 'CONDITIONAL_PAIR_CONTINUE_TO_500' if not any(source_alerts.values()) else 'NO_EXTENSION_SOURCE_ALERT'
    write(output, dict(plan='8e6ffa192d3e6c0adce7e2f89ca97e8ea20f461a', source_step=250,
          decision=decision, source_complete_at_decision=source_complete, source_alerts=source_alerts,
          shared_squared_distance_reduction=parameters['shared_squared_distance_reduction'],
          new_raw_illegal_relative_to_control=new_illegal, lift_requests=lifts,
          lower_lift_request_relative_to_control=suppressed, original_safe_states=safe,
          scientific_promotion=False, dynamic_L1_or_WBC_tested=False,
          proposed_new_formal_updates=500 if decision == 'CONDITIONAL_PAIR_CONTINUE_TO_500' else 0,
          cumulative_formal_updates=5100, formal_cap=8000, engineering_actual=24, engineering_cap=24,
          preserved_student_native_screen_slots=4, new_native_integrations=0, new_optimizer_updates=0,
          inputs=[dict(path=str(p), sha256=sha(p)) for p in
                  (panel, root/'CORE-PROVIDERS/ACTUAL.json', root/'TRAIN-PANEL-OUTPUT/SUMMARY.json',
                   root/'TRAIN-PANEL-OUTPUT/ACTUAL.json', root/'PARAMETER-DISTANCE.json')]))
    print(output, sha(output), flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'panel', 'output'):
        cli.add_argument(name, type=Path)
    decide(**vars(cli.parse_args()))
