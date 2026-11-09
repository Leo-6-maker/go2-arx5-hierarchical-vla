"""Centimetre-scale TRAIN layout feasibility, separately versioned and never BC.

The first native population varies X only. Y is static preparation; objects,
cameras, materials, Teacher math, controller and scoring stay unchanged.
One restart per position is a feasibility screen, not a reliability estimate.
"""
from pathlib import Path
from . import task_learning_grasp_v1 as task
from . import task_learning_teacher_panel_v1 as teacher

SCHEMA = 'TASK_LEARNING_NEXT_TRAIN_FEASIBILITY_PANEL_V1'
PURPOSE = 'NEXT_TRAIN_TEACHER_FEASIBILITY_NOT_BC'
EXPERIMENT = 'TASK_LEARNING_NEXT_TRAIN_LAYOUT_FEASIBILITY_NO_BC'
NATIVE_SCHEMA = 'TASK_LEARNING_NEXT_TRAIN_FEASIBILITY_NATIVE_ACTUAL_V1'
OFFSETS = {f'NEXTTRAIN_{axis}{sign}{cm:02d}':
           ((direction*cm*.01, 0.) if axis == 'X' else (0., direction*cm*.01))
           for axis in ('X', 'Y') for cm in (1, 2, 3)
           for sign, direction in (('M', -1), ('P', 1))}
NATIVE_LAYOUTS = tuple(k for k in OFFSETS if k.startswith('NEXTTRAIN_X'))


def validate(panel):
    from .task_learning_scene_v1 import scene_id
    rows = panel.get('assignments') or []
    task.need(panel.get('schema') == SCHEMA
        and panel.get('task_learning_role') == task.EXPANSION_QUALIFICATION_ROLE
        and panel.get('protocol') == task.protocol(task.EXPANSION_QUALIFICATION_ROLE)
        and panel.get('Teacher_implementation_source_sections_sha256')
            == teacher.teacher_fingerprint() == teacher.FROZEN_TEACHER
        and len(panel.get('source_sha', '')) == 40
        and tuple(r['layout'] for r in rows) == NATIVE_LAYOUTS
        and len({r['native_run'] for r in rows}) == len(NATIVE_LAYOUTS),
        'FIXED_SIX_NEXT_TRAIN_FEASIBILITY_REQUESTS')
    for row in rows:
        task.need(row.get('restart') == 1 and row.get('split') == 'TRAIN'
            and row.get('parent') == scene_id(row['layout'])
            and row.get('command') == task.COMMAND and row.get('purpose') == PURPOSE
            and row.get('Teacher_mode') == 'NORMAL_TEACHER_V7'
            and row.get('diagnostic_control') is None
            and Path(row['native_run']).name == row['native_run']
            and not row['native_run'].startswith('.')
            and len(row.get('model_sha256', '')) == 64,
            'INDEPENDENT_FEASIBILITY_NEVER_OLD_TRAIN_DEV_OR_TEST')
    return rows


def classify(row, **kwargs):
    return teacher.classify(row, **kwargs,
        expected_role=task.EXPANSION_QUALIFICATION_ROLE,
        expected_native_schema=NATIVE_SCHEMA)


def summarize(panel, outcomes):
    rows = validate(panel)
    expected = {r['native_run'] for r in rows}
    task.need(len({v['native_run'] for v in outcomes}) == len(outcomes)
        and all(v['native_run'] in expected for v in outcomes),
        'NO_RETRY_OR_FOREIGN_NEXT_TRAIN_RESULT')
    groups = {}
    for row in rows:
        cm = round(abs(OFFSETS[row['layout']][0])*100)
        group = groups.setdefault(str(cm), dict(requested=0, attempted=0,
            reset_ready=0, task_successes=0, hard_safety_stops=0, technical_failures=0))
        group['requested'] += 1
        value = next((v for v in outcomes if v['native_run'] == row['native_run']), None)
        if value is None:
            continue
        group['attempted'] += value['attempted'] is True
        group['reset_ready'] += value['reset_ready'] is True
        group['task_successes'] += (value['task_success'] is True
            and value['reset_ready'] is True and value['technical_valid'] is True)
        group['hard_safety_stops'] += value['hard_safety_stop'] is True
        group['technical_failures'] += value['technical_valid'] is not True
    for group in groups.values():
        group['success_per_attempted'] = (group['task_successes']/group['attempted']
            if group['attempted'] else None)
        group['success_given_ready'] = (group['task_successes']/group['reset_ready']
            if group['reset_ready'] else None)
    return dict(schema='TASK_LEARNING_NEXT_TRAIN_FEASIBILITY_SUMMARY_V1',
        requested=len(rows), attempted=sum(v['attempted'] is True for v in outcomes),
        complete=len(outcomes) == len(rows) and all(v['attempted'] is True for v in outcomes),
        by_absolute_X_offset_cm=groups, outcomes=outcomes,
        qualified_training_demonstrations=0, Processor_admitted=False,
        GR00T_queries=0, model_updates=0, large_collection_authorized=False,
        reliability_established=False, Y_native_qualification='NOT_RUN',
        next_gate='FIRST_STUDENT_FULL_DEV_FAILURE_ANALYSIS_BEFORE_DATA_PRODUCTION')
