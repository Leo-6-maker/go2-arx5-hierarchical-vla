"""Matched BOOK data arms; no data collection, statistics fitting or launching.

All arms reset to the same b248 delta and use the same frozen common TRAIN
statistics. Only their actual expert population differs. Historical Phase
results remain a reference, not the control for this changed processing chain.
"""
import re

SCHEMA='BOOK_BC_DAGGER_MATCHED4096_V1'
DATA_SCHEMA='BOOK_BC_DAGGER_MATCHED_TRAIN_ARM_V1'
POPULATION_SCHEMA='BOOK_BC_DAGGER_MATCHED_ARM_POPULATION_V1'


def population_rows(arm, entries):
    from .book_m32_export import COUNTS
    old_ids=[*[f'BOOK_SOURCE_SEED{i:03}'for i in range(15)],'BOOK_RECOVERY_TRAIN000_EXPERT_SUFFIX',
             *[f'BOOK_M32_RECOVERY{i:03}_EXPERT_SUFFIX'for i in range(5)]]
    original=dict(zip(old_ids,COUNTS))
    if arm not in ('D0','D1','D2'):raise ValueError('THREE_MATCHED_DATA_ARMS_ONLY')
    result=[];seen=set()
    for row in entries:
        if (set(row)!={'id','windows'}or type(row['id'])is not str or type(row['windows'])is not int
                or not 1<=row['windows']<=601 or row['id']in seen):
            raise ValueError('UNIQUE_ACTUAL_FROZEN_EXPERT_ENTRIES_REQUIRED')
        name=row['id'];seen.add(name)
        if name in original:
            if row['windows']!=original[name]:raise ValueError('ORIGINAL_D0_EXPERT_WINDOWS_CHANGED')
        elif arm=='D1':
            if not re.fullmatch(r'BOOK_GEOMETRY_TRAIN00[1-6]',name)or row['windows']!=601:
                raise ValueError('D1_ONLY_NEW_FULL_GEOMETRY_EXPERT')
        elif arm=='D2':
            if not re.fullmatch(r'BOOK_LATE_RECOVERY_TRAIN00[0-7]_EXPERT_SUFFIX',name)or row['windows']>=601:
                raise ValueError('D2_ONLY_ACTUAL_LATE_TRAIN_EXPERT_SUFFIX')
        else:raise ValueError('D0_ORIGINAL_POPULATION_ONLY')
        result.append(dict(row))
    if not set(original)<=seen:raise ValueError('PRESERVE_ALL_ORIGINAL_D0_EXPERTS')
    if arm!='D0'and len(result)==len(original):raise PermissionError('NOT_ADMITTED_NO_NEW_QUALIFIED_DATA')
    return sorted(result,key=lambda r:r['id'])


def plan(arm,dataset_sha256,statistics_sha256,population_sha256,entries,*,steps=4096):
    from .book_m32_learning import curve_plan
    if type(steps)is not int or steps not in(10,4096):raise ValueError('NATIVE10_OR_FINITE4096_ONLY')
    rows=population_rows(arm,entries)
    p=curve_plan(dataset_sha256,statistics_sha256,population_sha256)
    extra=len(rows)-21
    p.update(schema=SCHEMA,data_arm=arm,TRAIN_entries=rows,TRAIN_episodes=len(rows),TRAIN_windows=sum(r['windows']for r in rows),
        pure_full_teacher_episodes=15+(extra if arm=='D1'else 0),mixed_recovery_episodes=6+(extra if arm=='D2'else 0),
        optimization_population='ALL_AND_ONLY_FROZEN_'+arm+'_ACTUAL_EXPERT_WINDOWS',
        sampling='SAME_NUMPY_PCG64_UNIFORM_CYCLIC_EXPERT_H40_RULE_FOR_ALL_ARMS',
        common_statistics_required=True,statistics_version_changed=True,statistics_refit=False,
        capacity_qualification_only=steps==10,student_DEV_and_rejected_commands_as_labels=False,
        optimizer_initialization='FRESH_ADAMW_FROM_SAME_CLOSED_B248_INITIAL_NOT_PHASE4096',
        phase4096_not_a_matched_statistics_control=True,
        purpose='Matched D0 vs new full expert BOOK vs actual student-visited TRAIN expert suffixes. Same common statistics, initial weights, loss, batch32 and4096 schedule; data population is the intervention.')
    if steps==10:p.update(max_steps=10,TRAIN_visits=320,checkpoint_steps=[10],comparison_endpoint=10,generation_queries=32,native_diagnostic_loss_forwards=32)
    return p


def is_plan(p):
    if p.get('schema')!=SCHEMA:return False
    try:return p==plan(p['data_arm'],p['dataset_manifest_sha256'],p['statistics_sha256'],p['TRAIN_population_sha256'],p['TRAIN_entries'],steps=p['max_steps'])
    except(KeyError,TypeError,ValueError,PermissionError):return False


def validate_population(manifest,population,p):
    if not is_plan(p):raise PermissionError('EXACT_FROZEN_BC_DAGGER_RECIPE')
    if (manifest.get('schema')!=DATA_SCHEMA or manifest.get('arm')!=p['data_arm']
            or manifest.get('training_admitted')is not True
            or manifest.get('statistics_sha256')!=p['statistics_sha256']
            or population.get('schema')!=POPULATION_SCHEMA or population.get('arm')!=p['data_arm']
            or population.get('statistics_sha256')!=p['statistics_sha256']
            or population.get('DEV_TEST_confirm_admitted')is not False):
        raise PermissionError('OFFICIAL_TRAIN_ONLY_MATCHED_ARM_ADMISSION_REQUIRED')
    rows=manifest['episodes']
    if any(r.get('split')!='TRAIN'or r.get('task')!='BOOK'for r in rows):
        raise PermissionError('BOOK_ONLY_TRAIN_NO_DEV_TEST_OR_OTHER_TASKS')
    selected=population_rows(p['data_arm'],[dict(id=r['id'],windows=r['windows'])for r in rows])
    if selected!=p['TRAIN_entries']or selected!=population_rows(p['data_arm'],[dict(id=r['id'],windows=r['windows'])for r in population['episodes']]):
        raise PermissionError('EXACT_FROZEN_POPULATION_NOT_WINDOW_PADDING')
    keys=('windows','sha256','raw_root')
    if ({r['id']:tuple(r[k]for k in keys)for r in rows}
            !={r['id']:tuple(r[k]for k in keys)for r in population['episodes']}):
        raise PermissionError('NUMERIC_AND_RAW_PROVENANCE_MUST_JOIN_EXACTLY')
    if manifest['TRAIN_windows']!=p['TRAIN_windows']:raise PermissionError('EXACT_ACTUAL_WINDOW_COUNT')
    return True
