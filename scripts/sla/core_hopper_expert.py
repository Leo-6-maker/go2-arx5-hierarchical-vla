"""Identify the new expert scope and its immutable simulation-only population."""

PLAN = '171b5bedb290258954aac140157743e92e3b152e'
SCOPE = 'SLA_CORE_HOPPER_V1_EXPERT'
AUTHORITY = 'CURRENT_USER_SLA_CORE_HOPPER_PARALLEL_V1'
POP = 'd92fe34f122637bb6dc29bb93d9f7a3d60464f65c5474b00962432bbc07b4914'
OWNER = 'SLA_CORE_HOPPER_V1_UID2009'
# Rotate the same six instructions in both families, giving the six previously
# underrepresented cells their first attempts within each 12-condition block.
CELLS = [('F1','HOLD-blue'),('F2','HOLD-red'),('F1','PLACE-red-left'),('F2','PLACE-blue-right'),
         ('F1','PLACE-blue-right'),('F2','PLACE-red-left'),('F1','HOLD-red'),('F2','HOLD-blue'),
         ('F1','PLACE-blue-left'),('F2','PLACE-red-right'),('F1','PLACE-red-right'),('F2','PLACE-blue-left')]


def order(pop):
    if pop['plan'] != PLAN or pop['assignment_uses_outcomes'] or pop['old_TEST_reused']:
        raise ValueError('Unexpected population identity')
    banks = {}
    for split, count in [('TRAIN',240),('VAL',24)]:
        bank = sorted(pop['conditions'][split], key=lambda r:(r['geometry_variant'],CELLS.index((r['family'],r['instruction']))))
        ids = {r['condition_id'] for r in bank}
        if (len(bank) != count or len(ids) != count or
                any(r['id'] != r['condition_id'] or r['split'] != split or r['native_request_limit']['max_actual_requests'] != 1 for r in bank)):
            raise ValueError('Expert membership mismatch')
        banks[split] = bank
    t, v = banks['TRAIN'], banks['VAL']
    return t[:12] + v[:12] + t[12:24] + v[12:] + t[24:]


def repair_rows(pop,repair):
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
    if 'completed_pilot' in repair:
        from scripts.sla.core_hopper_repair_series import qualified_prefix
        return qualified_prefix(pop,repair)
    from scripts.sla.core_hopper_blue_carry import PILOT, PRIOR
    audit=read_pin(repair['audit'])
    prior=read_pin(repair['original_producer'])
    if repair.get('kind') == 'F2_RED_PLACE_PITCH':
        return red_place_rows(pop,repair,prior,audit)
    if repair.get('kind'):
        from scripts.sla.core_hopper_red_approach import CELL, KIND, PILOT as RED_PILOT, PITCH_DEGREES
        if (repair['kind']!=KIND or audit['status']!='PILOT_PITCH_GEOMETRY_VERIFIED'
                or audit['pilot']!=RED_PILOT or audit['prior']!=[CELL+'-00',CELL+'-01']
                or audit['pitch_deg']!=PITCH_DEGREES or audit['minimum_forbidden_gap_mm']<=0
                or any(audit[k] for k in ('VAL_inputs','native_requests','simulation_integrations','model_generations'))
                or prior['source_sha']!=audit['source_expert'] or prior['scope']!=SCOPE
                or prior['compiled_population']['sha256']!=POP or audit['distinct_scenes']!=2):
            raise ValueError('Pitch pilot requires its two original TRAIN stops and a verified registered scene')
        row=next(r for r in pop['conditions']['TRAIN'] if r['id']==RED_PILOT)
        cluster=next(c for c in pop['clusters'] if c['id']==row['scene_family'])
        if row['split']!='TRAIN' or cluster['scene']['sha256']!=audit['pilot_scene_sha256']:
            raise ValueError('Pitch pilot numerical scene differs')
        return [row]
    if (audit['status']!='TRAIN_STATIC_CARRY_BYPASS_VERIFIED' or audit['pilot']!=PILOT
            or audit['prior']!=PRIOR or prior['source_sha']!=audit['producer']
            or prior['scope']!=SCOPE or prior['compiled_population']['sha256']!=POP
            or audit['new_native_requests'] or audit['validation_outcomes_used']
            or min(audit['pilot_two_leg_centered_clearance_mm'])<=8.):
        raise ValueError('Unqualified carry repair or original owner')
    row=next(r for r in pop['conditions']['TRAIN'] if r['id']==PILOT)
    cluster=next(c for c in pop['clusters'] if c['id']==row['scene_family'])
    if row['split']!='TRAIN' or cluster['scene']['sha256']!=audit['pilot_scene_sha256']:
        raise ValueError('Unexpected unopened training scene')
    return [row]



def red_place_rows(pop,repair,prior,audit):
    from scripts.sla.core_hopper_red_approach import PITCH_DEGREES

    if (repair['audit']['sha256']!='438026277d214041adb3bb27a663220523a245f59e967c6854386624d420a9df'
            or repair['original_producer']['sha256']!='89f058d5cd3a69cae8f68eb6a63f63a040f20151d07daf0d88e6a0ec5c3f8ffe'
            or prior['scope']!=SCOPE or prior['compiled_population']['sha256']!=POP
            or prior['source_sha']!=audit['source_producer'] or audit['status']!='SAVED_PLACE_APPROACH_GEOMETRY'
            or audit['candidate_world_pitch_deg']!=PITCH_DEGREES
            or any(audit[k] for k in ('VAL_inputs','native_requests','native_integrations','model_generations','optimizer_updates'))):
        raise ValueError('Expected the registered TRAIN-only wrist-contact projection')
    conditions={r['id']:r for r in pop['conditions']['TRAIN']}
    scenes={c['id']:c['scene']['sha256'] for c in pop['clusters']}
    rows=[]
    for side in ('left','right'):
        cell='CHV1-TRAIN-F2-PLACE-red-'+side
        row=conditions[cell+'-02']
        test=audit['measurements'][side]
        if (row['split']!='TRAIN' or row['instruction']!='PLACE-red-'+side or row['family']!='F2'
                or test['unopened_condition']!=row['id'] or scenes[row['scene_family']]!=test['scene_sha256']
                or [r['condition'] for r in test['rows']]!=[cell+'-00',cell+'-01']
                or any(r['summary']['0']['minimum_forbidden_gap_mm']>=0 or
                       r['summary']['20']['minimum_forbidden_gap_mm']<=0 or
                       not r['summary']['20']['bilateral_pad_intersection'] for r in test['rows'])):
            raise ValueError('Registered side, original contact pair or pitch projection differs')
        rows.append(row)
    return rows


def validate_member(case, authority):
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
    ref = case['compiled_population']
    if (ref['sha256'] != POP or authority['compiled_population_sha256'] != POP
            or case['owner'] != OWNER or case['registry_scope'] != SCOPE
            or case['sim_only_role'] != 'EXPERT' or case['resident_server_spec'] is not None
            or authority['T0_epoch']+14*86400 != authority['deadline_epoch']
            or case['source_sha'] != authority['frozen_producer']
            or case['new_model_generations_budget'] != 0):
        raise PermissionError('Exact Core/Hopper expert scope, producer and budgets required')
    population = read_pin(ref)
    repair=authority.get('carry_repair')
    if repair!=case.get('carry_repair'):
        raise PermissionError('Teacher version and authority disagree')
    if repair:
        allowed=repair_rows(population,repair)
        if (case['panel_entry_id'] not in {r['id'] for r in allowed} or authority['native_request_cap']!=len(allowed)
                or authority['native_tick_cap']!=sum(r['native_max_ticks'] for r in allowed)):
            raise PermissionError('Expected the registered TRAIN pilot membership and total tick budget')
    elif authority['native_request_cap']!=264 or authority['native_tick_cap']!=12065592:
        raise PermissionError('Original expert population and budget required')
    row = next(r for r in order(population) if r['condition_id'] == case['panel_entry_id'])
    cluster = next(c for c in population['clusters'] if c['id'] == row['scene_family'])
    condition = case['mobile_condition']
    if (row['split'] != case['collection_split'] or row['instruction'] != case['task_instruction_id']
            or row['native_max_ticks'] != case['max_ticks'] or cluster['reset'] != condition['reset']
            or row['scene_seed'] != condition['seed'] or cluster['id'] != condition['layout']
            or cluster['family'] != condition['family'] or cluster['scene']['path'] != case['model_path']
            or cluster['scoring_geometry'] != case['multitask_geometry']
            or case['frozen_files'][case['model_path']] != cluster['scene']['sha256']
            or case['expert_training_eligible'] != (row['split']=='TRAIN')):
        raise PermissionError('Expert case is not bound to its numerical scene, split and clocks')


def check_world_lease(case, fd):
    import os
    import time
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
    lease = read_pin(case['site_world_lease'])
    held = os.fstat(fd)
    if (case['owner'] != OWNER or lease['uid'] != os.getuid() or lease['owner_pid'] != os.getppid()
            or lease['condition'] != case['panel_entry_id'] or lease['lock'] != case['mobile_world_lock']
            or (held.st_dev, held.st_ino) != (lease['lock_device'], lease['lock_inode'])
            or lease['deadline_epoch'] != case['allocation_deadline_epoch']
            or not time.time() < case['deadline_epoch'] <= lease['deadline_epoch']
            or lease['deadline_epoch'] - lease['acquired_epoch'] > 2160):
        raise PermissionError('A live bounded single-world lease must precede the native child')
