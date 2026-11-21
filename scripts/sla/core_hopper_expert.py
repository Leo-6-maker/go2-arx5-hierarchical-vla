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


def validate_member(case, authority):
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
    ref = case['compiled_population']
    if (ref['sha256'] != POP or authority['compiled_population_sha256'] != POP
            or case['owner'] != OWNER or case['registry_scope'] != SCOPE
            or case['sim_only_role'] != 'EXPERT' or case['resident_server_spec'] is not None
            or authority['native_request_cap'] != 264 or authority['native_tick_cap'] != 12065592
            or authority['T0_epoch']+14*86400 != authority['deadline_epoch']
            or case['source_sha'] != authority['frozen_producer']
            or case['new_model_generations_budget'] != 0):
        raise PermissionError('Exact Core/Hopper expert scope, producer and budgets required')
    population = read_pin(ref)
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
