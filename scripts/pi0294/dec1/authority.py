"""Nonadditive DEC1 accounting and scope identity, without native side effects."""
import math

AUTHORITY = 'a7a7ecd9a6be62a7221af25a5e32a0009b076a84'
PARENT_AUTHORITY = '9eff29a44a193dd4d3620dc0bfe5764a402eff42'
PROTOCOL = 'docs/protocols/DEC1_CONTROLLER_DISJOINT_VLA_SCIENCE_GO_20261021.md'
CARD = 'reports/PI0294/design/DEC1_CONTROLLER_DISJOINT_SCIENCE_GO_20261021.json'
PARENT = '94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38'
R0 = 'fb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7'
BANK = '60f7c35a05766fe23974eba3a72f53ca4db5545a3768e6c1cfea164ab4e58294'
HARD_CLOSE = 1792952996


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise PermissionError('DEC1_FINITE_NONNEGATIVE_ACTUAL_COST')
    return value


def available(adoption, local, hopper):
    """Caps are classifications INSIDE PV1, not a second spendable budget."""
    if (adoption.get('authority_release') != PARENT_AUTHORITY
            or local.get('run_uuid') != adoption.get('run_uuid')
            or hopper.get('campaign_run_uuid') != adoption.get('run_uuid')
            or hopper.get('authority_release') != PARENT_AUTHORITY
            or adoption.get('hard_closeout_epoch') != HARD_CLOSE
            or local.get('T0_epoch') != adoption.get('T0_epoch')
            or local.get('hard_closeout_epoch') != HARD_CLOSE
            or local.get('unknown_effects') != [] or local.get('pending_updates') != []
            or hopper.get('unknown_effects') != [] or hopper.get('pending_updates') != []):
        raise PermissionError('DEC1_ORIGINAL_CAMPAIGN_IDENTITY_CLOCK_AND_EFFECTS')
    used, reserved = hopper['used'], hopper['reserved']
    if (used['PBS_submissions'] != len(hopper['PBS_submissions'])
            or used['optimizer_updates'] != hopper['optimizer_updates']
            or used['neural_queries'] != hopper['generation_queries']):
        raise PermissionError('DEC1_NATIVE_COUNTERS_MUST_RECONCILE')
    if (number(local['GPU_allocated_s']) > number(used['H200_s'])
            or number(local['GPU_reserved_s']) > number(reserved.get('H200_s', 0))):
        raise PermissionError('DEC1_NO_STALE_NATIVE_GPU_PROJECTION')
    floor = local['final_reserve_inside_total']
    if floor['H200_s'] != 72000:
        raise PermissionError('DEC1_FINAL_TWENTY_HOURS_PROTECTED')
    active = [s for s in hopper['PBS_submissions'] if s['status'] not in
              ('NATIVE_TERMINAL', 'NATIVE_QSUB_REJECTED_NO_JOB')]
    if any(not s.get('job_id') for s in active):
        raise PermissionError('DEC1_UNKNOWN_SUBMISSION_NO_PARALLEL_MIGRATION')
    native_reserved = sum(number(s['reserved_H200_s']) + number(s['accounting_pad_s']) for s in active)
    if native_reserved != reserved.get('H200_s', 0):
        raise PermissionError('DEC1_ALL_ACTIVE_JOB_RESERVATIONS_PRESERVED')
    gpu = 96*3600-number(used['H200_s'])-number(reserved.get('H200_s', 0))-72000
    cpu = local['CPU_s']; cpu_reserved = local['CPU_reserved_s']
    pending = {key:sum(number(c.get(key+'_reserved', 0)) for c in local.get('children', []))
               for key in ('episode_starts', 'physics_ticks', 'engine_starts')}
    from scripts.pi0294.pv1.resources import pbs_limit
    remaining = dict(GPU_s=gpu, GPU_submissions=pbs_limit(hopper)-len(hopper['PBS_submissions']),
        optimizer_updates=16384-number(hopper['optimizer_updates']),
        neural_queries=48000-number(hopper['generation_queries'])-number(floor['neural_queries']),
        CoreLab_CPU_s=320*3600-number(cpu['CoreLab'])-number(cpu_reserved['CoreLab']),
        Hopper_CPU_s=160*3600-max(number(cpu['Hopper']),number(hopper['CPU_s']))-number(cpu_reserved['Hopper']),
        local_CPU_s=48*3600-number(cpu['local_PC'])-number(cpu_reserved['local_PC']),
        episode_starts=1000-number(local['episode_starts'])-number(floor['episode_starts'])-pending['episode_starts'],
        physics_ticks=40000000-sum(number(n) for n in local['physics_ticks'].values())
            -number(floor['live_ticks'])-number(floor['shadow_ticks'])-pending['physics_ticks'],
        constructors=100000-number(local['engine_starts'])-pending['engine_starts'])
    if min(remaining.values()) < 0:
        raise PermissionError('DEC1_ORIGINAL_REMAINING_RESOURCE_NEGATIVE')
    caps = dict(GPU_s=48*3600,GPU_submissions=8,optimizer_updates=8192,neural_queries=14000,
        CoreLab_CPU_s=96*3600,Hopper_CPU_s=96*3600,local_CPU_s=8*3600,
        episode_starts=136,physics_ticks=5000000,constructors=12000)
    return dict(schema='DEC1_NONADDITIVE_AVAILABLE_CLASSIFICATION_V1',
        original_remaining_after_spent_reservations_and_final=remaining,
        DEC1_effective_caps={k:min(v,remaining[k]) for k,v in caps.items()},
        DEC1_authorized_caps=caps, old_costs_or_reservations_reset=False,
        active_job_ids=[s['job_id'] for s in active],
        no_new_GPU_while_held_may_recover=bool(active), site_permission='NOT_GRANTED_BY_ACCOUNTING')
