"""Bind the two original CLOCK strict RGB16 receipts to the frozen native order."""
import argparse
import json
from pathlib import Path

from go2arx5_vla.sla_adapter import clock500_package
from go2arx5_vla.sla_adapter.clock500 import ARMS, ENDPOINT, PLAN
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_evaluation import RESTORE
from scripts.pi0292.durable import write
from scripts.sla import adapt5k_endpoint_transport as transport
from scripts.sla.clock500_export import POPULATION
from scripts.sla.freeze_clock500_native_population import validate
from scripts.sla.replay_adapt0_core import trained_inputs


def ordered_conditions(pop):
    validate(pop)
    cases = {r['id']:r for selected in pop['conditions'].values() for r in selected}
    rows = []
    for entry in pop['native_order']:
        case = cases[entry['condition']]
        hold = case['instruction'].startswith('HOLD-')
        queries = 120 if hold else 180
        rows.append(dict(entry,reset=case['reset'],noise_seed=case['noise_seed'],family=case['family'],
                         instruction=case['instruction'],layout=case['cluster'],native_ticks=5703+250*queries,
                         model_generations=queries))
    if (len(rows),sum(x['native_ticks'] for x in rows),sum(x['model_generations'] for x in rows))!=(72,3290616,11520):
        raise ValueError('CLOCK500_CORE_NATIVE_DENOMINATOR_CHANGED')
    return rows


def prove_arm(arm, card):
    package = card['package']
    manifest = clock500_package.Provider.read_manifest(Path(package['path']),package['sha256'])
    receipt, formal = map(read_pin,(card['strict_replay'],card['evaluation']))
    cp = trained_inputs(dict(card,phase=ENDPOINT,arm=arm),manifest,formal)
    rows = receipt['rows']
    if (receipt['schema']!=transport.CLOCK_RGB or receipt['status']!='CORE_CLOCK500_RGB16_EXACT'
            or receipt['arm']!=arm or receipt['phase']!=ENDPOINT
            or receipt['comparison_sha256']!=manifest['comparison_sha256']
            or receipt['checkpoint_sha256']!=cp or receipt['package']!=package['sha256']
            or receipt['core_custody']!=card['core_custody'] or receipt['evaluation']!=card['evaluation']
            or receipt['panel']!=card['panel'] or receipt['exit_code'] or receipt['passed'] is not True
            or receipt['physics'] is not False or receipt['hardware_allowed'] is not False
            or receipt['site']['host']!='corelab-4080s'
            or receipt['site']['GPU_uuid']!=receipt['environment']['GPU_uuid']
            or receipt['environment']['backend']!='gpu' or receipt['environment']['precision']!='highest'
            or receipt['generation_attempts']!=len(RESTORE) or receipt['model_generations']!=len(RESTORE)
            or any(receipt[k] for k in ('optimizer_updates','optimization_visits','native_requests','native_ticks'))
            or tuple(r['ordinal'] for r in rows)!=RESTORE):
        raise PermissionError('CLOCK500_ARM_NEEDS_ITS_OWN_SIXTEEN_STRICT_CORE_RESULTS')
    for row in rows:
        original = formal['rows']['target'][row['ordinal']]
        if (not all(row[key] is True for key in ('normalized_exact','physical_exact','guard_exact'))
                or any(row[k]!=original[k] for k in ('index','noise_id','identity','normalized_sha256','decoded_sha256'))):
            raise ValueError('CLOCK500_ARM_STRICT_RGB_EVIDENCE_CHANGED')
    return manifest,receipt['environment']


def qualify(spec):
    population = spec['population']
    if spec['plan']!=PLAN or population['sha256']!=POPULATION or set(spec['arms'])!=set(ARMS):
        raise ValueError('CLOCK500_CORE_REQUIRES_EXACTLY_ITS_FROZEN_COMPARISON')
    rows = ordered_conditions(read_pin(population))
    a, b = (prove_arm(arm,spec['arms'][arm]) for arm in ARMS)
    shared = ('comparison_sha256','parent_state_sha256','D2_sha256','MIX_sha256','parent_manifest',
              'D1_freeze_sha256','state_statistics_sha256','action_statistics_sha256','tokenizer_sha256',
              'initial_state_sha256','partition_sha256','frozen_fingerprint','native_population_sha256','panel_sha256')
    if (any(a[0][k]!=b[0][k] for k in shared) or a[1]!=b[1]
            or a[0]['native_population_sha256']!=population['sha256']
            or a[0]['checkpoint_sha256']==b[0]['checkpoint_sha256']):
        raise ValueError('CLOCK500_CORE_REQUIRES_ONE_PAIRED_PARENT_POPULATION_AND_BACKEND')
    return dict(schema='SLA_CLOCK500_CORE_PAIRED_RGB16_ACTUAL_V1',plan=PLAN,population=population,
        comparison_sha256=a[0]['comparison_sha256'],GPU_environment=a[1],
        endpoints={arm:spec['arms'][arm]['strict_replay'] for arm in ARMS},archived_RGB_reads=32,
        planned_native_order=rows,native_dispatch_qualified=False,physics=False,hardware_allowed=False,
        optimizer_updates=0,model_generations=0,native_requests=0,native_ticks=0)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = qualify(read_pin(dict(path=args.spec,sha256=args.sha256)))
    write(args.output,result)
    print(json.dumps(dict(comparison=result['comparison_sha256'],archived_RGB_reads=result['archived_RGB_reads'],
                          planned_cases=len(result['planned_native_order']),native_requests=0)))
