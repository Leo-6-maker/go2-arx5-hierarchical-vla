"""Portable admission and original-prefix contract for the registered Wave A producer."""
from hashlib import sha256
import json
import re

SCHEMA = 'SLA_CORE_MULTIENV_TRAIN_PROVENANCE_V1'
COLD_SCHEMA = 'SLA_CORE_MULTIENV_TRAIN_PROVENANCE_V2'
COLD_AUTHORITY_SCHEMA = 'SLA_CORE_MULTIENV_WAVE_A_COLD_AUTHORITY_V1'
JOINED_SCHEMAS = (SCHEMA, COLD_SCHEMA)
REGISTRY_SCHEMA = 'SLA_CORE_MULTIENV_FIXED_TRAIN_MEMBERS_V1'
PLAN = '4fe2c8cf1b69c168b2e3079e061bb7c702089cd3'
PRODUCER_SPEC = '45d949a883c7c10ec31c0caef9a40bdd0a4c51225d5417dac68c7b05432e870b'
PRODUCER_SOURCE = 'eccd8e2e2089bb2505c50043f959959d07094d56'
ALLOWED_REGISTRY = '91d54168764a9f1c0d38f203aae343108f39ad8c81daf62fc2a6ede8bf2b6a05'
PREFIX_COUNT = 59
PREFIX_SEAL = 'd0f7269319b57a4edcc1cd7ca0b60904624831985d54fb6db72cf2b25e8c563b'
PREFIX_PROJECTION = '1c5d6da9a1d432d4ca5997bbd5adf0ad099a93bd4a3658b3136cd263c4728b70'
MAX_COUNT = PREFIX_COUNT+96
# Rebind the portable verifier only after the predecessor and the new spec have sealed.
COLD_AUTHORITIES = frozenset()


def fingerprint(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def registry(spec_sha, spec, population):
    scenes = {row['id']:row['scene']['sha256'] for row in population['clusters']}
    return dict(schema=REGISTRY_SCHEMA,plan=spec['goal_plan_sha'],wave=population['wave'],
                spec=spec_sha,source=spec['source_sha'],population=spec['compiled_population']['sha256'],
                registered=[dict(condition_id=row['id'],family=row['family'],instruction=row['instruction'],
                     scene_family=row['cluster'],scene_sha256=scenes[row['cluster']],reset_seed=row['reset'],
                     native_max_ticks=row['native_max_ticks'],split='TRAIN') for row in population['conditions']['TRAIN']])


def registered_cards(value):
    if (fingerprint(value)!=ALLOWED_REGISTRY or value['spec']!=PRODUCER_SPEC
            or value['source']!=PRODUCER_SOURCE or value['plan']!=PLAN or value['schema']!=REGISTRY_SCHEMA):
        raise ValueError('Unexpected Wave A TRAIN authority')
    return {row['condition_id']:row for row in value['registered']}


def check_prefix(episodes, cases, hashes):
    if fingerprint(dict(episodes=episodes[:PREFIX_COUNT],cases=cases[:PREFIX_COUNT],
                        hashes=hashes[:PREFIX_COUNT]))!=PREFIX_PROJECTION:
        raise ValueError('The frozen TRAIN-059 prefix changed')


def cold_admissions(authority, old, new, cards, registry, trusted):
    fields = {'schema','plan','original_spec','original_terminal','original_records','population',
              'spec','source','prior_admissions','conditions'}
    if (set(authority) != fields or fingerprint(authority) not in trusted
            or authority['schema'] != COLD_AUTHORITY_SCHEMA or authority['plan'] != PLAN
            or authority['original_spec'] != PRODUCER_SPEC or authority['population'] != registry['population']
            or authority['spec'] == PRODUCER_SPEC or authority['source'] == PRODUCER_SOURCE
            or not re.fullmatch('[0-9a-f]{40}', authority['source'])
            or any(not re.fullmatch('[0-9a-f]{64}', authority[key]) for key in ('original_terminal','original_records','spec'))):
        raise ValueError('Unexpected cold Wave A authority')
    conditions = authority['conditions']
    retained = [(r['id'],r['episode_sha256'],r['native_result_sha256']) for r in old]
    if (not conditions or len(set(conditions)) != len(conditions) or not set(conditions) <= cards.keys()
            or set(conditions) & {r[0] for r in retained}
            or retained != [tuple(r) for r in authority['prior_admissions']]):
        raise ValueError('Published predecessor admissions or cold condition roster changed')
    if (not new or any(c['id'] not in conditions or c['producer_spec'] != authority['spec']
                       or c['producer_source'] != authority['source'] for c in new)):
        raise ValueError('Cold Wave A admission did not originate from the authorized suffix')
    ranks = {cid:i for i,cid in enumerate(conditions)}
    admitted = [ranks[c['id']] for c in new]
    if admitted != sorted(set(admitted)):
        raise ValueError('Cold Wave A admissions are duplicated or out of order')


def admitted_cards(proof, episodes, hashes):
    from scripts.sla.hopper_snapshot_producers import SCHEMA as OLD_SCHEMA, check_admission_provenance

    check_prefix(episodes,proof['cases'],hashes)
    if proof['canonical_plan']!=PLAN or proof['prefix_seal']!=PREFIX_SEAL or 'producer_spec' in proof:
        raise ValueError('Unexpected Wave A data provenance')
    old, new = proof['cases'][:PREFIX_COUNT], proof['cases'][PREFIX_COUNT:]
    check_admission_provenance(dict(schema=OLD_SCHEMA,original_campaign_spec=proof['original_campaign_spec'],cases=old))
    registry = proof['registered_producer']
    cards = registered_cards(registry)
    inherited = new
    if proof['schema'] == COLD_SCHEMA:
        authority = proof['cold_continuation']
        cut = len(authority['prior_admissions'])
        inherited = new[:cut]
        cold_admissions(authority,inherited,new[cut:],cards,registry,COLD_AUTHORITIES)
    elif proof['schema'] != SCHEMA or 'cold_continuation' in proof:
        raise ValueError('Unknown Wave A admission schema')
    if not inherited or any(c['id'] not in cards or c['producer_spec']!=PRODUCER_SPEC
                            or c['producer_source']!=PRODUCER_SOURCE for c in inherited):
        raise ValueError('Wave A publication requires a real registered TRAIN admission')
    return cards
