"""Portable admission and original-prefix contract for the registered Wave A producer."""
from hashlib import sha256
import json

from scripts.sla.hopper_snapshot_producers import SCHEMA as OLD_SCHEMA, check_admission_provenance

SCHEMA = 'SLA_CORE_MULTIENV_TRAIN_PROVENANCE_V1'
REGISTRY_SCHEMA = 'SLA_CORE_MULTIENV_FIXED_TRAIN_MEMBERS_V1'
PLAN = '4fe2c8cf1b69c168b2e3079e061bb7c702089cd3'
PRODUCER_SPEC = '45d949a883c7c10ec31c0caef9a40bdd0a4c51225d5417dac68c7b05432e870b'
PRODUCER_SOURCE = 'eccd8e2e2089bb2505c50043f959959d07094d56'
ALLOWED_REGISTRY = '91d54168764a9f1c0d38f203aae343108f39ad8c81daf62fc2a6ede8bf2b6a05'
PREFIX_COUNT = 59
PREFIX_SEAL = 'd0f7269319b57a4edcc1cd7ca0b60904624831985d54fb6db72cf2b25e8c563b'
PREFIX_PROJECTION = '1c5d6da9a1d432d4ca5997bbd5adf0ad099a93bd4a3658b3136cd263c4728b70'
MAX_COUNT = PREFIX_COUNT+96


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


def admitted_cards(proof, episodes, hashes):
    check_prefix(episodes,proof['cases'],hashes)
    if proof['canonical_plan']!=PLAN or proof['prefix_seal']!=PREFIX_SEAL or 'producer_spec' in proof:
        raise ValueError('Unexpected Wave A data provenance')
    old, new = proof['cases'][:PREFIX_COUNT], proof['cases'][PREFIX_COUNT:]
    check_admission_provenance(dict(schema=OLD_SCHEMA,original_campaign_spec=proof['original_campaign_spec'],cases=old))
    cards = registered_cards(proof['registered_producer'])
    if not new or any(c['id'] not in cards or c['producer_spec']!=PRODUCER_SPEC
                      or c['producer_source']!=PRODUCER_SOURCE for c in new):
        raise ValueError('Wave A publication requires a real registered TRAIN admission')
    return cards
