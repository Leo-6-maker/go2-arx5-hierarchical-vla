"""Identity and portable evidence for the single D4 matched DEV panel."""
from pathlib import Path

from .multitask_score_v6 import INSTRUCTIONS
from .storage import digest

SCOPE = 'SLA_D4_C20_G7_NATIVE_DEV'
PLAN = '32fae139c9c2c18d9a1dc01fc8490aa9723deafb'
STAGE = 'D4_DEV'
AUTHORITY_SOURCE = 'CURRENT_USER_D4_MATCHED_DEV_CONTINUATION'
ARMS = ('D4-A', 'D4-B')
POPULATION_SHA = '4a601d1ed2126e7acf6e522b10da569098bda37be0c206baf7069a47c783b746'
PARENT_SHA = '232442c8a8c934118cbcdb890c6f307d7fc857819d7e45c2286ed8e8e5d5f3a3'
REQUEST_CAP = 48
GENERATION_CAP = 7680
TICK_CAP = 2193744

SEALED = {
    'D4-A': dict(package='a4b4b0dc2746d28f493c982e6aa65a915c8ebeebb237e6960b5297afeff16ab8',
                 result='8174773e1052a0f23fe7cfda403ca94c434cc5005904188b67696b8f63e2b7c0',
                 restore='f17af3ab7fd90e9c775dec5b3c5c599fe00202bf27e053fee2c98504f63ddee8'),
    'D4-B': dict(package='db2f4c31165dd442b712f286e8589ee2f86ff91f968fdcef1cd7e12d8f74e290',
                 result='69f5e2d7cf2efde7655214a7c625614cb621f24c1251a375bb6ce00fdae7d242',
                 restore='fff2587396498ab67233f10b59d2de6bd93bd46bf52b63dc9c2e42b434cb9768'),
}


def endpoint(arm, package, result, restore):
    from .multitask_scope_v6 import read_pin

    pins = SEALED[arm]
    if (package['sha256'], result['sha256'], restore['sha256']) != tuple(pins[k] for k in ('package', 'result', 'restore')):
        raise ValueError('D4_REQUIRES_SEALED_VANDA_ENDPOINT_AND_RGB_RESTORE')
    manifest = read_pin(dict(path=str(Path(package['path'])/'MANIFEST.private.json'), sha256=pins['package']))
    actual, rgb = read_pin(result), read_pin(restore)
    full = actual['full_restore']
    observations = rgb['observations']
    if (manifest['bindings']['arm'] != arm or manifest['parent_manifest']['sha256'] != PARENT_SHA
            or (manifest['step'], manifest['cursor']) != (500, 16000)
            or actual['arm'] != arm or actual['exit_code'] or actual['new_updates'] != 500
            or actual['new_visits'] != 16000 or actual['new_source_visits'] or actual['new_native_requests']
            or actual['new_integrations'] or actual['actual']['training_full_generations'] != 2000
            or full['bit_exact'] is not True or full['arrays'] != 87 or full['step'] != 500 or full['cursor'] != 16000
            or full['optimizer_updates_added'] or actual['package'] != rgb['package']
            or rgb['package']['sha256'] != pins['package'] or rgb['exit_code'] or rgb['passed'] is not True
            or rgb['actual_generations'] != 16 or rgb['new_native_requests'] or len(observations) != 16
            or len({(row['index'], row['noise']) for row in observations}) != 16
            or not all(row['normalized_exact'] and row['EEF10_exact'] for row in observations)):
        raise ValueError('D4_REQUIRES_EXACT_500_16000_AND_INDEPENDENT_RGB16')
    return manifest, actual


def portable_endpoints(root):
    """Verify all delivered files on the receiving site without loading a model."""
    root = Path(root).resolve(strict=True)
    packages, results, restores = {}, {}, {}
    members = 0
    for arm, pins in SEALED.items():
        folder = root/arm
        asset = folder/'RGB_PACKAGE'
        packages[arm] = dict(path=str(asset), sha256=pins['package'])
        results[arm] = dict(path=str(folder/'RESULT.private.json'), sha256=pins['result'])
        restores[arm] = dict(path=str(folder/'RESTORE.private.json'), sha256=pins['restore'])
        manifest, _ = endpoint(arm, packages[arm], results[arm], restores[arm])
        files = [(row['file'], row['sha256']) for row in manifest['adapter_arrays']]
        files += [('STATISTICS.json', manifest['statistics_sha256']),
                  ('paligemma_tokenizer.model', manifest['tokenizer_sha256'])]
        asset = asset.resolve(strict=True)
        if (not asset.is_relative_to(root) or len({name for name, _ in files}) != 30
                or any(not (asset/name).resolve(strict=True).is_relative_to(asset)
                       or digest(asset/name) != sha for name, sha in files)):
            raise ValueError('D4_PORTABLE_MODEL_ASSET_IDENTITY_CHANGED')
        members += len(files) + 3
    if digest(root/'foundation/BYTE_MANIFEST.private.json') != PARENT_SHA:
        raise ValueError('D4_PORTABLE_PARENT_MANIFEST_CHANGED')
    return dict(packages=packages, endpoints=results, ordinary_RGB_restores=restores,
                verified_file_count=members+1, parent_manifest_sha256=PARENT_SHA)


def conditions(population):
    clusters = {row['id']: row for row in population['clusters']}
    rows = population['conditions'][STAGE]
    if (len(clusters) != 4 or len(rows) != 24 or population['historical_TEST_reused']
            or not population['scene_instruction_independent'] or population['new_science_request_planned'] != REQUEST_CAP
            or sorted(c['family'] for c in clusters.values()) != ['F1', 'F1', 'F2', 'F2']
            or len({c['scene']['sha256'] for c in clusters.values()}) != 4):
        raise ValueError('D4_REQUIRES_FOUR_FRESH_FIXED_LAYOUTS')
    expected = {(cluster, instruction) for cluster in clusters for instruction in INSTRUCTIONS}
    planned, seen = [], set()
    for row in rows:
        pair = row['cluster'], row['instruction']
        order = row['arm_order']
        if (pair not in expected or pair in seen or row['id'] != '-'.join(pair)
                or row['reset'] != clusters[row['cluster']]['reset']
                or row['family'] != clusters[row['cluster']]['family']
                or tuple(order) not in (ARMS, ARMS[::-1])):
            raise ValueError('D4_REQUIRES_ORIGINAL_MATCHED_SCENE_SEED_AND_ORDER')
        seen.add(pair)
        planned.extend((row, arm) for arm in order)
    if sum(arm == ARMS[0] for _, arm in planned[::2]) != 12:
        raise ValueError('D4_REQUIRES_TWELVE_FIRST_PLACEMENTS_PER_ARM')
    return planned
