"""Bind cross-layout native admission to the frozen DEV roster and completed Core evidence."""
import hashlib
from pathlib import Path

from scripts.sla.core_canonical_package import validate
from scripts.sla.core_hopper_expert import PLAN, POP
from scripts.sla.core_hopper_native_scope import SCOPE
from scripts.sla.core_main3800_native_proof import ARM, CHECKPOINT, MANIFEST, load_manifest
from scripts.sla.hopper_parent_custody import digest, write_once
from scripts.sla.hopper_site import pinned
from scripts.sla.main3800_auxiliary_readback import verify_package
from scripts.sla.main3800_event_panel import SEED, TRAIN052


STATUS = 'MAIN3800_CROSS_LAYOUT_CORE_READY_V1'
ROSTER = '146a13948b2c08b9d27684ad9ba0dc834fef6eaf74099132b70894adce49e978'
SIX_TERMINAL = '482415683e29ddc7e0d59b79c1e0790ecd113b4b42ea9937680d8c188b06ed54'
PHASE0 = 'MAIN-phase00000'
COHORTS = {'main': (PHASE0,ARM), 'auxiliary': ('A-phase04300','B-phase04300')}
AUXILIARY_ARMS = COHORTS['auxiliary']


def pin(path):
    return dict(path=str(path), sha256=digest(path))


def roster_rows(arm, roster, population):
    if roster['sha256'] != ROSTER or population['sha256'] != POP:
        raise ValueError('Cross-layout population changed after registration')
    value = pinned(roster)
    if (value['population'] != population or value['plan'] != PLAN or value['selected_from_outcomes']
            or value['request_cap_per_arm'] != 6 or value['tick_cap_per_arm'] != 274218
            or value['query_cap_per_arm'] != 960):
        raise ValueError('Cross-layout roster identity or request budget changed')
    return [dict(row,id=row['condition']+'-'+arm,arm=arm) for row in value['candidates']]


def manifest_for_gate(gate):
    package = gate['package']
    folder = Path(package['path']).resolve(strict=True)
    if gate['arm'] == PHASE0:
        return validate(folder,package['sha256'])
    if gate['arm'] == ARM:
        return load_manifest(package,gate['original_package'])
    result = verify_package(folder,package['sha256'])
    if gate['arm'] != result['arm']+'-phase04300':
        raise ValueError('Auxiliary arm package identity changed')
    return result


def matched_rows(root, terminal, panel, checkpoint, arm):
    """Every ordinal must have one real prediction and the original paired noise."""
    reference = [(split,row['tag']) for split,rows in panel['roster'].items() for row in rows]
    actual = terminal['rows'][arm]
    if len(reference)!=40 or len(actual)!=len(reference):
        raise ValueError('Incomplete Core event panel')
    folder = root/arm
    files = {f'{i:03d}.{kind}.json' for i in range(len(reference)) for kind in ('intent','generated','result')}
    if {p.name for p in folder.iterdir()} != files:
        raise ValueError('Core event arm contains unexpected or incomplete artifacts')
    for i,((split,tag),outcome) in enumerate(zip(reference,actual,strict=True)):
        seed = int.from_bytes(hashlib.sha256(f'{SEED}/{split}/{i}'.encode()).digest()[:4],'little')
        if (pinned(pin(folder/f'{i:03d}.intent.json')) != dict(ordinal=i,split=split,tag=tag,noise_id=seed,checkpoint=checkpoint)
                or pinned(pin(folder/f'{i:03d}.generated.json'))['ordinal'] != i
                or pinned(pin(folder/f'{i:03d}.result.json')) != outcome
                or (outcome['ordinal'],outcome['split'],outcome['tag']) != (i,split,tag)):
            raise ValueError('Core event panel lost the input, seed or real output association')


def publish(path, cfg, arm):
    base = pinned(cfg['base_native_config'])
    cohort = cfg['cohort']
    if arm not in COHORTS[cohort] or any(cfg[k]!=base[k] for k in ('receiver','population','expert','historic_run')):
        raise ValueError('Cross-layout comparison changed original Core admission or endpoint')
    terminal = pinned(cfg['main_native_terminal'])
    block = pinned(terminal['actual']['block'])
    students = [r for r in block['records'] if r['role']=='STUDENT']
    if (cfg['main_native_terminal']['sha256']!=SIX_TERMINAL
            or terminal['status']!='SIX_MATCHED_CONDITIONS_RECONCILED' or block['exit_code']
            or len(students)!=6 or {r['arm'] for r in students}!={ARM}
            or sum(r['native_grasp_ever'] for r in students)<1):
        raise ValueError('The initial MAIN3800 experiment has not qualified cross-layout work')
    offline = pinned(cfg['offline_config'])
    panel = pinned(offline['panel'])
    evidence = pinned(cfg['offline_terminal'])
    folder = Path(cfg['offline_config']['path']).parent
    if (offline['native_config']!=cfg['base_native_config'] or panel['seed']!=SEED or panel['training_manifest']!=TRAIN052
            or Path(cfg['offline_terminal']['path'])!=folder/'TERMINAL.json' or evidence['status']!='COMPLETE'
            or evidence['native_requests'] or evidence['updates'] or evidence['visits']
            or {k:len(v) for k,v in panel['roster'].items()}!={'TRAIN':24,'VAL':16}):
        raise ValueError('Missing the real paired event evaluation on the original Core resource')

    original = pinned(cfg['original_core0'])
    if original['gate']['arm']!=PHASE0 or original['gate']['status']!='ENDPOINT_CORE_CANONICAL_VERIFIED':
        raise ValueError('Missing the original protected MAIN0 package')
    original_package = original['gate']['package']
    if cohort == 'main':
        if offline.get('auxiliary_export') is not None or set(evidence['rows'])!={PHASE0,'MAIN-phase02000',ARM}:
            raise ValueError('The original three endpoints must close the same-Core diagnostic')
        package = original_package if arm==PHASE0 else base['package']
    else:
        export = pinned(offline['auxiliary_export'])
        if export['status']!='MATCHED_PAIR_RECONCILED' or export['matched_FM_updates']!=500 or set(evidence['rows'])!=set(AUXILIARY_ARMS):
            raise ValueError('Both reconciled auxiliary arms must close the same-Core diagnostic')
        arm_id = arm.split('-')[0]
        endpoint = export['arms'][arm_id]
        package = dict(path=str(Path(offline['auxiliary_export']['path']).parent/arm_id),sha256=endpoint['manifest_sha256'])
    if evidence['attempts']!=evidence['generations'] or evidence['generations']!=40*len(evidence['rows']):
        raise ValueError('Wrong completed event generation budget')
    gate = dict(status=STATUS,arm=arm,package=package,manifest=package['sha256'],original_package=original_package,
                receiver=cfg['receiver'],core_result=cfg['offline_terminal'])
    manifest = manifest_for_gate(gate)
    receiver = pinned(cfg['receiver'])
    if (manifest['foundation_sha256']!=receiver['model']['parent_manifest']['sha256']
            or manifest['state_sha256']!=receiver['model']['state_sha256'] or manifest['action_sha256']!=receiver['action']['sha256']
            or manifest['tokenizer_sha256']!=receiver['tokenizer']['sha256'] or manifest['panel']!=receiver['panel']['sha256']):
        raise ValueError('Cross-layout endpoint changed its frozen model or processor')
    gate['checkpoint'] = manifest['checkpoint_sha256']
    if cohort=='auxiliary' and (gate['checkpoint']!=endpoint['training_state_sha256']
                               or manifest['training_result_sha256']!=endpoint['result_sha256']):
        raise ValueError('Private auxiliary package changed after verified export')
    matched_rows(folder,evidence,panel,gate['checkpoint'],arm)
    proof = dict(gate=gate,plan=PLAN,source=cfg['source'],core_profile=receiver['gpu'],compiled_population=cfg['population'],
                 roster=cfg['roster'],cohort=cohort,panels={'DEV':roster_rows(arm,cfg['roster'],cfg['population'])},
                 hardware_allowed=False,physical_requests=0,model_generations=40)
    write_once(path,proof)
    return pin(path)


def check_binding(spec, proof, gate, receiver):
    arm = spec['arm']
    if (arm not in COHORTS[proof['cohort']] or spec['scope']!=SCOPE or spec['evaluation_split']!='DEV'
            or spec['plan_commit']!=PLAN or proof['plan']!=PLAN or receiver['plan']!=PLAN
            or spec['source_sha']!=proof['source'] or proof['core_profile']!=receiver['gpu']
            or spec['compiled_population']!=proof['compiled_population']
            or proof['hardware_allowed'] is not False or proof['physical_requests'] or proof['model_generations']!=40
            or gate['arm']!=arm or gate['receiver']!=spec['receiver'] or gate['manifest']!=gate['package']['sha256']
            or spec['packages']!={arm:gate['package']}
            or proof['panels']!={'DEV':roster_rows(arm,proof['roster'],proof['compiled_population'])}
            or (arm==ARM and (gate['checkpoint']!=CHECKPOINT or gate['manifest']!=MANIFEST))):
        raise PermissionError('The frozen cross-layout comparison or site identity changed')
