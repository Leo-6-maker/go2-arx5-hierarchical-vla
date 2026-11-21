"""Bind the three actual Core replays to the original, unstarted student population."""
from pathlib import Path
import time

from scripts.sla.core_canonical_cold import INITIAL, pin
from scripts.sla.core_canonical_ipc import cold_pair
from scripts.sla.core_canonical_package import CANONICAL_PLAN, PROTOCOL, validate
from scripts.sla.hopper_parent_custody import PLAN, write_once
from scripts.sla.hopper_public_checkpoint import ORDINALS
from scripts.sla.hopper_site import pinned

VERIFIED = 'ENDPOINT_CORE_CANONICAL_VERIFIED'
MATCHES = {'normalized', 'decoded', 'guard', 'inputs'}


def check_artifacts(cold_path, ipc_path, receiver, arm):
    """Validate the registered order, both cold denominators and original IPC replies."""
    r = pinned(receiver)
    worker = Path(ipc_path)
    spec = pinned(pin(worker/'EXECUTION_SPEC.json'))
    cfg, root, original = cold_pair(spec)
    if (root != cold_path or spec['arm'] != arm or spec['scope'] != PROTOCOL
            or spec['receiver'] != receiver or spec['physics'] or spec['GPU_uuid'] != r['gpu']
            or spec['new_complete_generations_budget'] != 16
            or spec['new_native_tick_budget'] or spec['new_physical_requests_budget']):
        raise PermissionError('Incorrect IPC site, endpoint, or allocation')
    package = dict(path=str(root/'packages'/arm),sha256=cfg['packages'][arm])
    m = validate(Path(package['path']),package['sha256'])
    if (m['checkpoint_sha256'] != cfg['checkpoints'][arm] or m['foundation_sha256'] != r['model']['parent_manifest']['sha256']
            or m['state_sha256'] != r['model']['state_sha256'] or m['action_sha256'] != r['action']['sha256']
            or m['tokenizer_sha256'] != r['tokenizer']['sha256'] or m['panel'] != r['panel']['sha256']):
        raise ValueError('The Core endpoint or frozen deployment identity changed')
    if original['profile']['gpu'] != r['gpu'] or [x['ordinal'] for x in original['rows']] != list(ORDINALS):
        raise ValueError('Core cold prediction scope differs')
    for mode in ('cold-A', 'cold-B'):
        stage = root/arm/mode
        for ext in ('intent','generated','result'):
            if sorted(x.name for x in stage.glob(f'*.{ext}.json')) != [f'{i:03d}.{ext}.json' for i in ORDINALS]:
                raise ValueError('Cold attempt/complete denominator differs from the registered panel')
        if any(pinned(pin(stage/f"{row['noise_id']}.input.json")) != row['inputs'] for row in original['rows']):
            raise ValueError('A recorded cold input fingerprint changed')
    ipc = pin(worker/'IPC_COMPARISON.json')
    replay = pinned(ipc)
    result_pin = pin(worker/'RESULT.private.json')
    result = pinned(result_pin)
    lease = pinned(pin(worker/'control/LEASE.json'))
    if (replay['status'] != 'IPC_MATCH' or replay['arm'] != arm or replay['cold'] != spec['cold_comparison']
            or replay['worker'] != result_pin or replay['updates'] or replay['native_requests']
            or result['exit_code'] or result['new_native_requests'] or result['new_integrations']
            or not result['requests'] == result['attempted_generations'] == result['completed_generations'] == 16
            or result['site']['GPU_uuid'] != r['gpu'] or result['site']['host'] != r['host']
            or lease['status'] != 'CLOSED' or lease['worker_exit_code'] != 0
            or lease['spec_sha256'] != pin(worker/'EXECUTION_SPEC.json')['sha256']):
        raise ValueError('The real G7 worker did not close an exact 16-request panel')
    rows = replay['rows']
    if [x['ordinal'] for x in rows] != list(ORDINALS):
        raise ValueError('Unexpected IPC replay roster')
    if sorted(x.parent.name for x in worker.glob('*/IPC_INTENT.json')) != [f'{i:03d}' for i in ORDINALS]:
        raise ValueError('Missing or extra IPC attempt')
    for row, expected in zip(rows, original['rows'], strict=True):
        folder = worker/f"{row['ordinal']:03d}"
        intent = pinned(pin(folder/'IPC_INTENT.json'))
        response = pinned(row['response'])
        if (row != pinned(pin(folder/'MATCH.json')) or row['index'] != expected['index']
                or intent['input_sha256'] != expected['input_sha256'] or intent['request'] != row['request']
                or row['response'] != pin(folder/'RESPONSE.json') or response['status'] != 'GENERATED_NOT_INSTALLED'
                or response['request_id'] != row['request'] or response['new_generations_this_request'] != 1
                or set(row['bitwise']) != MATCHES or not all(row['bitwise'].values())):
            raise ValueError('IPC result differs from the sealed cold outputs')
    return dict(status=VERIFIED,protocol=PROTOCOL,canonical_plan=CANONICAL_PLAN,arm=arm,
                checkpoint=m['checkpoint_sha256'],manifest=package['sha256'],package=package,
                core_result=ipc,cold=spec['cold_comparison'],worker=pin(worker/'EXECUTION_SPEC.json'),
                worker_result=result_pin,receiver=receiver,qualification_attempts=48,exact=16)


def publish(path, cold_path, ipc_path, receiver, population, arm, source):
    from scripts.sla.core_hopper_student_endpoint import cases

    if arm not in INITIAL:
        raise ValueError('Initial DEV admission covers the two preregistered endpoints')
    gate = check_artifacts(cold_path,ipc_path,receiver,arm)
    original = pinned(population)
    proof = dict(plan=PLAN,canonical_plan=CANONICAL_PLAN,gate=gate,panels={'DEV':cases(original,arm,'DEV')},
                 compiled_population=population,core_profile=pinned(receiver)['gpu'],source=source,
                 check_epoch=time.time(),hardware_allowed=False,physical_requests=0,model_generations=48)
    if path.exists():
        sealed=pinned(pin(path))
        if {k:v for k,v in sealed.items() if k!='check_epoch'} != {k:v for k,v in proof.items() if k!='check_epoch'}:
            raise ValueError('Existing canonical proof differs')
        return sealed
    write_once(path,proof)
    return proof


def check_native_binding(spec,proof,gate,receiver):
    """Small immutable checks for every native consumer. Publication verifies the full panel."""
    package=spec['packages'][spec['arm']]
    if (proof['canonical_plan'] != CANONICAL_PLAN or gate['canonical_plan'] != CANONICAL_PLAN
            or gate['protocol'] != PROTOCOL or gate['receiver'] != spec['receiver']
            or gate['arm'] != spec['arm'] or gate['arm'] not in INITIAL or gate['package'] != package
            or gate['qualification_attempts'] != 48 or proof['model_generations'] != 48
            or gate['manifest'] != package['sha256'] or proof['core_profile'] != receiver['gpu']):
        raise PermissionError('Missing the original same-Core deployment qualification')
    replay=pinned(gate['core_result'])
    cold=pinned(gate['cold'])
    worker=pinned(gate['worker'])
    if (replay['status'] != 'IPC_MATCH' or replay['arm'] != gate['arm'] or replay['cold'] != gate['cold']
            or replay['worker'] != gate['worker_result'] or cold['status'] != 'COLD_MATCH'
            or cold['complete'] != [16,16] or worker['GPU_uuid'] != receiver['gpu']
            or worker['receiver'] != spec['receiver'] or len(replay['rows']) != 16
            or not all(set(row['bitwise']) == MATCHES and all(row['bitwise'].values()) for row in replay['rows'])):
        raise PermissionError('The cold/IPC proof no longer matches the native allocation')
