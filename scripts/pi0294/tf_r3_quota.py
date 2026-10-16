"""Bind the adapter duration allowance to a live atomic reservation and fixed card."""
import hashlib
import json
from pathlib import Path
from .tf_r3_runtime import read, sha, validate_entry


def quota_bound(warmup_s, task_s):
    if (isinstance(warmup_s, bool) or isinstance(task_s, bool)
            or not isinstance(warmup_s, int) or not isinstance(task_s, int)
            or not 0 <= warmup_s <= 5 or task_s not in (0, 32, 64, 96, 128)):
        raise ValueError('R1_INTEGER_DURATION_DOMAIN')
    return (warmup_s+task_s)*500+3


def bind(root, output, card, ticket):
    """Caller holds R3_RESOURCE_OWNER.lock while storing the reservation/binding."""
    root, output = Path(root), Path(output)
    validate_entry(root, physical_asset='STARTUP' if card['role']=='STARTUP' else 'TASK')
    ledger = read(root/'R3_RESOURCE_LEDGER.json')
    if ledger['CPU_reservations'].get(ticket['stage']) != ticket:
        raise PermissionError('R1_QUOTA_REQUIRES_ACTUAL_LIVE_RESERVATION')
    if ticket['run_uuid'] != root.name or card['id'] != ticket['episode']:
        raise PermissionError('R1_QUOTA_CROSS_RUN_OR_CARD')
    cap = quota_bound(card['warmup_sim_s'], card['duration_s'])
    if ticket['physics_steps_cap'] != cap or not 0 < cap <= 66503:
        raise PermissionError('R1_QUOTA_EXCEEDS_ADOPTED_CARD')
    card_bytes=(json.dumps(card,sort_keys=True,separators=(',',':'))+'\n').encode()
    card_sha=hashlib.sha256(card_bytes).hexdigest()
    reservation=hashlib.sha256(json.dumps(ticket,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    data='\n'.join(['GRTQUOTA2',root.name,card_sha,reservation,str(cap),str(card['warmup_sim_s']),str(card['duration_s'])])+'\n'
    (output/'CARD_QUOTA_INPUT.json').write_bytes(card_bytes)
    path=output/'QUOTA_BINDING.txt'
    with path.open('x') as f:f.write(data)
    return dict(GR00T_TASK_QUOTA_BINDING=str(path),GR00T_TASK_RUN_UUID=root.name,
                GR00T_TASK_CARD_SHA256=card_sha,GR00T_TASK_RESERVATION_ID=reservation)
