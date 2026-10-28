"""R3's strict policy-condition boundary on the unchanged R2 processor."""
from pi0260_r2_inputs import batch as r2_batch, build, identity, inventory

ALLOWED=frozenset(('body_rgb','wrist_rgb','state_v1','language'))


def batch(processor,packet,label=None):
    if frozenset(packet)!=ALLOWED:
        raise ValueError(('ILLEGAL_POLICY_CONDITION_FIELDS',sorted(set(packet)-ALLOWED),sorted(ALLOWED-set(packet))))
    return r2_batch(processor,packet,label)
