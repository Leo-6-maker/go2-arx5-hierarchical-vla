"""Read-only legacy identity view for original GEN1 saved-data auditing.

The geometry seed was sealed before rendering. Native layout/initial state is
explicitly recorded and is not regenerated here. No raw CARD, manifest, sample,
audit, label, score or random state is changed by this adapter.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from . import amd_adoption as a, amd_capture as c, static_preview as preview


def geometry_seed(packet_bytes, card):
    expected={x['id']:x for i in (0,1)for x in c.cards(packet_bytes,i)}
    a.need(card.get('id')in expected and card==expected[card['id']],
        'SEED_AUDIT_EXACT_UNMODIFIED_GEN_CARD')
    specs=preview.specifications(json.loads(packet_bytes))
    matches=[x['V0']for x in specs.values()if x['V0']['design_hash']==card['GEN1_design_hash']]
    a.need(len(matches)==1 and matches[0]['geometry_id']==card['geometry_id'],
        'SEED_AUDIT_PREDECLARED_GEOMETRY_LINEAGE')
    seed=matches[0]['seeds']['geometry']
    a.need(type(seed)is int and 0<=seed<2**64,'SEED_AUDIT_FIXED_UINT64_PROVENANCE')
    return seed


@contextmanager
def legacy_identity_read_view(base, root, card, packet_bytes):
    root=Path(root).resolve();folder=root/'episodes'/card['id'];card_path=folder/'CARD.json'
    attempt_path=root/'ATTEMPT_PLAN.json';original_read=base.read
    a.need(not getattr(base,'_GEN1_identity_view_active',False),'SEED_AUDIT_NO_NESTED_READ_VIEW')
    seed=geometry_seed(packet_bytes,card);saved=original_read(card_path);attempts=original_read(attempt_path)
    a.need(saved==card and attempts['attempts'][card['id']]['logical_id']==card['id']
        and attempts['attempts'][card['id']]['card']==card and 'seed'not in card,
        'SEED_AUDIT_ACTUAL_RAW_CARD_AND_ORIGINAL_ATTEMPT_BINDING')
    def read_view(path):
        path=Path(path).resolve();value=original_read(path)
        if path==card_path:
            a.need(value==saved,'SEED_AUDIT_CARD_CHANGED_DURING_READ')
            return dict(value,seed=seed)
        if path==attempt_path:
            a.need(value==attempts,'SEED_AUDIT_ATTEMPT_PLAN_CHANGED_DURING_READ')
            result=deepcopy(value);result['attempts'][card['id']]['card']['seed']=seed
            return result
        return value
    base._GEN1_identity_view_active=True;base.read=read_view
    try:
        yield dict(schema='GEN1_ORIGINAL_GEOMETRY_SEED_IDENTITY_READ_VIEW_V1',
            attempt_id=card['id'],design_hash=card['GEN1_design_hash'],geometry_id=card['geometry_id'],
            seed=seed,seed_is_predeclared_geometry_provenance_not_new_execution_RNG=True,
            original_files_changed=False,policy_or_supervision_changed=False,
            original_auditor_numeric_gates_changed=False)
    finally:
        base.read=original_read;del base._GEN1_identity_view_active
