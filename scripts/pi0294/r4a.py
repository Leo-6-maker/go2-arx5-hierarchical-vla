"""The adopted full-body child shares the exact parent R4 clock and ledger."""
from pathlib import Path
from scripts.pi0292.durable import read
from .native_dataset import sha
from .r4 import validate_entry,cpu_stage

SCOPE_SHA='0e7a0045948b4151e6579a0acee3f64ee420584d7e063019997a1290c872cf57'
SCOPE_RELEASE='0ee6211fbdfd8b7cb6449e5a4d814a68ca29c0b9'


def validate_scope(root):
    root=Path(root);manifest=validate_entry(root,profile='4080-simulation')
    path=root/'evidence/R4A_SCOPE_ADOPTION_R1.json'
    if sha(path)!=SCOPE_SHA:raise PermissionError('EXACT_HUMAN_FULLBODY_SCOPE_REQUIRED')
    value=read(path)
    if (value['status']!='ADOPTED_BY_DIRECT_HUMAN_WITH_SAME_PARENT_BUDGET'
            or value['scope_release']!=SCOPE_RELEASE or value['scope_canonical']!=6175301986
            or value['additional_resources'] or value['reset_on_resume']):
        raise PermissionError('NO_ADDITIVE_FULLBODY_WINDOW_OR_BUDGET')
    return manifest
