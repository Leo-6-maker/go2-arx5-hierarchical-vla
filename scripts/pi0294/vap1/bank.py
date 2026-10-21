"""Exact D24/D48 profiles sharing the mature raw/reader/pack implementation."""
from . import plan, scope


def configuration(profile):
    from .attempts import qualify
    if profile not in ('VAP1_D24','VAP1_D48'):
        raise PermissionError('VAP1_EXACT_TWO_BANKS_NO_ARBITRARY_EXPANSION')
    cumulative=profile=='VAP1_D48'
    name='VAP1_R1_D48' if cumulative else 'VAP1_R1_D24'
    return dict(auth=plan.AUTH,validate=scope.validate,priority=scope.priority,cards=plan.cards,
        gate=lambda rows,groups:plan.bank_gate(rows,groups,cumulative=cumulative),
        gate_status='PASS_VAP1_DATA_BANK_GATE',schema='PI0294_VAP1_FIXED_R1_BANK_V1',bank=name,
        family='VAP1_DATA',local='VAP1_LOCAL_FROZEN_FULL_READER_QUALIFIED_NOT_HF_READY',
        adoption='VAP1_ADOPTION_AND_SCOPE.json',operation='VAP1_ASSEMBLE_AND_PACK_'+name,
        attempts='ALL_PAIRED_ATTEMPTS.json',output_cap=32*1024**3,archive_cap=(5 if cumulative else 3)*1024**3,
        pack_status='PASS_VAP1_LOCAL_BANK_AND_PACK_NOT_HF_READY',
        local_schema='PI0294_VAP1_LOCAL_FROZEN_V1',index_schema='PI0294_VAP1_ARCHIVE_INDEX_V1',qualify_attempts=qualify)


def run(root):
    from pathlib import Path
    from scripts.pi0292.durable import read
    from scripts.pi0294.cr2.bank import run as pack
    work=read(Path(root)/'plans/WORK_PLAN.private.json')
    profile=work.get('bank_profile');configuration(profile)
    return pack(root,profile=profile)
