"""Exact archived original source packet for the unchanged isolated wire test.

These three byte identities were independently read from the closed original
source. They are a historical input, not a fresh read of unavailable Core raw
files and never a replacement for this allocation's actual 600s/80 cycles.
"""
from pathlib import Path
from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha

PINS={
    'reference/RECORDED_OBSERVATION.json':'71558ff5bcbc03e83eef9fbe79c868e75a2fece5c712f708682c8f23a090a594',
    'reference/RAW_REFERENCE_BINDING.private.json':'faf194d32b2415cd90f9e2c0bf3e10cadd605fae5c5b6853c83ac5e3865321f5',
    'reference/REFERENCE_PROOF_FROM_CORE.json':'efc4039afdaf7370ca31bfc47d2c22555c040c70d5014e80df95ea3ebd3af2d9'}


def need(ok,why):
    if not ok:raise PermissionError('V5_VANDA_RECORDED_REFERENCE_'+why)


def proof(root,plan,manifest):
    root=Path(root)
    for name,pin in PINS.items():
        p=root/name
        need(manifest['execution']['sealed_inputs'].get(name)==pin and p.is_file() and not p.is_symlink()
             and not p.stat().st_mode&0o222 and sha(p)==pin,'EXACT_HISTORICAL_FULL_BYTES_'+name)
    value=read(root/'reference/REFERENCE_PROOF_FROM_CORE.json')
    need(plan['reference_proof_sha256']==PINS['reference/REFERENCE_PROOF_FROM_CORE.json']
         and plan['recorded_observation_file_sha256']==PINS['reference/RECORDED_OBSERVATION.json']
         and plan['raw_reference_binding_sha256']==PINS['reference/RAW_REFERENCE_BINDING.private.json']
         and value['recorded_payload_sha256']==plan['recorded_payload_sha256']
         and value['status']=='PASS_ACTUAL_CLOSED_SOURCE_RECORDED_TWO_CAMERA_REFERENCE'
         and value['new_models']==value['new_queries']==value['new_physics']==0,
         'UNCHANGED_ORIGINAL_RECORDED_PACKET_PROVENANCE')
    from .transport_fixture import check_observation,immutable
    body=check_observation(immutable(root/'reference/RECORDED_OBSERVATION.json'),plan)
    # The mature check validates both PNG payloads/codec/sha; the archived
    # proof additionally pins exactly which actual original camera pixels.
    images=body['payload']['images']
    need({name:image['sha256'] for name,image in zip(('body_rgb','wrist_rgb'),images)}==value['actual_two_camera_pixels'],
         'EXACT_ORIGINAL_DUAL_CAMERA_PIXELS')
    return value
