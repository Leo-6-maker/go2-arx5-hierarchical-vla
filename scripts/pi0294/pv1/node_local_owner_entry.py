"""Cold-only entry path adapter for the exact original resident methods.

The two subprocesses keep their original arguments, affinity, limits and
protocol. Only the entry location follows the new role's sealed local copy.
No running owner, scientific policy or historical file is modified.
"""
import hashlib

ORIGINAL_SHA='4a268f0fa51ebcffda189f6eb169060b3a1299ef53a7ebd18a53a8403f6645c8'
CHANGES=(
    ("args=['/home/liuyu-nfs/Projects/GR00T-WBC/infrastructure/enter.sh',profile,info['python']",
     "args=[str(Path(info['root'])/'entry/infrastructure/enter.sh'),profile,info['python']"),
    ("entry=Path(read(r/'OUTPUT_BINDING.private.json')['registry']).parent/'enter.sh'",
     "from scripts.pi0294.pv1 import node_local_custody as custody\nentry=custody.entry_directory(r,m,read(r/'OUTPUT_BINDING.private.json'))/'enter.sh'"),
)


def derive(original_bytes):
    if hashlib.sha256(original_bytes).hexdigest()!=ORIGINAL_SHA:
        raise PermissionError('NODE_LOCAL_OWNER_EXACT_ORIGINAL_METHODS_BYTES')
    text=original_bytes.decode()
    for old,new in CHANGES:
        if text.count(old)!=1:
            raise PermissionError('NODE_LOCAL_OWNER_TWO_EXACT_COLD_ENTRY_LOCATIONS')
        text=text.replace(old,new)
    body=text.encode();compile(body,'<cold-node-local-original-owner-methods>','exec')
    return body,dict(schema='COLD_NODE_LOCAL_ORIGINAL_OWNER_TWO_ENTRY_PATHS_V1',
        original_sha256=ORIGINAL_SHA,derived_sha256=hashlib.sha256(body).hexdigest(),
        changed_entry_locations=2,scientific_variables_changed=False,native_admission=False)
