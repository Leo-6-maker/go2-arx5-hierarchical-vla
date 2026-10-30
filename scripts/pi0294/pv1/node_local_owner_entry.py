"""Cold-only entry path adapter for the exact original resident methods.

The two subprocesses keep their original arguments, affinity, limits and
protocol. Only the entry location follows the new role's sealed local copy.
No running owner, scientific policy or historical file is modified.
"""
import hashlib

ORIGINAL_SHA='4a268f0fa51ebcffda189f6eb169060b3a1299ef53a7ebd18a53a8403f6645c8'
OWNER_SHA='15b636f2e6f05545c7e3b7d22c3de57616d3f500ce8daf8681180fa446244177'
LOCAL_ENVIRONMENT={'TMPDIR':'tmp','XDG_CACHE_HOME':'cache/xdg','HF_HOME':'cache/huggingface',
    'TORCH_HOME':'cache/torch','CUDA_CACHE_PATH':'cache/cuda','TRITON_CACHE_DIR':'cache/triton',
    'MPLCONFIGDIR':'cache/matplotlib'}
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


def derive_owner_environment(original_bytes):
    """Create a separately attested cold entry; retain the sealed original."""
    if hashlib.sha256(original_bytes).hexdigest()!=OWNER_SHA:
        raise PermissionError('NODE_LOCAL_OWNER_EXACT_ORIGINAL_FINITE_OWNER_BYTES')
    text=original_bytes.decode();anchor='REMOTE_HEADER=None\n'
    bootstrap="body='import os,sys,fcntl,runpy;from pathlib import Path;fd=os.open('"
    if text.count(anchor)!=1 or text.count(bootstrap)!=1:
        raise PermissionError('NODE_LOCAL_OWNER_EXACT_LOCAL_AND_GPU_BOOTSTRAPS')
    addition=("\n_NODE_LOCAL_ENVIRONMENT="+repr(LOCAL_ENVIRONMENT)+"\n"
        "def _node_local_environment(root):\n return {k:str(root/'control'/v) for k,v in _NODE_LOCAL_ENVIRONMENT.items()}\n"
        "LOCAL_HEADER+='\\nos.environ.update({k:str(r/\\\"control\\\"/v) for k,v in '+repr(_NODE_LOCAL_ENVIRONMENT)+'.items()})\\n'\n")
    text=text.replace(anchor,anchor+addition).replace(bootstrap,
        "body='import os;os.environ.update('+repr(_node_local_environment(root))+');'+"+bootstrap[5:])
    body=text.encode();compile(body,'<separate-cold-owner-local-environment-entry>','exec')
    return body,dict(schema='COLD_FINITE_OWNER_LOCAL_TEMP_AND_CACHE_ENVIRONMENT_V1',
        original_sha256=OWNER_SHA,derived_sha256=hashlib.sha256(body).hexdigest(),
        local_environment=LOCAL_ENVIRONMENT,original_entry_retained=True,
        resource_or_scientific_variables_changed=False,native_admission=False)
