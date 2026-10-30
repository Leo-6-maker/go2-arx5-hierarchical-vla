"""Cold adapters of the exact existing Core owner; original driver untouched."""
import hashlib
from scripts.pi0294.pv1.node_local_owner_entry import derive_owner_environment,LOCAL_ENVIRONMENT

PINS={'v5_core_resident_owner.py':'15b636f2e6f05545c7e3b7d22c3de57616d3f500ce8daf8681180fa446244177',
 'v5_core_resident_methods.py':'040fcac62470c43eca2773b58c584e4d2b61d6e81cac35fa42d3e93bd107cbdc',
 'fixed_file_rpc_client.py':'59a4156d8212a1c16839b223cfa192b9d6b4283ff7d19609a8e8b94fbb78ece0',
 'ORIGINAL_RESIDENT_KERNEL_HELPERS.py':'b2d48f4a385f427daef3d87adeb15d7aea3c7fa30394ea302f1637e9a1d7e501'}
ENVIRONMENT=dict(LOCAL_ENVIRONMENT,HUGGINGFACE_HUB_CACHE='cache/huggingface/hub')


def replace(text,old,new):
    if text.count(old)!=1:raise PermissionError('FIXED512_EXACT_ORIGINAL_OWNER_SEAM:'+old[:60])
    return text.replace(old,new)


def derive(files):
    if set(files)!=set(PINS) or any(hashlib.sha256(b).hexdigest()!=PINS[n]for n,b in files.items()):
        raise PermissionError('FIXED512_ALL_FOUR_ORIGINAL_OWNER_BYTES')
    out=dict(files);owner,_=derive_owner_environment(files['v5_core_resident_owner.py']);text=owner.decode()
    text=replace(text,
        'from scripts.pi0294.pv1 import mix_core_accounting as c,mix_accounting as a,mix_core_settlement as terminal,mix_core_admission as admission,mix_core_identity as identity,successor_cd as successor',
        'from scripts.pi0294.pv1 import mix_accounting as a,mix_core_settlement as terminal\nfrom scripts.pi0294.generalization import minimal_core_accounting as c,minimal_core_native as admission,minimal_core_identity as identity,minimal_core_accounting as successor')
    text=replace(text,"assert self.arm=='D25'","assert self.arm=='E1'")
    text=replace(text,'from scripts.pi0294.pv1.mix_core_d25_local_successor import finite_control_scope',
        'from scripts.pi0294.generalization.minimal_core_accounting import finite_control_scope')
    text=replace(text,'_NODE_LOCAL_ENVIRONMENT='+repr(LOCAL_ENVIRONMENT),'_NODE_LOCAL_ENVIRONMENT='+repr(ENVIRONMENT))
    out['v5_core_resident_owner.py']=text.encode()
    methods=files['v5_core_resident_methods.py'].decode()
    anchor="  code=info['code'];bootstrap="
    methods=replace(methods,anchor,"  local_env={k:str(Path(info['root'])/'control'/v)for k,v in "+repr(ENVIRONMENT)+".items()}\n"+anchor)
    anchor="  profile='4080-simulation' if role=='core' else '4080-neural-evaluation'"
    methods=replace(methods,anchor,"  bootstrap='import os;os.environ.update('+repr(local_env)+');'+bootstrap\n"+anchor)
    out['v5_core_resident_methods.py']=methods.encode()
    for n,b in out.items():compile(b,'<new-cold-fixed512-'+n+'>','exec')
    return out,dict(schema='ARX5_FIXED512_EXACT_ORIGINAL_CORE_OWNER_ADAPTER_V1',
        original_sha256=PINS,derived_sha256={n:hashlib.sha256(b).hexdigest()for n,b in out.items()},
        local_environment=ENVIRONMENT,original_driver_controller_safety_and_score_unchanged=True,
        native_admission=False)
