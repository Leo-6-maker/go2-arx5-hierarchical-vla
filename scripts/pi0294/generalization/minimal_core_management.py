"""New512 inputs for the original cold finite-custody management programs."""
from scripts.pi0294.pv1 import finite_startup_management as original
from .minimal_core_owner import replace

COMMON=replace(original.COMMON,"Q=N/'pbs/alloc-02'","Q=N/'pbs/alloc-01'")
COMMON=replace(COMMON,'LOCAL=Path(cfg[\'canonical_LOCAL_ledger\'])',
    "from scripts.pi0294.generalization import minimal_core_accounting as c\nLOCAL=Path(cfg['canonical_LOCAL_ledger'])")
PROGRAMS={}
for name,body in original.PROGRAMS.items():
    if body.startswith(original.COMMON):body=COMMON+body[len(original.COMMON):]
    PROGRAMS[name]=body
PROGRAMS['prepare_bind']=replace(PROGRAMS['prepare_bind'],
    "b['proposed_global']['revision']+=1;c.checked(b['proposed_global']);c.paired(b['proposed_global'],b['proposed_LOCAL'])",'')
old=original.ISSUE[len(original.COMMON):]
start=old.index('from scripts.pi0294.pv1.evaluation_admission import resource_floor')
end=old.index('inputs=dict(')
old=old[:start]+'''from scripts.pi0294.generalization import minimal_core_native as admission
physical=read(F/'RESOURCE_LEDGER.json');assert physical['reservations']=={}and physical['unknown_effects']==[]
snap=admission.snapshot(N,r,global_bytes=gb,local_bytes=lb,PC_free_bytes=P['PC_free_bytes'],PC_observed_epoch=P['PC_observed_epoch'])
'''+old[end:]
old=old.replace("g,l=map(json.loads,(gb,lb));c.paired(g,l)","g,l=map(json.loads,(gb,lb))")
old=replace(old,"'plans/V5_CD_CLOSED_TRAINING_CAPTURE.private.json'","'plans/E1_FIXED512_ENDPOINT_REFERENCE.json'")
PROGRAMS['issue']=COMMON+old
PROGRAMS['GLOBAL']=replace(original.GLOBAL,
    'from scripts.pi0294.pv1 import mix_core_accounting as c,mix_core_delegation as d',
    'from scripts.pi0294.pv1 import mix_core_delegation as d\nfrom scripts.pi0294.generalization import minimal_core_accounting as c')
for _name,_body in PROGRAMS.items():compile(_body,'<same-original-cold-custody-fixed512-'+_name+'>','exec')
