"""Cold registered Core GPU role. Validation never grants model/physics GO."""
import hashlib
import json
import os
from pathlib import Path
import time

from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from . import evaluation_core_accounting as c,evaluation_stage as stage
from .evaluation_transfer import CAPTURE_SHA
from .evaluation_plan import development,POPULATION_FILE
from .dataset import POPULATION
from .scope import SCOPE,PROTOCOL
from scripts.pi0294.dec1.authority import BANK
from scripts.pi0294.e1.scope import DATA,STATS

PARENT_BINDING_SHA='7903f13a83ec13ffd6e6737709eb7cc38f3e68af96003ae3339473dad54b6818'
ORIGINAL_NUMERICAL_TABLE_SHA='7666e26ed616dc9facc93fc5b5ac554ab737ba99f1f3e6c2c9887bf5dd207f9b'
COSMOS_PROCESSOR_PINS={
    'chat_template.json':'6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4',
    'config.json':'bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b',
    'generation_config.json':'1e241830b48b397cb0900101421df5450baddc7adf01e5fc86b5615865f3bae4',
    'merges.txt':'599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3',
    'preprocessor_config.json':'27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516',
    'tokenizer.json':'a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7',
    'tokenizer_config.json':'c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399',
    'video_preprocessor_config.json':'7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13',
    'vocab.json':'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'}


def need(ok,why):
    if not ok:raise PermissionError('PV1_CORE_COLD_'+why)


def member(root,name):
    root=Path(root);relative=Path(name);p=root/relative
    need(not relative.is_absolute() and relative.parts and '..' not in relative.parts
         and not any((root/Path(*relative.parts[:i])).is_symlink() for i in range(1,len(relative.parts)+1))
         and p.resolve().is_relative_to(root.resolve()),'OWN_NON_SYMLINK_RELATIVE_MEMBER')
    return p


def validate(root,*,closeout=False):
    need(type(closeout) is bool,'EXPLICIT_CLOSEOUT_PHASE')
    from scripts.pi0294.generalization import minimal_core_accounting as first512
    if first512.typed(read(Path(root)/'RUN_MANIFEST.json')):
        from scripts.pi0294.generalization.minimal_core_context import validate as validate512
        return validate512(root,closeout=closeout)
    from . import mix_core_identity as cd
    if cd.typed(read(Path(root)/'RUN_MANIFEST.json')):
        from .mix_core_context import validate as validate_successor
        return validate_successor(root,closeout=closeout)
    root=Path(root);need(root.is_absolute() and root.is_dir() and not root.is_symlink()
        and root.stat().st_uid==os.getuid()==2009 and os.uname().nodename=='corelab-4080s',
        'ACTUAL_OWN_CORE_ROOT_NOT_AMD_OR_HPC')
    for name in ('RUN_MANIFEST.json','OUTPUT_BINDING.private.json'):
        p=root/name
        need(p.is_file() and not p.is_symlink() and p.stat().st_uid==2009
             and not p.stat().st_mode&0o222,'IMMUTABLE_OWN_ROOT_IDENTITY:'+name)
    m=read(root/'RUN_MANIFEST.json');binding=read(root/'OUTPUT_BINDING.private.json')
    need(m.get('schema')=='PI0294_PV1_NEURAL_CHILD_RUN_MANIFEST_V1' and m.get('scope')==c.ROLE
         and m.get('platform_evaluation_version')==c.VERSION and m.get('owner')=='EXISTING_LOCAL_PC'
         and m.get('owner_user')==os.environ.get('USER')=='liuyu-nfs' and m.get('owner_uid')==2009
         and m.get('run_uuid')==root.name and m.get('execution',{}).get('profile')==c.PROFILE
         and m['execution'].get('uid')==2009 and m['execution'].get('host')=='corelab-4080s'
         and root.parent.name=='neural-children','EXACT_VERSIONED_CORE_GPU_PROFILE_AND_NAMESPACE')
    need(sha(root/'OUTPUT_BINDING.private.json')==m['execution']['output_binding_sha256']
         and binding['run_uuid']==root.name and binding['uid']==2009
         and binding['role']==m['execution']['private_path_role']==c.ROLE+'__'+root.name
         and Path(binding['run_root']).resolve()==root.resolve()
         and Path(binding['registered_parent']).resolve()==root.parent.resolve(),
         'ORIGINAL_CANONICAL_REGISTERED_OUTPUT_BINDING')
    parent=root.parent.parent;adoption=read(parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json')
    need(parent.name==adoption['run_uuid']==c.scope.CAMPAIGN and adoption['authority_release']==c.AUTHORITY
         and adoption['T0_epoch']==c.old.T0 and adoption['stop_new_jobs_epoch']==c.old.STOP_NEW
         and adoption['stop_new_physics_epoch']==c.old.STOP_PHYSICS
         and adoption['hard_closeout_epoch']==c.old.HARD_CLOSE
         and m['authority']['protocol_ref']==c.AUTHORITY
         and m['limits']['PV1_T0_epoch']==c.old.T0 and m['limits']['PV1_hard_close_epoch']==c.old.HARD_CLOSE,
         'ORIGINAL_AUTHORITY_AND_ALL_UNRESET_CLOCKS')
    need(sha(parent/'OUTPUT_BINDING.private.json')==PARENT_BINDING_SHA,
         'EXACT_ORIGINAL_NATIVE_CORE_PARENT_NOT_A_COPIED_CAMPAIGN')
    pb=read(parent/'OUTPUT_BINDING.private.json');pl=read(pb['lease']);lease=read(binding['lease'])
    need(Path(pb['run_root']).resolve()==parent.resolve() and pb['registry']==binding['registry'],
         'SAME_REAL_ORIGINAL_PARENT_AND_CANONICAL_REGISTRY')
    now=time.time();cutoff=c.old.HARD_CLOSE if closeout else c.old.STOP_PHYSICS
    need(c.old.T0<=now<cutoff and lease['status']=='ACTIVE' and lease['run_uuid']==root.name
         and lease['owner']=='EXISTING_LOCAL_PC' and lease['authority_release']==c.AUTHORITY
         and lease['deadline_epoch']==m['limits']['operation_deadline_epoch']
         and (closeout or now<lease['deadline_epoch']<=c.old.STOP_PHYSICS)
         and pl['status']=='ACTIVE_CONTROL_PLANE_ONLY' and pl['run_uuid']==c.scope.CAMPAIGN
         and pl['authority_release']==c.AUTHORITY and pl['deadline_epoch']==c.old.HARD_CLOSE,
         'CURRENT_CHILD_AND_ORIGINAL_PARENT_LEASES')
    entries=read(binding['registry'])['entries'];matches=[x for x in entries if x.get('role')==binding['role']]
    need(len(matches)==1 and matches[0]['run_uuid']==root.name and Path(matches[0]['path']).resolve()==root.resolve(),
         'ONE_CANONICAL_GPU_ROLE_WITH_ITS_OWN_REAL_PATH')
    code=member(root,m['code']['capsule_relative']);seal=member(root,m['code']['source_seal_relative'])
    need(sha(seal)==m['code']['source_seal_sha256'] and Path(__file__).resolve()==
         (code/'scripts/pi0294/pv1/evaluation_core_scope.py').resolve(),'ACTUAL_IMPORTED_SEALED_COLD_HELPER')
    worker=code.parent/'WORKER_RUN_MANIFEST.json'
    need(os.environ.get('PV1_RUN_MANIFEST')==str(worker) and not worker.stat().st_mode&0o222
         and worker.read_bytes()==(root/'RUN_MANIFEST.json').read_bytes(),'IMMUTABLE_STANDARD_ENTRY_WORKER_MANIFEST')
    source=read(seal)
    need(source['source_sha']==m['code']['vla_commit'],'EXACT_FIXED_SOURCE')
    for name,pin in source['files'].items():
        need(sha(member(code,name))==pin,'SOURCE_MEMBER_CHANGED:'+name)
    frozen=m['code']['frozen_numerical_files']
    need(len(frozen)==15 and hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()==ORIGINAL_NUMERICAL_TABLE_SHA
         and all(source['files'].get(n)==v and sha(member(code,n))==v for n,v in frozen.items()),
         'ALL_ORIGINAL_15_NUMERICAL_FILES')
    need(sha(code/SCOPE)==m['authority']['scope_sha256'] and sha(code/PROTOCOL)==m['authority']['protocol_sha256'],
         'ORIGINAL_SCOPE_AND_PROTOCOL_BYTES')
    for name,pin in m['execution']['sealed_inputs'].items():
        p=member(root,name);need(p.is_file() and not p.stat().st_mode&0o222 and sha(p)==pin,
             'IMMUTABLE_COLD_INPUT:'+name)
    def sealed(name):
        need(name in m['execution']['sealed_inputs'],'REQUIRED_PRESEALED_INPUT:'+name)
        return read(member(root,name))
    st=sealed('PV1_STAGE_BINDING.json')
    need(st['schema']=='PI0294_PV1_ADMITTED_SHARED_POLICY_STAGE_V1' and st['authority_release']==c.AUTHORITY
         and st['campaign_run_uuid']==c.scope.CAMPAIGN and st['population_plan_sha256']==POPULATION
         and st['Teacher_takeover'] is False and st['normalizer_refit'] is False
         and st['evaluation_data_for_training'] is False and st['route']=='COVERAGE_FIRST'
         and st['new_bank_manifest_sha256']==BANK,'EXACT_FROZEN_SCIENTIFIC_STAGE')
    population=sealed(POPULATION_FILE);need(sha(root/POPULATION_FILE)==POPULATION,'FULL_ORIGINAL_POPULATION')
    plan=sealed('plans/SOURCE_EVALUATION_PLAN.json');candidate=plan['candidate']
    need(candidate in stage.CANDIDATES and plan==development(population['groups'],route=st['route'],
         candidate=candidate,new_bank_manifest_sha256=st['new_bank_manifest_sha256']),
         'UNCHANGED_FULL_DEV48_PLAN_NOT_AN_EASIER_SUBSET')
    sealed('plans/PRIOR_EVALUATION_OUTCOMES.json');sealed('plans/INITIAL_CORR_PARAMETER_INVENTORY.json')
    for name,pin in ((stage.PLAN,stage.PLAN_SHA),(stage.CARD,stage.CARD_SHA),
            ('plans/V4_S2_GLOBAL_RESOURCE_AUTHORIZATION.json',c.scope.GRANT_SHA),
            ('plans/FULL_AB_CLOSED_ENDPOINT_CAPTURE.private.json',CAPTURE_SHA)):
        need(m['execution']['sealed_inputs'].get(name)==pin and sha(member(root,name))==pin,
             'EXACT_STAGE_GRANT_AND_CLOSED_SOURCE_CAPTURE:'+name)
    capture=sealed('plans/FULL_AB_CLOSED_ENDPOINT_CAPTURE.private.json')['endpoints'][candidate]
    need(capture['checkpoint_sha256']==c.old.ENDPOINTS[candidate] and capture['lease_status']=='CLOSED',
         'EXACT_ALREADY_AUDITED_CLOSED_SAVED4096')
    if candidate=='B_COVERAGE':
        need(m['execution']['sealed_inputs'].get('plans/B_SAVED4096_CLOSED_PROOF.private.json')==c.scope.B_CLOSURE,
             'HUMAN_APPROVED_EXACT_B_OWNER_FAILURE_REUSE_PROOF')
    checkpoint=sealed('plans/EVALUATION_CHECKPOINT.private.json')
    need(checkpoint['candidate']==candidate and checkpoint['global_step']==4096
         and checkpoint['checkpoint_manifest_sha256']==c.old.ENDPOINTS[candidate]
         and checkpoint['full_resume_files_present'] is False
         and Path(checkpoint['checkpoint']).resolve()==(root/'model_views'/candidate).resolve(),
         'INFERENCE_ONLY_LOCAL_VIEW_NOT_A_NEW_FULL_RESUME')
    bindings=sealed('NEURAL_BINDINGS.private.json');inputs=sealed('plans/PV1_TRAIN_INPUTS.private.json')
    need(m['code']['official_gr00t_commit']=='51d4c89f72fda44cbf77285c6a8114b52676b8a1'
         and Path(bindings['source']).is_dir() and Path(bindings['python']).is_file()
         and Path(bindings['cosmos']).name=='9ce19a195e423419c349abfc86fd07178b230561',
         'EXISTING_PINNED_OFFICIAL_SOURCE_AND_COSMOS_RUNTIME')
    need(all(sha(Path(bindings['cosmos'])/n)==pin for n,pin in COSMOS_PROCESSOR_PINS.items()),
         'ORIGINAL_DEPLOYED_ENTRY_TOKENIZER_AND_PROCESSOR_BYTES')
    original=Path(bindings['dataset']);bank=Path(inputs['new_dataset'])
    need(sha(original/'DATASET_MANIFEST.json')==DATA and sha(original/'statistics.json')==STATS
         and sha(bank/'DATASET_MANIFEST.json')==BANK and sha(bank/'statistics.json')==STATS
         and sha(bank/'READY.json')==st['new_bank_ready_sha256'],
         'ACTUAL_ORIGINAL_DATA_R0_AND_FROZEN_NEW_BANK_METADATA')
    if not closeout:
        need(not any(p.exists() for p in (root/'STOP.json',parent/'STOP.json',
             root/'SOURCE_PHYSICS_STOP.json',root/'SOURCE_SAFETY_STOP_FROM_CORE.json')),'ORIGINAL_STOP_FENCE')
    return m,bindings,inputs,st,adoption
