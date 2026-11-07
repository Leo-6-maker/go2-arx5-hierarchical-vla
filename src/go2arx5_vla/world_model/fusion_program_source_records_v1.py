"""Read only exact CLOSED native/GPU artifacts; assign no physical authority."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
from .pilot1_storage import local_path
from . import fusion_live_program_replay_v1 as replay
from .fusion_program_response_data_v1 import FEATURE_DIM,SCHEMA,FEATURE_NAMES,validate,summarize
from .fusion_contract import FROZEN_GR00T_CHECKPOINT_SHA256 as CHECKPOINT

MANIFEST_SCHEMA='WM02_PROGRAM_RESPONSE_SOURCE_MANIFEST_V1'
NATIVE_SHA='f30450e669236724a63c125cc88df86bb28aad9136ee644b1c8328e088b0adea'
OBSERVER_SHA='07192cd562d79500a48f13069199e73a62d03d567dea59dfd14d175670e3038e'


def _need(ok,why):
    if not ok:raise ValueError('PROGRAM_SOURCE_'+why)


def _file(spec,descriptor,io,*,decode=True,cap=512*1024**2):
    _need(set(descriptor)=={'path','sha256'},'EXACT_ARTIFACT_DESCRIPTOR')
    path=Path(descriptor['path']);local_path(spec,path)
    _need(path.is_file()and not path.is_symlink()and path.resolve()==path and path.stat().st_size<=cap,'CANONICAL_BOUNDED_ARTIFACT')
    h=hashlib.sha256()
    with path.open('rb')as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    _need(h.hexdigest()==descriptor['sha256'],'ARTIFACT_SHA256:'+path.name)
    io['artifacts_hashed']+=1;io['logical_bytes_hashed']+=path.stat().st_size
    if not decode:return path
    with path.open()as stream:value=json.load(stream)
    io['JSON_bytes_read']+=path.stat().st_size
    return value


def _pid0(root):
    q=subprocess.run(['systemctl','--user','show','wm02-'+root.name,'-p','MainPID'],capture_output=True,text=True,timeout=10)
    _need(q.returncode==0 and q.stdout.strip()=='MainPID=0','ACTUAL_TERMINAL_PID0_REQUIRED')


def _closed(spec,bundle,side,registry,io):
    actual=_file(spec,bundle[side+'_spec'],io,cap=4*1024**2);root=Path(actual['registered_root'])
    _need(Path(bundle[side+'_spec']['path'])==root/'EXECUTION_SPEC.json','EXACT_ORIGINAL_SPEC_LOCATION')
    lease=_file(spec,bundle[side+'_lease'],io,cap=64*1024);worker=_file(spec,bundle[side+'_worker'],io,cap=64*1024)
    _need(Path(bundle[side+'_lease']['path'])==root/'control/LEASE.json'
          and Path(bundle[side+'_worker']['path'])==root/'WORKER_EXIT.private.json'
          and lease.get('status')=='CLOSED'and lease.get('run_uuid')==root.name
          and lease.get('spec_sha256')==bundle[side+'_spec']['sha256']
          and lease.get('source_sha')==actual['source_sha']==bundle['source_sha']
          and type(worker.get('exit_code'))is int and worker['exit_code']==lease.get('worker_exit_code')
          and worker.get('failure')is None,'ORIGINAL_CLOSED_WORKER_AND_SPEC_JOIN')
    _need(actual['owner']==spec['owner']and actual['registry']==spec['registry']
          and any(e.get('path')==str(root)and e.get('role')==actual['role']
             and e.get('source_sha')==actual['source_sha']and e.get('owner')==actual['owner']
             and e.get('local_lease')==str(root/'control/LEASE.json')for e in registry['entries']),
          'ORIGINAL_OWNER_REGISTERED_SOURCE_NOT_A_DATA_PERMISSION_BOOLEAN')
    _pid0(root)
    return actual,worker


def load_sources(spec,*,required_role='TRAIN'):
    io=dict(artifacts_hashed=0,logical_bytes_hashed=0,JSON_bytes_read=0)
    manifest=_file(spec,dict(path=spec['source_manifest'],sha256=spec['source_manifest_sha256']),io,cap=4*1024**2)
    _need(manifest.get('schema')==MANIFEST_SCHEMA and bool(manifest.get('records')),'EXPLICIT_NONEMPTY_ORIGINAL_SOURCE_MANIFEST')
    registry=json.loads(Path(spec['registry']).read_text());records=[];attempts=[];seen=set()
    for bundle in manifest['records']:
        n,nworker=_closed(spec,bundle,'native',registry,io);g,gworker=_closed(spec,bundle,'gpu',registry,io)
        root=Path(n['registered_root']);groot=Path(g['registered_root']);nr=root/'native-01';gr=groot/'fit-01'
        _need(str(root)not in seen,'UNIQUE_PHYSICAL_SOURCE_RUN');seen.add(str(root))
        a=replay.assignment(n)
        _need(a['split']==required_role and a['panel_sha256']is not None,'OLD_DIAGNOSTIC_DEV_OR_TEST_CANNOT_BECOME_TRAIN')
        panel=_file(spec,bundle['learning_panel'],io,cap=4*1024**2)
        _need(bundle['learning_panel']['path']==n['response_learning_panel']
              and bundle['learning_panel']['sha256']==n['response_learning_panel_sha256']==a['panel_sha256']
              and any(p['native_run']==root.name and p['GPU_run']==groot.name
                     and p['parent']==a['parent']and p['split']==a['split']for p in panel['assignments']),
              'EXACT_PREPHYSICS_PANEL_ASSIGNMENT')
        _need(n.get('phase')==replay.PHASE and n.get('live_skill_start_step')==replay.B
              and n.get('WBC_changes')is False and n.get('physics_dt_s')==.002
              and g.get('producer_native_spec_sha256')==bundle['native_spec']['sha256']
              and Path(g['producer_native_spec'])==root/'EXECUTION_SPEC.json'
              and g.get('source_sha')==n['source_sha'],'SAME_FROZEN_PHYSICAL_PRODUCER_AND_FROZEN_QUERY')
        _need(g.get('kind')=='FROZEN_VLA_LIVE_PAUSED_QUERY'and g.get('max_queries')==1
              and g.get('checkpoint_manifest_sha256')==CHECKPOINT
              and g.get('numeric_mode')=='ORIGINAL_FP32_WEIGHTS_BF16_AUTOCAST'
              and g.get('Actor_updates')==g.get('WM_updates')==g.get('GR00T_updates')==0,
              'EXACT_FROZEN_GR00T_PROPOSAL_CONDITION')
        attempt=dict(run=root.name,parent=a['parent'],split=a['split'],source_sha=n['source_sha'],
            native_exit_code=nworker['exit_code'],GPU_exit_code=gworker['exit_code'],anchors=0,valid_label_atoms=0)
        attempts.append(attempt)
        if not all(bundle.get(k)for k in ('decision','action','gpu_result')):
            attempt['excluded_reason']='NO_GENERATED_INSTALL_DECISION_OR_QUERY_NO_DYNAMICS_LABELS';continue
        for key,expected in [('native_result',nr/'RESULT.json'),('decision',nr/'LIVE_SKILL_SEGMENT_DECISION.private.json'),
                             ('gpu_result',gr/'RESULT.private.json'),('action',gr/'REAL_VLA_ACTION.npz')]:
            _need(Path(bundle[key]['path'])==expected,'ORIGINAL_ARTIFACT_LOCATION:'+key)
        native=_file(spec,bundle['native_result'],io)
        _need(native.get('source_sha')==n['source_sha']and native.get('phase')==replay.PHASE
              and native.get('native_binary_sha256')==NATIVE_SHA and native.get('observer_sha256')==OBSERVER_SHA,
              'ACTUAL_FROZEN_NATIVE_AND_OBSERVER_IDENTITIES')
        complete=native.get('reference_qualification')or{}
        if complete.get('generated_segment'):
            segment=complete['generated_segment']
            prefix=dict(native_endpoint_step=replay.B,windows=[r for r in complete['windows']if r['native_end_step']<=replay.B][-9:])
        elif bundle.get('initialization')and bundle.get('segment'):
            prefix=_file(spec,bundle['initialization'],io);prefix=dict(native_endpoint_step=prefix['native_endpoint_step'],windows=prefix['windows'][-9:])
            segment=_file(spec,bundle['segment'],io)
        else:
            attempt['excluded_reason']='NO_CONFIRMED_GENERATED_SUFFIX';continue
        _need((nworker['exit_code']!=0 or(native.get('status')=='NATIVE_STAGE_PASSED'and native.get('shutdown_clean')is True
              and native.get('original_joint_or_warning_samples')==0)),'CLOSED0_MUST_HAVE_ORIGINAL_PASS_AND_CLEAN_SHUTDOWN')
        decision=_file(spec,bundle['decision'],io,cap=16*1024**2);gpu=_file(spec,bundle['gpu_result'],io,cap=64*1024)
        action=_file(spec,bundle['action'],io,decode=False,cap=1024**2)
        _need(gpu.get('actual_VLA_inferences')==1 and gpu.get('action_sha256')==bundle['action']['sha256']
              and gpu.get('producer_native_spec_sha256')==bundle['native_spec']['sha256']
              and gpu.get('source_sha')==g['source_sha']
              and gpu.get('execution_spec_sha256')==bundle['gpu_spec']['sha256']
              and gpu.get('checkpoint_manifest_sha256')==g['checkpoint_manifest_sha256'],
              'ONE_REAL_FROZEN_PROPOSAL_NOT_SYNTHETIC_OR_EXPERT')
        with np.load(action,allow_pickle=False)as z:
            _need(np.array_equal(z['physical_H40'],decision['raw_H40']),'ORIGINAL_ALREADY_DECODED_ALL40')
        for atom in (range(0,len(segment.get('windows',[])),7)if segment.get('windows')else [0]):
            row=replay.extract(n,prefix,decision,segment,install_atom=atom);x,names=replay.flatten_features(row)
            _need(tuple(names)==FEATURE_NAMES,'FIXED_FEATURE_WHITELIST')
            phase=('POST_QUERY_HOLD_CONTROL'if decision['adaptation'].get('post_query_HOLD_control')is True
                   else'PROGRAM'if atom<=decision['adaptation']['clock']['end_atom']else'POST_PROGRAM_HOLD')
            records.append(dict(X=x,Y=row['labels']['TCP_SE3_and_finger_displacement'],valid=row['labels']['observed_horizon_mask'],
                split=a['split'],parent=a['parent'],run_id=root.name,program_atom=atom,record_phase=phase))
            attempt['anchors']+=1;attempt['valid_label_atoms']+=int(row['labels']['observed_horizon_mask'].sum())
        del native,complete,segment,prefix
    if records:data={k:np.stack([r[k]for r in records])for k in records[0]}
    else:data=dict(X=np.empty((0,FEATURE_DIM)),Y=np.empty((0,16,8)),valid=np.empty((0,16),bool),
                   split=np.array([],dtype='U10'),parent=np.array([],dtype='U1'),run_id=np.array([],dtype='U1'),program_atom=np.array([],dtype=int),record_phase=np.array([],dtype='U24'))
    validate(data)
    metadata=dict(schema=SCHEMA,source_manifest_sha256=spec['source_manifest_sha256'],attempts=attempts,
        summary=summarize(data),IO=io,feature_names=list(FEATURE_NAMES),required_role=required_role,
        object_truth_features=False,unobserved_suffix_labels=False,TEST_read=False,exact_hidden_state_forks=False)
    return data,metadata
