"""CPU-bound delivery identity and independent qualified Trainer protection."""
import argparse,json,os,shutil,tempfile
from pathlib import Path
from contract import INSTRUCTIONS,sha
from data import immutable
from r3_shared import checked_panel

TRAIN_CPU='PASS_ACTUAL_NATIVE_TRAINER_INPUT_PLAN_NOISE_LOSS_MODES_SCOPE_AND_COMPLETE_LOCAL_STATE'


def input_schema(path,instruction=None):
    import numpy as np
    with np.load(path,allow_pickle=False) as f:
        assert set(f.files)=={'base_raw','wrist_raw','actual_state','instruction'}
        assert f['actual_state'].shape==(10,) and np.isfinite(f['actual_state']).all()
        assert all(f[n].shape==(128,128,3) and f[n].dtype==np.uint8 for n in ('base_raw','wrist_raw'))
        assert str(f['instruction']) in INSTRUCTIONS.values()
        if instruction:assert str(f['instruction'])==instruction
    return dict(path=str(path),sha256=sha(path),schema='REAL_CURRENT_RGB_STATE_LANGUAGE_ONLY')


def entries(root):
    root=Path(root);out=[]
    for p in sorted(root.rglob('*')):
        assert not p.is_symlink(),f'Protection must contain original independent bytes: {p}'
        if p.is_file():out.append(dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    assert out
    return out


def verify_entries(root,inventory):
    root=Path(root)
    assert entries(root)==inventory,'Partial, mixed or changed asset inventory'
    return True


def qualify(root,point):
    root=Path(root);stage_b=point.startswith(('SHARED','REPAIR_B'));tasks=['T1','T2'] if stage_b else ['T1'];name='CONFIRM_B' if stage_b else 'CONFIRM_A'
    proofs={};counts={}
    for split in ('TRAIN','DEV',name):
        if not (root/'source/evidence'/f'{point}_{split}.json').exists():
            return dict(status='DEVELOPMENT_ONLY_NOT_CONFIRMED',qualified=False,point=point,missing=split,proofs=proofs,counts=counts,selected='NONE',deployment='NONE')
        counts[split],proofs[split],_,_=checked_panel(root,point,split,tasks)
    candidate=root/'qualification'/(name+'_CANDIDATE.json');c=json.loads(candidate.read_text());assert c['point']==point and c['confirm_allowed']
    binding=json.loads((root/'design/MODEL_POINTS.json').read_text())[point];assert binding==c['model_binding']
    for n,h in c['development_panels'].items():assert sha(root/'source/evidence'/n)==h
    passed=all(counts['TRAIN'][t]>=7 and counts['DEV'][t]>=13 and counts[name][t]>=26 for t in tasks)
    if stage_b:passed=passed and counts['DEV']['paired_both']>=12 and counts[name]['paired_both']>=24
    return dict(status=('SHARED_TWO_TASK_SOURCE_QUALIFIED' if stage_b else 'PLACE_V1_SOURCE_QUALIFIED') if passed else 'VALID_QUALITY_FAIL',
                qualified=bool(passed),point=point,counts=counts,proofs=proofs,candidate_sha256=sha(candidate),model_binding=binding,selected='NONE',deployment='NONE')


def build(root,point):
    import importlib.util,importlib.metadata,sys
    root=Path(root);transfer=root/'transfers'/point;view=root/'models'/point
    a=json.loads((transfer/'TRAINING_CPU.json').read_text());assert a['status']==TRAIN_CPU and a['point']==point
    export=transfer/'EXPORT.json';assert a['export_sha256']==sha(export)
    material=view/'MATERIALIZATION.json';m=json.loads(material.read_text());assert m['status']=='PASS_EXACT_TRANSFER_MASTER_READBACK_NATIVE_BF16_VIEW'
    for n,h in m['files'].items():assert sha(view/n)==h,n
    binding=json.loads((root/'design/MODEL_POINTS.json').read_text())[point]
    assert Path(binding['model_root']).resolve()==view.resolve() and binding['files']==m['files']
    for n in ('processor_config.json','statistics.json','embodiment_id.json'):assert sha(view/n)==sha(root/'data'/n)
    seal=json.loads((root/'design/CODE_SEAL.json').read_text());code={}
    for n,h in seal['files'].items():assert sha(root/'code'/n)==h;code[n]=h
    dependencies={}
    for p,h in seal.get('native_dependencies',{}).items():assert sha(p)==h;dependencies[p]=h
    # Resolve the existing evaluator import path without importing a model or torch.
    inherited=[root.parent/'PI0283/code',root.parent/'PI0280/official',root/'project-src',root/'project-src/src']
    for p in inherited:
        if str(p) not in sys.path:sys.path.append(str(p))
    resolutions={}
    for name in ('native_model','scope','learning','codec','phase','gr00t'):
        spec=importlib.util.find_spec(name);assert spec and spec.origin and Path(spec.origin).is_file(),name
        resolutions[name]=dict(path=spec.origin,sha256=sha(spec.origin));dependencies[spec.origin]=sha(spec.origin)
    official=Path(resolutions['gr00t']['path']).parent
    for name in ('policy/gr00t_policy.py','data/state_action/state_action_processor.py','data/utils.py'):
        p=official/name;assert p.is_file();dependencies[str(p)]=sha(p)
    versions={}
    for name in ('torch','transformers','safetensors','numpy','Pillow'):
        versions[name]=importlib.metadata.version(name)
    qualification=qualify(root,point)
    admission=json.loads((root/'data/R2_SCOPED_DATA_ADMISSION.json').read_text())
    canary=root/'data/CANARY_INPUT.npz';assert sha(canary)==admission['canary_input_sha256']
    reused_inputs=dict(canary=input_schema(canary,INSTRUCTIONS['T1']),native24_sha256=sha(root/'data/NATIVE_INPUTS_CHECK.json'))
    diag=root/'design/DIAG96_PER_TASK.json'
    if diag.exists():
        dd=json.loads(diag.read_text());assert dd['status']=='FROZEN_PRE_SCORE_DIAG96_PER_ADMITTED_TASK'
        for row in dd['rows']:
            assert sha(row['input'])==row['input_sha256'] and sha(row['target'])==row['target_sha256']
            input_schema(row['input'],INSTRUCTIONS[row['task']])
        reused_inputs['DIAG']=dict(plan_sha256=sha(diag),rows=len(dd['rows']),all_input_target_hashes_verified=True,input_schema_verified=True)
    videos=[]
    for split in ('TRAIN','DEV','CONFIRM_A','CONFIRM_B','ACQUISITION'):
        panel=root/'source/evidence'/f'{point}_{split}.json'
        if not panel.exists():continue
        for row in json.loads(panel.read_text()).get('rows',[]):
            folder=Path(row['folder'])
            for v in sorted(folder.glob('*.mp4')):videos.append(dict(path=str(v),sha256=sha(v),bytes=v.stat().st_size,task=row['task'],layout=row['layout'],split=row['split']))
    return dict(status='IMMUTABLE_SOURCE_POLICY_BUNDLE_CPU_READBACK',owner='R3_WP8_SHARED_DELIVERY',state='COMPLETE',point=point,
                model_root=str(view),model_files=m['files'],effective_BF16_tensor_identity=binding['effective_bf16_tensor_sha256'],
                materialization_sha256=sha(material),export_sha256=sha(export),training_CPU_sha256=sha(transfer/'TRAINING_CPU.json'),reused_CPU_inputs=reused_inputs,
                full_Trainer=dict(host_role='HOPPER_TRAINING_HOST',root=a['checkpoint'],entries=a['entries'],transferred_to_evaluator=False),
                processor_statistics_mapping={n:sha(root/'data'/n) for n in ('processor_config.json','statistics.json','embodiment_id.json')},
                pristine_revision='2fc962b973bccdd5d8ce4f67cc63b264d6886495',official_code='51d4c89f72fda44cbf77285c6a8114b52676b8a1',
                language=INSTRUCTIONS,row=10,K=4,H=40,active_dimensions=10,padded_dimensions=132,
                observation=dict(cameras=['base_camera_raw','hand_camera_raw'],raw_shape=[128,128,3],processor_shape=[256,256],EEF='actual_pre_command_TCP_world_XYZ_row_rot6d',gripper='measured_opening',history=False),
                action=dict(EEF='world_task_frame_XYZ_row_rot6d',native_encoding='relative_EEF_absolute_gripper',gripper='independent_open_command',decode_precision='FP32',inference_precision='BF16',first=8,feedback_seconds=.4,control_Hz=20,physics_Hz=100,task_seconds=32),
                code_files=code,native_import_files=dependencies,resolved_import_paths=resolutions,source_environment=dict(platform='LINUX_SOURCE_NON_REALTIME',python=sys.executable,versions=versions,environment_seal_sha256=sha(root/'design/CODE_SEAL.json')),
                qualification=qualification,videos=videos,load_neural_calls='OWNER_ONLY_EXISTING_LOAD_PARITY_ALLOWANCE',consumed=dict(model_queries=0,optimizer_updates=0,physics=0),selected='NONE',deployment='NONE')


def copy_independent(source,target,inventory):
    source=Path(source).resolve();target=Path(target)
    assert source!=target.resolve() and not target.resolve().is_relative_to(source)
    if target.exists():verify_entries(target,inventory)
    else:
        partial=target.with_name(target.name+'.partial');assert not partial.exists(),'Audit interrupted copy before retry'
        assert shutil.disk_usage(target.parent).free>sum(x['bytes'] for x in inventory)+1024**3
        shutil.copytree(source,partial,symlinks=False);verify_entries(partial,inventory)
        for item in inventory:
            with (partial/item['path']).open('r+b') as f:os.fsync(f.fileno())
        partial.replace(target)
    for item in inventory:
        src=source/item['path'];dst=target/item['path']
        assert not src.is_symlink() and not dst.is_symlink() and not os.path.samefile(src,dst)
        assert dst.stat().st_nlink==1,'Hardlinks are not an independent protected recovery'
    return dict(source_root=str(source),protected_root=str(target.resolve()),entries=inventory,independent_bytes=True,links_verified=True)


def protect(root,point,training_audit,view_manifest,destination):
    root=Path(root);q=qualify(root,point);assert q['qualified'] and q['status']=='PLACE_V1_SOURCE_QUALIFIED'
    audit=Path(training_audit);a=json.loads(audit.read_text());assert a['status']==TRAIN_CPU and a['point']==point
    manifest=Path(view_manifest);m=json.loads(manifest.read_text())
    assert m['status']=='IMMUTABLE_SOURCE_POLICY_BUNDLE_CPU_READBACK' and m['point']==point
    mq=m['qualification'];assert mq['qualified'] and mq['status']==q['status'] and mq['counts']==q['counts']
    assert mq['candidate_sha256']==q['candidate_sha256'] and mq['model_binding']==q['model_binding']
    for split in q['proofs']:
        for key in ('source_sha256','CPU_sha256'):assert mq['proofs'][split][key]==q['proofs'][split][key]
    assert m['training_CPU_sha256']==sha(audit) and m['full_Trainer']['entries']==a['entries']
    source=Path(a['checkpoint']);assert source.resolve().is_relative_to((root/'learning').resolve())
    original_export=root/'delivery'/point/'EXPORT.json';e=json.loads(original_export.read_text());assert sha(original_export)==a['export_sha256']
    pristine=Path(e['original_base']);assert pristine.is_dir() and (pristine/'model.safetensors.index.json').is_file()
    reference=root/'design/PRISTINE_LOADING_REFERENCE.json';assert reference.is_file()
    bank=root/'data/SOURCE_BANK.json';b=json.loads(bank.read_text());assert b['status']=='PASS_NEW_TEACHER_BANKS_CPU_READBACK'
    latest_audits=sorted((root/'evidence').glob('*_TRAINING_CPU.json'),key=lambda p:p.stat().st_mtime_ns)
    assert latest_audits;last=json.loads(latest_audits[-1].read_text());assert last['status']==TRAIN_CPU
    latest=Path(last['checkpoint']);assert latest.is_dir() and (latest/'optimizer.pt').is_file()
    assert entries(source)==a['entries']
    saved=json.loads((source/'PI0282_RECOVERY.json').read_text());assert all((source/n).is_file() for n in saved['required_files'])
    assert (source/'optimizer.pt').is_file() and saved['cursor']==int(point.rsplit('-',1)[1])*8
    destination=Path(destination);assert destination.resolve().is_relative_to((root/'protected').resolve())
    destination.parent.mkdir(parents=True,exist_ok=True)
    full=copy_independent(source,destination,a['entries'])
    fullmanifest=root/'qualification/PLACE_V1_FULL_POLICY_MANIFEST.json'
    immutable(fullmanifest,dict(status='QUALIFIED_EXACT_INDEPENDENT_FULL_TRAINER',point=point,full_Trainer=full,immutable_evaluator_manifest_sha256=sha(manifest),immutable_evaluator_manifest=m))
    value=dict(status='PLACE_V1_SOURCE_QUALIFIED_AND_IMMUTABLY_PROTECTED',point=point,CONFIRM_A_success=q['counts']['CONFIRM_A']['T1'],CONFIRM_A_requested=32,
               full_policy_manifest=str(fullmanifest),full_policy_manifest_sha256=sha(fullmanifest),full_Trainer=full,qualification=q,
               active_references=dict(pristine=dict(root=str(pristine),loading_reference_sha256=sha(reference)),original_raw=dict(bank_manifest=str(bank),sha256=sha(bank),source_host_role='CORELAB_ORIGINAL_RAW_EPISODES'),last_audited_resume=dict(root=str(latest),audit_sha256=sha(latest_audits[-1])),qualified_original=str(source),qualified_recovery=str(destination),shared_parent=str(destination)),
               model_queries=0,optimizer_updates=0,physics=0,selected='NONE',deployment='NONE')
    immutable(root/'qualification/PLACE_V1_PROTECTED.json',value)
    return value


def verify_protection(path):
    v=json.loads(Path(path).read_text());assert v['status']=='PLACE_V1_SOURCE_QUALIFIED_AND_IMMUTABLY_PROTECTED'
    assert v['CONFIRM_A_requested']==32 and v['CONFIRM_A_success']>=26
    assert sha(v['full_policy_manifest'])==v['full_policy_manifest_sha256']
    assert v['full_Trainer']['independent_bytes'] and v['full_Trainer']['links_verified']
    verify_entries(v['full_Trainer']['protected_root'],v['full_Trainer']['entries'])
    source=Path(v['full_Trainer']['source_root']);target=Path(v['full_Trainer']['protected_root'])
    for x in v['full_Trainer']['entries']:
        assert not (target/x['path']).is_symlink() and (target/x['path']).stat().st_nlink==1
        if (source/x['path']).exists():assert not os.path.samefile(source/x['path'],target/x['path'])
    return v


def fixture_panels(root,point,split,tasks):
    from contract import SPLITS
    from r3_shared import CPU_STATUS
    size={'TRAIN':8,'DEV':16,'CONFIRM_A':32,'CONFIRM_B':32}[split];start=SPLITS[split][0]
    rows=[dict(point=point,task=t,split=split,layout=s,instruction=INSTRUCTIONS[t],status='VALID_SOURCE_COMPLETE',persistent_success=True) for s in range(start,start+size) for t in tasks]
    source=root/'source/evidence'/f'{point}_{split}.json'
    immutable(source,dict(rows=rows,tasks={t:dict(requested=size,success=size) for t in tasks},paired_both_success=size if len(tasks)==2 else None))
    immutable(root/'audit'/f'{point}_{split}_CPU.json',dict(status=CPU_STATUS,source_sha256=sha(source),rows=[dict(r,score=dict(subpredicates={'T1_hold':dict(hold_pass=0),'T2_hold':dict(hold_pass=0)})) for r in rows],pairs=[dict(layout=s,same_initial_actual_RGB_state_and_cameras={'RGB':True,'state':True}) for s in range(start,start+size)] if len(tasks)==2 else []))


def self_check():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);point='PLACE4096-1024'
        assert not qualify(root,point)['qualified']
        for split in ('TRAIN','DEV','CONFIRM_A'):fixture_panels(root,point,split,['T1'])
        binding={'files':{'model':'fixture'}};immutable(root/'design/MODEL_POINTS.json',{point:binding})
        immutable(root/'qualification/CONFIRM_A_CANDIDATE.json',dict(point=point,confirm_allowed=True,model_binding=binding,development_panels={f'{point}_{s}.json':sha(root/'source/evidence'/f'{point}_{s}.json') for s in ('TRAIN','DEV')}))
        assert qualify(root,point)['qualified']
        p=root/'source/evidence'/f'{point}_CONFIRM_A.json';v=json.loads(p.read_text())
        for row in v['rows'][25:]:row['persistent_success']=False
        v['tasks']['T1']['success']=25;p.write_text(json.dumps(v))
        c=root/'audit'/f'{point}_CONFIRM_A_CPU.json';cpu=json.loads(c.read_text());cpu['source_sha256']=sha(p)
        for row in cpu['rows'][25:]:row['persistent_success']=False
        c.write_text(json.dumps(cpu));assert not qualify(root,point)['qualified']
        src=root/'source';src.mkdir(exist_ok=True);(src/'weight').write_bytes(b'CPU fixture only')
        target=root/'copy';copy_independent(src,target,entries(src));assert not os.path.samefile(src/'weight',target/'weight')
        (target/'weight').write_bytes(b'bad')
        try:verify_entries(target,entries(src))
        except AssertionError:pass
        else:raise AssertionError('Corrupt protected bytes accepted')
        linked=root/'linked';linked.mkdir()
        for x in entries(src):
            dst=linked/x['path'];dst.parent.mkdir(parents=True,exist_ok=True);os.link(src/x['path'],dst)
        try:copy_independent(src,linked,entries(src))
        except AssertionError:pass
        else:raise AssertionError('Linked recovery accepted as independent')
    return dict(status='PASS_CPU_QUALIFICATION_REAL_PANEL_GATES_INDEPENDENT_COPY_CORRUPTION_REJECTION',synthetic_fixtures=True,model_queries=0,physics=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('build','qualify','protect','verify-protection','self-check'));p.add_argument('--root');p.add_argument('--point');p.add_argument('--training-audit');p.add_argument('--view-manifest');p.add_argument('--destination');p.add_argument('--output');a=p.parse_args()
    v=self_check() if a.mode=='self-check' else build(a.root,a.point) if a.mode=='build' else qualify(a.root,a.point) if a.mode=='qualify' else protect(a.root,a.point,a.training_audit,a.view_manifest,a.destination) if a.mode=='protect' else verify_protection(a.destination)
    if a.output:immutable(a.output,v)
    print(json.dumps(v))
