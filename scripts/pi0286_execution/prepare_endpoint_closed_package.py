"""Fixed small endpoint evidence, all32 saved videos; no model, physics or RAW arrays."""
import hashlib, ipaddress, json, os, re, resource, shutil, tarfile, time
from pathlib import Path

R=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0286')
V=R/'evidence/video/endpoint-saved-rgb/V2-complete-text'
A=R/'archive/endpoint-closed'
B=A/'bundle'
POINTS=('C0_REPLAY1024-1024','C1_CORRECT1024-1024')
PANELS=((POINTS[0],'TRAIN'),(POINTS[1],'TRAIN'),(POINTS[1],'DEV'))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def put(p,v):
    p.write_text(json.dumps(v,indent=2)+'\n')

def scan_public_text(name,raw):
    text=raw.decode('utf8')
    assert not re.search(r'\bhf_[A-Za-z0-9]{20,}\b|-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----',text),('CREDENTIAL_LITERAL',name)
    private=[ipaddress.ip_network(n) for n in ('10.0.0.0/8','172.16.0.0/12','192.168.0.0/16')]
    for found in re.findall(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])',text):
        try:addr=ipaddress.ip_address(found)
        except ValueError:continue
        assert not any(addr in n for n in private),('PRIVATE_ADDRESS_LITERAL',name)

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    resource.setrlimit(resource.RLIMIT_AS,(16<<30,16<<30))
    assert json.loads((V/'ALL32_VIDEO_CLOSED.json').read_text())['status']=='PASS_CLOSED_ALL32_ACTUAL_ENDPOINT_SAVED_RGB_FULL_DECODE'
    assert not (A/'PACKAGE_READY.json').exists(),'Frozen package remains immutable'
    B.mkdir(parents=True);members=[];copied={};conditions=[]
    def copy(rel,name):
        source=R/rel;target=B/name;assert source.is_file() and not target.exists()
        shutil.copyfile(source,target);assert sha(source)==sha(target)
        copied[name]=dict(source=str(source),source_sha256=sha(source))
    for point in POINTS:
        short='C0' if point==POINTS[0] else 'C1'
        closure=json.loads((R/'runtime'/(point+'_endpoint_EVALUATION_CLOSED.json')).read_text())
        assert closure['status'] in ('ENDPOINT_CLOSED_VALID_TRAIN_FAIL','ENDPOINT_CLOSED_VALID_DEVELOPMENT_FAIL')
        for rel,name in [('runtime/'+point+'_endpoint_EVALUATION_CLOSED.json',short+'_ENDPOINT_CLOSED.json'),
                         ('qualification/'+point+'_ROUTE.json',short+'_ROUTE.json'),
                         ('diagnostics/'+point+'/RESULT.json',short+'_DIAG_ACTUAL.json'),
                         ('audit/'+point+'_DIAG_CPU.json',short+'_DIAG_CPU.json'),
                         ('models/'+point+'/MATERIALIZATION.json',short+'_MODEL_MATERIALIZATION.json'),
                         ('transfers/'+point+'/READY.json',short+'_MODEL_READY.json'),
                         ('transfers/'+point+'/EXPORT.json',short+'_EXPORT_IDENTITY.json'),
                         ('transfers/'+point+'/TRAINING_CPU.json',short+'_TRAINING_CPU.json'),
                         ('transfers/'+point+'/DIRECT_READBACK.json',short+'_DIRECT_TRANSFER_READBACK.json'),
                         ('runtime/'+point+'_ENDPOINT_EVALUATION_DISPATCH.json',short+'_EVALUATION_DISPATCH.json')]:copy(rel,name)
        diag=json.loads((R/'audit'/(point+'_DIAG_CPU.json')).read_text())
        assert diag['status']=='PASS_CPU_FIXED_DIAG_NATIVE_DECODE_NOISE_PREFIX_FULL_H40_AND_GATE' and len(diag['rows'])==96
        assert diag['source_sha256']==sha(R/'diagnostics'/point/'RESULT.json')
        timers=[]
        for p in (R/'runtime').glob('*EVAL_COMPLETE.json'):
            t=json.loads(p.read_text())
            if point in str(t):timers.append(dict(path=str(p),sha256=sha(p),receipt=t))
        put(B/(short+'_ACTUAL_STAGE_TIMERS.json'),dict(point=point,timers=timers,note='Same-argument qualification clock filename holds its final call; dispatch-to-closure clocks cover the whole finite evaluator.'))
    for point,panel in PANELS:
        group=point+'_'+panel;short=('C0' if point==POINTS[0] else 'C1')+'_'+panel
        source=R/'source/evidence'/(group+'.json');cpu=R/'audit'/(group+'_CPU.json')
        s=json.loads(source.read_text());c=json.loads(cpu.read_text())
        assert c['status']=='PASS_INDEPENDENT_CPU_CONTACT_SCORE_CLOCK_INSTALLATION_AND_PAIRED_INITIAL_INPUTS' and c['source_sha256']==sha(source)
        copy(str(source.relative_to(R)),short+'_SOURCE.json');copy(str(cpu.relative_to(R)),short+'_CPU.json')
        for suffix in ['.mp4','_SOURCE_INDEX.json','_VIDEO.json','_DELIVERY.json']:
            copy(str((V/(group+suffix)).relative_to(R)),short+suffix)
        video=json.loads((V/(group+'_VIDEO.json')).read_text());assert sha(V/(group+'.mp4'))==video['sha256']
        for row in s['rows']:
            folder=Path(row['folder']);result=folder/'RESULT.json';assert json.loads(result.read_text())==row
            conditions.append(dict(point=point,panel=panel,layout=row['layout'],persistent_success=row['persistent_success'],result_sha256=sha(result),source_file_hashes=row['files'],video=short+'.mp4'))
            for name in ['RESULT.json','START.json','EPISODE_CARD.json','TASK_SCORE_CPU.json','QUERIES.json','NATIVE_INVENTORY.json']:
                p=folder/name;assert p.is_file()
                h=sha(p)
                if name!='RESULT.json':assert h==row['files'][name]
                members.append(dict(source=str(p),member=group+'/'+str(row['layout'])+'/'+name,bytes=p.stat().st_size,sha256=h))
    assert len(conditions)==32 and len(members)==192
    tarpath=B/'SOURCE_SMALL_RECORDS.tar.gz'
    with tarfile.open(tarpath,'w:gz',compresslevel=6,dereference=True) as tar:
        for item in members:tar.add(item['source'],arcname=item['member'],recursive=False)
    with tarfile.open(tarpath,'r:gz') as tar:
        actual=tar.getmembers();assert [m.name for m in actual]==[m['member'] for m in members]
        for m,item in zip(actual,members):
            f=tar.extractfile(m);h=hashlib.sha256();total=0
            for block in iter(lambda:f.read(8<<20),b''):h.update(block);total+=len(block)
            assert total==item['bytes'] and h.hexdigest()==item['sha256']
    for rel,name in [('design/COMPARISON_CLOSE.json','COMPARISON_CLOSE.json'),
                     ('qualification/FINAL_NO_ELIGIBLE_CANDIDATE.json','FINAL_NO_ELIGIBLE_CANDIDATE.json'),
                     ('runtime/STOP_NEW_ARMS.json','STOP_NEW_ARMS.json'),
                     ('design/CODE_SEAL.json','CORE_CODE_SEAL.json'),('design/MODEL_POINTS.json','MODEL_POINTS.json'),
                     ('design/DIAG96_PER_TASK.json','DIAG96_FIXED_PLAN.json'),
                     ('evidence/evaluation/USER_BOTH_ARMS_COMPLETION_PRIORITY_NO_1400_CUTOFF.json','USER_NO_1400_CUTOFF_CPU.json'),
                     ('evidence/evaluation/C1_SSD_MODEL_ONLY_ALIAS.json','C1_SSD_MODEL_ONLY_ALIAS.json')]:copy(rel,name)
    for name in ['video_saved_endpoints_v2.py','VIDEO_TECHNICAL_REPAIR.json','ALL32_VIDEO_CLOSED.json']:
        copy(str((V/name).relative_to(R)),name)
    assert sha(R/'qualification/FINAL_NO_ELIGIBLE_CANDIDATE.json')=='ae3e4d81072a1dc664b21cb908972e8185fb0fa8668e1ddb55c1f8e550e6bb1e'
    seal=json.loads((R/'design/CODE_SEAL.json').read_text());code_members=[]
    for name,h in sorted(seal['files'].items()):
        p=R/'code'/name;assert sha(p)==h
        code_members.append(dict(source=str(p),member=name,bytes=p.stat().st_size,sha256=h))
    with tarfile.open(B/'FROZEN_CORE_CODE.tar.gz','w:gz',compresslevel=6,dereference=True) as tar:
        for item in code_members:tar.add(item['source'],arcname=item['member'],recursive=False)
    with tarfile.open(B/'FROZEN_CORE_CODE.tar.gz','r:gz') as tar:
        actual=tar.getmembers();assert [m.name for m in actual]==[x['member'] for x in code_members]
        for member,item in zip(actual,code_members):assert hashlib.sha256(tar.extractfile(member).read()).hexdigest()==item['sha256']
    put(B/'INVENTORY.json',dict(status='PASS_FIXED_SMALL_SIDECARS_AND_SEALED_CODE_TAR_MEMBER_READBACK',conditions=conditions,source_small_members=members,code_members=code_members,copied_sources=copied,raw_arrays='Original Core source retained; not copied into publication package',runtime_keys_tokens_caches=False,new_model_queries=0,new_physics=0,new_render=0))
    (B/'README.md').write_text('''# PI0286: actual C0/C1 endpoint evidence\n\nBoth original 1024-update endpoints completed. C0: fixed TRAIN2/8, valid TRAIN gate failure, DEV omitted by original gate. C1: fixed TRAIN8/8 and DEV5/16, valid development gate failure. Source CPU audits and both DIAG96 CPU audits passed. Original selector returned NO_DEVELOPMENT_QUALIFIED_CANDIDATE. No candidate or CONFIRM_C; selected/deployment NONE. These outcomes are source Panda native PD, NON-REALTIME, not target robot/WBC qualification.\n\nThree videos cover all32 actually evaluated source conditions: C0 TRAIN8, C1 TRAIN8 and C1 DEV16. Each video page shows four fixed layouts with both saved cameras. T1 persistent hold/41 and original stack hold/41 are labelled from actual saved results. Playback follows saved simulation time, not measured wall execution. No new renderer, physics or model query was used. Source indices bind every segment to original RESULT, observations, physics and independent CPU hashes.\n\nSOURCE_SMALL_RECORDS.tar.gz contains exactly192 declared JSON sidecars from those32 conditions, including full native input/output/RGB hash inventories. Original large RGB, physics and model output arrays remain in Core source storage; no arrays or weights, original64 dataset, runtime keys, tokens or caches are uploaded here. FROZEN_CORE_CODE.tar.gz contains only the Core seal's declared files. The public package is separate from the earlier correction-closed historical snapshot and Hopper full-model archive.\n\nThe user explicitly authorized both arms to complete beyond14:00. The archived CPU receipt records that calendar extension while cumulative scientific/GPU budgets and original gates remained unchanged.\n''',encoding='utf8')
    for p in B.iterdir():
        if p.suffix in ('.json','.md','.py'):scan_public_text(p.name,p.read_bytes())
        elif p.name.endswith('.tar.gz'):
            with tarfile.open(p,'r:gz') as tar:
                for member in tar.getmembers():scan_public_text(p.name+'/'+member.name,tar.extractfile(member).read())
    files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in B.iterdir() if p.is_file()}
    ready=dict(status='PASS_FIXED_CLOSED_ENDPOINT_SMALL_EVIDENCE_AND_ALL32_VIDEO_PACKAGE',scope='C0_C1_ENDPOINT_CLOSED_NO_DEVELOPMENT_QUALIFIED_CANDIDATE',repo_id='leoyuer/pi0246-gate-pass-campaign',prefix='evidence/PI0286/recovery-vs-replay-20261009/endpoint-closed',files=files,total_bytes=sum(x['bytes'] for x in files.values()),condition_count=32,original_source_retained=True,contains_weights=False,contains_raw_arrays=False,new_model_queries=0,new_physics=0,new_render=0,epoch=time.time())
    put(A/'PACKAGE_READY.json',ready);print(json.dumps(ready))

if __name__=='__main__':main()
