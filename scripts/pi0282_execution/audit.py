"""PI0282 finite CPU evidence audit; no model/controller/physics imports."""
import argparse,hashlib,json,os,re,time
from pathlib import Path

ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0282')
BASE=ROOT.parent
SEEDS=set(range(30032,30048))
TOKEN=re.compile(r'(?<![A-Za-z0-9.])(3003[2-9]|3004[0-7])(?![A-Za-z0-9.])')
SEEDKEY=re.compile(r'(seed|condition|episode|slot|trial|case|uid|request|query|path|folder|id)',re.I)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def identity(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))

def save(folder,name,value):
    p=ROOT/folder/name;p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        h=p.parent/'history';h.mkdir(exist_ok=True);(h/(p.stem+'-'+sha(p)+p.suffix)).write_bytes(p.read_bytes())
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)
    return p

def seed_mentions(value,key='',location='$'):
    out=[]
    if isinstance(value,dict):
        for k,v in value.items():out.extend(seed_mentions(v,k,location+'.'+k))
    elif isinstance(value,list):
        for i,v in enumerate(value):out.extend(seed_mentions(v,key,location+f'[{i}]'))
    elif SEEDKEY.search(key):
        if isinstance(value,int) and not isinstance(value,bool) and value in SEEDS:out.append(dict(seed=value,field=location,value=value))
        elif isinstance(value,str):
            for m in TOKEN.finditer(value):out.append(dict(seed=int(m[1]),field=location,value=value[:600]))
    return out

def exposure():
    """Narrow metadata/journal/RESULT index inventory, excludes payloads and model trees."""
    roots=[]
    for n in range(276,283):
        r=BASE/f'PI0{n}'
        roots += [r/p for p in ('evidence','design','source','runtime') if (r/p).is_dir()]
        for phase in ('R2','R3'):
            roots += [r/phase/p for p in ('evidence','design','contracts','runtime') if (r/phase/p).is_dir()]
        r=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5')/f'PI0{n}'
        roots += [r/p for p in ('evidence','episodes','quality','source','fixed_inputs') if (r/p).is_dir()]
        for phase in ('R2','R3'):
            roots += [r/phase/p for p in ('evidence','episodes','quality') if (r/phase/p).is_dir()]
    skipped_names={'code','official','cache','hf_home','venv','ManiSkill','project-src','training','checkpoints','model_bindings','archive','audit','videos','assets'}
    files=[];matches=[];directories=[];errors=[]
    for root in roots:
        for path,dirs,names in os.walk(root):
            dirs[:]=[d for d in dirs if d not in skipped_names and not d.startswith('q0') and d!='history']
            for d in dirs:
                if TOKEN.search(d):directories.append(str(Path(path)/d))
            for name in names:
                p=Path(path)/name
                if p.suffix not in ('.json','.jsonl'):continue
                if name in ('rows.json','contacts.jsonl','actor_inputs.jsonl','actor_outputs.jsonl','GATES.json','TIMINGS.jsonl'):continue
                if p.stat().st_size>256<<20:errors.append(dict(path=str(p),reason='METADATA_FILE_EXCEEDS_256MiB'));continue
                record=identity(p);record['parsed_rows']=0;files.append(record)
                try:
                    values=(json.loads(line) for line in p.open() if line.strip()) if p.suffix=='.jsonl' else [json.loads(p.read_text())]
                    for row,value in enumerate(values):
                        record['parsed_rows']+=1
                        found=seed_mentions(value)
                        if found:
                            actual=(name in ('RESULT.json','QUERIES.json','START.json','PARENT.json') or any(x in name.lower() for x in ('model_attempt','model_head','model_budget','budget.jsonl')))
                            matches.append(dict(path=str(p),row=row,classification='EXECUTION_RECEIPT_OR_JOURNAL' if actual else 'CARD_OR_INDEX_METADATA_REQUIRES_CONTEXT',mentions=found))
                except (OSError,ValueError) as e:errors.append(dict(path=str(p),reason=str(e)))
    execution=[r for r in matches if r['classification']=='EXECUTION_RECEIPT_OR_JOURNAL']
    status='EXPOSURE_FOUND_PRESERVE_SEEDS' if execution or directories else ('NO_PRIOR_QUERY_FOUND_IN_SCOPED_RECEIPTS' if not errors else 'UNRESOLVED_INVENTORY_ERRORS')
    v=dict(schema='PI0282_CONFIRM_EXPOSURE_READBACK_V1',status=status,seeds=sorted(SEEDS),observed_unix_s=time.time(),roots=[str(p) for p in roots],
        confirm_allowed=status=='NO_PRIOR_QUERY_FOUND_IN_SCOPED_RECEIPTS',
        confirm_allowed_semantics='Exposure prerequisite only in the explicitly listed retained Corelab roots; selection and five-part candidate freeze must independently pass. This receipt does not unlock execution by itself.',
        files=files,matches=matches,seed_directories=directories,errors=errors,
        scope='Existing NFS PI0276..PI0282 source/evidence/design/runtime plus R2/R3 metadata; actual NVMe PI0276..PI0279 evidence/episodes/quality/source/fixed_inputs and R2/R3; semantic seed/id/path fields only. No model/array payload scan.',
        limitations='Absence in these retained receipts is not a proof about unrecorded execution or unrelated external machines; owner must bind historical source locations.',
        no_seed_replacement=True,new_model_calls=0,new_physics_steps=0)
    save('audit','CONFIRM_EXPOSURE.json',v)
    print(json.dumps(dict(status=status,files=len(files),execution_hits=len(execution),metadata_hits=len(matches)-len(execution),directory_hits=directories,errors=errors)))

def fixed_rules():
    v=dict(schema='PI0282_FIXED_AUDIT_AND_SELECTION_V1',development=dict(eligible_endpoints=['INTERFACE2048','ACTION2048'],TRAIN_denominator=4,TRAIN_min=3,DEV_denominator=8,DEV_min=6),
        ordering=['DEV persistent count descending','TRAIN persistent count descending','INTERFACE2048 preferred'],
        confirmation=dict(seeds=sorted(SEEDS),denominator=16,min_persistent=12,one_frozen_candidate_only=True,previously_unqueried_required=True,failed_confirm_no_second_candidate=True),
        metric='Full valid32s and unchanged on_top & static & not_grasped at every41 control samples28..30s; early valid failures retained in denominator, missing suffix never fabricated',
        identity_freeze=['model','processor','statistics','mapping','decode'],TRAIN_statistics_only=True,retention_not_selection=True,
        expected_rows=dict(index=10,scalars=10180740,slices=14,packed_scalars=325783680),training=dict(updates_each=2048,chunks_each=16384,phase_chunks_each=4096,effective_batch=8,visits_per_episode_phase=[44,45],episodes=93),
        required_actual_receipts=['original row10 slice hashes','frozen-row byte hashes','optimizer moments/grad masks/zero projector decay','actual sample cursor/order','actual full132 noise/time hash per forward','same initial/model snapshot both arms','canary fresh-process resume parity','TRAIN codec physical errors','cache actual parity'],
        completed_receipt_audits=[],source_candidate='NONE_UNTIL_FIXED_GATES',selected='NONE',deployment='NONE',new_model_calls=0,new_physics_steps=0)
    save('audit','FIXED_RULES.json',v);print('FIXED_RULES_READY_NOT_EXECUTED')

if __name__=='__main__':
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only audit requires hidden CUDA'
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['exposure','rules']);a=p.parse_args();exposure() if a.phase=='exposure' else fixed_rules()
