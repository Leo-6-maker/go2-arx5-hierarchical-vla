"""Finite GEN1 first attempts through unchanged PV1 capture.run_cell.

New G_TRAIN identity/authority only. Original Teacher, S1, raw audit, commands,
full-task score and video audit are not replaced. The M2 native counter proof
is reused only with the exact same capture.py, SIF and frozen numerical files.
"""
import argparse,hashlib,json,math,os,resource,time
from pathlib import Path
from . import amd_adoption as a,amd_pilot_scope as p,static_preview as preview

ROLE='GEN1_AMD_FIRST_ATTEMPT_TEACHER_V1'
PROFILE='amd-gen1-teacher-capture'


def read(path):return json.loads(Path(path).read_bytes())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def cards(packet_bytes,index):
    from .spec import design_cards
    a.need(type(index)is int and index in(0,1),'TWO_FINITE_WORKERS')
    plan=p.pilot_plan(packet_bytes);groups=preview.specifications(json.loads(packet_bytes));by_design={}
    for pair in groups.values():
        for card in design_cards(pair['V0']):by_design[(pair['V0']['design_hash'],card['task'])]=card
    values=[]
    for row in plan['attempts'][index*6:(index+1)*6]:
        card=by_design[(row['design_hash'],row['task'])]
        card.update(id=row['attempt_id'],nominal_clearance_m=.004,
            origin='GEN1_REGISTERED_TRAIN_TEACHER_FIRST_ATTEMPT',study_split='G_TRAIN',
            geometry_id=row['geometry_id'],GEN1_design_hash=row['design_hash'],
            GEN1_lineage_id=row['lineage_id'],visual_profile='V0')
        a.need(card['split']=='G_TRAIN'and'/'not in card['id']and len(card['id'])<100,
            'SAFE_RAW_ID_AND_EXPLICIT_GEN_TRAIN_NOT_OLD_PV1')
        values.append(card)
    a.need(len(values)==6 and {c['task']for c in values}=={'T1','T2'},'SIX_FIXED_CELLS_PER_WORKER')
    return values


def binding(root):
    r=Path(root);a.need(os.getuid()==2009 and os.uname().nodename=='corelab-amd-server'
        and r.is_absolute()and not r.is_symlink(),'GEN1_CAPTURE_HOST_UID')
    m=read(r/'RUN_MANIFEST.json');b=read(r/'OUTPUT_BINDING.private.json')
    a.need(m['scope']==ROLE and m['execution']['profile']==PROFILE
        and m['output_root']==b['run_root']==str(r)and m['run_uuid']==r.name
        and b['run_manifest_sha256']==sha(r/'RUN_MANIFEST.json'),'OWN_CAPTURE_MANIFEST')
    g=read(r/'plans/ACTUAL_GLOBAL_CAPTURE_ALLOCATION.private.json');mark=a.validate(g);pool=mark['teacher_pilot']
    row=pool['workers'][str(m['worker_index'])]
    a.need(pool['status']=='RESERVED_FIRST_ATTEMPTS_ONLY'and row['manifest_sha256']==b['run_manifest_sha256']
        and row['root']==str(r)and row['attempt_ids']==[c['id']for c in cards((r/preview.PACKET).read_bytes(),m['worker_index'])],
        'REAL_GLOBAL_UNIQUE_FIRST_ATTEMPT_ALLOCATION')
    limits=m['limits'];a.need(all(limits.get(k)==v for k,v in dict(CPU_s=21600,physics_ticks=192000,
        engine_starts=504,policy_episodes=6,per_cell_CPU_s=3600,per_cell_ticks=32000,
        per_cell_task_shadow_constructors=81,per_cell_readonly_audit_constructors=3,
        memory_bytes=8*1024**3).items()),'ORIGINAL_TWELVE_TASK_AND_EXPLICIT_READONLY_AUDIT_ENVELOPE')
    a.need(type(limits['cpus'])is list and len(limits['cpus'])==1
        and set(limits['cpus'])<=os.sched_getaffinity(0),'ACTUAL_SINGLE_CORE_WORKER')
    lease=read(b['lease']);a.need(lease['status']=='ACTIVE'and lease['run_uuid']==r.name
        and lease['manifest_sha256']==b['run_manifest_sha256']and lease['deadline_epoch']==mark['deadline_epoch']
        and mark['T_CD']<=time.time()<min(lease['deadline_epoch'],limits['operation_deadline_epoch']),
        'OWN_ACTIVE_ORIGINAL_WINDOW_LEASE')
    a.need(not any(x.exists()for x in(r/'STOP.json',r.parent.parent/'STOP.json')),'ORIGINAL_SOURCE_STOP_FENCE')
    return m,b


def validate(root):
    r=Path(root);m,b=binding(r);code=r/m['code']['capsule_relative'];seal=read(r/'SOURCE_SEAL.json')
    a.need(Path(__file__).resolve()==(code/'scripts/pi0294/generalization/amd_capture.py').resolve()
        and sha(r/'SOURCE_SEAL.json')==m['code']['source_seal_sha256'],'ACTUAL_IMMUTABLE_CAPTURE_SOURCE')
    for name,h in seal['files'].items():a.need(sha(code/name)==h,'CAPTURE_SOURCE:'+name)
    for name,h in seal['frozen15'].items():a.need(sha(code/name)==h,'ORIGINAL_NUMERICAL:'+name)
    for name,h in m['execution']['sealed_inputs'].items():a.need(sha(r/name)==h,'CAPTURE_INPUT:'+name)
    prior=read(r/'plans/ACTUAL_M2_COUNTER_PROOF.private.json')
    a.need(prior['counter']['status']=='PASS_ORIGINAL_VS_COUNTED_NATIVE_MODEL_STATE_XML_AND_TWO_RGB'
        and prior['counter']['integrations']==0 and prior['SIF_sha256']==a.SIF_SHA
        and prior['capture_py_sha256']==sha(code/'scripts/pi0294/pv1/capture.py')
        and prior['static24_independent_audit_status']=='PASS_INDEPENDENT_ACTUAL_AMD_STATIC24_ZERO_INTEGRATION',
        'REUSE_EXACT_ORIGINAL_NATIVE_COUNTER_PROOF_NO_CHANGED_WRAPPER')
    assets=read(r/'plans/AMD_ASSETS.private.json');a.need(len(assets['files'])==26,'ORIGINAL_ASSETS')
    for name,h in assets['files'].items():a.need(sha(Path(assets['root'])/name)==h,'ASSET:'+name)
    reg=read(b['registry']);a.need(sum(x.get('role')==b['role']and x['path']==str(r)for x in reg['entries'])==1,'REGISTERED_CAPTURE_ROLE')
    a.need(os.environ.get('CUDA_VISIBLE_DEVICES')==''and os.environ.get('MUJOCO_GL')=='egl'
        and os.environ.get('LIBGL_ALWAYS_SOFTWARE')=='1'
        and os.environ.get('__EGL_VENDOR_LIBRARY_FILENAMES')=='/usr/share/glvnd/egl_vendor.d/50_mesa.json'
        and 'MUJOCO_EGL_DEVICE_ID'not in os.environ and 'PBS_JOBID'not in os.environ,'ORIGINAL_MESA_CPU_NO_NEURAL_GPU')
    return m,assets


def run(root):
    from scripts.pi0294.pv1 import capture as original
    from scripts.pi0294.pv1.mix_vanda_codec import validate as codec_validate
    from scripts.pi0292.durable import write,OwnerLock
    r=Path(root);m,assets=validate(r);coordinator=r.parent.parent
    codec_validate(coordinator,read(coordinator/'RUN_MANIFEST.json'))
    a.need(read(coordinator/'CODEC_NATIVE_ABI_RESULT.json')['status']=='PASS_ORIGINAL_AUDITOR_CODEC_IN_ORIGINAL_SIF',
        'REAL_CODEC_ABI_BEFORE_ANY_CONSTRUCTOR')
    plan=cards((r/preview.PACKET).read_bytes(),m['worker_index']);source_sha=m['code']['vla_commit']
    lock=OwnerLock(r/'GEN1_CAPTURE_OWNER.lock');a.need(not(r/'GEN1_CAPTURE_STARTED.json').exists(),'FIRST_ATTEMPTS_NOT_REPLAYED')
    class Meter(original.Meter):
        def __init__(self):
            self.root=r;self.started=time.process_time();self.children=resource.getrusage(resource.RUSAGE_CHILDREN)
            self.m=m;self.inputs={'asset_root':assets['root']};self.plan={'cards':plan};self.ledger=read(r/'RESOURCE_LEDGER.json')
            a.need(self.ledger['status']=='RESERVED'and self.ledger['unknown_effects']==[]and self.ledger['attempted_cells']==[],
                'FRESH_WORKER_EFFECT_LEDGER')
            self.ticks=self.live_ticks=self.shadow_ticks=self.starts=0;self.cell=None;self.cell_cpu=0.;self.cell_ticks=self.cell_starts=0
            self.audit_phase=False;self.audit_starts=self.cell_audit_starts=0
            self.lease_path=read(r/'OUTPUT_BINDING.private.json')['lease'];self.lease_check_at=0.
            original.ensure_legacy_attempt_plan(r,plan);self.progress(status='RUNNING')
        def guard(self):
            original.Meter.guard(self)
            if self.cell is not None:
                a.need(time.process_time()-self.cell_cpu<3570,'FINITE_CELL_CPU_SAVING_MARGIN')
                audit=self.audit_starts-self.cell_audit_starts
                a.need(self.ticks-self.cell_ticks<=32000 and self.starts-self.cell_starts-audit<=81
                    and audit<=3,'FINITE_TASK_AND_SEPARATE_READONLY_AUDIT_CONSTRUCTORS')
            if any(x.exists()for x in(r/'STOP.json',coordinator/'STOP.json')):raise PermissionError('GEN1_SOURCE_STOP')
            now=time.monotonic()
            if now-self.lease_check_at>=.25:
                a.need(read(self.lease_path)['status']=='ACTIVE','LIVE_CAPTURE_LEASE');self.lease_check_at=now
        def start(self):
            if self.cell is not None:
                audit=self.audit_starts-self.cell_audit_starts
                a.need(audit<3 if self.audit_phase else self.starts-self.cell_starts-audit<81,
                    'TASK_OR_READONLY_AUDIT_CAP_BEFORE_CALL')
            value=original.Meter.start(self)
            if self.audit_phase:self.audit_starts+=1
            return value
        def step(self,engine,method,target,opening,live):
            if self.cell is not None:a.need(self.ticks-self.cell_ticks<32000,'CELL_TICK_CAP_BEFORE_CALL')
            return original.Meter.step(self,engine,method,target,opening,live)
    meter=Meter();rows=[];failure=None;started=time.process_time();children=resource.getrusage(resource.RUSAGE_CHILDREN)
    from scripts.pi0294 import audit as raw_audit
    original_audit=raw_audit.audit_episode
    def audit_with_classified_cost(*args,**kwargs):
        meter.audit_phase=True
        try:return original_audit(*args,**kwargs)
        finally:meter.audit_phase=False
    raw_audit.audit_episode=audit_with_classified_cost
    def stop(reason):
        try:original.put(coordinator/'STOP.json',dict(reason=reason,worker=r.name))
        except FileExistsError:pass  # First safety signal remains authoritative.
    resource.setrlimit(resource.RLIMIT_CPU,(21600,21600));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    write(r/'GEN1_CAPTURE_STARTED.json',dict(epoch=time.time(),pid=os.getpid(),
        procstart=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source_sha=source_sha,
        reused_M2_counter_qualification=True,new_counter_probe=False))
    try:
        for card in plan:
            meter.guard();a.need(card['id']not in meter.ledger['attempted_cells'],'FIRST_ATTEMPT_ONLY')
            meter.cell=card['id'];meter.cell_cpu=time.process_time();meter.cell_ticks=meter.ticks;meter.cell_starts=meter.starts;meter.cell_audit_starts=meter.audit_starts
            meter.ledger['attempted_cells'].append(card['id']);meter.progress(current_cell=card['id'])
            value=original.run_cell(meter,card)
            rows.append(dict(attempt_id=card['id'],study_split='G_TRAIN',geometry_id=card['geometry_id'],
                design_hash=card['GEN1_design_hash'],visual_profile='V0',native_scope=ROLE,
                original_capture_receipt=value,legacy_receipt_split_is_compatibility_only=True,
                full_task_success=value['task_success'],codec_for_training='NOT_RUN_REQUIRES_VANDA_OFFICIAL_EXPORT'))
            write(r/'GEN1_CAPTURE_PROGRESS.json',dict(attempts=rows,expected_ids=[c['id']for c in plan]))
            meter.cell=None
            if(r/'STOP.json').exists():stop('CAPTURE_NATIVE_SAFETY_OR_INTEGRITY');break
        return rows
    except BaseException as exc:
        failure=type(exc).__name__+': '+str(exc)
        meter.ledger['unknown_effects'].append(dict(error=failure,full_remaining_reservation_retained=True))
        stop('CAPTURE_EFFECTS_UNKNOWN_OR_ENGINEERING_FAILURE')
        raise
    finally:
        raw_audit.audit_episode=original_audit
        after=resource.getrusage(resource.RUSAGE_CHILDREN);cost=time.process_time()-started+after.ru_utime+after.ru_stime-children.ru_utime-children.ru_stime
        meter.progress(status='TERMINAL_KNOWN'if failure is None else'FAILED_PRESERVED_UNKNOWN_EFFECTS',
            CPU_s=cost,source_sha=source_sha)
        write(r/'GEN1_CAPTURE_RESULT.json',dict(status='KNOWN_FIRST_ATTEMPTS_TERMINAL'if failure is None else'FAILED_PRESERVED',
            attempts=rows,expected_ids=[c['id']for c in plan],error=failure,CPU_s=cost,
            updates=0,model_loads=0,queries=0,counter_probe_reused=True,source_sha=source_sha,
            readonly_audit_constructor_attempts=meter.audit_starts))
        lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    print(json.dumps(run(args.root)))
