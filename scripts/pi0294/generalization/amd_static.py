"""Registered AMD M2 using original GEN1 inspectors and Core effect counters.

44 constructors maximum; mj_step is prohibited. No Teacher or neural model.
"""
import argparse,json,os,resource,time
from pathlib import Path
from . import amd_adoption as a,static_preview as preview

ROLE='GEN1_AMD_STATIC_PREVIEW_V1'
PROFILE='amd-gen1-static-preview'
SCHEMA='GEN1_AMD_STATIC_NATIVE_COUNTERS_V1'
LIMITS=dict(CPU_s=1200,physics_ticks=0,policy_episodes=0,engine_starts=44)


def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    import hashlib
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def binding(root):
    r=Path(root);m=read(r/'RUN_MANIFEST.json');b=read(r/'OUTPUT_BINDING.private.json')
    a.need(os.getuid()==2009 and os.uname().nodename=='corelab-amd-server'
        and r.is_absolute()and not r.is_symlink()and r.stat().st_uid==2009,'AMD_STATIC_HOST_UID')
    a.need(m['scope']==ROLE and m['run_uuid']==r.name=='gen1-amd-static24-01'
        and m['execution']['profile']==PROFILE
        and m['output_root']==b['run_root']==str(r)
        and b['run_manifest_sha256']==sha(r/'RUN_MANIFEST.json'),'OWN_STATIC_MANIFEST')
    a.need(all(m['limits'][k]==v for k,v in LIMITS.items())
        and type(m['limits']['cpus'])is list and len(m['limits']['cpus'])==1
        and set(m['limits']['cpus'])<=os.sched_getaffinity(0),'STATIC_LIMITS_REAL_CPU')
    g=read(r/'plans/ACTUAL_GLOBAL_STATIC_ALLOCATION.private.json');mark=a.validate(g)
    row=mark['static_previews'];receipt=read(r/'plans/ACTUAL_GLOBAL_STATIC_RECEIPT.private.json')
    a.need(row['run_uuid']==r.name and row['status']=='RESERVED_NOT_SPENT'
        and row['manifest_sha256']==b['run_manifest_sha256']
        and row['engine_starts']==44 and row['physics_ticks']==0
        and receipt['after_sha256']==sha(r/'plans/ACTUAL_GLOBAL_STATIC_ALLOCATION.private.json'),
        'ACTUAL_ONCE_ALLOCATED_M2_NOT_A_NEW_BUDGET')
    a.need(receipt['status']=='CONFIRMED_SINGLE_METADATA_WRITE_NOT_CROSS_SITE_ADMISSION',
        'ORIGINAL_STATIC_ONCE_RECEIPT')
    lease=read(b['lease']);a.need(lease['status']=='ACTIVE'and lease['run_uuid']==r.name
        and lease['manifest_sha256']==b['run_manifest_sha256']
        and lease['deadline_epoch']==mark['deadline_epoch'],'LIVE_STATIC_LEASE')
    a.need(mark['T_CD']<=time.time()<min(mark['deadline_epoch'],m['limits']['operation_deadline_epoch']),
        'ORIGINAL_CLOCK_STATIC_WINDOW')
    a.need(not any(p.exists()for p in(r/'STOP.json',r/'SOURCE_PHYSICS_STOP.json',r.parent.parent/'STOP.json')),
        'ORIGINAL_STOP_FENCES')
    return m,b


def validate(root):
    r=Path(root);m,b=binding(r);code=r/m['code']['capsule_relative'];seal=read(r/'SOURCE_SEAL.json')
    a.need(Path(__file__).resolve()==(code/'scripts/pi0294/generalization/amd_static.py').resolve()
        and sha(r/'SOURCE_SEAL.json')==m['code']['source_seal_sha256'],'ACTUAL_IMMUTABLE_STATIC_SOURCE')
    for name,h in seal['files'].items():
        p=code/name;a.need(not p.is_symlink()and not p.stat().st_mode&0o222 and sha(p)==h,'STATIC_SOURCE:'+name)
    for name,h in seal['frozen15'].items():a.need(sha(code/name)==h,'ORIGINAL_NUMERICAL:'+name)
    for name,h in m['execution']['sealed_inputs'].items():
        p=r/name;a.need(not p.is_symlink()and not p.stat().st_mode&0o222 and sha(p)==h,'STATIC_INPUT:'+name)
    assets=read(r/'plans/AMD_ASSETS.private.json');a.need(len(assets['files'])==26,'EXACT_26_ASSETS')
    for name,h in assets['files'].items():a.need(sha(Path(assets['root'])/name)==h,'ACTUAL_ASSET:'+name)
    a.need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('MUJOCO_GL')=='egl'
        and os.environ.get('LIBGL_ALWAYS_SOFTWARE')=='1'
        and os.environ.get('__EGL_VENDOR_LIBRARY_FILENAMES')=='/usr/share/glvnd/egl_vendor.d/50_mesa.json'
        and 'MUJOCO_EGL_DEVICE_ID'not in os.environ
        and 'PBS_JOBID'not in os.environ,'ORIGINAL_SOFTWARE_RENDER_NO_GPU_ALIAS')
    reg=read(b['registry']);env=read(b['environment'])
    a.need(sum(x.get('role')==ROLE and x['path']==str(r)for x in reg['entries'])==1
        and sum(x['name']==PROFILE and x['manifest']==str(r/'RUN_MANIFEST.json')for x in env['profiles'])==1,
        'ACTUAL_CANONICAL_REGISTRY_AND_PROFILE')
    return m


def ledger(root):
    """Adapt only authority binding; original counter transaction methods stay."""
    from scripts.pi0294.pv1.core_ledger import Ledger
    class StaticLedger(Ledger):
        def __init__(self,root):
            self.root=Path(root);self.path=self.root/'RESOURCE_LEDGER.json'
            self.m=read(self.root/'RUN_MANIFEST.json');self.limits=self.m['limits']
            self.schema=SCHEMA;self.authority='5b5de6f5d26ac90a6002c83ebcb60005acf06462'
            self.started=time.process_time()
            self._guard_at=0.;self._guard_value=None;self._guard_closeout=None
            self.check(read(self.path))
        def guard(self,*,closeout=False):
            binding(self.root)
            if not closeout and time.process_time()-self.started>=1170:
                raise TimeoutError('GEN1_AMD_STATIC_CPU_SAVING_MARGIN')
            self.check(read(self.path),limits=not closeout,allow_unknown=closeout)
    return StaticLedger(root)


def run(root):
    r=Path(root);m=validate(r);started=time.process_time()
    from scripts.pi0292.durable import write,OwnerLock
    from .spec import design_cards
    from .validation import common_home_preparation
    from scripts.pi0294.pv1.capture import qualify_counter
    a.need(not(r/'GEN1_STATIC_STARTED.json').exists(),'STATIC_ONCE_NO_REPLAY')
    lock=OwnerLock(r/'GEN1_STATIC_OWNER.lock');counter=ledger(r)
    os.sched_setaffinity(0,set(m['limits']['cpus']))
    resource.setrlimit(resource.RLIMIT_CPU,(1200,1200));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    write(r/'GEN1_STATIC_STARTED.json',dict(epoch=time.time(),pid=os.getpid(),
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        procstart=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
    out=dict(schema='GEN1_AMD_ACTUAL_STATIC24_V1',status='FAILED_PRESERVED',variants=[],homes=[],
        expected_images=24,actual_images=0,integrations=0,Teacher_attempts=0,models=0,
        common_home_selected=False,visual_generalisation_proven=False)
    try:
        # No native module is imported before all host/source/allocation gates.
        import mujoco
        from PIL import Image
        assets=read(r/'plans/AMD_ASSETS.private.json')['root'];groups=preview.specifications(read(r/preview.PACKET))
        original=mujoco.mj_step
        def prohibited(*args,**kwargs):raise PermissionError('GEN1_AMD_ZERO_INTEGRATION_NO_MJ_STEP')
        mujoco.mj_step=prohibited
        try:
            class Meter:
                root=r;plan={'cards':[design_cards(next(iter(groups.values()))['V0'])[0]]};inputs={'asset_root':assets}
                guard=staticmethod(counter.guard);start=staticmethod(lambda:counter.charge('engine_starts',1))
            qualify_counter(Meter())
            for i,(geometry,pair)in enumerate(sorted(groups.items())):
                baseline=None
                for visual,spec in sorted(pair.items()):
                    renderer=[]
                    def save_images(images):
                        from OpenGL.GL import glGetString,GL_VENDOR,GL_RENDERER,GL_VERSION
                        strings={name:(glGetString(enum)or b'').decode()for name,enum in
                            [('vendor',GL_VENDOR),('renderer',GL_RENDERER),('version',GL_VERSION)]}
                        a.need('Mesa'in strings['vendor']and 'llvmpipe'in strings['renderer'].lower(),
                            'ACTUAL_ORIGINAL_MESA_LLVMPIPE_NOT_HARDWARE')
                        renderer.append(strings)
                        saved={}
                        for camera,data in images.items():
                            rel=f'preview/{i:02d}_{visual}_{camera}.png';p=r/rel;p.parent.mkdir(exist_ok=True,mode=0o700)
                            a.need(not p.exists(),'PRESERVE_EVERY_ACTUAL_IMAGE');Image.fromarray(data).save(p);p.chmod(0o400)
                            saved[camera]=dict(relative=rel,sha256=sha(p),bytes=p.stat().st_size);out['actual_images']+=1
                        return saved
                    value=preview.inspect_task_pair(spec,assets,construct=lambda:counter.charge('engine_starts',1),
                        guard=counter.guard,save_images=save_images)
                    if baseline is None:baseline=value['first']
                    else:preview.compare_reset(baseline,value['first'],same_visual=False)
                    out['variants'].append(dict(geometry_id=geometry,visual=visual,renderer=renderer[0],**value))
            anchor=design_cards(next(iter(groups.values()))['V0'])[0]
            for home in common_home_preparation()['candidates']:
                card=dict(anchor,initial_tcp=home['initial_tcp'])
                try:
                    x=preview.inspect(card,assets,construct=lambda:counter.charge('engine_starts',1),guard=counter.guard)
                    y=preview.inspect(card,assets,construct=lambda:counter.charge('engine_starts',1),guard=counter.guard)
                    preview.compare_reset(x,y,same_visual=True)
                    out['homes'].append(dict(id=home['id'],status='FINITE_INITIAL_STATIC_ONLY',initial=x,duplicate=y,
                        approach_path='NOT_RUN',selected=False,dynamic_qualified=False))
                except ValueError as exc:out['homes'].append(dict(id=home['id'],status='INITIAL_IK_REJECT',error=str(exc),selected=False))
            a.need(out['actual_images']==24,'ALL_PREDECLARED_IMAGES')
            out['status']='COMPLETE_STATIC24_NOT_TEACHER_OR_POLICY_QUALIFICATION'
            out['initial_safety_and_FOV_pass']=all(v['first']['initial_safety_valid']and v['first']['field_of_view']['passed']for v in out['variants'])
        finally:mujoco.mj_step=original
        return out
    except BaseException as exc:out['error']=type(exc).__name__+': '+str(exc);raise
    finally:
        out['actual_constructor_attempts']=read(counter.path)['engine_starts'];out['Core_CPU_s']=time.process_time()-started
        counter.cpu(out['Core_CPU_s']);write(r/'GEN1_STATIC_RESULT.json',out);lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    args=parser.parse_args();print(json.dumps(run(args.root)))
