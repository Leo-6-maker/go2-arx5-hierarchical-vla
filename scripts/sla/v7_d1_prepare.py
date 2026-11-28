"""Bind one qualified red PREGRASP D1 card; retain the unqualified LIFT card unstarted."""
import argparse
from contextlib import closing
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil

from m38_gpu_preflight_v1 import pin,read,require,sha
from scripts.pi0292.durable import OwnerLock,write

PLAN='67d15fb8fad688a2bbd82e225111bb49ec34559b'
REVIEW='8e6ffa192d3e6c0adce7e2f89ca97e8ea20f461a'
IDS=('V7_D1_EARLY_TRAIN_15','V7_D1_EARLY_TRAIN_16')
OLD='v7-d0-followup.vNPlBk/binding.bAKrxj'


def build(site,stage):
    require(not stage.exists(),'Resolve the existing D1 preparation rather than regenerating seeds')
    old=site/OLD
    previous=read(pin(site/'v7-d0-followup.vNPlBk/PROSPECTIVE.private.json',
                      '3a719a4f644cf2b5513eb1b0b5dcaaff6c040cd5cdfad9ebcab1e11cdae3310d'))
    closed=read(pin(site/'V7-D0-EARLY-SEALED-02/RESULT.json','47455a46b49c09124c98cba7a9d5b910221ff19b0362d0ceac1e40b7beca7068'))
    replay=read(pin(site/'v7-d1-teacher-replay._fjo98ci/ACTUAL.json','4ea185c077832f560e2d94cde9773ef64f9304cdc748b46f7ca428d1f3f9215e'))
    require(closed['complete_reconciliation'] and len(closed['cases'])==2
            and all(row['state']=='CLOSED' and not row['recovery_admitted'] for row in closed['cases']),
            'Retain the whole closed earlier cohort')
    red,= [r for r in replay['cases'] if r['stage']=='PREGRASP']
    require(red['first_warning']['preview_prepared'] and red['first_warning']['static_prefix']['passed_knots']==11
            and len(red['invalid_boundary_fixtures_rejected'])==4,'Requires the actual safe-boundary first-prefix replay')
    require(all(r['first_warning'] is None for r in replay['cases'] if r['stage']=='LIFT'),
            'The one-second stall rule status changed; review its independent eligibility')
    used_resets=set();used_noise=set()
    protected=read(pin(site/'v63-place-core-code-01/PROTECTED_POPULATION.json',
                       'a1dfa32f329cb91d474e20488cbeeafa3dc41858f32dacd6cd593dbed61bbb26'))
    used_resets.update(row['reset'] for row in protected['clusters'])
    for i in range(1,6):
        for row in read(site/f'RECOVERY-TRAIN-ROSTER-{i:02d}.json')['cases']:
            used_resets.add(row['reset']);used_noise.update((row['policy_seed']+q)%2**31 for q in range(180))
    for wave,ids in (('01',('V7_D0_EARLY_TRAIN_11','V7_D0_EARLY_TRAIN_12')),
                     ('02',('V7_D0_EARLY_TRAIN_13','V7_D0_EARLY_TRAIN_14'))):
        for name in ids:
            case=site/f'V7-D0-EARLY-NATIVE-{wave}'/name
            lease=read(case/'control/LEASE.json')
            spec=read(pin(case/'EXECUTION_SPEC.json',lease['spec_sha256']))
            require(lease['status']=='CLOSED','A preceding native owner remains active')
            used_resets.add(spec['mobile_condition']['reset'])
            used_noise.update((spec['comparison_policy_seed']+q)%2**31 for q in range(180))
    roster=copy.deepcopy(previous)
    sources=read(old/'overlay/V7-D0-NATIVE-ROSTER.private.json')['teacher_sources']
    cases=[];teachers={}
    for source_name,name,enabled in (('V7_D0_EARLY_TRAIN_14',IDS[0],True),('V7_D0_EARLY_TRAIN_13',IDS[1],False)):
        card=copy.deepcopy(next(r for r in previous['cases'] if r['id']==source_name))
        key=lambda purpose:int.from_bytes(hashlib.sha256(f'{REVIEW}|{name}|{purpose}'.encode()).digest()[:4],'little')%2**31
        reset,noise=key('reset'),key('noise')
        require(reset not in used_resets and all((noise+q)%2**31 not in used_noise for q in range(180)),
                'Fresh D1 reset or noise overlaps actual previous attempts')
        card.update(id=name,reset=reset,policy_seed=noise,native_enabled=enabled,
                    trigger='PRECONTACT_20MM' if enabled else 'LIFT_STALL_1S',
                    not_started_reason=None if enabled else 'NO_ELIGIBLE_ONE_SECOND_STALL_IN_ACTUAL_CLOSED_REPLAY')
        cases.append(card);teachers[name]=copy.deepcopy(sources[source_name])
    roster.update(cases=cases,past_attempts=14,past_admissions=4,max_new_requests=1,max_ticks=2*50703,
        max_queries=360,allowed_trigger='PRECONTACT_20MM_OR_ORIGINAL_UNINSTALLED_G7',
        reviewed_plan=REVIEW,actual_starts=0,native_dispatch_ids=[IDS[0]],
        conditional_unstarted_ids=[IDS[1]],native_dispatch_max_ticks=50703,
        precontact_replay=dict(path=str(site/'v7-d1-teacher-replay._fjo98ci/ACTUAL.json'),sha256=sha(site/'v7-d1-teacher-replay._fjo98ci/ACTUAL.json')))
    stage.mkdir(mode=0o700)
    write(stage/'PROSPECTIVE.private.json',roster)
    write(stage/'TEACHER-SOURCES.private.json',teachers)
    print(stage,sha(stage/'PROSPECTIVE.private.json'),flush=True)


def noise(site,stage,worker):
    import jax
    import numpy as np
    spec=read(pin(worker,'361709e35c2bf427d5644e7af37a925014ab52861c075d4782749afd371d6bfc'))
    roster=read(stage/'PROSPECTIVE.private.json')
    old=site/OLD
    bank=read(old/'CUDA-NOISE.private.json')
    source=read(pin(spec['sample_noise']['path'],spec['sample_noise']['sha256']))['tensors']
    require(jax.default_backend()=='gpu' and os.environ['CUDA_VISIBLE_DEVICES']==spec['GPU_uuid'],
            'Use the actual original CUDA sampling backend')
    with closing(OwnerLock(Path(spec['neural_owner_lock']))):
        draw=lambda seed:hashlib.sha256(np.asarray(jax.random.normal(jax.random.PRNGKey(seed),(1,40,32),dtype=np.float32)).tobytes()).hexdigest()
        for row in (source[0],source[len(source)//2],source[-1]):require(draw(row['noise_id'])==row['gpu_sha256'],'CUDA sentinel changed')
        used={row['gpu_sha256'] for row in source+bank['tensors']}
        from v7_d0_noise import OLD_NOISE,FIRST_NOISE
        for path,expected in (*OLD_NOISE,FIRST_NOISE):
            used.update(r['gpu_sha256'] for r in read(pin(site/path,expected))['tensors'])
        result=[]
        for card in roster['cases']:
            for query in range(180):
                seed=(card['policy_seed']+query)%2**31;value=draw(seed)
                require(value not in used,'D1 actual noise overlaps a prior actual bank')
                used.add(value);result.append(dict(case=card['id'],query=query,noise_id=seed,gpu_sha256=value))
        write(stage/'CUDA-NOISE.private.json',dict(plan=PLAN,split='TRAIN',roster_sha256=sha(stage/'PROSPECTIVE.private.json'),
            worker_spec_sha256=sha(worker),prior_noise_manifest_sha256=spec['sample_noise']['sha256'],
            prior_recovery=[sha(old/'CUDA-NOISE.private.json')],matched_archived_cuda_sentinels=3,tensors=result,
            model_forwards=0,native_integrations=0,optimizer_updates=0))
    print('D1_NOISE',sha(stage/'CUDA-NOISE.private.json'),flush=True)


def finalize(site,stage,code):
    old=site/OLD;output=stage/'overlay'
    require(not output.exists(),'Preserve the previous D1 overlay')
    prospective=stage/'PROSPECTIVE.private.json';roster=read(prospective)
    noise_path=stage/'CUDA-NOISE.private.json';sample=read(noise_path)
    require(sample['roster_sha256']==sha(prospective),'CUDA bank no longer binds the prospective cards')
    shutil.copytree(old/'overlay',output,symlinks=True)
    scripts=output/'scripts/sla'
    previous=read(old/'overlay/V7-D0-NATIVE-ROSTER.private.json')
    runtime=dict(previous,cases=roster['cases'],prospective=dict(path=str(prospective),sha256=sha(prospective)),
        teacher_sources=read(stage/'TEACHER-SOURCES.private.json'),handoff_protocol='PRECONTACT_OR_LIFT_STALL_V1',
        realized_noise=dict(path=str(noise_path),sha256=sha(noise_path)),reviewed_plan=REVIEW,
        precontact_replay=roster['precontact_replay'],past_attempts=14,max_new_requests=1,
        dispatch_ids=roster['native_dispatch_ids'])
    runtime_path=output/'V7-D1-NATIVE-ROSTER.private.json';write(runtime_path,runtime)
    for name in ('v7_precontact_teacher.py',):shutil.copyfile(code/name,scripts/name)
    content=(old/'v7_d0_native.py').read_text()
    changes={
        'PROSPECTIVE = "4e10cfdd865ade7fe6e360664d2601530ba5737e9160e14d1c9abdde4edfe761"':f'PROSPECTIVE = "{sha(prospective)}"',
        'FOLLOWUP_PROSPECTIVE = "3a719a4f644cf2b5513eb1b0b5dcaaff6c040cd5cdfad9ebcab1e11cdae3310d"':f'FOLLOWUP_PROSPECTIVE = "{sha(prospective)}"',
        'NOISE = "1073abde8dae989e6534fe298b30063a6c770767808add685ce565ea4e7f48e6"':f'NOISE = "{sha(noise_path)}"',
        'IDS = ("V7_D0_EARLY_TRAIN_11", "V7_D0_EARLY_TRAIN_12")':f'IDS = {IDS!r}',
        'FOLLOWUP_IDS = ("V7_D0_EARLY_TRAIN_13", "V7_D0_EARLY_TRAIN_14")':f'FOLLOWUP_IDS = {IDS!r}',
        'STAGES = ("LIFT", "PREGRASP")':'STAGES = ("PREGRASP", "LIFT")',
        'V7-D0-NATIVE-ROSTER.private.json':'V7-D1-NATIVE-ROSTER.private.json',
        'EARLY_UNINSTALLED_G7_ONLY':'PRECONTACT_OR_LIFT_STALL_V1'}
    for before,after in changes.items():
        require(before in content,'D0 binding seam changed: '+before[:70]);content=content.replace(before,after)
    (scripts/'v7_d1_native.py').write_text(content)
    for name in ('v64_recovery_launcher.py','v64_recovery_cohorts.py'):
        p=scripts/name;require(not p.is_symlink(),'Only a fresh private overlay can be edited')
        p.write_text(p.read_text().replace('from v7_d0_native import','from v7_d1_native import'))
    p=scripts/'v64_m38_recovery_wave5.py';text=p.read_text()
    before='policy_seed=c["policy_seed"],instruction=c["task"],checkpoint=roster["checkpoint_sha256"]) for i,c in enumerate(roster["cases"])]'
    require(text.count(before)==1,'Native dispatch seam changed')
    p.write_text(text.replace(before,'policy_seed=c["policy_seed"],instruction=c["task"],enabled=c["native_enabled"],checkpoint=roster["checkpoint_sha256"]) for i,c in enumerate(roster["cases"])]'))
    p=scripts/'v64_recovery_native.py';text=p.read_text()
    require(text.count('        if self.early_only:\n            return None\n')==1,'Post-consumption seam changed')
    text=text.replace('EARLY_UNINSTALLED_G7_ONLY','PRECONTACT_OR_LIFT_STALL_V1')
    text=text.replace('        if self.early_only:\n            return None\n',
        '        if self.early_only:\n            from v7_d1_live_hook import maybe_prepare\n            return maybe_prepare(self, adapter, score, initial, query, row)\n')
    before='if prepared["event"]["trigger"] == "EARLY_UNINSTALLED_G7":'
    require(text.count(before)==1,'The qualified contiguous sequence seam changed')
    text=text.replace(before,'if prepared["event"]["trigger"] in ("EARLY_UNINSTALLED_G7", "PRECONTACT_20MM", "LIFT_STALL_1S"):')
    p.write_text(text)
    shutil.copyfile(code/'v7_d1_live_hook.py',scripts/'v7_d1_live_hook.py')
    for path in (p,scripts/'v7_d1_native.py',scripts/'v7_d1_live_hook.py',scripts/'v7_precontact_teacher.py'):
        compile(path.read_text(),str(path),'exec')
    write(stage/'OVERLAY-ACTUAL.json',dict(roster_sha256=sha(runtime_path),source_overlay=str(old/'overlay'),
        actual_new_native_requests=0,active_dispatch_ids=roster['native_dispatch_ids'],
        unqualified_unstarted_ids=roster['conditional_unstarted_ids'],
        changed_files={str(p.relative_to(output)):sha(p) for p in scripts.glob('v7_d1*.py')},
        WBC_changes=False,safety_changes=False))
    print('D1_RUNTIME',runtime_path,sha(runtime_path),flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('mode',choices=('build','noise','finalize'))
    cli.add_argument('site',type=Path);cli.add_argument('stage',type=Path);cli.add_argument('extra',type=Path,nargs='?')
    a=cli.parse_args()
    if a.mode=='build':build(a.site,a.stage)
    elif a.mode=='noise':noise(a.site,a.stage,a.extra)
    else:finalize(a.site,a.stage,a.extra)
