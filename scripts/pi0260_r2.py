"""PI0260-R2 metadata overlay and admission. No IK, rendering or model imports."""
import copy
import hashlib
import json
from pathlib import Path
import sys

OLD=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0260')
ROOT=OLD/'R2'
AUTHORITY='c5070dc1b6eaff9d3070c92e3a0248ff4c0a83be'
OLD_REG='d4880cc2cf5e9ef2d20913ea3821eac50323e505065dba176b1a5d04c75eaede'
H0='3e57b86adfb06de239f20f36e44f4babd3a250de088f3f3377665edb13b30fa8'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    assert not p.exists(),('IMMUTABLE_OUTPUT_EXISTS',str(p))
    p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')

def split_kind(s):
    if s=='TEST' or s.startswith(('TEST_','MT-TEST_','MT_TEST_')):return 'TEST'
    aliases={'PILOT':'DEV','DEVELOPMENT':'DEV','CALIBRATION':'CALIBRATION','TRAIN':'TRAIN',
             'KIN-VAL':'VAL','MT-VAL':'VAL','UNUSED_RESERVED':'UNUSED'}
    assert s in aliases,('UNRESOLVED_SPLIT_ALIAS',s)
    return aliases[s]

def allocation(i):
    assert 0<=i<204
    return 'DEVELOPMENT' if i<20 else 'TRAIN' if 20<=i<120 or 160<=i<192 else 'KIN-VAL'

def geometric_key(f,r):
    center=f.get('center_world')
    if center is None:center=[a+f['factors'][k] for a,k in zip(r['geometry']['nominal_center_world'],('x_shift','y_shift','z_shift'))]
    return tuple(round(float(x),10) for x in [*center,f['factors']['radius']])

def full_identity(f,r,config,defaults):
    """Numeric scene AND initial state, excluding color and names. Keep cue revision separately."""
    center=geometric_key(f,r)[:3];g=r['geometry']
    root=f.get('initial_root_position',r.get('root_pose',config['nominal_base_world'])[:3])
    quat=f.get('initial_root_quaternion',r.get('root_pose',[0,0,0,0,0,0,1])[3:])
    q=f.get('initial_arm_joints',r.get('arm_default',config['default_q']))
    add=lambda v:[round(float(a+b),10) for a,b in zip(center,v)]
    return dict(center=list(center),radius=geometric_key(f,r)[3],
        targets=[add(v) for v in g['target_offsets_world_m']],yaw=g['target_yaw_rad'],
        obstacle=add(g['obstacle_offset_world_m']),obstacle_half_extents=g['obstacle_half_extents_m'],
        goal=add(g['route_goal_offset_world_m']),root=root,root_quaternion=quat,arm=q,
        all_default_joints=r.get('source_default_joints',defaults),
        initial_state_scope='DECLARED_NOMINAL_NOT_STOCHASTIC_PHYSICAL_RESET')

def tier(train,val,integrity=True):
    if not integrity:return 'DENY_DATA_INTEGRITY'
    if train<16 or val<6:return 'R2_ACTUAL_DATASET_SHORTFALL'
    assert train<=40 and val<=10
    return 'STANDARD_40_10' if (train,val)==(40,10) else f'COMPACT_{train}_{val}'

def route(expected,actual,h0=True,registry=True,owner=False,job=False,saved=False,**unused):
    if expected!=actual:return 'DENY_CHANGED_INPUT'
    if owner:return 'DENY_DUPLICATE_OWNER'
    if job:return 'ADOPT_EXISTING_JOB'
    if not h0 or not registry:return 'DENY_DATA_INTEGRITY'
    return 'REUSE_CANONICAL_RESULT' if saved else 'R2_PRODUCTION_ADMITTED_FROM_EXISTING_DIAGNOSTIC_EVIDENCE'

def post_codec(selected,valid,statistics_fit_count=1):
    assert statistics_fit_count==1 and set(valid)<=set(selected),'no refill or statistics refit'
    return [x for x in selected if valid.get(x,False)]

def conditioning(packet):
    return {k:packet[k] for k in ('body_rgb','wrist_rgb','state_v1','language')}

def val_windows(sequences):
    """Freeze phase slots; actual qualifying indices must be bound before model calls."""
    assert 12<=len(sequences)<=20 and len(set(sequences))==len(sequences)
    return [dict(sequence=sequences[i%len(sequences)],quartile=(i%len(sequences)+i//len(sequences))%4,
                 visit=i//len(sequences),correlated_repeat=i>=len(sequences)) for i in range(16)]

def test():
    cases=[]
    assert route('x','x',old_dev='FAIL',helper=None).startswith('R2_PRODUCTION');cases.append('old_DEV_FAIL_preserved_H0_overlay_admits')
    assert tier(16,6)=='COMPACT_16_6' and tier(16,6,False)=='DENY_DATA_INTEGRITY';cases.append('both_arms_minimum_and_integrity')
    assert tier(15,10)==tier(40,5)=='R2_ACTUAL_DATASET_SHORTFALL';cases.append('below_minimum_denied')
    assert tier(27,8)=='COMPACT_27_8' and tier(40,10)=='STANDARD_40_10';cases.append('truthful_variable_tier')
    assert all(split_kind(s)=='TEST' for s in ['TEST','TEST_COMPOSITION','TEST_GEOMETRY','MT-TEST_COMPOSITION','MT-TEST_GEOMETRY','MT_TEST_GEOMETRY'])
    try:split_kind('UNKNOWN')
    except AssertionError:pass
    else:raise AssertionError('unknown split admitted')
    a=dict(center_world=[.1,.2,.3],factors=dict(radius=.04,colors=['red','blue']))
    b=copy.deepcopy(a);b['factors']['colors']=['green','red'];b['family_id']='alias'
    assert geometric_key(a,{})==geometric_key(b,{})
    b['center_world'][0]+=.01;assert geometric_key(a,{})!=geometric_key(b,{})
    cases.append('test_aliases_and_independent_numeric_keys')
    assert [i for i in range(204) if allocation(i)=='TRAIN']==list(range(20,120))+list(range(160,192))
    assert [i for i in range(204) if allocation(i)=='KIN-VAL']==list(range(120,160))+list(range(192,204));cases.append('fixed_reserved_slots_no_DEV_or_backfill')
    assert {(i+c)%2+2 for i in range(192,204) for c in (0,1)}=={2,3}
    assert set(conditioning(dict(body_rgb=1,wrist_rgb=2,state_v1=3,language=4,task=5,time=6)))=={'body_rgb','wrist_rgb','state_v1','language'}
    cases.append('VAL_language_and_conditioning_boundary')
    assert post_codec(['a','b','c'],dict(a=True,b=False,c=False))==['a']
    for valid,fit in [(dict(d=True),1),(dict(a=True),2)]:
        try:post_codec(['a','b'],valid,fit)
        except AssertionError:pass
        else:raise AssertionError('refill/refit admitted')
    cases.append('failure_preserved_no_codec_refill_or_refit')
    for n in range(6,11):
        rows=val_windows([f'f{i//2}-c{i%2}' for i in range(2*n)])
        assert len(rows)==16 and len({(r['sequence'],r['quartile']) for r in rows})==16
        assert sum(r['correlated_repeat'] for r in rows)==max(0,16-2*n)
        assert 4*2*(2*(32+len(rows)+12+12+6)+64)==1760
    cases.append('variable_cohort_1760_slots_honest_correlation')
    assert route('a','a',saved=True)=='REUSE_CANONICAL_RESULT' and route('a','a',job=True)=='ADOPT_EXISTING_JOB'
    assert route('a','b',saved=True)=='DENY_CHANGED_INPUT' and route('a','a',owner=True)=='DENY_DUPLICATE_OWNER'
    cases.append('reuse_hash_owner_helper_routing')
    return dict(status='PASS',cases=cases,model_queries=0,scope='CPU routing; not actual H1/codec/model evidence')

def freeze():
    assert sha(OLD/'design/registry.json')==OLD_REG and sha(OLD/'state/H0_ACCEPTED.json')==H0
    old=read(OLD/'design/registry.json');reg=copy.deepcopy(old);assert len(reg['families'])==204
    before=read(OLD/'state/design_freeze.json');assert before['selected_rig']==0
    p55=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0255')
    config=read(p55/'evidence/resolved_native_config_cpu.json')
    assert sha(p55/'evidence/resolved_native_config_cpu.json')==old['source_hashes']['resolved_native_config_cpu.json']
    sources=[p55/'design/task_registry_v1.json',OLD.parent/'PI0258/design/registry.json',OLD.parent/'PI0259/design/registry.json',OLD/'design/registry.json']
    actual55={read_row['episode']['family_id'] for read_row in map(json.loads,(p55/'attempt_results.jsonl').read_text().splitlines())}
    actual={}
    # Only file paths/started metadata are inspected for exposure; no TEST payloads.
    for stage in ['PI0258','PI0259','PI0260']:
        base=OLD.parent/stage
        candidates=list((base/'calibration').glob('*/started.json'))+list((base/'proposals').glob('*/*/started.json'))
        if stage=='PI0258':candidates+=list((base/'development').glob('*/*-geometry.npz'))
        actual[stage]=sorted({p.parent.name.rsplit('-M',1)[0] for p in candidates})
    rows=[];oldunion=set();corrected=set();protected=set();identities={}
    for path in sources:
        prior=read(path);stage=path.parents[1].name
        for f in prior['families']+prior.get('calibration_families',[]):
            kind=split_kind(f['split']);key=geometric_key(f,prior)
            exposed=(f['family_id'] in (actual55 if stage=='PI0255' else actual[stage]) or kind=='CALIBRATION')
            if stage=='PI0260' and f.get('proposal_index',204)<20:exposed=True
            if stage=='PI0258' and f in prior['families'][:6]:
                assert f['family_id'] in actual[stage],('missing actual prior DEV metadata',f['family_id'])
            old_protected=f['split'].startswith('TEST')
            if stage!='PI0260' and (old_protected or exposed):oldunion.add(key)
            if kind=='TEST':protected.add(key)
            if kind=='TEST' or exposed:
                corrected.add(key)
                ident=full_identity(f,prior,config,old['source_default_joints'])
                rows.append(dict(source=str(path),source_sha256=sha(path),family=f['family_id'],split=f['split'],
                    reason='TEST' if kind=='TEST' else 'PRIOR_EXPOSURE',numeric_key=list(key),full_identity=ident))
                identities.setdefault(key,[]).append(ident)
    original_excluded={tuple(x['geometry_key']) for x in read(OLD/'design/exclusions.json')['sources']}
    assert oldunion==original_excluded,('old exclusion reconciliation differs',oldunion^original_excluded)
    protected_delta=protected-original_excluded
    bindings=[];skips=[];contamination=[]
    for f in reg['families']:
        i=f['proposal_index'];assert f['family_id']==f'pi0260-family-{i:03d}'
        previous=f['split'];f['split']=allocation(i);k=geometric_key(f,reg)
        ident=full_identity(f,reg,config,old['source_default_joints'])
        f['r2_full_identity']=ident
        # Conservative scene exclusion also protects stochastic old resets / changed cue sizes.
        if k in corrected:
            f['exclusion']=[dict(family=x['family'],source=x['source'],reason=x['reason'],
                full_nominal_identity_equal=x['full_identity']==ident) for x in rows if tuple(x['numeric_key'])==k]
            if i>=20:skips.append(f['family_id'])
        if i<20 and k in protected:contamination.append(f['family_id'])
        for s in [s for s in reg['sequences'] if s['family_id']==f['family_id']]:
            before_s=copy.deepcopy(s);s['split']=f['split'];s['execute']=i>=20 and not f.get('exclusion')
            if i>=192:s['instruction']=s['paraphrases'][(i+s['command'])%2+2]
            bindings.append(dict(before=before_s,after=copy.deepcopy(s)))
        assert f['factors']==old['families'][i]['factors'] and f['center_world']==old['families'][i]['center_world']
    assert reg['renderer']==old['renderer'] and reg['solver']==old['solver'] and reg['geometry']==old['geometry']
    reg.update(schema='PI0260_R2_MOTION2_V1',authority=AUTHORITY,parent_registry_sha256=OLD_REG)
    reg['budgets'].update(proposal_families=204,proposal_sequences=816,technical_sequence_reserve=24,active_windows=90000,
                          calibration_image_pairs=84,additional_calibration_image_pairs=0)
    oldcount=read(OLD/'state/data_progress.json')['counters']
    assert (oldcount['new_kinematic_sequence_attempts'],oldcount['production_image_pairs'],oldcount['active_raw_windows'])==(80,2134,3520)
    report=dict(status='PASS',sources=rows,actual_executed_ids=actual,old_union=sorted(oldunion),corrected_union=sorted(corrected),
        new_protected_numeric_keys=sorted(protected_delta),all_delta_including_oldDEV=sorted(corrected-oldunion),
        excluded_prospective_slots=skips,old_DEV_protected_overlap=contamination,
        comparison='full numeric scene/root/DOF identity recorded; conservative scene-only exclusion also retained; color/ID never distinguish geometry',
        TEST_payload_reads=0,new_IK=0,new_render=0)
    write(ROOT/'design/exclusions.json',report);write(ROOT/'design/row_bindings.json',bindings)
    write(ROOT/'design/registry.json',reg)
    frozen=dict(before,registry_sha256=sha(ROOT/'design/registry.json'),parent_registry_sha256=OLD_REG,r2_authority=AUTHORITY)
    write(ROOT/'state/design_freeze.json',frozen)
    baseline={'design/registry.json':OLD_REG,'state/H0_ACCEPTED.json':H0,
              'state/STOP_SCIENCE.json':sha(OLD/'state/STOP_SCIENCE.json'),
              'state/selection_DEVELOPMENT.json':sha(OLD/'state/selection_DEVELOPMENT.json')}
    write(ROOT/'evidence/route_cpu.json',test())
    bound={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'design/registry.json',ROOT/'design/exclusions.json',ROOT/'design/row_bindings.json',ROOT/'state/design_freeze.json',ROOT/'evidence/route_cpu.json']}
    bound.update({str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'runtime').glob('*.py')})
    write(ROOT/'design/overlay.json',dict(authority=AUTHORITY,workflow=before['workflow'],parent_science=old['authority'],
        owner=before['owner'],old_root=str(OLD),old_bindings=baseline,bindings=bound,baseline_counters=oldcount,
        old_DEV_status=read(OLD/'state/STOP_SCIENCE.json'),status=route('a','a'),
        TRAIN_indices=list(range(20,120))+list(range(160,192)),VAL_indices=list(range(120,160))+list(range(192,204))))
    print(json.dumps(dict(status=route('a','a'),registry_sha256=sha(ROOT/'design/registry.json'),
        overlay_sha256=sha(ROOT/'design/overlay.json'),protected_delta=len(protected_delta),skipped=skips,contamination=contamination)))

def admission(root=ROOT):
    card=read(root/'design/overlay.json');assert card['authority']==AUTHORITY
    for rel,h in card['old_bindings'].items():assert sha(OLD/rel)==h,('OLD_CHANGED',rel)
    for rel,h in card['bindings'].items():assert sha(root/rel)==h,('R2_CHANGED',rel)
    assert read(root/'evidence/route_cpu.json')['status']=='PASS'
    assert read(root/'design/exclusions.json')['status']=='PASS'
    reg=read(root/'design/registry.json');reg['renderer']['selected_rig']=0
    return reg

if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='test':print(json.dumps(test(),indent=2))
    elif sys.argv[1]=='check':print(json.dumps(dict(status='PASS',families=len(admission()['families']))))
