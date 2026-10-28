"""Fixed PI0264 source bank, eight-pass language schedule; no model calls."""
import argparse
import copy
import hashlib
from collections import Counter,defaultdict
from pathlib import Path

from pi0263_plan import read,sha,write,NONLANGUAGE

AUTHORITY='8245e324b21b25f643eba26df6b2f189981bb40e'
SOURCE_SHA='602dc213ca91f2dcd5099bd97a45587fe5fe1e0d205243596d48e07c5f662617'
TEMPLATES={
 'T0':'Approach the {color} target and align the tool with its axes.',
 'T1':'Bring the tool to the {color} marker with the pose shown by its axes.',
 'T2':'Position the tool at the {color} target and match the orientation of its displayed axes.',
 'T3':"Guide the tool to the {color} marker and align it with the marker's axes.",
 'T4':'The {color} target is the destination for the tool; match the orientation shown by its axes.',
 'T5':"Match the tool's position and orientation to the {color} marker and its displayed axes.",
 'T6':'Bring the tool into alignment with the displayed axes at the {color} target.',
 'T7':"Use the {color} marker as the tool's target position and use its axes as the target orientation.",
}
CHALLENGE={
 'Q0':'Place the tool at the {color} marker, oriented to agree with the axes shown there.',
 'Q1':'The tool should go to the {color} target with its orientation matching the displayed coordinate frame.',
 'Q2':"Take the tool to the {color} marker and give it the orientation represented by the marker's axes.",
 'Q3':'Make the tool coincide with the {color} target in position and with its shown frame in orientation.',
}


def seed_for(unit):
    return int.from_bytes(hashlib.sha256(f'PI0264:26401:{unit}'.encode()).digest()[:8],'big')%2**31


def phases(unit):
    if unit['slot']=='PAIR_SLOT':
        assert all(c['index']==0 and c['time_s']==0 and 'quartile' not in c for c in unit['selected'])
        return {'COMMON_START'}
    assert unit['slot']=='REPLAY_SLOT' and all(c['quartile'] in range(4) for c in unit['selected'])
    return {str(c['quartile']) for c in unit['selected']}


def select_bank(old):
    assert len(old['units'])==2000
    bank=[]
    for i in range(0,2000,2):
        a,b=old['units'][i:i+2]
        assert (a['visit'],b['visit'])==(0,1)
        assert all(a[k]==b[k] for k in ('original_unit','task','slot','family','selected','ordinary'))
        bank.append({k:v for k,v in a.items() if k not in ('unit','visit','order','x_template')})
    assert len({u['original_unit'] for u in bank})==1000
    strata=defaultdict(list)
    for u in bank:strata[u['task'],u['slot']].append(u)
    assert set(strata)=={(t,s) for t in ('M1','M3') for s in ('PAIR_SLOT','REPLAY_SLOT')}
    chosen=[]
    for rows in strata.values():
        rows.sort(key=lambda u:u['original_unit'])
        first={}
        for u in rows:first.setdefault(u['family'],u)
        assert len(first)==30
        keep={u['original_unit'] for u in first.values()}
        represented=set().union(*(phases(u) for u in rows))
        covered=set().union(*(phases(u) for u in first.values()))
        for phase in sorted(represented):
            if phase not in covered:
                u=next(u for u in rows if u['original_unit'] not in keep and phase in phases(u))
                keep.add(u['original_unit']);covered.update(phases(u))
        assert covered==represented and len(keep)<=64
        for u in rows:
            if len(keep)==64:break
            keep.add(u['original_unit'])
        chosen.extend(u for u in rows if u['original_unit'] in keep)
    return sorted(chosen,key=lambda u:u['original_unit'])


def language(arm,unit,sequence,color):
    assert arm in ('B','L')
    if unit['task']=='M3':return sequence['instruction'],'ORIGINAL_M3'
    assert sequence['instruction'] in [TEMPLATES[t].format(color=color) for t in ('T0','T1')]
    template=unit['b_template'] if arm=='B' else unit['l_template']
    assert template in TEMPLATES
    return TEMPLATES[template].format(color=color),template


def validate(plan):
    units=plan['units'];bank=plan['source_bank']
    assert len(bank)==256 and len(units)==2048
    assert Counter((u['task'],u['slot']) for u in bank)=={(t,s):64 for t in ('M1','M3') for s in ('PAIR_SLOT','REPLAY_SLOT')}
    for task in ('M1','M3'):
        for slot in ('PAIR_SLOT','REPLAY_SLOT'):
            assert len({u['family'] for u in bank if (u['task'],u['slot'])==(task,slot)})==30
    for i,u in enumerate(units):
        visit,index=divmod(i,256);source=bank[index]
        assert u['unit']==i+1 and u['visit']==visit and u['bank_index']==index
        assert u['order']==([0,1] if visit%2==0 else [1,0])
        assert all(u[k]==source[k] for k in source)
        if u['task']=='M1':
            offset=int(hashlib.sha256(f"PI0264:26401:text-order:{u['original_unit']}".encode()).hexdigest(),16)%8
            assert u['text_offset']==offset and u['l_template']==f'T{(visit+offset)%8}' and u['b_template']==f'T{(visit+offset)%2}'
        else:assert u['b_template']==u['l_template']=='ORIGINAL_M3'
    assert Counter(u['l_template'] for u in units if u['task']=='M1')=={f'T{i}':128 for i in range(8)}
    assert Counter(u['b_template'] for u in units if u['task']=='M1')=={'T0':512,'T1':512}


def freeze(source,output):
    assert sha(source)==SOURCE_SHA
    old=read(source);bank=select_bank(old);units=[]
    for visit in range(8):
        for index,source_unit in enumerate(bank):
            offset=int(hashlib.sha256(f"PI0264:26401:text-order:{source_unit['original_unit']}".encode()).hexdigest(),16)%8
            m1=source_unit['task']=='M1'
            units.append(dict(source_unit,unit=len(units)+1,visit=visit,bank_index=index,
                order=[0,1] if visit%2==0 else [1,0],text_offset=offset,
                b_template=f'T{(visit+offset)%2}' if m1 else 'ORIGINAL_M3',
                l_template=f'T{(visit+offset)%8}' if m1 else 'ORIGINAL_M3'))
    plan=dict(schema='PI0264_LANGUAGE_DIVERSITY_PLAN_V1',authority_commit=AUTHORITY,source_plan_sha256=SOURCE_SHA,
        seed=26401,namespace='PI0264:26401',source_bank=bank,units=units,common_start_bank=old['common_start_bank'],
        templates=TEMPLATES,challenge=CHALLENGE,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')
    validate(plan);write(output,plan)
    family=Counter((u['task'],u['slot'],u['family']) for u in bank)
    phase=Counter((u['task'],u['slot'],p) for u in bank for p in phases(u))
    observed={(c['frame_sha256'],c['sequence'],c['index']) for u in bank for c in u['selected']}
    write(Path(output).with_name('INHERITED_EXPOSURE.json'),dict(base_units=256,visits=8,optimizer_units=2048,
        examples_per_arm=4096,unique_source_observation_label_records=len(observed),
        unique_frames=len({c['frame_sha256'] for u in bank for c in u['selected']}),
        phase_definition='Existing selected-record quartile tags for REPLAY_SLOT; actual bank index0/time0 for PAIR_SLOT COMMON_START.',
        family=[dict(task=t,slot=s,family=f,base_units=n) for (t,s,f),n in sorted(family.items())],
        phase=[dict(task=t,slot=s,phase=p,base_units_containing_phase=n) for (t,s,p),n in sorted(phase.items())],forced_uniformity=False))
    # Exercise actual validation on forbidden visit/template alterations before any model call.
    for changed in (units[1:],units[:-1]+[units[-2]]):
        try:validate(dict(plan,units=changed))
        except AssertionError:pass
        else:raise AssertionError('BAD_VISIT_ACCEPTED')
    bad=copy.deepcopy(units);m1=next(u for u in bad if u['task']=='M1');m1['l_template']='Q0'
    try:validate(dict(plan,units=bad))
    except AssertionError:pass
    else:raise AssertionError('Q_SUPERVISION_ACCEPTED')
    print('PASS_PI0264_FIXED_256_BANK_8_PASS_PLAN',sha(output))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();freeze(args.old,args.output)
