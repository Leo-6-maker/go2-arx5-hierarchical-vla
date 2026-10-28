"""Fixed old/new matched banks and eight-pass schedule; no model imports."""
import argparse
import hashlib
import itertools
import json
from collections import Counter,defaultdict
from pathlib import Path
from pi0264_plan import TEMPLATES,NONLANGUAGE
from pi0263_plan import read,sha,write


def seed_for(update):
    visit,index=divmod(update-1,512)
    assert 0<=visit<8 and 0<=index<512
    return int.from_bytes(hashlib.sha256(f'PI0266:TRAIN:26611:{index+1}:{visit}'.encode()).digest()[:8],'big')%2**31


def family_order(rows):
    groups=defaultdict(list)
    for row in sorted(rows,key=lambda r:r['original_unit']):groups[row['family']].append(row)
    return [r for group in itertools.zip_longest(*(groups[k] for k in sorted(groups))) for r in group if r is not None]


def phase_order(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['bucket']].append(row)
    return [r for group in itertools.zip_longest(*(groups[k] for k in sorted(groups))) for r in group if r is not None]


def language(unit,color):
    return (TEMPLATES[unit['template']].format(color=color),unit['template']) if unit['task']=='M1' else (None,'ORIGINAL_M3')


def validate(plan):
    banks=plan['banks'];units=plan['units']
    assert len(units)==4096 and all(len(banks[a])==512 for a in ('R','F'))
    assert all(banks['R'][i]==banks['F'][i] for i in range(0,512,2))
    for arm in ('R','F'):
        assert Counter((b['task'],b['slot']) for b in banks[arm])=={(t,s):128 for t in ('M1','M3') for s in ('COMMON','TRANSFER')}
        assert Counter(b['source'] for b in banks[arm])==({'OLD_TRAIN':512} if arm=='R' else {'OLD_TRAIN':256,'ACTUAL_FEEDBACK':256})
    for i,u in enumerate(units):
        visit,index=divmod(i,512)
        assert (u['unit'],u['visit'],u['bank_index'],u['order'])==(i+1,visit,index,[0,1] if visit%2==0 else [1,0])
        assert u['task']==banks['R'][index]['task']==banks['F'][index]['task']
        assert u['template']==('T'+str((visit+u['text_offset'])%8) if u['task']=='M1' else 'ORIGINAL_M3')
        assert u['seed']==seed_for(i+1)
    for index in range(512):
        if banks['R'][index]['task']=='M1':assert {units[p*512+index]['template'] for p in range(8)}==set(TEMPLATES)
    assert {b['source'] for b in banks['F'][:8]}=={'OLD_TRAIN','ACTUAL_FEEDBACK'}
    assert len({b['bucket'] for b in banks['F'][:8] if b['source']=='ACTUAL_FEEDBACK'})>=2


def freeze(old_path,legacy_path,windows_path,output):
    old=read(old_path);legacy=read(legacy_path);windows=read(windows_path)
    assert old['schema']=='PI0264_LANGUAGE_DIVERSITY_PLAN_V1' and len(old['source_bank'])==256
    assert legacy['schema']=='PI0263_COUNTERBALANCED_PLAN_V1'
    assert windows['status']=='PASS_ACTUAL_WINDOW_ADMISSION_CONTINUE_MATCHED_TRAINING' and not windows['shortages']
    common=old['source_bank'];excluded={u['original_unit'] for u in common}
    remaining=[u for u in legacy['units'] if u['visit']==0 and u['original_unit'] not in excluded]
    shared={t:family_order([u for u in common if u['task']==t]) for t in ('M1','M3')}
    control={t:family_order([u for u in remaining if u['task']==t])[:128] for t in ('M1','M3')}
    feedback={t:phase_order([u for u in windows['transfer_bank'] if u['task']==t]) for t in ('M1','M3')}
    assert all(len(pool[t])==128 for pool in (shared,control,feedback) for t in ('M1','M3'))
    banks={'R':[],'F':[]}
    for i in range(128):
        for task in ('M1','M3'):
            original=shared[task][i]
            source=dict(task=task,slot='COMMON',source='OLD_TRAIN',family=original['family'],selected=original['selected'],original_unit=original['original_unit'])
            for arm in banks:banks[arm].append(source)
            r=control[task][i];f=feedback[task][i]
            banks['R'].append(dict(task=task,slot='TRANSFER',source='OLD_TRAIN',family=r['family'],selected=r['selected'],original_unit=r['original_unit']))
            banks['F'].append(dict(task=task,slot='TRANSFER',source='ACTUAL_FEEDBACK',bucket=f['bucket'],selected=f['selected'],cycled=f['cycled']))
    units=[]
    offsets=[]
    for r,f in zip(banks['R'],banks['F']):
        # Matched source-slot hash fixes one common template schedule, even where observations differ.
        identities=[[c.get('frame_sha256'),c.get('proposal_sha256'),c.get('sequence'),c.get('index')] for b in (r,f) for c in b['selected']]
        offset=int(hashlib.sha256(json.dumps(identities,separators=(',',':')).encode()).hexdigest(),16)%8;offsets.append(offset)
    for visit in range(8):
        for index,source in enumerate(banks['R']):
            units.append(dict(unit=len(units)+1,visit=visit,bank_index=index,task=source['task'],slot=source['slot'],
                order=[0,1] if visit%2==0 else [1,0],text_offset=offsets[index],
                template='T'+str((visit+offsets[index])%8) if source['task']=='M1' else 'ORIGINAL_M3',seed=seed_for(len(units)+1)))
    plan=dict(schema='PI0266_MATCHED_FEEDBACK_BANK_AND_VISITS_V1',authority_commit='5f728de34c900523fa933a75c82be1b310ad814b',
        namespace='PI0266:TRAIN:26611:<base_unit_1_to_512>:<pass_0_to_7>',seed=26611,banks=banks,units=units,templates=TEMPLATES,
        source_hashes=dict(old_plan=sha(old_path),legacy_plan=sha(legacy_path),window_admission=sha(windows_path)),
        template_offset='SHA256 of paired slot source identities; shared across arms; each base observation visits all eight M1 forms',
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')
    validate(plan)
    bad=dict(plan,units=units[:-1]);
    try:validate(bad)
    except AssertionError:pass
    else:raise AssertionError('missing update accepted')
    write(output,plan)
    print(json.dumps(dict(status='PASS_FIXED_PI0266_MATCHED_BANK_8_PASSES',updates_per_arm=4096,microbatches_per_arm=8192,sha256=sha(output))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('old','legacy','windows','output'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();freeze(a.old,a.legacy,a.windows,a.output)
