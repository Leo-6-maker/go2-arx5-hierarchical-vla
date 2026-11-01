"""TRAIN-only phase exposure, outside the student observation/label interface."""
import hashlib
import json
from collections import defaultdict

import numpy as np

SCHEMA = 'BOOK_PHASE4096_V1'
INDEX_SCHEMA = 'BOOK_SAVED_TRAIN_PHASE_INDEX_V1'
SLOTS = (('uniform',16),('approach_align',2),('close_verify',2),
         ('carry_boundary',4),('carry_other',4),('release',1),
         ('withdraw',1),('early_recovery',2))
STAGES = {'approach','align','close','verify_grasp','lift','preinsert',
          'insert','release','withdraw','hold'}


def digest(value):
    return hashlib.sha256((json.dumps(value,indent=2,allow_nan=False)+'\n').encode()).hexdigest()


def pools(rows,index,recipe):
    if (index['schema']!=INDEX_SCHEMA or digest(index)!=recipe['phase_index_sha256']
            or any(index[k]!=recipe[k] for k in
                ('dataset_manifest_sha256','statistics_sha256','TRAIN_population_sha256'))):
        raise PermissionError('SEALED_TRAIN_PHASE_INDEX_IDENTITY')
    metadata={};out={name:defaultdict(list)for name,count in SLOTS}
    for e in index['episodes']:
        name=e['id'];n=e['windows'];source=e['source_observation_index']
        stages=e['stages'];closed=e['prefix8_closed'];boundaries=set(e['source_planning_indices'])
        if (len(source)!=n or len(stages)!=n or len(closed)!=n
                or len(set(source))!=n or source!=list(range(source[0],source[0]+n))
                or not set(stages)<=STAGES or any(type(c)is not bool for c in closed)):
            raise ValueError('ACTUAL_SOURCE_INDEX_PHASE_PREFIX_ALIGNMENT')
        for a in range(n):
            key=name,a
            if key in metadata:raise ValueError('DUPLICATE_TRAIN_PHASE_WINDOW')
            metadata[key]=(stages[a],closed[a],source[a] in boundaries,e['expert_suffix'])
    if len(rows)!=len(metadata) or len(rows)!=12533:
        raise ValueError('EXACT_ORIGINAL_TRAIN_WINDOW_PHASE_COVERAGE')
    for i,(r,a) in enumerate(rows):
        if r['split']!='TRAIN':raise PermissionError('PHASE_NO_DEV_OR_TEST')
        stage,closed,boundary,recovery=metadata[r['id'],a]
        def add(name):out[name][r['id']].append(i)
        add('uniform')
        if stage in ('approach','align'):
            add('approach_align')
            if recovery:add('early_recovery')
        if stage in ('close','verify_grasp'):add('close_verify')
        if stage in ('lift','preinsert','insert') and closed:
            add('carry_boundary' if boundary else 'carry_other')
        if stage in ('release','withdraw'):add(stage)
        # Final Teacher hold is open, after release; it belongs only to uniform.
    if any(not group or any(not v for v in group.values())for group in out.values()):
        raise ValueError('ALL_DECLARED_PHASE_STREAMS_NONEMPTY')
    return {k:dict(sorted(v.items()))for k,v in out.items()}


def sequence(rows,index,recipe):
    """Generate once; independent PRNGs never consume model/noise RNG state.

    Uniform cycles all windows. Specialized streams cycle shuffled episodes,
    then draw a shuffled window cycle within the selected episode. Each stream
    therefore balances eligible episodes before windows, including on resume.
    """
    groups=pools(rows,index,recipe);seed=recipe['seed'];updates=recipe['max_steps']
    streams={};slots=[]
    for stream,(name,count) in enumerate(SLOTS):
        visits=updates*count;group=groups[name];rng=np.random.Generator(np.random.PCG64(seed+1000000*(stream+1)))
        result=[]
        if name=='uniform':
            eligible=np.asarray(sorted(i for v in group.values()for i in v),dtype=np.int64)
            for cycle in range((visits+len(eligible)-1)//len(eligible)):
                result.extend(eligible[np.random.Generator(np.random.PCG64(seed+cycle)).permutation(len(eligible))].tolist())
        else:
            names=sorted(group);orders={};positions=defaultdict(int)
            while len(result)<visits:
                for ep in rng.permutation(len(names)):
                    ep=names[int(ep)];pool=group[ep];position=positions[ep]
                    if position%len(pool)==0:orders[ep]=rng.permutation(len(pool))
                    result.append(pool[int(orders[ep][position%len(pool)])]);positions[ep]+=1
                    if len(result)==visits:break
        streams[name]=np.asarray(result[:visits],dtype=np.int64).reshape(updates,count)
        slots.extend((name,j)for j in range(count))
    result=np.empty((updates,32),dtype=np.int64)
    for step in range(updates):
        order=np.random.Generator(np.random.PCG64(seed+9000000+step)).permutation(32)
        result[step]=[streams[slots[j][0]][step,slots[j][1]]for j in order]
    return result.reshape(-1)


def load_index(root,recipe):
    from pathlib import Path
    body=(Path(root)/'TRAIN_PHASE_INDEX.json').read_bytes()
    if hashlib.sha256(body).hexdigest()!=recipe['phase_index_sha256']:
        raise PermissionError('SEALED_PHASE_INDEX_BYTES')
    return json.loads(body)
