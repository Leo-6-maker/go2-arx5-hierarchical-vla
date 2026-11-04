"""Numeric paired S dataset; common real history, original branch provenance, no target padding."""
from pathlib import Path
import json
import numpy as np
from .replay import tensor_state,SCHEMA,aligned,NumericProjectUnpickler
from .storage import digest,write_json
from .contact_identification import ANCHORS,CANDIDATES
import gzip


def read(path):
    with gzip.open(path,'rb')as f:return NumericProjectUnpickler(f).load()


def build(roots,output):
    output=Path(output);output.mkdir(exist_ok=False);arrays={k:[]for k in ('state','reference','events','done','timeout','valid','clock','context','raw','partition','scene','anchor','candidate','initial_event')}
    files=[];denominator=[];repeats=[];scenes={};counts=dict(main=0,repeat=0,tail=0,prefix=0)
    for root in map(Path,roots):
        report=json.loads((root/'RESULT.json').read_text())
        folders={json.loads(p.read_text())['ident_index']:p.parent for p in root.glob('scene-*/CARD.json')}
        for s in report['scenes']:
            i=s['scene_index']
            if i in scenes:raise ValueError('A predeclared scene was collected twice')
            scenes[i]=s;counts['prefix']+=s['prefix_transitions'];counts['main']+=s['main_transitions'];counts['repeat']+=s['repeat_transitions'];counts['tail']+=s['tail_transitions']
            folder=folders[i] # Bind actual card identity, including the original batch-local folder numbering.
            for anchor in ANCHORS:
                for candidate in CANDIDATES:denominator.append(dict(scene=i,anchor=anchor,candidate=candidate,status='UNRUN'))
                if anchor not in s['anchors']:continue
                parent=folder/anchor;common=read(parent/'COMMON_HISTORY.pkl.gz');history=common['rows']
                if len(history)!=8:raise ValueError('Exactly eight real history atoms required')
                for r in history:aligned(r)
                repeats.append(dict(scene=i,anchor=anchor,repeatability=s['anchors'][anchor]['repeatability'],tail_physical_equivalence=s['anchors'][anchor]['tail_physical_equivalence']))
                for candidate in CANDIDATES:
                    path=parent/(candidate+'.pkl.gz');pin=digest(path);record=read(path);rows=record['rows'];files.append(dict(relative=f'{root.name}/{folder.name}/{anchor}/{candidate}',scene_index=i,sha256=pin))
                    status='COMPLETE16'if len(rows)==16 else'PARTIAL'if rows else'REJECTED_ZERO_PHYSICS'
                    next(x for x in denominator if x['scene']==i and x['anchor']==anchor and x['candidate']==candidate)['status']=status
                    if not rows:continue
                    for r in rows:aligned(r)
                    if not np.array_equal(tensor_state(history[-1]['state_after']),tensor_state(rows[0]['state_before'])):raise ValueError('Branch does not start from common real history')
                    seq=history+rows;states=[seq[0]['state_before']]+[r['state_after']for r in seq]
                    n=len(seq);valid=[True]*n
                    while len(seq)<24:seq.append(seq[-1]);states.append(states[-1]);valid.append(False)
                    arrays['state'].append(np.stack([tensor_state(s)for s in states]))
                    arrays['reference'].append(np.stack([r['reference_compiled']for r in seq]));arrays['events'].append([[r['state_after'][k]for k in SCHEMA['events']]for r in seq])
                    arrays['done'].append([r['terminal']if v else False for r,v in zip(seq,valid)]);arrays['timeout'].append([r['truncated']if v else False for r,v in zip(seq,valid)]);arrays['valid'].append(valid)
                    arrays['clock'].append([[r['t0']/32,r['native_progress']['point']*.05]for r in seq])
                    c=seq[0]['context'];arrays['context'].append([*c['goal_xyz'],c['table_top_m'],c['cube_edge_m']]);arrays['raw'].append(np.stack([r['action_raw']for r in rows[:8]]))
                    if len(rows)<8:arrays['raw'][-1]=np.pad(arrays['raw'][-1],((0,8-len(rows)),(0,0)),mode='edge')
                    arrays['partition'].append(s['partition']);arrays['scene'].append(i);arrays['anchor'].append(anchor);arrays['candidate'].append(candidate)
                    arrays['initial_event'].append([rows[0]['state_before'][k]for k in SCHEMA['events']])
    if set(scenes)!=set(range(24)):raise ValueError('Retain all24 intended scene denominators, including incomplete scenes')
    for k in arrays:arrays[k]=np.asarray(arrays[k])
    arrays.update(purpose=np.array('WM02_S_CONTACT_PAIRED_REPLAY_V1'),schema_json=np.array(json.dumps(SCHEMA)))
    with (output/'PAIRED.npz').open('xb')as stream:np.savez_compressed(stream,**arrays)
    report=dict(version='WM02_S_CONTACT_IDENTIFICATION_V1',scene_count=24,branch_denominator=480,
        complete16=sum(x['status']=='COMPLETE16'for x in denominator),partial=sum(x['status']=='PARTIAL'for x in denominator),
        rejected_zero_physics=sum(x['status']=='REJECTED_ZERO_PHYSICS'for x in denominator),unrun=sum(x['status']=='UNRUN'for x in denominator),
        actual_transitions=counts,numeric_pairs=len(arrays['state']),data_sha256=digest(output/'PAIRED.npz'),
        actual_source_sha=report['source_sha'],source_files=files,denominator=denominator,repeatability=repeats,
        partitions={p:sorted({int(i)for i in arrays['scene'][arrays['partition']==p]})for p in ('FIT','DIAGNOSTIC_DEV','ONCE_HOLDOUT')},
        original_split='TRAIN only; diagnostic group partitions distinct from original DEV/TEST',Teacher_phase_inputs=False,target_transitions=0)
    write_json(output/'AUDIT.json',report);return report
