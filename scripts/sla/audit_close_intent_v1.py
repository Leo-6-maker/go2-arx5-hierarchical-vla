"""Read-only command-intent diagnostics; never replace original gate scores.

Threshold22mm is the existing halfway crossing definition. This does not
claim native grasp, classify measured object-limited aperture, or rescore tasks.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def classify(width):
    return 'OPEN'if width>.022 else'CLOSED'


def summarize(records):
    result={}
    for phase in ('ALL','PREGRASP','DESCEND','CLOSE','LIFT','HOLD'):
        rows=[r for r in records if phase=='ALL'or r['phase']==phase]
        if not rows:continue
        annotated=[r for r in rows if r['annotated_crossing']]
        result[phase]=dict(predictions=len(rows),original_full_H40_pass=sum(r['original_full_H40_pass']for r in rows),
            terminal_opposite_command_state=sum(r['terminal_opposite_command_state']for r in rows),
            terminal_width_MAE_m=float(np.mean([r['terminal_width_error_m']for r in rows])),
            terminal_width_error_max_m=max(r['terminal_width_error_m']for r in rows),
            H40_width_MAE_m=float(np.mean([r['width_MAE_m']for r in rows])),
            H40_width_error_max_m=max(r['width_error_max_m']for r in rows),
            annotated_crossings=len(annotated),crossing_within100ms=sum(
                r['crossing_error_s']is not None and r['crossing_error_s']<=.100000001 for r in annotated),
            missed_crossings=sum(r['crossing_error_s']is None for r in annotated))
    return result


def audit(dataset,conditions):
    root=Path(dataset);manifest=json.loads((root/'MANIFEST.private.json').read_bytes());bank=[]
    for ep in manifest['episodes']:
        path=root/ep['labels']['file']
        if sha(path)!=ep['labels']['sha256']:raise ValueError('SEALED_ORIGINAL_EXPERT_LABELS_CHANGED')
        with np.load(path,allow_pickle=False)as z:labels=z['nominal_policy_H40'].copy()
        bank.extend((ep['id'],anchor,.002+.040*row[:,9])for anchor,row in enumerate(labels))
    outputs={}
    for name,path in conditions.items():
        rows=json.loads(Path(path).read_bytes())
        if name=='C20_1000':rows=[r for r in rows if r['route']=='cached_actual']
        if len(rows)!=96:raise ValueError('ORIGINAL_FIXED48_TIMES_NOISE2')
        records=[]
        for r in rows:
            w=r.get('window',r);index=w['index'];episode,anchor,truth=bank[index]
            if(w['episode'],w['anchor'])!=(episode,anchor):raise ValueError('ORIGINAL_WINDOW_IDENTITY_MISMATCH')
            output=r['output']
            if sha(output['path'])!=output['sha256']:raise ValueError('ORIGINAL_PREDICTION_CHANGED')
            with np.load(output['path'],allow_pickle=False)as z:predicted=.002+.040*z['physical_H40'][:,9]
            crossing=lambda a:next((int(k)for k,v in enumerate(a)if v<=.022),None)
            annotated=bool(truth[0]>.022 and truth[-1]<=.022);actual_cross=crossing(predicted)
            event_error=None if not annotated or actual_cross is None else .05*abs(crossing(truth)-actual_cross)
            records.append(dict(index=index,noise=r['noise'],phase=w['phase'],
                original_full_H40_pass=r['envelope']['original_full_H40_pass'],
                expected_terminal_command_state=classify(truth[-1]),predicted_terminal_command_state=classify(predicted[-1]),
                terminal_opposite_command_state=classify(truth[-1])!=classify(predicted[-1]),
                terminal_width_error_m=float(abs(predicted[-1]-truth[-1])),
                width_MAE_m=float(abs(predicted-truth).mean()),width_error_max_m=float(abs(predicted-truth).max()),
                annotated_crossing=annotated,crossing_error_s=event_error))
        outputs[name]=dict(input_sha256=sha(path),summary=summarize(records),records=records)
    return dict(schema='SLA_FIXED96_COMMAND_INTENT_DIAGNOSTICS_V1',conditions=outputs,
        original_gate_or_task_scores_changed=False,grasp_success_inferred=False,new_model_generations=0,
        new_optimizer_updates=0,new_physical_requests=0,physical_admission=False)


if __name__=='__main__':
    import sys
    config=json.loads(Path(sys.argv[1]).read_bytes())
    print(json.dumps(audit(config['dataset'],config['conditions'])))
