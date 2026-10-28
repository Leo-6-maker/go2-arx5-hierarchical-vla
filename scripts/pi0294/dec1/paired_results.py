"""Strict saved DEC1 rows and eight-geometry paired analysis; no execution."""
import csv
import itertools
import json
from pathlib import Path
import numpy as np

MODEL='ORIGINAL_CORR_AT2048_R0'
CHECKPOINT='94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38'
R0='fb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7'
VALID={'SUCCESS','VALID_TASK_FAILURE','REFERENCE_REJECT','S1_REJECT'}
OTHER={'NATIVE_SAFETY_ERROR','ENGINEERING_MISSING','NOT_RUN'}
KEYS=set(itertools.product(range(1,9),('T1','T2'),('E_SOURCE','E_TARGET'),('I_TIME','I_PROGRESS')))
UNKNOWN={'UNKNOWN','',None}


def aggregate(rows, *, geometry_identities, noise_identity, bootstrap_draws=2000, seed=20261021):
    """All 64 explicit rows required; unknown outcomes never become failures.

    The eight geometry identities and noise-bank identity come from the
    owner's sealed panel, not inferred from observed favorable results.
    Missing/native-fault panels expose counts but have no effect estimate.
    """
    if (set(geometry_identities)!=set(range(1,9)) or
            len(set(geometry_identities.values()))!=8 or
            any(v in UNKNOWN or v=='NOT_SEALED' for v in geometry_identities.values()) or
            noise_identity in UNKNOWN or noise_identity=='NOT_BOUND'):
        raise ValueError('DEC1_SEALED_EIGHT_GEOMETRIES_AND_NOISE_REQUIRED')
    if type(bootstrap_draws) is not int or not 100<=bootstrap_draws<=100000:
        raise ValueError('DEC1_BOUNDED_EXPLICIT_CLUSTER_RESAMPLING')
    indexed={};counts={k:0 for k in sorted(VALID|OTHER)}
    for row in rows:
        try:
            slot=int(row['geometry_slot_not_sealed'])
            key=(slot,row['task'],row['executor'],row['interface'])
            status=row['status'];outcome=row['task_success']
            geom=row['geometry_identity'];noise=row['noise_identity']
            model=row['model']
        except (KeyError,TypeError,ValueError) as error:
            raise ValueError('DEC1_REQUIRED_RESULT_KEYS') from error
        if key not in KEYS or key in indexed:raise ValueError('DEC1_UNKNOWN_OR_DUPLICATE_CASE')
        if (model!=MODEL or status not in VALID|OTHER or geom!=geometry_identities[slot]
                or noise!=noise_identity):
            raise ValueError('DEC1_FIXED_CORR_R0_GEOMETRY_NOISE_AND_STATUS')
        if status in VALID|{'NATIVE_SAFETY_ERROR'}:
            if (row.get('checkpoint_manifest_sha256')!=CHECKPOINT or row.get('statistics_sha256')!=R0):
                raise ValueError('DEC1_ACTUAL_CORR_CHECKPOINT_AND_R0_DIGEST_REQUIRED')
            expected=status=='SUCCESS'
            if outcome not in (expected,str(expected).lower(),str(int(expected))):
                raise ValueError('DEC1_STATUS_OUTCOME_CONTRADICTION')
        elif row.get('checkpoint_manifest_sha256',CHECKPOINT)!=CHECKPOINT or row.get('statistics_sha256',R0)!=R0:
            raise ValueError('DEC1_NO_FOREIGN_POLICY_IN_UNRUN_ROWS')
        elif outcome not in UNKNOWN:
            raise ValueError('DEC1_UNRUN_OR_MISSING_OUTCOME_MUST_REMAIN_UNKNOWN')
        if status not in {'NOT_RUN','ENGINEERING_MISSING'} and row.get('native_qualification')!='QUALIFIED':
            raise ValueError('DEC1_COMPLETED_CASE_NATIVE_QUALIFICATION_REQUIRED')
        indexed[key]=row;counts[status]+=1
    if set(indexed)!=KEYS:raise ValueError('DEC1_MISSING_CASE_KEYS_NO_SILENT_ZERO_FILL')
    strata={}
    for task,executor,interface in itertools.product(('T1','T2'),('E_SOURCE','E_TARGET'),('I_TIME','I_PROGRESS')):
        values=[indexed[i,task,executor,interface]['status'] for i in range(1,9)]
        strata['/'.join((task,executor,interface))]=dict(planned=8,
            status_counts={s:values.count(s) for s in sorted(VALID|OTHER)},
            successes=values.count('SUCCESS'),
            success_rate=values.count('SUCCESS')/8 if all(s in VALID for s in values) else None)
    complete=all(r['status'] in VALID for r in indexed.values())
    summary=dict(schema='DEC1_STRICT_PAIRED_GEOMETRY_RESULTS_V1',model=MODEL,checkpoint_manifest_sha256=CHECKPOINT,statistics_sha256=R0,
        planned_cells=64,independent_geometry_blocks=8,status_counts=counts,
        complete_for_paired_inference=complete,effect_estimates=None,intervals=None,
        by_task_executor_interface=strata,
        no_target_data_backflow=True,selected='NONE',deployment='NONE')
    if not complete:return summary
    blocks=[]
    for slot in range(1,9):
        deltas={}
        for executor in ('E_SOURCE','E_TARGET'):
            deltas[executor]=np.mean([int(indexed[slot,t,executor,'I_PROGRESS']['status']=='SUCCESS')-
                int(indexed[slot,t,executor,'I_TIME']['status']=='SUCCESS') for t in ('T1','T2')])
        blocks.append([deltas['E_SOURCE'],deltas['E_TARGET'],deltas['E_TARGET']-deltas['E_SOURCE']])
    blocks=np.asarray(blocks)
    samples=np.random.default_rng(seed).integers(0,8,size=(bootstrap_draws,8))
    means=blocks[samples].mean(axis=1)
    names=('Delta_source','Delta_target','interaction')
    summary.update(effect_estimates=dict(zip(names,blocks.mean(axis=0).tolist())),
        geometry_block_differences=blocks.tolist(),
        intervals={name:np.quantile(means[:,i],[.025,.975]).tolist() for i,name in enumerate(names)},
        resampling=dict(unit='PAIRED_GEOMETRY_BLOCK_ALL_EIGHT_CELLS_TOGETHER',blocks=8,
            draws=bootstrap_draws,seed=seed),
        interpretation='PILOT_SIGNAL_OR_INCONCLUSIVE' if any(
            np.quantile(means[:,i],.025)<=0<=np.quantile(means[:,i],.975) for i in (1,2))
            else 'BOUNDED_PILOT_ESTIMATE_NOT_GENERAL_WBC_CAUSAL_EFFECT')
    return summary


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('rows_tsv');p.add_argument('sealed_identity_json')
    args=p.parse_args();identity=json.loads(Path(args.sealed_identity_json).read_text())
    with Path(args.rows_tsv).open(newline='') as f:rows=list(csv.DictReader(f,delimiter='\t'))
    result=aggregate(rows,geometry_identities={int(k):v for k,v in identity['geometry_identities'].items()},
        noise_identity=identity['noise_identity'])
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
