"""Predeclared per-condition arithmetic; never aggregate marginal medians."""
import math
from statistics import median


def finite(value):
    return value is not None and math.isfinite(value)


def compare(direct, stochastic):
    """One deterministic output versus the complete two/eight-noise bank."""
    assert len(stochastic) in (2,8)
    assert len({r['j'] for r in stochastic})==len(stochastic)
    assert {r['j'] for r in stochastic}==set(range(len(stochastic)))
    assert all(r['input_id']==direct['input_id'] and r['condition_sha256']==direct['condition_sha256'] for r in stochastic)
    errors=[r.get('prefix_xyz_mse_m2') for r in stochastic]
    own=direct.get('prefix_xyz_mse_m2')
    assert all(v>=0 for v in [own,*errors] if finite(v))
    complete=all(finite(v) for v in [own,*errors])
    mean=sum(errors)/len(errors) if complete else None
    cosines=[r.get('direction_cosine') for r in stochastic];cosine=direct.get('direction_cosine')
    assert all(-1-1e-12<=v<=1+1e-12 for v in [cosine,*cosines] if finite(v))
    directions=all(finite(v) for v in [cosine,*cosines])
    return dict(input_id=direct['input_id'],condition_sha256=direct['condition_sha256'],
        direct_outputs=1,stochastic_outputs=len(stochastic),logical_direct_join_multiplicity=len(stochastic),
        complete_MSE_comparison=complete,direct_prefix_MSE_m2=own if finite(own) else None,
        mean_stochastic_prefix_MSE_m2=mean,delta_prefix_MSE_m2=own-mean if complete else None,
        direct_prefix_RMSE_m=math.sqrt(own) if finite(own) else None,
        stochastic_bank_RMSE_m=math.sqrt(mean) if complete else None,
        complete_direction_comparison=directions,
        direct_wrong_direction=None if not finite(cosine) else int(cosine<0),
        stochastic_wrong_direction_outputs=sum(v<0 for v in cosines if finite(v)),
        stochastic_defined_direction_outputs=sum(finite(v) for v in cosines),
        stochastic_wrong_direction_fraction=sum(v<0 for v in cosines)/len(cosines) if all(finite(v) for v in cosines) else None,
        direct_gate=direct['gate'],stochastic_gates=[r['gate'] for r in stochastic],
        finite_REJECT_is_retained=True,independent_direct_samples=1)


def summarize(rows):
    assert rows and len({r['input_id'] for r in rows})==len(rows)
    values=[r['delta_prefix_MSE_m2'] for r in rows if r['complete_MSE_comparison']]
    direct_defined=[r['direct_wrong_direction'] for r in rows if r['direct_wrong_direction'] is not None]
    return dict(conditions=len(rows),defined_MSE_conditions=len(values),undefined_MSE_conditions=len(rows)-len(values),
        median_per_condition_delta_prefix_MSE_m2=median(values) if values else None,
        direct_wrong_direction_conditions=sum(direct_defined),direct_defined_direction_conditions=len(direct_defined),
        stochastic_wrong_direction_outputs=sum(r['stochastic_wrong_direction_outputs'] for r in rows),
        stochastic_defined_direction_outputs=sum(r['stochastic_defined_direction_outputs'] for r in rows),
        stochastic_total_outputs=sum(r['stochastic_outputs'] for r in rows),
        direct_REJECT_conditions=sum(r['direct_gate']=='REJECT' for r in rows),
        stochastic_REJECT_outputs=sum(g=='REJECT' for r in rows for g in r['stochastic_gates']))


def self_check():
    def row(i,e,j=None,cos=-1,gate='ACCEPT'):
        return dict(input_id=str(i),condition_sha256='fixture-'+str(i),prefix_xyz_mse_m2=e,j=j,direction_cosine=cos,gate=gate)
    a=[0.,2.,100.];b=[1.,80.,90.]
    pairs=[compare(row(i,x),[row(i,y,j) for j in (0,1)]) for i,(x,y) in enumerate(zip(a,b))]
    assert summarize(pairs)['median_per_condition_delta_prefix_MSE_m2']==-1
    assert median(a)-median(b)==-78
    bad=compare(row(0,1.),[row(0,math.nan,0),row(0,1.,1)])
    assert not bad['complete_MSE_comparison'] and bad['delta_prefix_MSE_m2'] is None
    rejected=compare(row(0,1.,gate='REJECT'),[row(0,2.,j) for j in (0,1)])
    assert rejected['complete_MSE_comparison'] and rejected['direct_gate']=='REJECT'
    undefined=compare(row(0,1.,cos=None),[row(0,2.,j) for j in (0,1)])
    assert not undefined['complete_direction_comparison'] and undefined['direct_wrong_direction'] is None
    k8=compare(row(0,1.),[row(0,2.,j) for j in range(8)])
    assert k8['independent_direct_samples']==1 and k8['logical_direct_join_multiplicity']==8
    try:compare(row(0,1.),[row(1,2.,j) for j in (0,1)])
    except AssertionError:pass
    else:raise AssertionError('Mismatched conditions must be rejected')
    print('PASS_PER_CONDITION_DELTAS_COMPLETE_BANKS_INVALID_RETENTION_AND_DENOMINATORS')


if __name__=='__main__':self_check()
