"""Preregistered repeated native panels and parent/goal/restart analysis."""
import math
import numpy as np

METHODS=('TRAIN_MEAN_RESPONSE_RULE','FROZEN_ARX_SELECTOR','TRAINED_ACTOR')


def panel(scenes,task,*,seed,split='DEV',goal_indices=(8,72,136,200),primary_method='TRAINED_ACTOR'):
    from .pilot1_numpy import goal_offsets
    if (split not in ('DEV','TEST') or len(scenes)<2 or len({s['parent']for s in scenes})!=len(scenes)
            or any(s['split']!=split for s in scenes) or type(seed)is not int
            or len(goal_indices)!=4 or len(set(goal_indices))!=4
            or {i//64 for i in goal_indices}!={0,1,2,3} or primary_method not in METHODS[1:]):
        raise ValueError('DECLARED_INDEPENDENT_PARENT_SPLIT_AND_FOUR_GOAL_STRATA_REQUIRED')
    goals=goal_offsets(task);rng=np.random.default_rng(seed);rows=[];pair=0
    for index in rng.permutation(len(scenes)):
        scene=scenes[int(index)]
        for gi in rng.permutation(goal_indices):
            order=list(rng.permutation(METHODS));a,b,c=order
            pairs=[(a,b),(c,a),(b,c)];rng.shuffle(pairs)
            first_repeat={m:int(rng.integers(2))for m in METHODS};seen={m:0 for m in METHODS}
            for methods in pairs:
                for slot,method in enumerate(methods):
                    repeat=first_repeat[method]^seen[method];seen[method]+=1
                    rows.append(dict(parent=scene['parent'],scene_id=scene['scene_id'],split=split,
                        goal_index=int(gi),task_goal_offset_xy_m=goals[int(gi)].tolist(),method=str(method),
                        repeat_index=repeat,slot=slot,pair_index=pair))
                pair+=1
    return dict(schema='WM02_PILOT2_REPEATED_NATIVE_PANEL_V1',seed=seed,split=split,
        physical_parents=len(scenes),goal_indices=list(goal_indices),methods=list(METHODS),repeats=2,
        primary_method=primary_method,inference_minimum_parents=6,
        rows=rows,planned_evaluations=len(rows),exact_hidden_state_fork=False,native_integrations=0)


def minimum_practical_gain(rule,task):
    gaps=np.asarray(rule['same_action_restart_gap_m'],float)
    if (rule['fit_split']!='TRAIN' or gaps.ndim!=2 or gaps.shape[1]!=4
            or not np.isfinite(gaps).all() or np.any(gaps<0)):
        raise ValueError('REAL_TRAIN_RESTART_NOISE_REQUIRED_BEFORE_TEST')
    noise=float(np.quantile(gaps,.9));utility=.25*float(task['success_tolerance_m'])
    return dict(minimum_practical_improvement_m=max(.5*noise,utility),
        TRAIN_restart_pair_gap_p90_m=noise,task_utility_floor_m=utility,
        formula='MAX_HALF_TRAIN_P90_RESTART_GAP_AND_QUARTER_ORIGINAL_TASK_TOLERANCE',
        TEST_used=False)


def analyze(plan,observations,*,minimum_gain_m,seed=363501,bootstrap=10000,baseline='TRAIN_MEAN_RESPONSE_RULE'):
    """Missing/invalid/native fail stay separate; no exact-fork oracle regret."""
    def key(r):return(r['parent'],r['goal_index'],r['method'],r['repeat_index'])
    expected={key(r):r for r in plan['rows']}
    if len(expected)!=len(plan['rows'])or minimum_gain_m<=0 or bootstrap<100 or baseline not in plan['methods']:
        raise ValueError('UNIQUE_FROZEN_PANEL_AND_PREREGISTERED_ANALYSIS_REQUIRED')
    found={}
    for r in observations:
        k=key(r)
        if k not in expected or k in found:raise ValueError('UNPLANNED_OR_DUPLICATE_NATIVE_CONDITION')
        if r.get('slot')!=expected[k]['slot']:raise ValueError('ACTUAL_NATIVE_SLOT_MUST_MATCH_PREREGISTRATION')
        found[k]=r
    result=dict(schema='WM02_PILOT2_PARENT_GOAL_RESTART_ANALYSIS_V1',planned=len(expected),
        attempted=sum(bool(r['attempted'])for r in found.values()),not_run=len(expected)-sum(bool(r['attempted'])for r in found.values()),
        invalid_evidence=0,native_failed=0,safety_stops=0,methods={},comparisons={},
        exact_fork_oracle=False,bootstrap_replicates=bootstrap,bootstrap_seed=seed,
        minimum_practical_improvement_m=minimum_gain_m)
    valid={}
    for k,r in found.items():
        if not r['attempted']:continue
        result['safety_stops']+=int(bool(r.get('safety_stop')))
        if not r['native_passed']:result['native_failed']+=1
        elif not r['evidence_valid']:result['invalid_evidence']+=1
        else:
            if not np.isfinite([r['XY_error_m'],r['objective_m']]).all()or min(r['XY_error_m'],r['objective_m'])<0:
                raise ValueError('FINITE_ORIGINAL_MEASURED_NATIVE_COST_REQUIRED')
            valid[k]=r
    parents=sorted({r['parent']for r in plan['rows']});goals=plan['goal_indices'];methods=plan['methods']
    for m in methods:
        actual=[r for k,r in found.items()if k[2]==m and r['attempted']];measured=[r for k,r in valid.items()if k[2]==m]
        err=np.array([r['XY_error_m']for r in measured]);objective=np.array([r['objective_m']for r in measured])
        result['methods'][m]=dict(attempted=len(actual),valid_measured=len(measured),
            strict_task_successes=sum(bool(r['task_success'])for r in measured),
            strict_task_success_rate=sum(bool(r['task_success'])for r in measured)/len(actual)if actual else None,
            valid_measured_mean_XY_error_m=float(err.mean())if len(err)else None,
            valid_measured_median_XY_error_m=float(np.median(err))if len(err)else None,
            valid_measured_p90_XY_error_m=float(np.quantile(err,.9))if len(err)else None,
            valid_measured_mean_objective_m=float(objective.mean())if len(objective)else None)
    rng=np.random.default_rng(seed)
    for method in methods:
        if method==baseline:continue
        complete=[p for p in parents if all((p,g,m,r)in valid for g in goals for m in (baseline,method)for r in range(2))]
        summary=dict(complete_physical_parents=len(complete),total_physical_parents=len(parents),
            incomplete_parents_excluded_from_paired_cost_only=[p for p in parents if p not in complete],
            LOCAL_CONTROL_GAIN=False)
        if complete:
            values=np.asarray([[[[valid[(p,g,m,r)]['objective_m']for r in range(2)]
                for m in (baseline,method)]for g in goals]for p in complete])
            parent_effect=(values[:,:,0,:].mean(-1)-values[:,:,1,:].mean(-1)).mean(-1)
            draws=[]
            for _ in range(bootstrap):
                selected=rng.integers(len(complete),size=len(complete));effects=[]
                for i in selected:
                    gi=rng.integers(len(goals),size=len(goals));v=values[i,gi]
                    # Independent native restarts are resampled separately per method.
                    rr=rng.integers(2,size=(len(goals),2,2));resampled=np.take_along_axis(v,rr,axis=2).mean(2)
                    effects.append(float((resampled[:,0]-resampled[:,1]).mean()))
                draws.append(float(np.mean(effects)))
            low,high=np.quantile(draws,[.025,.975]);mean=float(parent_effect.mean())
            safety_method=sum(bool(r.get('safety_stop'))for k,r in found.items()if k[2]==method and r['attempted'])
            safety_base=sum(bool(r.get('safety_stop'))for k,r in found.items()if k[2]==baseline and r['attempted'])
            candidate=bool(len(complete)==len(parents)and mean>=minimum_gain_m and low>0
                and int((parent_effect>0).sum())>=math.ceil(2*len(parents)/3)and safety_method<=safety_base
                and not result['invalid_evidence']and not result['native_failed']and result['not_run']==0)
            summary.update(parent_equal_mean_cost_improvement_m=mean,parent_effects_m=dict(zip(complete,parent_effect.tolist())),
                hierarchical_95_interval_m=[float(low),float(high)],positive_parent_count=int((parent_effect>0).sum()),
                DEV_GAIN_CANDIDATE=bool(plan['split']=='DEV'and candidate),
                LOCAL_CONTROL_GAIN=bool(plan['split']=='TEST'and candidate
                    and method==plan['primary_method']and len(parents)>=plan['inference_minimum_parents']))
        result['comparisons'][method]=summary
    return result
