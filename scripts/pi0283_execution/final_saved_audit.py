"""Final source funnel from retained arrays; no neural, render or physics calls."""
import json,os
from pathlib import Path
import numpy as np
from closed_loop_audit import ROOT,longest,predicates,sha


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert longest([False,True,True,False,True])==2
    points=[]
    for point in ('MORE1024','MORE2048','EVENT1024'):
        if not (ROOT/'source/evidence'/f'RESULT_{point}_TRAIN.json').exists():continue
        panel=ROOT/'source/evidence'/f'RESULT_{point}_TRAIN.json'
        receipt=json.loads(panel.read_text());rows=[]
        for row in receipt['rows']:
            folder=Path(row['folder'])
            with np.load(folder/'physics.npz',allow_pickle=False) as f:p={k:f[k] for k in f.files}
            flags=predicates(p)
            for k,v in flags.items():assert np.array_equal(v,p[k]),(point,row['condition'],k)
            # ponytail: use existing native predicates; the absent historical lift rule is not invented.
            control={k:v[::5][20:] for k,v in flags.items()};success=control['success']
            indices=np.flatnonzero(success);transitions=np.flatnonzero(success[:-1]&~success[1:])+1
            first=float(indices[0]*.05) if len(indices) else None
            assert first==row['first_success_s']
            lift=p['cubeA_pose'][100:,2]-p['cubeA_pose'][0,2]
            height=lift>.02
            with np.load(folder/'observations.npz') as f:opening=f['state_v1'][20:,-1]
            with np.load(folder/'commands.npz') as f:command_opening=f['action'][20:,-1]
            clip=[]
            for q in sorted(folder.glob('q*')):
                if (q/'output.npz').exists():
                    with np.load(q/'output.npz') as f:
                        value=f['raw'][...,:10];fraction=float((np.abs(value)>1).mean())
                        assert fraction==float(f['normalized_OOB_fraction']);clip.append(fraction)
            rows.append(dict(condition=row['condition'],status=row['status'],queries=row['queries'],task_controls=row['task_control_calls'],
                sustained_grasp_0p3=longest(flags['is_cubeA_grasped'][100:])>=30,
                original_sustained_lift_predicate='UNOBSERVED_NOT_RECORDED',
                descriptive_lift_gt2cm_0p3=longest(height)>=30,
                descriptive_lift_rule='Height above task-initial cubeA >2cm for >=30 physics samples; descriptive, not task qualification',
                max_lift_m=float(lift.max()) if len(lift) else None,on_top=bool(control['is_cubeA_on_cubeB'].any()),
                first_success_s=first,last_success_s=float(indices[-1]*.05) if len(indices) else None,
                first_success_loss_s=float(transitions[0]*.05) if len(transitions) else None,
                last_success_loss_s=float(transitions[-1]*.05) if len(transitions) else None,
                persistent=row['persistent_success'],hold_success_samples=int(success[560:601].sum()),hold_observed_samples=len(success[560:601]),
                hold_on_top_samples=int(control['is_cubeA_on_cubeB'][560:601].sum()),hold_static_samples=int(control['is_cubeA_static'][560:601].sum()),
                hold_grasped_samples=int(control['is_cubeA_grasped'][560:601].sum()),
                commanded_opening_range=[float(command_opening.min()),float(command_opening.max())] if len(command_opening) else None,
                measured_opening_range=[float(opening.min()),float(opening.max())] if len(opening) else None,
                peak_left_contact_force_N=float(np.linalg.norm(p['left_force'][100:],axis=1).max()),
                peak_right_contact_force_N=float(np.linalg.norm(p['right_force'][100:],axis=1).max()),ik_holds=row['ik_holds'],
                output_pre_inverse_OOB_fraction_mean=float(np.mean(clip)) if clip else None,
                output_pre_inverse_OOB_fraction_max=float(max(clip)) if clip else None,
                result_sha256=sha(folder/'RESULT.json')))
        diagpath=ROOT/'source/diagnostics'/point/'CPU_RECOMPUTE.json';diag=json.loads(diagpath.read_text())
        metrics={}
        for split in ('TRAIN','VAL'):
            subset=[r for r in diag['rows'] if r['split']==split];assert len(subset)==24
            metrics[split]=dict(rows=24,fit_count=sum(r['fit_pass'] for r in subset),
                phase_fit={str(k):dict(passed=sum(r['fit_pass'] for r in subset if r['phase']==k),denominator=sum(r['phase']==k for r in subset)) for k in range(4)},
                metrics={k:dict(median=float(np.median([r[k] for r in subset])),p95=float(np.percentile([r[k] for r in subset],95)),maximum=float(max(r[k] for r in subset))) for k in ('XYZ_RMS_m','SO3_RMS_rad','gripper_MAE','normalized_OOB_fraction')})
        counts={k:sum(bool(r[k]) for r in rows) for k in ('persistent','sustained_grasp_0p3','descriptive_lift_gt2cm_0p3','on_top')}
        counts['first_success']=sum(r['first_success_s'] is not None for r in rows)
        points.append(dict(point=point,rows=rows,counts=counts,diagnostics=metrics,panel_sha256=sha(panel),diag_sha256=sha(diagpath)))
    assert points
    best=max(points,key=lambda p:(p['counts']['persistent'],p['counts']['first_success'],-p['diagnostics']['TRAIN']['metrics']['XYZ_RMS_m']['median'],-('MORE1024','MORE2048','EVENT1024').index(p['point'])))
    output=dict(status='PASS_CPU_FINAL_SAVED_FUNNEL_NOT_TASK_PASS',points=points,best_development=best['point'],
        new_model_queries=0,new_optimizer_updates=0,new_render=0,new_physics=0,source_candidate='NONE',selected='NONE',deployment='NONE')
    path=ROOT/'audit/FINAL_SAVED_FUNNEL.json';path.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=output['status'],counts={p['point']:p['counts'] for p in points},best_development=best['point'])))


if __name__=='__main__':main()
