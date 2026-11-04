"""Post-hoc ranking decomposition of already-opened prospective DEV; no model or physics."""
import argparse,json,time
from pathlib import Path
import numpy as np
from go2arx5_vla.world_model.storage import digest,write_json
from go2arx5_vla.world_model.replay import SLICES,load_rows
from go2arx5_vla.world_model.decision_diagnostics import stable_choice

def components(predicted_opening,predicted_grasp,actual_opening,actual_grasp):
    arrays=[np.asarray(a)for a in(predicted_opening,predicted_grasp,actual_opening,actual_grasp)]
    if any(a.ndim!=1 or a.shape!=arrays[0].shape or not np.isfinite(a).all()for a in arrays):raise ValueError('Finite aligned candidate vectors required')
    gp,pp,gt,pt=arrays
    if any(np.any((a<0)|(a>1))for a in(pp,gt,pt)):raise ValueError('Probability/measured fraction outside physical range')
    return dict(saved_WM_proxy=(1-pp)*np.clip(gp,0,1),PRIVILEGED_true_opening=(1-pp)*gt,PRIVILEGED_true_grasp=(1-pt)*np.clip(gp,0,1),PRIVILEGED_true_both=(1-pt)*gt)

def validate_population(original,cards):
    # The first frozen new8 result predates the optional study field. Bind its
    # actual hashed population rather than inventing/requiring later metadata.
    if (original.get('study','NEW8_F0_VALIDATION_V1')!='NEW8_F0_VALIDATION_V1'
            or original['intended_parents']!=8 or original['intended_anchors']!=24
            or len(cards)!=8 or [c['ident_index']for c in cards]!=list(range(40,48))
            or any(c['split']!='DEV'or c['ident_partition']!='PROSPECTIVE_B2_NEVER_FIT'for c in cards)):
        raise ValueError('Only original already-opened prospective DEV8 permitted')

def run(root,manifest,pin):
    start=time.monotonic();cpu=time.process_time();root=Path(root);manifest=Path(manifest)
    if digest(manifest)!=pin:raise ValueError('Original audited stream manifest differs')
    pins=json.loads(manifest.read_text())['files'];used={}
    def checked(relative):
        p=root/relative
        if relative not in pins or digest(p)!=pins[relative]:raise ValueError('Unmanifested/changed original evidence:'+relative)
        used[relative]=pins[relative];return p
    original=json.loads(checked('RESULT.json').read_text())
    validate_population(original,json.loads(checked('CARDS.json').read_text()))
    scenes=[];counts={k:0 for k in('saved_WM_proxy','PRIVILEGED_true_opening','PRIVILEGED_true_grasp','PRIVILEGED_true_both')}
    for i in range(40,48):
        prefix=f'scene-{i:02d}/supported_release/';selection=json.loads(checked(prefix+'BEFORE_ANY_BRANCH.json').read_text());truth=json.loads(checked(prefix+'RESULT.json').read_text());names=selection['names']
        if selection['uses_future_truth']or not selection['prospective_validation']:raise ValueError('Original causal selection required')
        with np.load(checked(prefix+'PREDICTIONS_BEFORE_EFFECT.npz'),allow_pickle=False)as f:pred=f['state'];event=f['event'];raw=f['raw']
        if pred.shape!=(5,16,50)or event.shape!=(5,16,4)or raw.shape!=(5,8,7)or len(names)!=5:raise ValueError('Original candidate ABI differs')
        opening=[];grasp=[];support=[];quality=[]
        for name in names:
            relative=prefix+'candidate-'+name+'.pkl.gz';p=checked(relative);rows,rejections=load_rows(dict(path=str(p),sha256=pins[relative]))
            if len(rows)!=16 or rejections or any(r['terminal']or r['truncated']for r in rows):raise ValueError('Complete existing candidate required')
            state=rows[-1]['state_after'];opening.append(state['gripper_fraction']);grasp.append(float(state['native_grasp']));support.append(float(truth['candidates'][name]['native_support']));quality.append(truth['candidates'][name]['Q'])
        gp=pred[:,-1,SLICES['gripper_fraction']].ravel();pp=event[:,-1,2];scores=components(gp,pp,opening,grasp)
        if not np.allclose(scores['saved_WM_proxy'],selection['methods']['pure_WM']['scores'],rtol=0,atol=1e-7):raise ValueError('Saved WM score formula differs')
        if not np.allclose(np.asarray(support)*scores['PRIVILEGED_true_both'],quality,rtol=0,atol=1e-12):raise ValueError('Original diagnostic Q no longer matches measured state')
        oracle=max(quality);choices={}
        for k,s in scores.items():
            index=stable_choice(s,raw);regret=max(0.,oracle-quality[index]);counts[k]+=int(regret>1e-12);choices[k]=dict(selected=names[index],actual_Q=quality[index],regret_Q=regret)
        if choices['saved_WM_proxy']['selected']!=truth['online']['pure_WM']['selected']:raise ValueError('Original prospective choice differs')
        r=names.index('release');wm=names.index(choices['saved_WM_proxy']['selected'])
        scenes.append(dict(scene=i,candidate_names=names,predicted_gripper_fraction=gp.tolist(),actual_gripper_fraction=opening,predicted_grasp=pp.tolist(),actual_grasp=grasp,native_support=support,actual_Q=quality,scores={k:s.tolist()for k,s in scores.items()},choices=choices,WM_margin_over_immediate_release=float(scores['saved_WM_proxy'][wm]-scores['saved_WM_proxy'][r]),actual_immediate_minus_selected_Q=quality[r]-quality[wm]))
    return dict(schema='WM02_POSTHOC_RELEASE_RANK_V1',purpose='Already-opened prospective DEV8 mechanism audit; truth-substitution columns are PRIVILEGED explanations, never deployable new methods or blind validation.',original_collection_source_sha=original['source_sha'],original_model=original['model'],stream_manifest_sha256=pin,consumed_files_sha256=used,parents=8,candidates=40,positive_regret_parent_counts=counts,scenes=scenes,new_physics=0,new_inference=0,new_optimizer_updates=0,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in('root','manifest','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True);a=p.parse_args(argv);v=run(a.root,a.manifest,a.manifest_sha256)
    # Bind original host paths privately; publish only hashes and logical paths.
    v['original_model']={k:x for k,x in v['original_model'].items()if k not in('path',)}
    write_json(a.output,v);print(json.dumps({k:v[k]for k in('parents','candidates','positive_regret_parent_counts','new_physics','new_inference','new_optimizer_updates','CPU_s','wall_s')}));return 0
if __name__=='__main__':raise SystemExit(main())
