"""Audit saved source command/clock and limiter memory; zero model inference/plant steps."""
import argparse,json,time
from pathlib import Path
import numpy as np
from go2arx5_vla.world_model.storage import digest,write_json
from go2arx5_vla.world_model.replay import load_rows
from go2arx5_vla.world_model.actuator_observer import SourceFingerTarget
from scripts.wm02.diagnose_saved_release import validate_population

def audit_numeric(path,pin):
    if digest(path)!=pin:raise ValueError('Original FIT/DEV numeric replay differs')
    with np.load(path,allow_pickle=False)as f:
        keep=np.flatnonzero(np.isin(f['partition'],['FIT','DIAGNOSTIC_DEV']))
        if len(keep)!=480:raise ValueError('Only original FIT360/DEV120 windows required')
        raw=f['raw'][keep];reference=f['reference'][keep,8:24];clock=f['clock'][keep,8:24]
    h40=np.concatenate((raw[:,:,6],np.repeat(raw[:,-1:,6],32,axis=1)),axis=1)
    expected=np.repeat(h40[:,None],16,axis=1);expected[:,8:]=raw[:,-1,None,6:7]
    opening_error=float(np.abs(reference[:,:,:,9]-expected).max())
    absolute_error=float(np.abs(clock[:,:,0]-(clock[:,0:1,0]+np.arange(16)[None]*.05/32)).max())
    progress_error=float(np.abs(clock[:,:,1]-(np.arange(16)%8)[None]*.05).max())
    if opening_error>1e-12 or absolute_error>1e-10 or progress_error>1e-12:raise ValueError('Actual grip/time action contract differs')
    return dict(FIT_DEV_windows=480,old_holdout_windows_scored=0,H40_gripper_max_abs=opening_error,absolute_clock_max_abs=absolute_error,consumed_row_clock_max_abs_s=progress_error)

def consume_row(observer,row):
    consumed=row['reference_consumed']
    if len(consumed)!=25 or row['physics_ticks']!=25:raise ValueError('Complete recorded25-tick source atom required')
    before=np.asarray(row['controller_memory']['finger_targets']);after=np.asarray(row['controller_memory_after']['finger_targets'])
    if before.shape!=(2,)or after.shape!=(2,)or not np.isfinite(np.r_[before,after]).all():raise ValueError('Actual two-finger controller evidence required')
    lo,hi=observer.interval();error=float(max(np.max(np.abs(lo-before)),np.max(np.abs(hi-before))))
    for i,c in enumerate(consumed):
        if abs(c['time_s']-(row['t0']+i*.002))>1e-9:raise ValueError('Consumed2ms command clock differs')
        observer.consume(float(c['eef10'][9]))
    lo,hi=observer.interval();error=max(error,float(np.max(np.abs(lo-after))),float(np.max(np.abs(hi-after))))
    return error

def run(root,manifest,manifest_sha,replay,replay_sha,controller_source):
    start=time.monotonic();cpu=time.process_time();root=Path(root);manifest=Path(manifest)
    controller_pins={'scripts/pi0291_arx5_data/engine.py':'623dc5ebf59fd9c470152ee3d60bb15b11ea554ef4b41bf911e62c20dec46c30','scripts/pi0291_arx5_data/config.py':'debc6d9d0358bc6dcf5a1107c5723bb6eb8ff4b396fc35f7901fb67f771f593c'}
    for name,pin in controller_pins.items():
        if digest(Path(controller_source)/name)!=pin:raise ValueError('Original controller/reset/limiter source differs:'+name)
    if digest(manifest)!=manifest_sha:raise ValueError('Original audited prospective manifest differs')
    pins=json.loads(manifest.read_text())['files'];used={}
    def checked(relative):
        p=root/relative
        if relative not in pins or digest(p)!=pins[relative]:raise ValueError('Saved source bytes differ:'+relative)
        used[relative]=pins[relative];return p
    def rows(relative):
        p=checked(relative);r,events=load_rows(dict(path=str(p),sha256=pins[relative]))
        if events or not r or any(v['terminal']or v['truncated']for v in r):raise ValueError('Complete original source records required')
        return r
    original=json.loads(checked('RESULT.json').read_text());validate_population(original,json.loads(checked('CARDS.json').read_text()))
    numeric=audit_numeric(replay,replay_sha);scenes=[];total_prefix=total_branch=0;maximum=0.
    for scene in range(40,48):
        prefix=rows(f'scene-{scene:02d}/construction/PREFIX.pkl.gz')
        if abs(prefix[0]['t0'])>1e-12 or prefix[0]['controller_memory']['steps']!=0:raise ValueError('Known original reset/complete command prefix required')
        observer=SourceFingerTarget(known_reset=True);times=[];states=[];prefix_error=0.
        for row in prefix:
            if times and abs(row['t0']-times[-1])>1e-9:raise ValueError('Prefix has missing command time')
            prefix_error=max(prefix_error,consume_row(observer,row));times.append(row['t1']);states.append(observer.interval())
        parts=[];total_prefix+=len(prefix)
        for name in('hold','release','slow_open','delay2','delay4'):
            future=rows(f'scene-{scene:02d}/supported_release/candidate-{name}.pkl.gz')
            if len(future)!=16:raise ValueError('All16 existing actual rows required')
            indices=[i for i,t in enumerate(times)if abs(t-future[0]['t0'])<1e-9]
            if len(indices)!=1:raise ValueError('Actual command prefix must join saved anchor')
            index=indices[0];lo,hi=states[index];branch=SourceFingerTarget(known_reset=True);branch.lower=lo.copy();branch.upper=hi.copy()
            burn=prefix[max(0,index-7):index+1]
            if len(burn)!=8:raise ValueError('Exactly8 causal observed history rows required')
            unknown=SourceFingerTarget()
            for row in burn:
                for c in row['reference_consumed']:unknown.consume(float(c['eef10'][9]))
            low,high=unknown.interval();start_targets=np.asarray(future[0]['controller_memory']['finger_targets'])
            if np.any(start_targets<low-1e-12)or np.any(start_targets>high+1e-12):raise ValueError('Commands-only interval excludes actual controller')
            error=0.;curve=[]
            for k,row in enumerate(future):
                error=max(error,consume_row(branch,row));target=branch.interval()[0]
                curve.append(dict(horizon=k+1,known_reset_reconstructed_target_fraction=float(target.mean()/.044),measured_opening=row['state_after']['gripper_fraction'],measured_speed_fraction_s=row['state_after']['gripper_speed'],actual_grasp=bool(row['state_after']['native_grasp'])))
            parts.append(dict(candidate=name,anchor_time_s=future[0]['t0'],initial_measured_opening=future[0]['state_before']['gripper_fraction'],reconstructed_initial_target_fraction=float(lo.mean()/.044),burn8_commands_only_target_fraction_interval=[float(low.mean()/.044),float(high.mean()/.044)],controller_target_max_abs_error_m=error,curve=curve));total_branch+=len(future);maximum=max(maximum,error,prefix_error)
        scenes.append(dict(scene=scene,prefix_rows=len(prefix),prefix_controller_target_max_abs_error_m=prefix_error,candidates=parts))
    if maximum>1e-12:raise ValueError('Reconstructed deterministic limiter differs from actual saved controller')
    return dict(schema='WM02_SOURCE_ACTUATOR_MEMORY_AUDIT_V1',scope='POSTHOC already-opened DEV command-history audit; reconstructs only internal references, not physical finger/contact motion. Unknown-reset interval uses commands only, so it does not prove full state/history non-observability.',original_collection_source_sha=original['source_sha'],controller_source_sha256=controller_pins,replay_sha256=replay_sha,stream_manifest_sha256=manifest_sha,consumed_files_sha256=used,numeric_action_clock=numeric,parents=8,branch_candidates=40,existing_prefix_rows=total_prefix,existing_branch_rows=total_branch,controller_target_max_abs_error_m=maximum,scenes=scenes,new_physics=0,new_inference=0,new_optimizer_updates=0,closed_form_fits=0,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in('root','manifest','replay','output','controller-source'):p.add_argument('--'+key,type=Path,required=True)
    for key in('manifest-sha256','replay-sha256'):p.add_argument('--'+key,required=True)
    a=p.parse_args(argv);v=run(a.root,a.manifest,a.manifest_sha256,a.replay,a.replay_sha256,a.controller_source);write_json(a.output,v)
    print(json.dumps({k:v[k]for k in('parents','branch_candidates','existing_prefix_rows','existing_branch_rows','controller_target_max_abs_error_m','new_physics','new_inference','new_optimizer_updates','CPU_s','wall_s')}));return 0
if __name__=='__main__':raise SystemExit(main())
