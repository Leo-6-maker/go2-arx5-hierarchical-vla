"""Read closed virtual arrays once; add prefix/support diagnostics without rescoring."""
from cpu_common import *
import ast
import io
import time
import resource
import numpy as np


def matrices(x):
    u=x[...,:3].astype(float);u/=np.linalg.norm(u,axis=-1,keepdims=True)
    v=x[...,3:].astype(float);v-=np.sum(u*v,axis=-1,keepdims=True)*u
    v/=np.linalg.norm(v,axis=-1,keepdims=True)
    return np.stack((u,v,np.cross(u,v)),axis=-2)


def angle(a,b):
    return np.arccos(np.clip((np.sum(a*b,axis=(-2,-1))-1)/2,-1,1))


def main():
    restrict();started=time.monotonic()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scene=read(BASE/'PI0267/design/SCENE_CARD.json');anchor=np.array(scene['policy_anchor_world_m'])
    snapshot=read(OUT/'SOURCE_SNAPSHOT.json');sources={};rows=[];episodes=[];plot_data={}
    source=BASE/'PI0267/code/pi0267_r1_cpu.py'
    fn=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='independent_entry')
    ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns);entry=ns['independent_entry']
    sources[str(source)]=sha(source)
    meta=read(OUT/'TRAIN_NUMERIC_INDEX.json')
    with np.load(PREP/'cache/train_numeric.npz') as z:train_s=z['state'].astype(float)
    train_p=train_s[:,:3]+anchor;train_r=matrices(train_s[:,3:9]);support={}
    for task in ('M1','M3'):
        for command in (0,1):
            for recovery in (False,True):
                support[task,command,recovery]=np.array([i for i,m in enumerate(meta) if m['task']==task and m['command']==command and bool(m['variant'])==recovery])
    def load(p,expected=None):
        b=p.read_bytes();h=hashlib.sha256(b).hexdigest()
        if expected:assert h==expected,(p,h,expected)
        sources[str(p)]=h
        return np.load(io.BytesIO(b))
    for point in ('donor','N4096','R4096'):
        part=snapshot['completed_virtual_partitions'][point];assert part['completed']==64 and part['eligible_for_complete_point_analysis']
        for path,h in sorted(part['results'].items()):
            p=Path(path);assert sha(p)==h;sources[path]=h;r=read(p);assert r['validity']=='VALID'
            with load(p.parent/'trajectory.npz',r['arrays_sha256']) as z:
                t=z['time'];ap=z['actual_p'];ar=z['actual_r'];ip=z['installed_p'];ir=z['installed_r'];spec=json.loads(str(z['spec_json']))
            goal=np.array(spec['goal']);goal_r=np.array(spec['goal_rotation'])
            dist=np.linalg.norm(ap-goal,axis=1);ori=angle(ar,goal_r);good=(dist<=.035)&(ori<=.10)
            hold=r['scorer']['time_to_success_s'];after=np.flatnonzero(t>=hold-1e-9) if hold is not None else np.array([],int)
            exits=after[~good[after]];aentry=ientry=None
            target=goal
            if r['task']=='M3':
                _,aentry,_=entry(t,ap,ar,spec);_,ientry,_=entry(t,ip,ir,spec)
                assert aentry==r['scorer']['entry_index']
                d=1 if r['command']==0 else -1;a=-d*3*np.pi/4
                target=np.array(spec['center'])+[spec['radius']*np.cos(a),spec['radius']*np.sin(a),0]
            gates={g['replan']:g['result']['decision'] for g in r['gates']};requested_entry=0;begin=len(rows)
            for q in r['queries']:
                k=q['replan'];tick=20*k;assert tick<len(t);op=Path(q['query']['output']);obs=Path(q['query']['input'])
                with load(obs,q['observation_sha256']) as z:
                    assert np.array_equal(ap[tick],z['position_world']) and np.array_equal(ar[tick],z['rotation_world'])
                    assert abs(float(z['timestamp'])-t[tick])<1e-9
                with load(op,q['prediction_sha256']) as z:pred=z['decoded'][0].astype(float)
                pp=pred[:,:3]+anchor;pr=matrices(pred[:,3:9]);first=pp[0]-ap[tick];prefix=pp[8]-pp[0];total=pp[8]-ap[tick]
                assert np.allclose(total,first+prefix,rtol=0,atol=1e-12)
                req=False
                if r['task']=='M3':
                    _,ei,_=entry(np.arange(40)*.05,pp,pr,spec);req=ei is not None;requested_entry+=req
                stop=min(tick+20,len(t)-1);accepted=gates.get(k)=='ACCEPT'
                row=dict(point=point,case=p.parent.name,family=r['family'],task=r['task'],command=r['command'],backend=r['backend'],replan=k,time_s=float(t[tick]),gate=gates.get(k),
                    first_error_m=float(np.linalg.norm(first)),prefix_motion_m=float(np.linalg.norm(prefix)),commanded_total_m=float(np.linalg.norm(total)),
                    first_so3_rad=float(angle(pr[0],ar[tick])),prefix_so3_rad=float(angle(pr[8],pr[0])),
                    algebra_max_abs=float(np.max(abs(total-first-prefix))),requested_entry=req,
                    target_distance_before_m=float(np.linalg.norm(ap[tick]-target)),target_distance_after_m=float(np.linalg.norm(ap[stop]-target)) if accepted else None,
                    target_progress_m=float(np.linalg.norm(ap[tick]-target)-np.linalg.norm(ap[stop]-target)) if accepted else None,
                    same_tick_tracking_max_m=float(np.linalg.norm(ap[tick:stop+1]-ip[tick:stop+1],axis=1).max()),
                    requested_max_segment_m=float(np.linalg.norm(np.diff(pp,axis=0),axis=1).max()),
                    collision_in_next_prefix=any(tick<c['tick']<=stop for c in r['collision_events']))
                for recovery in (False,True):
                    ids=support[r['task'],r['command'],recovery]
                    dp=np.linalg.norm(train_p[ids]-ap[tick],axis=1);dr=angle(train_r[ids],ar[tick]);metric=np.sqrt((dp/.01)**2+(dr/.06)**2)
                    j=int(np.argmin(metric));m=meta[ids[j]];label='recovery' if recovery else 'nominal'
                    row.update({label+'_support_distance':float(metric[j]),label+'_nearest_family':m['family'],label+'_nearest_index':m['index']})
                rows.append(row)
            own=rows[begin:];collision=[]
            for c in r['collision_events']:
                idx=min((c['tick']-1)//20,len(own)-1)
                collision.append(dict(event=c,time_s=float(t[c['tick']]),preceding_query=own[idx]))
            episodes.append(dict(point=point,case=p.parent.name,task=r['task'],command=r['command'],backend=r['backend'],original_success=r['success'],
                stop_reason=r['stop_reason'],queries=len(own),gate_reject=sum(x['gate']=='REJECT' for x in own),collision_events=collision,
                requested_chunks_with_entry=requested_entry,installed_entry_s=None if ientry is None else float(t[ientry]),achieved_entry_s=None if aentry is None else float(t[aentry]),
                first_hold_s=hold,first_exit_s=None if not len(exits) else float(t[exits[0]]),final_inside_original_pose_bounds=bool(good[-1]),
                post_hold_path_m=None if not len(after) else float(np.linalg.norm(np.diff(ap[after],axis=0),axis=1).sum()),
                post_hold_excess_xyz_integral_ms=None if not len(after) else float(np.trapz(np.maximum(dist[after]-.035,0),t[after])),
                post_hold_max_xyz_m=None if not len(after) else float(dist[after].max()),final_xyz_m=float(dist[-1]),final_so3_rad=float(ori[-1]),
                approach_first2s_progress_m=float(np.linalg.norm(ap[0]-target)-np.linalg.norm(ap[min(100,len(t)-1)]-target)),
                same_tick_tracking_max_m=float(np.linalg.norm(ap-ip,axis=1).max()),result_sha256=h,trajectory_sha256=r['arrays_sha256']))
            if r['family']=='pi0265-family-028' and r['command']==0 and r['backend']=='E0':plot_data[point,r['task']]=(t.copy(),dist.copy(),hold)
        print('CLOSED_POINT_READ',point,len(rows),flush=True)
    csv_write('CLOSED_PREFIX_ROWS.csv',rows);write('CLOSED_EPISODE_ROWS.json',episodes);write('CLOSED_SOURCE_HASHES.json',sources)
    groups=[]
    for point in ('donor','N4096','R4096'):
        for task in ('M1','M3'):
            for backend in ('E0','E1'):
                e=[e for e in episodes if (e['point'],e['task'],e['backend'])==(point,task,backend)]
                q=[q for q in rows if (q['point'],q['task'],q['backend'])==(point,task,backend)]
                g=dict(point=point,task=task,backend=backend,episodes=len(e),queries=len(q),success=sum(x['original_success'] for x in e),
                    collisions=sum(bool(x['collision_events']) for x in e),REJECT=sum(x['gate']=='REJECT' for x in q),
                    requested_entry_episodes=sum(x['requested_chunks_with_entry']>0 for x in e),installed_entry=sum(x['installed_entry_s'] is not None for x in e),achieved_entry=sum(x['achieved_entry_s'] is not None for x in e),
                    successful_final_outside=sum(x['original_success'] and not x['final_inside_original_pose_bounds'] for x in e))
                for name in ('first_error_m','prefix_motion_m','commanded_total_m','first_so3_rad','prefix_so3_rad','nominal_support_distance','recovery_support_distance'):
                    g['median_'+name]=float(np.median([x[name] for x in q]))
                groups.append(g)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    for ax,task in zip(axes,('M1','M3')):
        for point in ('donor','N4096','R4096'):
            t,d,h=plot_data[point,task];ax.plot(t,d*1000,label=point)
            if h is not None:ax.axvline(h,alpha=.2)
        ax.set(title=task+' / family028 c0 E0',xlabel='Saved virtual time (s)',ylabel='Actual-to-final-goal distance (mm)');ax.legend()
    fig.suptitle('Fixed metadata representatives; virtual EEF, not WBC; no rescoring');fig.tight_layout();fig.savefig(OUT/'FIXED_CLOSED_TRAJECTORIES.png',dpi=160);plt.close(fig)
    write('CLOSED_SUMMARY.json',dict(groups=groups,rows=len(rows),episodes=len(episodes),max_prefix_identity_abs=max(x['algebra_max_abs'] for x in rows),
        nearest_support_metric='Exact minimum within same task/command TRAIN: sqrt((XYZ distance / .010m)^2 + (SO3 geodesic / .060rad)^2); nominal and recovery separate. No visual-equivalence claim.',
        requested_entry_scope='Each saved H40 independently; does not concatenate unexecuted futures. Installed/achieved use full actual timeline.',
        runtime_s=time.monotonic()-started,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,**ZERO))
    print('COMPLETE_CLOSED_CPU',flush=True)


if __name__=='__main__':main()
