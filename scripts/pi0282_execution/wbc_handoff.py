"""Read-only PI0281 WBC evidence handoff. No simulator/actor/model imports."""
import json,os,shutil,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from audit import ROOT,BASE,identity,sha,save

OLD=BASE/'PI0281'

def main():
    out=ROOT/'wbc_handoff';out.mkdir(parents=True,exist_ok=True)
    bindingpath=BASE/'PI0280/evidence/WBC_BINDING.json';binding=json.loads(bindingpath.read_text())
    original_index=[]
    for r in binding['files']:
        p=Path(r['path']);row=dict(r,exists=p.exists(),readback='SEALED_PRIOR_HASH_INDEX_NOT_REHASHED')
        if p.exists():
            row.update(current_bytes=p.stat().st_size,current_mtime_ns=p.stat().st_mtime_ns)
            assert p.stat().st_size==r['bytes'],('Bound original changed size',str(p))
            if p.stat().st_size<=1<<20:
                row['current_sha256']=sha(p);row['readback']='CURRENT_SMALL_FILE_HASH_MATCH';assert row['current_sha256']==r['sha256']
        original_index.append(row)
    byname={r['name']:r for r in binding['files']}
    snapshots=[]
    copied=[]
    small=[bindingpath,BASE/'PI0280/design/WBC_CONFIG.json',OLD/'wbc/CPU_PROJECTION.json',OLD/'wbc/REPAIR_CARD.json',OLD/'wbc/REPAIR_INSTALLATION_DEFECT.json',OLD/'wbc/REPAIR_INSTALLATION_CPU_CHECK.json',OLD/'wbc/WBC_VIDEOS.json',OLD/'wbc/FIXTURE_READOUT.json',OLD/'execution/EXECUTED_WBC_SOURCE_BYTES.json']
    small += [Path(byname[k]['path']) for k in ('native_urdf','native_config','compatibility_shim','scripts/native_wbc_benchmark.py','scripts/stage21_native_receding_horizon.py','wbc_wrapper','wbc_env','legged_robot','trajectory_module')]
    small += [Path(byname['wbc_env']['path']).parents[3]/'go1_gym/utils/math_utils.py',OLD/'audit/code/audit_wbc.py',OLD/'audit/code/audit_saved.py']
    small += list((OLD/'audit').glob('WBC-native*.json'))+list((OLD/'audit').glob('WBC-*-engineering-001.json'))
    small += [p for p in (OLD/'code').glob('wbc*.py')]
    for p in small:
        if not p.exists():continue
        dest=out/'metadata'/(sha(p)[:12]+'-'+p.name);dest.parent.mkdir(exist_ok=True);shutil.copyfile(p,dest);copied.append(dict(original=identity(p),copy=identity(dest)))
    videos=json.loads((OLD/'wbc/WBC_VIDEOS.json').read_text())['videos']
    for d in sorted((OLD/'wbc').iterdir()):
        if not d.is_dir() or not (d/'RESULT.json').exists() or not (d/'rows.json').exists():continue
        result=json.loads((d/'RESULT.json').read_text());rows=json.loads((d/'rows.json').read_text());name=d.name
        auditpath=OLD/'audit'/('WBC-'+name+'.json');audit=json.loads(auditpath.read_text()) if auditpath.exists() else None
        bindfile=d/'REPAIR_RUNTIME_BINDING.json';repair=json.loads(bindfile.read_text()) if bindfile.exists() else None
        with np.load(d/'reference.npz',allow_pickle=False) as f:ref={k:f[k] for k in f.files}
        arrays={
            'reference_timestamp_s':ref['timestamp'], 'raw_reference_world_position_m':ref['position'],
            'requested_chunk_position_m':ref['position']-ref['anchor'],'requested_world_quaternion_xyzw':ref['quaternion'],'anchor_world_position_m':ref['anchor'],
            'actual_sim_time_s':np.array([r['time'] for r in rows]),'actual_task_time_s':np.array([r['time']-result['initialization_s'] for r in rows]),
            'installed_position_world_m':np.array([r['installed'] for r in rows]),'installed_quaternion_world_xyzw':np.array([r['installed_quaternion'] for r in rows]),
            'actual_native_TCP_world_m':np.array([r['native_TCP'] for r in rows]),'actual_native_quaternion_world_xyzw':np.array([r['native_quaternion'] for r in rows]),
            'actual_URDF_TCP_world_m':np.array([r['TCP'] for r in rows]),'actual_URDF_TCP_rotation':np.array([r['rotation'] for r in rows]),
            'actual_root_state':np.array([r['root'] for r in rows]),'actual_joint_position_SI':np.array([r['q'] for r in rows]),
            'native_orientation_proxy':np.array([r['orientation_error_proxy'] for r in rows]),'progress_m':np.array([r['progress'] for r in rows]),
            'time_law_progress_m':np.array([r['s_ref'] for r in rows]),'timing_error_m':np.array([r['timing_err'] for r in rows]),
            'hard_penetration_SI':np.array([list(r['hard_penetration'].values()) for r in rows]),'soft_margin_SI':np.array([r['telemetry']['joint_limit_margin'] for r in rows])}
        native=Rotation.from_quat(arrays['actual_native_quaternion_world_xyzw']);installed=Rotation.from_quat(arrays['installed_quaternion_world_xyzw'])
        theta=(installed*native.inv()).magnitude();arrays['true_sameTCP_SO3_error_rad']=theta
        arrays['native_proxy_norm_dimensionless']=np.linalg.norm(arrays['native_orientation_proxy'],axis=1)
        dest=out/(name+'-poses.npz');np.savez_compressed(dest,**arrays)
        selected={0,max(0,result.get('controlled_ticks',0)//2),max(0,result.get('controlled_ticks',0)-1)}
        actor=[];actorfile=d/'actor_inputs.jsonl'
        if actorfile.exists():
            with actorfile.open() as f:
                for line in f:
                    a=json.loads(line)
                    if a['tick'] in selected:actor.append(a)
        actorarr=dict(tick=np.array([a['tick'] for a in actor]),obs_history=np.array([a['obs_history'] for a in actor],dtype=np.float64),
            preview_s=np.array([a['preview_s'] for a in actor]),preview_position=np.array([a['preview_position'] for a in actor]),preview_quaternion=np.array([a['preview_quaternion'] for a in actor]))
        actorout=out/(name+'-representative-actor.npz');np.savez_compressed(actorout,**actorarr)
        original_files=[identity(p) for p in d.iterdir() if p.is_file() and p.name in ('RESULT.json','DONE.json','START.json','rows.json','reference.npz','installation.json','actor_inputs.jsonl','actor_outputs.jsonl','frames.npz','REPAIR_RUNTIME_BINDING.json','IO_COMPLETENESS.json','observation_layout.json')]
        video=next((v for v in videos if v['condition']==name),None)
        if video:
            p=Path(video['video']);assert p.exists() and sha(p)==video['sha256'];video=dict(video,current_identity=identity(p),delivery='REFERENCE_ONLY_EXISTING_REAL_VIDEO_NO_NEW_RENDER')
        tail=(arrays['actual_task_time_s']>=8)&(arrays['actual_task_time_s']<=10+1e-5)
        ep=np.linalg.norm(arrays['installed_position_world_m']-arrays['actual_native_TCP_world_m'],axis=1)
        metrics=dict(position_RMS_m=float(np.sqrt(np.mean(ep[tail]**2))),position_max_m=float(ep[tail].max()),SO3_RMS_rad=float(np.sqrt(np.mean(theta[tail]**2))),SO3_max_rad=float(theta[tail].max()),samples=int(tail.sum())) if tail.any() else None
        if audit and metrics:
            assert abs(metrics['SO3_RMS_rad']-audit['rotation_final2s_rad']['rms'])<1e-10
            assert abs(metrics['position_RMS_m']-audit['position_final2s_m']['rms'])<1e-10
        receipts=[]
        for p in (d/'REPAIR_RUNTIME_BINDING.json',d/'START.json',d/'installation.json',d/'observation_layout.json'):
            if p.exists():receipts.append(dict(identity=identity(p),value=json.loads(p.read_text())))
        snapshots.append(dict(condition=name,result_status=result['status'],result=result,
            repair_consumption=repair or 'NO_REPAIR_RUNTIME_BINDING_SAVED',repair_annotation='TRUE_REPAIR_CONSUMED' if repair and repair.get('bound') else ('NATIVE' if name.startswith('native') else 'ENGINEERING_REPAIR_NOT_INSTALLED'),
            final2s=metrics,quality=audit.get('quality') if audit else None,prior_independent_readback=identity(auditpath) if audit else None,
            pose_bundle=identity(dest),representative_actor_bundle=identity(actorout),representative_actor_ticks=sorted(selected),originals=original_files,video=video,receipts=receipts,
            final_progress_m=float(arrays['progress_m'][-1]),final_time_reference_m=float(arrays['time_law_progress_m'][-1]),final_timing_error_m=float(arrays['timing_error_m'][-1]),
            proxy_sin_half_angle_abs_max=float(abs(arrays['native_proxy_norm_dimensionless']-np.sin(theta/2)).max()),
            hard_penetration_max_SI=float(arrays['hard_penetration_SI'].max()),soft_margin_min_mixed_SI=float(arrays['soft_margin_SI'].min())))
    intake=Path('/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage13')
    intakeinventory={sub:[dict(name=p.name,bytes=p.stat().st_size,is_dir=p.is_dir()) for p in (intake/sub).iterdir()] for sub in ('intake','manifests','extracted') if (intake/sub).exists()}
    v=dict(schema='PI0282_SENIOR_WBC_HANDOFF_V1',status='SEALED_EXISTING_EVIDENCE_HANDOFF_TARGET_BLOCKED',authority_release='8fdaab73b8b6686409317d9562833235c2839acc',canonical=6043060523,
        source=identity(bindingpath),frozen_original_index=original_index,core_metadata_copies=copied,fixtures=snapshots,
        upgraded_WBC=dict(status='UNRECEIVED',inventory_scope=str(intake),observed_entries=intakeinventory,owner_delivery_path=None,ABI_comparison='UNOBSERVED_NO_DELIVERED_UPGRADE',new_execution=0),
        normalizer=dict(identity='Original loader/config/shim/HistoryWrapper are indexed and copied; actor receives saved full obs_history. No substitute normalizer or fitted stats introduced.',
            normalization_trace='Resolved native WBC_CONFIG plus wrapper get_arm_observations and native actor loader source; no new normalizer evaluation or checkpoint tensor loading',
            learned_normalizer_load='Original safe_load_actor loads only actor_body and actor_history_encoder prefixes. Observation scales/clipping come from native config/source; checkpoint non-inference keys are not silently treated as an executed normalizer.'),
        sameTCP='URDF root→x5_link6 plus identity-rotation offset [.1424,0,.0001057]m; native quaternion is link6 quaternion; all prior independent readbacks indexed.',
        orientation_proxy='Native sign-corrected quaternion vector part, dimensionless norm sin(theta/2). Report true SO3 angle in radians; proxy is unchanged actor input.',
        requested_semantics='Requested chunk XYZ is raw world reference minus fixed anchor; installed rows are evaluated at native time-law progress. Actual TCP/URDF FK remain distinct; preview uses spatial native progress.',
        frozen_weights_software='Frozen delivered arm/dog checkpoint hashes kept separate from wrapper/one progress repair implementation and actual per-attempt runtime consumption receipts.',
        target_block='All actually consumed repair fixtures fail SO3 RMS<=.15/max<=.30; no target composition or high-level compensating training qualification.',
        original_files_unchanged=True,new_model_calls=0,new_actor_calls=0,new_physics_steps=0,new_render_calls=0,new_teacher_calls=0,new_training_updates=0,selected='NONE',deployment='NONE')
    save('wbc_handoff','HANDOFF.json',v)
    lines=['# PI0281 已保存 WBC 证据交接','', 'PI0282只读交接。升级版 UNRECEIVED；没有执行新模型、actor、物理或渲染。', '', '|实际尝试|修复接入|末2s位置RMS mm|真实SO3 RMS rad|资格|','|---|---|---:|---:|---|']
    for r in snapshots:
        m=r['final2s'];lines.append(f"|{r['condition']}|{r['repair_annotation']}|{m['position_RMS_m']*1000:.3f}" if m else f"|{r['condition']}|{r['repair_annotation']}|UNOBSERVED")
        lines[-1]+=f"|{m['SO3_RMS_rad']:.6f}|{r['quality']}|" if m else f"|UNOBSERVED|{r['quality']}|"
    lines += ['', 'HANDOFF.json 给出每个原件及真实视频路径/SHA、actor/资产/软件身份与修复实际调用回执。poses.npz 包含完整已保存时间、requested/raw/installed/actual姿态、关节、真SO3、原代理、progress/time-law和硬软限；representative-actor.npz 仅原first/middle/last实际history/preview，完整actor日志按原件索引引用。大模型与完整actor日志未复制。', '', '原代理范数为无量纲 sin(theta/2)，不等于弧度角误差。full10s不是资格PASS。真正return修复完成10s仍有progress残差及姿态FAIL；源任务学习成功不能解除这个目标执行器阻断。', '', '重现算术：打开poses.npz，SO3=Rotation.from_quat(installed)*Rotation.from_quat(actual).inv()的magnitude；final2s用actual_task_time_s∈[8,10]；原actor pack/history/FK验算脚本及既有审计SHA见索引。不得用此包启动WBC或新版本。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=v['status'],fixtures=len(snapshots),upgrade='UNRECEIVED',new_execution=0)))

if __name__=='__main__':
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    main()
