"""Seal the actual R2 data result and its observable-input admission stop."""
import json
from pathlib import Path
import sys
import numpy as np
import cv2
from pi0260_r2 import ROOT,read,write,sha

def run(commit):
    codec=read(ROOT/'dataset/codec_receipt.json');alias=read(ROOT/'evidence/observability_aliases.json')
    actual=read(ROOT/'evidence/actual_processor_aliases.json');geometry=read(ROOT/'review_geometry/receipt.json')
    assert codec['status']=='COMPACT_30_10' and alias['affected_families']==40 and len(actual['pairs'])==40
    assert geometry['status']=='PASS_R2_COLLECTION_READONLY_REPLAY'
    selected=codec['selected_post_codec'];assert set(alias['family_worst'])==set(selected['TRAIN']+selected['KIN-VAL'])
    assert all(x['rows'][0]['identity']==x['rows'][1]['identity'] for x in actual['pairs'])
    out=ROOT/'delivery';out.mkdir(exist_ok=True)
    sys.path.insert(0,str(ROOT/'qa/visualization'))
    from plotting import plot,padded
    images=[]
    for task in ('M1','M3'):
        pair=max((r for r in alias['conflicts'] if r['first']['task']==task),key=lambda r:r['max_xyz_m'])
        canvas=np.full((720,1280,3),24,np.uint8)
        def text(s,x,y,size=.55):cv2.putText(canvas,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,(230,230,230),1,cv2.LINE_AA)
        text('KINEMATIC REFERENCE / IDENTICAL POLICY INPUT / NO MODEL OR PHYSICS',20,28,.65)
        text(pair['first']['sequence'],20,57)
        futures=[]
        for col,side in enumerate(('first','second')):
            row=pair[side];x=20+col*320
            record=read(ROOT/'dataset'/(row['sequence']+'.json'))
            with np.load(ROOT/row['frame']) as z:
                for stream,y in [('body',125),('wrist',365)]:
                    canvas[y:y+192,x:x+192]=cv2.resize(z[stream+'_rgb'][:,:,::-1],(192,192),interpolation=cv2.INTER_NEAREST)
                    text(stream+' actual 96x96 RGB',x,y-12,.42)
            text(f't={row["time"]:.2f}s, same RGB + FK + language',x,85,.39)
            with np.load(ROOT/'dataset'/(row['sequence']+'.npz')) as z:futures.append(z['raw_action'][row['row_index'],:,:3])
        t=np.arange(40)*.05
        for axis in range(3):
            a,b=[f[:,axis] for f in futures]
            plot(canvas,(655,100+axis*175,600,165),[(t,a,(100,210,250)),(t,b,(190,130,250))],(0,1.95),padded(np.r_[a,b],.002),
                'same-input label '+str('XYZ'[axis]),'future seconds','native metres')
        text(f'Future XYZ RMSE {pair["xyz_rmse_m"]*1000:.2f} mm; maximum point gap {pair["max_xyz_m"]*1000:.2f} mm',20,625,.6)
        text('No hidden time/phase input added. This is a DATA observability result, not a model quality result.',20,660,.5)
        name=task+'_same_input_different_future.png';assert cv2.imwrite(str(out/name),canvas)
        images.append(dict(path=name,sha256=sha(out/name),pair=pair))
    counts=read(ROOT/'state/collection_owner.json')['counters']
    result=dict(status='H1_OBSERVABILITY_BLOCKED_EXACT_INPUT_FUTURE_ALIAS',code_commit=commit,
        authority='c5070dc1b6eaff9d3070c92e3a0248ff4c0a83be',old_DEV_FAIL_preserved=True,
        dataset_quantity_geometry_visibility_codec='COMPACT_30_10',TRAIN_families=30,VAL_families=10,
        trained_models=0,canary_updates=0,main_updates=0,quality_queries=0,planned_quality_tuples=1760,
        experiment_complete=False,model_quality='NOT_EVALUATED',H1_admitted=False,
        stop_basis=['PI0258 section B geometry/observability: same permitted observations must not require conflicting futures; structural non-observability blocks that family',
            'PI0260 section4.2 item5; R2 preserves per-example observation/input integrity and forbids hidden phase inputs'],
        observation_conflict=dict(affected_families=40,usable_windows=alias['windows'],unique_allowed_inputs=alias['unique_inputs'],
            same_input_pairs=alias['same_input_pairs'],nonidentical_future_pairs=alias['nonidentical_future_pairs'],
            actual_processor_verified_pairs=40,max_xyz_rmse_m=alias['maximum_xyz_rmse_m'],max_pointwise_xyz_m=alias['maximum_pointwise_xyz_m'],
            type='WITHIN_SAME_COMMAND_SYNTHETIC_TIME_ALIAS',label_roundtrip_tolerance_is_not_model_quality_threshold=True),
        qualification_scope='Quantity, geometry and codec PASS are retained. Exact current-input/future-label mapping is not well-defined; no claim that the 20mm model threshold is mathematically impossible.',
        cause='Fixed residual-tolerant IK holds exactly the same joint state over multiple prescribed times; the clock-driven H40 desired reference continues to change.',
        no_remedy_applied=['no clock/phase/source ID conditioning','no reference smoothing or label replacement','no IK tolerance/time-law/camera change','no family replacement'],
        counters=counts,source_bindings={
            'collection_manifest':sha(ROOT/'handoff/H1_geometry/manifest.json'),
            'geometry_replay':sha(ROOT/'review_geometry/receipt.json'),
            'codec_receipt':sha(ROOT/'dataset/codec_receipt.json'),
            'raw_aliases':sha(ROOT/'evidence/observability_aliases.json'),
            'actual_processor_aliases':sha(ROOT/'evidence/actual_processor_aliases.json')},
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE',next='STOP_BEFORE_MODEL_CANARY_FOR_DATA_OBSERVABILITY_REVIEW',
        natural_RGB_review='No H1 visual signoff claimed; diagnostic source RGB cards retained')
    write(ROOT/'state/H1_OBSERVABILITY_BLOCKED.json',result);write(out/'alias_cards.json',images)
    (out/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>PI0260-R2 data result</title>
<style>body{font:18px system-ui;max-width:1200px;margin:35px auto;padding:20px;background:#141820;color:#e7edf5}img{width:100%}a{color:#8dccff}</style>
<h1>PI0260-R2：实际数据与观测准入结果</h1>
<p>30 TRAIN / 10 KIN-VAL 家族通过几何、可见性及真实 codec 检查。独立保存数组复核通过。</p>
<h2>H1 观测准入停止</h2><p>40 个家族均出现相同 RGB、FK 状态和同一指令，对应不同合成时间的未来标签。
真实 processor 的像素、token、state 等输入字节完全相同。最大未来逐点差 25.91 mm，最大轨迹 XYZ RMSE 差 15.77 mm。</p>
<p>这是数据可观测性问题，不是模型质量 FAIL，也不证明 20 mm 模型门限不可能通过。未加载或训练 Base/G，未运行模型评价。</p>
<p>旧 DEV 4/20 FAIL 保留。未加入时间/phase，不改参考、IK、gate 或相机。selected/deployment=NONE。</p>
<img src="M1_same_input_different_future.png"><img src="M3_same_input_different_future.png">
<p>KINEMATIC REFERENCE / NO MODEL / NO PHYSICS。图片与曲线来自已保存原件。</p>''',encoding='utf-8')
    base=read(ROOT/'handoff/H1_geometry/manifest.json');entries={x['path']:x for x in base['entries']}
    extras=[ROOT/'handoff/H1_geometry/manifest.json',ROOT/'state/H1_OBSERVABILITY_BLOCKED.json']
    for folder in ('dataset','review_geometry','evidence','delivery','result_code'):
        extras += [p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    extras += [ROOT/'runtime/pi0260_r2_inputs.py',ROOT/'runtime/pi0260_r2_observability.py']
    for p in extras:entries[str(p.relative_to(ROOT))]=dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p))
    manifest=ROOT/'handoff/R2_result/manifest.json'
    write(manifest,dict(schema='PI0260_R2_DATA_AND_OBSERVABILITY_RESULT_V1',code_commit=commit,result=result,
        entries=sorted(entries.values(),key=lambda x:x['path'])))
    print(json.dumps(dict(status=result['status'],manifest_sha256=sha(manifest),files=len(entries),source_bytes=sum(x['bytes'] for x in entries.values()))),flush=True)

if __name__=='__main__':run(sys.argv[1])
