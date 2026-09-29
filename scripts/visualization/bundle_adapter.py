"""Finite local evidence export, rendering, verification and deterministic packaging."""
import csv,html,json,shutil,subprocess,sys,zipfile
from pathlib import Path
from eef_video_workflow import read,write,sha,rows,decode,saved_path,scientific_key,FFMPEG,AUTH
from actual_video import render as actual

def export(root):
    from recorded_adapter import BASE,copy_bound,stage,cpu
    from preparation_adapter import stage as preparation
    out=root/'delivery';receipt=read(root/'source_hf_readback.json');entries=receipt['local']['entries']
    source=root/'inputs/PI0252'
    if not source.exists():source=BASE/'PI0252/pre_full_sim'
    for item in entries:
        p=source/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
        # Only own synthetic observations/results. Private runtime source stays at its producer.
        if item['path'].startswith(('G/A/','controls/')) and p.suffix in ('.mp4','.npz','.json','.jsonl'):
            if not item['path'].startswith('controls/A_attempt1/'):
                copy_bound(p,out/'actual_evidence'/item['path'],item['sha256'])
    for label,relative in [('G','G/A/output'),('expert_A','controls/A_attempt2'),('expert_B','controls/B_attempt1'),('expert_C','controls/C_attempt1'),('expert_D','controls/D_attempt1')]:
        copy_bound(source/relative/'actual_body_wrist.mp4',out/'originals'/f'{label}_actual_body_wrist.mp4')
    copy_bound(root/'source_hf_readback.json',out/'provenance/PI0252_fixed_HF_readback.json')
    preparation(root);stage(root);cpu(root)

def guard_render_imports():
    # Render-only process must never load a model or construct a native simulator.
    class Block:
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in {'torch','transformers','isaacgym','isaacsim'}:
                raise RuntimeError('FORBIDDEN_RENDER_IMPORT: '+fullname)
    block=Block();sys.meta_path.insert(0,block);return block

def immutable_hashes(out):
    files=[p for directory in ('actual_evidence','originals','recorded','preparation') for p in (out/directory).rglob('*') if p.is_file()]
    files += [out/n for n in ('display_matrix.json','query_attempts.jsonl','head_calls.jsonl','query_counters.json') if (out/n).exists()]
    return {str(p.relative_to(out)):sha(p) for p in sorted(files)}

def render(root,kind='all',playback=.25):
    out=root/'delivery';before=immutable_hashes(out);block=guard_render_imports()
    try:
        if kind in ('all','actual'):actual(root,playback)
        if kind in ('all','predictions'):
            from prediction_video import render as predictions
            predictions(root)
        if kind in ('all','preparation'):
            from preparation_adapter import render as preparation
            preparation(root,playback)
    finally:sys.meta_path.remove(block)
    assert before==immutable_hashes(out),'Source evidence changed during render'
    write(out/'render_receipt.json',dict(status='PASS_IMMUTABLE_RENDER_NO_MODEL_OR_SIMULATOR_IMPORT',kind=kind,playback=playback,evidence_hashes=before,new_queries=0,new_physics=0))

def verify_corelab(root):
    from recorded_adapter import cpu
    out=root/'delivery';cpu(root)
    complete=read(out/'completeness.json');matrix=read(out/'display_matrix.json');metrics=read(out/'recorded_cpu_qa.json');counts=read(out/'query_counters.json')
    assert len(complete['cells'])==len(metrics['cells'])==96 and complete['missing']==0
    assert counts['query_attempts']==len(rows(out/'query_attempts.jsonl'))<=104
    assert counts['head_calls']==len(rows(out/'head_calls.jsonl'))<=416
    assert counts['valid_new_queries']==complete['new']<=96
    assert counts['query_attempts']-counts['valid_new_queries']<=8
    assert all(counts[k]==0 for k in ('training','backward','optimizer','physics_steps','wbc_actor_calls'))
    for item in complete['cells']:
        r=item['historical_record'];contract=[r[k] for k in ('condition_identity','source_tensor_sha256','noise_source_sha256','runtime','steps','runner_sha256')]
        assert item['cache_key']==scientific_key(complete['model_bindings'][item['model']]['checkpoint_tree_sha256'],metrics['processor_config_sha256'],metrics['statistics_sha256'],metrics['decoder_source_sha256'],item['source_episode_id'],contract)
        assert sha(saved_path(out,item['path'])/'tensors.npz')==r['tensor_sha256']
    for receipt in (read(out/'preparation_manifest.json')['bindings'],):
        for binding in receipt:assert sha(saved_path(out,binding['path']))==binding['sha256']
    evidence=out/'actual_evidence'
    for f in read(out/'first_assets_receipt.json')['files']:
        if (evidence/f['path']).exists():assert sha(evidence/f['path'])==f['sha256'] and (evidence/f['path']).stat().st_size==f['bytes']
    videos=[decode(p) for p in sorted(out.rglob('*.mp4'))]
    cells={c['id']:c for c in complete['cells']}
    preparation={c['id']:c for c in read(out/'preparation_manifest.json')['examples']}
    for number in ('01','02','03'):
        video=next(v for v in videos if Path(v['path']).parent==out and Path(v['path']).name.startswith(number+'_') and '_1x' not in Path(v['path']).name)
        mapping=read(out/f'{number}_frame_mapping.json')['frames']
        assert len(mapping)==video['decoded_frames'] and [x['output_frame'] for x in mapping]==list(range(len(mapping)))
        assert video['shape']==[720,1280,3] and video['probe']['streams'][0]['pix_fmt']=='yuv420p'
        for f in mapping:
            if number=='01':assert all(p['source_timestamp_s']<=f['reference_elapsed_s']+1e-9 for p in f['panels'])
            elif number=='02':
                for p in f['panels']:
                    assert p['cache_key']==cells[p['cell']]['cache_key'] and p['noise_sha256']==cells[p['cell']]['historical_record']['noise_source_sha256']
            else:
                for p in f['panels']:
                    c=preparation[p['case']];source=next(x for x in c['frames'] if x['path']==p['source_frame'])
                    assert source['usable_as_pre_failure'] and not source['terminal']
                    assert p['missing_before'] or p['source_timestamp_s']<=f['simulation_time_s']+1e-9
    test=read(out/'contract_test_receipt.json');assert test['status']=='PASS'
    encoder=subprocess.check_output([str(FFMPEG),'-version'],text=True).splitlines()[0]
    write(out/'corelab_qa_receipt.json',dict(status='PASS_FULL_DECODE_CLOCKS_IDENTITIES_CPU_METRICS_AND_CONTRACT_TESTS',videos=videos,contract_tests=test,independent_cpu_cells=len(metrics['cells']),encoding_software=encoder,counters=counts,observer_noninterference='NOT_PROVEN_FOR_NEW_RECORDING; new capture and physical runner remain disabled',raw_link6_tcp_correction='UNAVAILABLE in PI0252 archive; historical telemetry retained',quota='UNKNOWN'))
    print('PASS_FULL_VIDEO_AND_EVIDENCE_QA',flush=True)

def package(root):
    from portable_verify import verify
    import numpy as np
    out=root/'delivery';qa=verify(root,manifest=False);complete=read(out/'completeness.json')
    qa['counters']=read(out/'query_counters.json');write(out/'qa_receipt.json',qa)
    videos=[]
    titles=['G 与专家：历史实际执行、参考与跟踪','G / U / FULL：固定尺度的离线预测','准备阶段：缓存修复与物理合格分开判断']
    for i,title in enumerate(titles,1):
        p=next(p for p in out.glob(f'{i:02d}_*.mp4') if '_1x' not in p.name)
        bound=next(v for v in qa['videos'] if Path(v['path'])==p)
        videos.append(dict(path=p.name,title=title,sha256=sha(p),bytes=p.stat().st_size,evidence_class='OFFLINE_PREDICTION_ON_RECORDED_OBSERVATIONS' if i==2 else 'ARCHIVED_PHYSICS_RECORDING',qa=bound))
    summary=dict(status='PI0257_VIDEO_REVIEW_V2_READY',authority_commit='1e760ba055106cb85b508711f1466d1cb78a9656',display_cells=96,v1_reused=88,v1_new=8,v1_counters=qa['counters'],v2_counters=read(out/'revision_counters.json'),models=complete['model_bindings'],deployment_selected_checkpoint='NONE',historical_verdicts={'G2000':'physical TRAIN-A VALID FAIL','U4608':'old-support FAIL','FULL10000':'selection NONE','PI0256':'A/B/Bfix 0/6; C not released'},capabilities={'offline_export_render':'IMPLEMENTED','new_physics_guard':'GUARD_STUB_ONLY','actual_receding_runner':'NOT_CONNECTED','portable_verify':'PACKAGE_ONLY_MEDIA_HASH_TIME_SAVED_ARRAYS','corelab_verify':'FULL_PROCESSOR_DECODER_GATE_REQUIRES_FROZEN_CORELAB'},legacy_runner_label='IMPLEMENTED_GUARDED_NOT_EXECUTED retained only in v1 archive; not proof of an implemented runner',independent_TCP_correction='UNAVAILABLE for PI0252; HISTORICAL_AS_RECORDED',wall_time_inference_latency='No new inference; v1 isolated per-query wall time unavailable')
    write(out/'summary.json',summary);write(out/'video_manifest.json',videos)
    records=read(out/'recorded_cpu_qa.json')['cells'];fields=['id','model','cohort','noise','source_episode_id','status','gate','metric_origin']+sorted(records[0]['metrics'])
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader()
        for r in records:writer.writerow({**{k:r[k] for k in fields if k in r},**r['metrics']})
    stats=[]
    for model in complete['model_bindings']:
        for cohort in ('O64','P64'):
            for noise in ('B0','B1'):
                values=[r['metrics']['xyz_rmse_m']*1000 for r in records if (r['model'],r['cohort'],r['noise'])==(model,cohort,noise)]
                stats.append(dict(model=model,cohort=cohort,noise=noise,windows=len(values),xyz_median_mm=float(np.median(values))))
    write(out/'display_subset_summary.json',dict(scope='Fixed eight windows per cohort/noise; not full64 metrics or success rate',rows=stats))
    cards=''.join(f'<section id="section{i}"><h2>{html.escape(v["title"])}</h2><p>{v["evidence_class"]}</p><video id="video{i}" controls preload="metadata" src="{v["path"]}"></video><p><a href="{v["path"]}" download>下载 MP4</a></p></section>' for i,v in enumerate(videos,1))
    navigation=[]
    for c in read(out/'02_chapters.json'):
        links=f'<a href="{c["zoom_card"]}">局部放大三模型卡（有独立刻度）</a> · '+' · '.join(f'<a href="{x["card"]}">{x["id"].split("_")[-2]} 局部静态卡</a>' for x in c['cells'])
        navigation.append(f'<details><summary><button onclick="jump({c["start_s"]})">跳到 {c["start_s"]} 秒</button> {c["cohort"]} / {c["noise"]} / 第 {c["chapter"]} 组</summary><p>{html.escape(c["source"])}</p><p>{links}</p><a href="{c["card"]}"><img loading="lazy" src="{c["card"]}" alt="同输入同噪声的三模型静态证据卡"></a></details>')
    originals=''.join(f'<li><a href="originals/{html.escape(p.name)}">{html.escape(p.name)}</a></li>' for p in sorted((out/'originals').glob('*.mp4')))
    table=''.join(f'<tr><td>{s["model"]}</td><td>{s["cohort"]}</td><td>{s["noise"]}</td><td>{s["xyz_median_mm"]:.2f}</td></tr>' for s in stats)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>PI0257 v2 视频证据</title>
<style>body{max-width:1100px;margin:32px auto;padding:0 20px;background:#111820;color:#e8edf4;font:17px/1.65 system-ui}a{color:#83c9ff}video,img{width:100%;background:#000}section,details{margin:28px 0}h1,h2{line-height:1.3}.note{border-left:4px solid #efb25b;padding:12px 18px;background:#202b38}button{padding:8px;background:#324a60;color:white;border:1px solid #8ca8be;cursor:pointer}td,th{padding:6px 18px;border-bottom:1px solid #43515f}summary{cursor:pointer}</style>
<h1>PI0257 v2：把尺度、姿态、跟踪和任务结果讲清楚</h1>
<p>本轮只用已保存证据重新绘图：新增模型查询、训练、反传、物理步、WBC 调用均为 0。原 96 格不变：v1 的 88 格历史复用与 8 格补推理仍按原来源保留。</p>
<div class="note"><strong>三个判定分别看：</strong>轨迹 gate 是否允许、WBC 是否跟上、任务是否命中目标。Gate WARN 不是任务成功。G 旧实际执行 FAIL；U 旧支持 FAIL；FULL 选择 NONE。专家背景不是模型驱动结果。</div>
<h2>如何看新版</h2><p>① 实际录像旁加入世界 XY/XZ 轨迹和随时间变化的距离／跟踪误差。97 个控制 tick 与 40 点 H40 原始输出分开命名，旧字段原封保留。所有 PI0252 TCP 数值仍为 HISTORICAL_AS_RECORDED。</p>
<p>② 全部 32 组使用同一物理尺度与数字刻度，改用 XY/XZ 视图和三轴误差曲线；同时显示 SO(3)、首点姿态跳变、峰值线速度与具体 gate issue。96 格都为 INITIAL_ROTATION_DISCONTINUITY_WARN，并非线速度拒绝。</p>
<p>③ 准备阶段展示姿态误差、角速度及连续合格时间。源码／CPU 缓存修复证据通过，与实际姿态跟踪／准备资格仍失败是两件事。时间走到 timeout 不能当成稳定通过。</p>'''+cards+'''
<h2>原速与慢放</h2><p>主实际录像使用 0.25× 慢放，真实时钟始终可见。<a href="01_G_vs_expert_actual_1x.mp4">01 原速</a> · <a href="03_preparation_failures_and_repairs_1x.mp4">03 原速</a>。02 的 64 秒版保留全部 32 组，每组 2 秒；可用下面章节定位并打开静态卡。</p>
<h2>02 章节与 96 格静态卡</h2>'''+''.join(navigation)+'''
<h2>固定展示子集的 XYZ 中位数（mm）</h2><p>每个 cohort／noise 仅 8 个窗口，不是完整 64 窗口结果，也不是成功率。它说明此展示集合上的学习收益与旧支持保持代价。</p><table><tr><th>模型</th><th>集合</th><th>噪声</th><th>XYZ 中位数 mm</th></tr>'''+table+'''</table>
<h2>工作流的真实能力</h2><p>离线导出／重渲染已实现；新物理入口只是 <code>GUARD_STUB_ONLY</code>，实际 receding 执行器尚未接入。保护保持开启。</p><p><code>verify-portable</code>（或 <code>verify</code>）仅使用包内媒体、hash、时间和保存数组复算，不导入 Corelab 模块。<code>verify-corelab</code> 另外复核冻结 processor／decoder／完整 gate 链，仍需要原 Corelab 环境。</p>
<h2>原始录像与历史版本</h2><ul>'''+originals+'''</ul><p><a href="https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/tree/f660d94fd100c3a76304c3e05bf5b4d5d4ca561e/evidence/PI0257/corelab-video-workflow-v1-20260929">v1 固定归档</a> · <a href="legacy_v1/02_G_U_FULL_recorded_input_predictions.mp4">原 v1 64秒证据版</a>。原始 PI0255 视频可能包含 autoreset 后的画面；新版诊断对照明确排除了这些失败姿态误用。</p>
<h2>数据和 QA</h2><p><a href="summary.json">能力及计数</a> · <a href="metrics.csv">96 格指标</a> · <a href="01_metrics_v2.json">H40／控制 tick 分栏指标</a> · <a href="01_metrics.json">保留的旧指标</a> · <a href="03_preparation_metrics_v2.json">准备阶段曲线数据</a> · <a href="qa_receipt.json">包内 QA</a> · <a href="corelab_qa_receipt.json">Corelab 完整链 QA</a> · <a href="source_manifest.json">文件 SHA256 / size</a></p>
<script>function jump(t){const v=document.getElementById('video2');v.currentTime=t;v.scrollIntoView({behavior:'smooth'});v.pause();}</script></html>'''
    (out/'index.html').write_text(page,encoding='utf-8')
    source=Path(__file__).parent;(out/'code').mkdir(exist_ok=True)
    names=('eef_video_workflow.py','recorded_adapter.py','preparation_adapter.py','prediction_video.py','bundle_adapter.py','actual_video.py','plotting.py','portable_verify.py','portable_numeric.py','evidence_so3.py','test_video_evidence_contract.py','README.md')
    for name in names:
        p=source/name
        if p.resolve()!=(out/'code'/p.name).resolve():shutil.copyfile(p,out/'code'/p.name)
    files=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'source_manifest.json']
    write(out/'source_manifest.json',dict(schema='PI0257_PORTABLE_BUNDLE_V2',files=files,rights='Own synthetic evidence and public workflow/NumPy SO3 code; no private WBC/asset source, tokenizers, weights or credentials',provenance='v1 evidence retained; legacy_v1 contains old presentation. No new inference or physics.'))
    destination=root/'PI0257_video_evidence_v2.zip'
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                info=zipfile.ZipInfo('delivery/'+p.relative_to(out).as_posix(),(2026,9,29,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16;archive.writestr(info,p.read_bytes())
    write(root/'package_receipt.json',dict(path=str(destination),bytes=destination.stat().st_size,sha256=sha(destination),members=len(files)+1))
    print(json.dumps(read(root/'package_receipt.json')),flush=True)
