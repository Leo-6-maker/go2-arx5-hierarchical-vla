"""Read closed ledgers and receipts. No inference, training, or physics entry point."""
import csv, hashlib, json, statistics, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote

ROOTS={'core':'/home/liuyu-nfs/Projects/go2-arx5/PI0284','hop':'/scratch/e1192548/Projects/go2-arx5/experiments/PI0284'}
OUT=Path('reports/PI0284/final')


def write(name, value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')


def quantile(values, fraction):
    values=sorted(values);i=(len(values)-1)*fraction;a=int(i);b=min(a+1,len(values)-1)
    return values[a]+(values[b]-values[a])*(i-a)


def collect(host):
    script='ROOT='+repr(ROOTS[host])+'\n'+'''from pathlib import Path
import json,collections,hashlib,subprocess
r=Path(ROOT);files={}
tops=['audit','source/evidence','videos','design','endpoint_diagnostics','natural'] if 'liuyu' in ROOT else ['evidence/CTRL2048','evidence/LR5X2048','runtime']
for top in tops:
 for p in (r/top).glob('*.json'):
  if p.stat().st_size<4_000_000:files[str(p.relative_to(r))]=p.read_text()
ledgers={}
for name in ['evidence/quality_budget.jsonl','evidence/model_processes.jsonl','source/evidence/budget.jsonl']:
 p=r/name
 if not p.exists():continue
 rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
 value=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rows=len(rows),kinds=dict(collections.Counter(v.get('kind',v.get('event')) for v in rows)))
 if 'model_processes' in name:
  ended={(v['pid'],v['start_epoch']) for v in rows if v['event']=='end'}
  unmatched=[v for v in rows if v['event']=='start' and (v['pid'],v['start_epoch']) not in ended]
  intervals=sorted((v['start_epoch'],v['epoch']) for v in rows if v['event']=='end')
  used=0.;end=float('-inf')
  for a,b in intervals:used+=max(0.,b-max(a,end));end=max(end,b)
  value.update(interval_union_seconds=used,unmatched_starts=unmatched,records=rows)
 ledgers[name]=value
owned=int(subprocess.check_output(['du','-sx','--block-size=1',str(r.resolve())],text=True).split()[0])
print(json.dumps(dict(files=files,ledgers=ledgers,owned_bytes=owned,quota_headroom='UNKNOWN')))
'''
    result=json.loads(remote(host,script))
    for name,body in result.pop('files').items():
        p=OUT/'originals'/host/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body,encoding='utf8')
    write(host+'_LEDGERS.json',result)
    return result


def load(host,name):
    return json.loads((OUT/'originals'/host/name).read_text(encoding='utf8'))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    existing=OUT/'TERMINAL_RECEIPT.json'
    if existing.exists() and json.loads(existing.read_text(encoding='utf8'))['archival'].startswith('PASS_'):
        print('ALREADY_SEALED_NO_NEW_CALLS');return
    core=collect('core');hop=collect('hop')
    neural=Counter()
    for v in (core,hop):
        neural.update(v['ledgers']['evidence/quality_budget.jsonl']['kinds'])
        assert not v['ledgers']['evidence/model_processes.jsonl']['unmatched_starts']
    physical=core['ledgers']['source/evidence/budget.jsonl']['kinds']
    gpu=sum(v['ledgers']['evidence/model_processes.jsonl']['interval_union_seconds'] for v in (core,hop))
    budget=dict(formal_updates=4096,canary_updates=32,completed_updates=4128,formal_chunks=32768,
                technical_training_recoveries=0,memory_updates=0,technical_rollback_updates=0,
                policy_quality_calls=neural['query'],head_evaluations=neural['head'],physical=physical,
                valid_quality_retries=0,technical_physical_replacements=0,CONFIRM=0,
                GPU_process_interval_union_seconds=gpu,GPU_process_interval_union_hours=gpu/3600,
                owned_bytes_before_compact_archive=core['owned_bytes']+hop['owned_bytes'],
                owned_GiB_before_compact_archive=(core['owned_bytes']+hop['owned_bytes'])/(1<<30),
                CPU_core_hours='UNOBSERVED_AS_AGGREGATE',quota_headroom='UNKNOWN',
                CPU_observed_PBS_cput='01:45:19',PBS_walltime='01:43:40',
                CPU_note='PBS cput is observed process CPU for the job, not a complete two-host CPU total; storage/archive/CPU-helper usage has no complete common telemetry.',
                model_downloads=0,new_teacher_calls=0,new_physical_training_data=0,WBC_target_physics_hardware=0)
    assert neural['query']<=6400 and neural['head']<=102400 and gpu<=12*3600
    for name,cap in [('starts',64),('control',45000),('physics',230000),('rgb',45000),('dispatch',90000)]:assert physical[name]<=cap
    assert budget['owned_GiB_before_compact_archive']<120
    write('BUDGET.json',budget)
    offline=[];closed=[];points={}
    for point in ('CTRL1024','CTRL2048','LR5X1024','LR5X2048'):
        audit=load('core','audit/'+point+'_DIAG_CPU.json');assert audit['status'].startswith('PASS') and len(audit['rows'])==96
        offline.extend(dict(point=point,**v) for v in audit['rows'])
        panels=['TRAIN','DEV'] if point.endswith('2048') else ['TRAIN']
        entry=dict(groups=audit['groups'],panels={})
        for panel in panels:
            original=load('core',f'source/evidence/RESULT_{point}_{panel}.json')
            cpu=load('core',f'audit/{point}_{panel}_CLOSED_LOOP_CPU.json');assert cpu['status'].startswith('PASS')
            rows=original['rows'];assert len(rows)==(4 if panel=='TRAIN' else 8)
            assert len(cpu['rows'])==len(rows) and all(v['status'].startswith('VALID') for v in rows)
            assert all(not v['persistent_success'] for v in rows)
            entry['panels'][panel]=dict(total=len(rows),persistent=0,first_success=sum(v['first_success_s'] is not None for v in rows),statuses=dict(Counter(v['status'] for v in rows)))
            closed.extend(dict(point=point,panel=panel,**v) for v in cpu['rows'])
        points[point]=entry
    write('OFFLINE384.json',offline);write('CLOSED_LOOP32.json',closed)
    assert len(offline)==384 and len(closed)==32
    columns=['point','panel','condition','status','persistent','first_success_s','hold_pass_samples','observed_hold_samples','reference_to_installed_XYZ_RMS_m','installed_to_achieved_after_control_XYZ_RMS_m','reference_to_achieved_after_control_XYZ_RMS_m','max_cube_lift_above_initial_m','rejected_gate_codes']
    with (OUT/'CLOSED_LOOP32.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,columns,extrasaction='ignore');writer.writeheader();writer.writerows(closed)
    retention={}
    for point in ('CTRL2048','LR5X2048'):
        v=load('core',f'audit/{point}_DROID_CPU.json');assert v['status'].startswith('PASS') and v['finite_count']==32
        retention[point]=dict(finite=32,total=32,original_decoder_exact=True,raw_RMS_vs_pristine_median=statistics.median(row['raw_RMS_vs_pristine'] for row in v['records']),raw_RMS_vs_pristine_p95=quantile([row['raw_RMS_vs_pristine'] for row in v['records']],.95),task_quality='NOT_MEASURED')
    trainers=[]
    for arm in ('CTRL2048','LR5X2048'):
        for step in (1024,2048):
            v=load('hop',f'evidence/{arm}/checkpoint-{step}.full_readback.json')
            assert v['status']=='PASS_FULL_FILE_READBACK' and v['complete_trainer'] and len(v['files'])==14
            trainers.append(v)
    write('FOUR_COMPLETE_TRAINERS_HF.json',trainers)
    terminal=dict(goal='PI0284_NATIVE_GENERATION_CONDITIONING_GO_V1',release='fc9647b2183e5bb9d3943f4a57565ed9751e0739',canonical=6053658526,
                  execution='COMPLETE',execution_closed=True,quality='VALID_QUALITY_FAIL',sampler=4,route=['CTRL2048','LR5X2048'],
                  points=points,retention=retention,CONFIRM='NOT_ADMITTED_NO_QUALIFIED_CANDIDATE',source_candidate='NONE',selected='NONE',deployment='NONE',
                  new_learning_allowed=False,new_quality_allowed=False,budget=budget,
                  interpretation='Offline fit improved in some conditions; approach still fails all diagnostic rows and no checkpoint has persistent object-task success. No sole loss/conditioning/controller cause established.',
                  archival='FOUR_FULL_TRAINER_FIXED_READBACK_PASS_COMPACT_EVIDENCE_IN_PROGRESS')
    write('TERMINAL_RECEIPT.json',terminal)
    for host,root in ROOTS.items():
        remote(host,'from pathlib import Path\np=Path('+repr(root)+')/"evidence/TERMINAL_RECEIPT.json"\np.write_text('+repr(json.dumps(terminal,indent=2)+'\n')+')\n')
    lines=['# PI0284 终局：执行完成，持续物体任务资格未通过','',
           '固定授权 fc9647b2183e5bb9d3943f4a57565ed9751e0739 / Issue #11 canonical 6053658526。',
           '两臂各2048更新、四份完整1024/2048 Trainer、384离线行、32实际源域闭环及64 DROID查询全部完成，独立CPU复算通过。没有合格候选，CONFIRM未准入；source_candidate、selected、deployment均为NONE。','',
           '## 固定质量表','',
           '| 点 | TRAIN fit B0/B1（各24） | VAL fit B0/B1（各24） | TRAIN持续成功 | DEV持续成功 | 首次成功TRAIN/DEV |',
           '|---|---|---|---|---|---|']
    for point,entry in points.items():
        groups={(v['split'],v['packet']):v for v in entry['groups']};train=entry['panels']['TRAIN'];dev=entry['panels'].get('DEV')
        lines.append(f"| {point} | {groups['TRAIN',0]['fit']}/24 · {groups['TRAIN',1]['fit']}/24 | {groups['VAL',0]['fit']}/24 · {groups['VAL',1]['fit']}/24 | 0/4 | {'0/8' if dev else '未安排'} | {train['first_success']}/4 · {str(dev['first_success'])+'/8' if dev else '—'} |")
    lines+=['','fit为原prefix8 XYZ≤10mm、SO3≤0.10rad及夹爪MAE≤0.10的联合门。持续成功仍按完整32秒及28–30秒全部41个20Hz样本的原on_top/static/not_grasped门；首次成功不能替代持续成功。',
            '四点的approach阶段均零fit。LR5X2048的TRAIN10001发生ANGULAR_SPEED_REJECT，原gate拒绝参考、未安装；保留为有效失败，没有重试。其余实际结果与每个失败分母见CLOSED_LOOP32.csv。',
            '', '## 实际干预与完整性','',
            '固定EVENT1024全Trainer/Adam5120起点；两臂同16384 draws、9152 unique UID、相同实际输入/完整132维BF16噪声/time，microbatch4×accum2、两个实际loader workers。共同projector LR1e-4，仅DiT LR1e-5/5e-5不同。Native flow objective、统计、相机、decoder、S4、SafetyGate及Panda native PD保持原绑定。',
            '每臂真实8 vs 4+fresh-restore+4 canary通过；两臂正式4096微批全量配对、四份14成员库存与冻结范围/Adam/S4审计通过。终点global Adam7168。Hopper 654134 Exit_status=0；Stageout_status=1不能改判为训练失败。',
            '保存的详细梯度只有每臂48/4096微批（更新1–8及128倍数），其余UNOBSERVED；旧紧凑status中的INTERVAL16文字不正确。完整输入/噪声/time、loss及LR记录不受此影响。',
            '', '## 生成与条件响应','',
            'K16/K4 TRAIN median J比1.12289，只有8/24改善，首次release夹爪回退0.030975>0.02；按预注册门冻结K4，没有K16物理panel。',
            '自然观测只有1个合格不相交pair，少于6：所有弱响应判断INCONCLUSIVE。真实pixels/backbone features不同，噪声字节相同。parent/CTRL/LR waypoint7响应比例约1.656/2.156/1.958，不能将单pair写成因果视觉消融或足够的条件泛化证明；endpoint39、方向和残差见ENDPOINT_NATURAL_CPU原件。',
            'DROID每臂32/32有限、原decoder精确复算通过；相对pristine的漂移只是描述性证据，没有完整多任务成功标准。',
            '', '## 参考、安装、跟踪与任务','',
            '逐episode分开列出reference→installed、installed→同次控制后实际TCP、reference→实际TCP和物体原生成功；不使用两minimum的差值冒充tracking。物体lift高度只作描述，未补造历史漏存的lift资格门。',
            '父模型首次丢失成功8.85s时仍on_top且not_grasped，但static=false（角速度0.725277>0.5rad/s）；同tick安装误差0.02069mm、实际跟踪7.116mm。8.95s曾恢复成功，随后再次失败。这不是单一接触/控制器/模型根因证据。',
            '全部实际闭环为SOURCE_PANDA/native PD、非实时；不是师兄WBC、Go2迁移或真机。四个视频用同一预固定TRAIN10000、真实保存的BASE/WRIST和原物理标志，未新渲染或重跑。',
            '', '## 预算与封存','',
            f"正式4096 + canary32 = 4128 completed；新quality {neural['query']}/6400，heads {neural['head']}/102400；starts {physical['starts']}/64，controls {physical['control']}/45000，physics {physical['physics']}/230000，RGB {physical['rgb']}/45000，dispatch {physical['dispatch']}/90000。",
            f"双机科学GPU进程闭合区间并集 {gpu/3600:.6f}/12h；归档前新增实际占用 {budget['owned_GiB_before_compact_archive']:.6f}/120GiB。个人quota UNKNOWN；df不代表个人额度。CPU聚合core-hours未完整观测，不将PBS cput当双机总量或宣称CPU精确总量PASS。",
            '四份Trainer均HF PUBLIC固定revision完整14成员readback；原RGB、唯一原始证据和本地恢复点仍受保护。compact evidence及真实视频独立封存；报告/归档问题不重训。',
            '', '## 科学停止点','',
            'COMPLETE / VALID_QUALITY_FAIL；不补CONFIRM，不按中期点、噪声或episode追成功。原科学失败及前序预算保留。下一任务设计仅NOT_AUTHORIZED/launch_allowed=false；需要另行明确任务对照与预算，不自动扩量、恢复采集或WBC。']
    (OUT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(status='TERMINAL_SCIENCE_SEALED_ARCHIVE_PENDING',budget=budget,quality=terminal['quality'])))


if __name__=='__main__':
    assert quantile([0,1,2,3],.5)==1.5
    main()
