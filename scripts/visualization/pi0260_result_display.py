"""Finite shortfall display from sealed arrays only. No model or new scene."""
from pathlib import Path
import html
import json
import sys
import cv2
import numpy as np
from pi0259_review import require
from eef_video_workflow import encoder,decode,text,sha,write
from plotting import plot

ROOT=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0260')


def run():
    review=ROOT.parent/'PI0260_review/DEV_v1';receipt=json.loads((review/'receipt.json').read_text())
    require(receipt['status']=='PI0260_ELIGIBLE_SCENE_SHORTFALL','display scope changed')
    out=ROOT/'delivery';require(not out.exists(),'first display already exists');out.mkdir()
    rows=[json.loads((review/('pi0260-family-%03d.json'%i)).read_text()) for i in range(20)]
    passing=next(r for r in rows if r['eligible'])
    failing=next(r for r in rows if all(s['complete_geometry'] for s in r['sequences']) and not r['eligible'])
    chosen=[passing,failing];inputs=[]
    for row in chosen:
        seq=next(s for s in row['sequences'] if s['task']=='M3' and s['command']==0)
        folder=ROOT/'proposals'/row['family']/seq['sequence']
        records=json.loads((folder/'observations.json').read_text())['rows']
        with np.load(folder/'geometry.npz') as z:a={k:z[k].copy() for k in z.files}
        inputs.append((row,seq,records,a))
    video=out/'PI0260_DEV_REFERENCE_shortfall.mp4';process=encoder(video,fps=10)
    for tick in range(161):
        canvas=np.full((720,1280,3),20,np.uint8)
        text(canvas,'PI0260 DEV: 4 / 20 eligible families; required 6. TRAINING NOT ADMITTED.',20,30,.65)
        text(canvas,'KINEMATIC REFERENCE | SAVED RGB | NO PHYSICS / WBC / MODEL PREDICTION',20,59,.5)
        for col,(row,seq,records,a) in enumerate(inputs):
            x=20+col*630;record=records[tick];t=record['time']
            with np.load(ROOT/'proposals'/row['family']/'frames'/(record['frame']+'.npz')) as z:
                pair=np.concatenate([z['body_rgb'],z['wrist_rgb']],axis=1)[...,::-1]
            canvas[130:418,x:x+576]=cv2.resize(pair,(576,288),interpolation=cv2.INTER_NEAREST)
            text(canvas,row['family']+' / M3 c0 / '+('ELIGIBLE' if row['eligible'] else 'VISIBILITY FAIL'),x,96,.55)
            text(canvas,'body 96x96                              wrist 96x96 (display 3x)',x,121,.41)
            checks=record['check'];black=max(checks['counts']['black_A'].values());white=max(checks['counts']['white_B'].values())
            text(canvas,'t=%.1fs | black=%dpx white=%dpx (required >=4)'%(t,black,white),x,445,.52)
            text(canvas,'M3 c0 visible active windows: %.3f%%'%(seq['visibility_fraction']*100),x,474,.52)
            times=np.array([r['time'] for r in records]);counts=np.array([min(max(r['check']['counts'][k].values()) for k in ['black_A','white_B']) for r in records])
            plot(canvas,(x,489,580,150),[(times,counts,(95,210,240))],(0,16),(0,max(8,int(counts.max()))),
                'Minimum endpoint pixels across black / white','synthetic time (s)','pixels',cursor=t,threshold=4)
        text(canvas,'First eligible family vs first observation-failed family in frozen order; no example/model rerolls.',20,674,.49)
        text(canvas,'Original images and geometry remain unchanged. Full 20-family negative denominator retained.',20,700,.49)
        process.stdin.write(canvas.tobytes())
        if tick==0:cv2.imwrite(str(out/'overview.png'),canvas)
    process.stdin.close();require(process.wait()==0,'encoder failure');video_receipt=decode(video)
    table=[]
    for row in rows:
        geom=sum(s['complete_geometry'] for s in row['sequences']);vis=[s.get('visibility_fraction') for s in row['sequences'] if s['task']=='M3']
        table.append('<tr><td>'+html.escape(row['family'])+'</td><td>'+str(geom)+'/4</td><td>'+html.escape(str(vis))+'</td><td>'+('ELIGIBLE' if row['eligible'] else 'REJECTED')+'</td></tr>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PI0260 DEV 结果</title><style>body{max-width:1200px;margin:32px auto;font:18px system-ui;line-height:1.6;background:#f5f7fa;color:#19232d}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #cbd2da}video,img{width:100%}</style><h1>PI0260：DEV 条件数据不足</h1><p><strong>4 / 20 家族合格，协议要求 6。</strong>12 个家族未完成全部有界 IK 轨迹；另 4 个家族的 M3 端点可见率低于95%。固定 DEV 提案已耗尽。</p><p>H0 相机检查通过，不能代表全部新场景可观测。CPU 从原始数组复算了80条序列、11,237个状态的原碰撞检查、2,134对图像及4,192个原 gate 窗口。</p><p><strong>未进行 TRAIN/VAL、processor 拟合、模型训练或评价。Base/G6000 和1760格评价均不存在；实验目标尚未完成。</strong>selected/deployment = NONE。</p><h2>已有参考的原始图像对比</h2><p>固定顺序的第一个合格家族与第一个观察失败家族；仅运动学参考，非模型预测、物理或WBC成功。实际96×96图像以3倍最近邻显示。</p><video controls src="PI0260_DEV_REFERENCE_shortfall.mp4"></video><h2>全部20个提案</h2><table><tr><th>家族</th><th>完整几何</th><th>M3双指令可见率（None=未渲染）</th><th>共同资格</th></tr>'''+''.join(table)+'''</table><p>依据 PI0260 / R1 停止该依赖链。未更换相机、阈值、IK、提案或借用 TRAIN 样本。后续需要科学设置的新授权，非例行执行 GO。</p></html>'''
    (out/'index.html').write_text(page,encoding='utf-8')
    write(out/'DISPLAY_RECEIPT.json',dict(selection_rule='first eligible / first complete-geometry observation-failed family, M3 c0; frozen proposal order',
        source_review_sha256=sha(review/'receipt.json'),families=[r['family'] for r in chosen],video=video_receipt,
        new_model_queries=0,new_scene_construction=0,physics_steps=0,evidence_class='KINEMATIC_REFERENCE_NOT_POLICY_PREDICTION'))
    print(json.dumps(dict(video=str(video),decoded_frames=video_receipt['decoded_frames'])))


if __name__=='__main__':run()
