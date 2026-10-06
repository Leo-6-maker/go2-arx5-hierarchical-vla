"""Reuse the source task report and saved-RGB video path; no new queries."""
import argparse,ast,csv,json,os,xml.etree.ElementTree as ET
from pathlib import Path
import sys
sys.path.insert(0,'/home/liuyu-nfs/Projects/go2-arx5/PI0277/R2/code')
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import cv2,numpy as np
from scipy.spatial.transform import Rotation
from evaluation_card import COMMON,ROOT,OLD,RAW,read,sha,write
from panel import function

def main(point,lane):
    receipt=read(ROOT/'evidence'/f'CPU_REPLAY_{point}_{lane}.json')
    assert receipt['status']=='PASS_SAVED_S4_DECODE_FRESH_INPUT_NATIVE_PREDICATES'
    reconstruct=function(OLD/'code/replay.py','reconstruct',dict(np=np))
    import sys
    sys.path.insert(0,str(OLD/'project-src'))
    episode=function(OLD/'code/report.py','episode',dict(np=np,ET=ET,Rotation=Rotation,ROOT=COMMON,sha=sha,reconstruct=reconstruct,json=json))
    cases=receipt['task']['rows'];rows=[dict(point=point,lane=lane,seed=r['seed'],**episode(Path(r['path']))) for r in cases]
    dst=RAW/'delivery';dst.mkdir(exist_ok=True)
    table=dst/f'PHYSICAL_{point}_{lane}.csv'
    with table.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    slots=[f'{lane}-{r["seed"]}-{point}' for r in cases]
    def valid(slot):return Path(next(r['path'] for r in cases if slot==f'{lane}-{r["seed"]}-{point}'))
    # Reuse the original frame/codec/readback loop, replacing only registered cases and labels.
    tree=ast.parse((OLD/'code/video.py').read_text());node=next(n for n in tree.body if getattr(n,'name',None)=='main')
    assert isinstance(node.body[0],ast.Assign) and isinstance(node.body[1],ast.AugAssign)
    node.body[:2]=ast.parse('slots=registered_slots').body
    class Labels(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value=='PI0276_':n.value='PI0277_R2_'
            if n.value=='PASS_ALL12_REGISTERED_SAVED_RGB_VIDEOS':n.value='PASS_REGISTERED_SAVED_RGB_PANEL_VIDEOS'
            if n.value=='evidence/VIDEOS.json':n.value=f'evidence/VIDEOS_{point}_{lane}.json'
            return n
    node=Labels().visit(node)
    scope=dict(ROOT=ROOT,RAW=RAW,write=write,sha=sha,json=json,cv2=cv2,np=np,valid=valid,registered_slots=slots)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(OLD/'code/video.py'),'exec'),scope)
    video_receipt=ROOT/'evidence'/f'VIDEOS_{point}_{lane}.json'
    legacy=ROOT/'evidence/VIDEOS.json'
    if not video_receipt.exists() and legacy.exists() and [r['slot'] for r in read(legacy)['rows']]==slots:
        value=read(legacy)
        for r in value['rows']:assert sha(r['path'])==r['sha256']
        value.update(status='PASS_REGISTERED_SAVED_RGB_PANEL_VIDEOS',legacy_label_corrected_from=str(legacy))
        write(video_receipt,value)
    elif not video_receipt.exists():scope['main']()
    write(ROOT/'evidence'/f'DELIVERY_{point}_{lane}.json',dict(point=point,lane=lane,table=str(table),table_sha256=sha(table),cases=rows,
        video_receipt_sha256=sha(video_receipt),new_model_queries=0,new_physics_steps=0,selected_checkpoint='NONE',deployment='NONE'))
    print(json.dumps(dict(point=point,lane=lane,table=str(table),cases=len(rows),status='ACTUAL_SAVED_ARRAY_TABLE_AND_RGB_VIDEOS')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('point');p.add_argument('lane',choices=('TRAIN_PHYSICS','TEST','CONFIRM'));a=p.parse_args();main(a.point,a.lane)
