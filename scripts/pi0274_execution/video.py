"""One EQUIL first-family ALL8 montage from existing RGB, never rerendered."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,ast,csv,json
import cv2
import numpy as np
from quality_gate import BASE,ROOT,RAW,read,sha,admitted
from quality_worker import write

def prepared():
    source=BASE/'PI0269/code/video.py';assert sha(source)=='4df5992ec759dda3bc9ae37d200b3585501c5c46366d245baa263d5af037873e'
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    loop=next(n for n in node.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='point')
    assert ast.literal_eval(loop.iter)==('UNIFORM4096','PREFIX4096');loop.iter=ast.parse("('EQUIL5120',)",mode='eval').body
    constants=[n for n in ast.walk(node) if isinstance(n,ast.Constant) and n.value=='PI0269_'];assert len(constants)==1;constants[0].value='PI0274_'
    scope=globals().copy();exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),scope);return scope['main']
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-check',action='store_true')
    if p.parse_args().self_check:prepared();print('PASS_QUALIFIED_ONE_POINT_ALL8_SAVED_RGB_VIDEO_SOURCE',flush=True)
    else:admitted();prepared()()
