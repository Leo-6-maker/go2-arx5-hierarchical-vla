"""Use the qualified saved-RGB montage, fixed first EVAL family and all eight AGG cases."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import ast
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from quality_gate import BASE, ROOT, read, sha, admitted
from admission import write

RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0270')


def prepared_main():
    source=BASE/'PI0269/code/video.py'
    assert sha(source)=='4df5992ec759dda3bc9ae37d200b3585501c5c46366d245baa263d5af037873e'
    tree=ast.parse(source.read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    loop=next(n for n in main.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='point')
    assert ast.literal_eval(loop.iter)==('UNIFORM4096','PREFIX4096')
    loop.iter=ast.parse("('AGG4096',)",mode='eval').body
    names=[n for n in ast.walk(main) if isinstance(n,ast.Constant) and n.value=='PI0269_']
    assert len(names)==1;names[0].value='PI0270_'
    scope=globals().copy()
    exec(compile(ast.fix_missing_locations(ast.Module(body=[main],type_ignores=[])),str(source),'exec'),scope)
    return scope['main']


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--self-check',action='store_true')
    if parser.parse_args().self_check:
        prepared_main();print('PASS_QUALIFIED_FIXED_ALL8_SAVED_RGB_VIDEO_SOURCE_NO_FRAME_PROCESSING')
    else:
        admitted();prepared_main()()
