"""Boundary/schema and retained-source checks before any endpoint call."""
import json,os,sys
from pathlib import Path
import numpy as np
from budget import interval_seconds
from prepare_core import sha,write

def check(root):
    root=Path(root);assert os.environ['CUDA_VISIBLE_DEVICES']==''
    card=json.loads((root/'design/SOURCE_CARD.json').read_text());plan=json.loads((root/'design/PRE_SCORE_INPUTS.json').read_text())['rows']
    assert len(plan)==48 and card['panels']['CONFIRM']==list(range(30100,30116)) and card['development_points']==['CTRL2048','LR5X2048']
    for row in plan:
        for packet in (0,1):
            noise=Path(row['input']).parent/f'noise-{packet}.npy';v=np.load(noise);assert v.shape==(1,40,132) and v.dtype==np.uint16
    pairs=json.loads((root/'design/NATURAL_PAIRS.json').read_text())['pairs']
    for i,pair in enumerate(pairs):
        for side in ('a','b'):assert sha(pair[side]['input'])==pair[side]['input_sha256']
        for packet in (0,1):assert np.load(root/'natural'/f'pair-{i}-noise-{packet}.npy').dtype==np.uint16
    for n,h in card['code_sha256'].items():assert sha(root/'code'/n)==h
    for v in card['predecessors'].values():assert sha(v['path'])==v['sha256']
    prior=Path(json.loads((root/'design/RUNTIME_CARD.json').read_text())['parent_code'])
    sys.path.append(str(prior));import source
    source.self_check();source.ROOT=root;source.OUT=root/'source'
    old_visible=os.environ['CUDA_VISIBLE_DEVICES']
    try:
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        for point in card['points']:source.guard(point,'TRAIN')
        for point in card['development_points']:source.guard(point,'DEV')
        try:source.guard('CTRL1024','DEV')
        except AssertionError:pass
        else:raise AssertionError('Intermediate incorrectly admitted to DEV')
    finally:os.environ['CUDA_VISIBLE_DEVICES']=old_visible
    assert interval_seconds([(0,4),(1,3),(3,8),(10,11)])==9
    retained=json.loads((root/'design/RETENTION32.json').read_text());assert len(retained['rows'])==32 and retained['history_frame_deltas']==[-15,0]
    for row in retained['rows']:
        assert sha(row['input_path'])==row['input_sha256'] and sha(row['noise_path'])==row['noise_sha256']
        with np.load(row['input_path']) as f:assert f['state'].shape==(17,) and f['actual_history_frame_indices'].tolist()==[row['frame']-15,row['frame']]
    result=dict(status='PASS_CPU_FROZEN_EVAL_SOURCE_ENTRY_NEGATIVE_GUARD_NOISE_HISTORY_AND_GPU_INTERVAL_ACCOUNTING',neural=0,optimizer=0,physics=0,render=0,panels=dict(TRAIN4=4,DEV8=2,CONFIRM16_max=1),DIAG_rows_per_point=96,natural_pairs=len(pairs),DROID_per_endpoint=32)
    write(root/'runtime/EVALUATION_CPU_CHECK.json',result);print(json.dumps(result))

if __name__=='__main__':check(sys.argv[1])
