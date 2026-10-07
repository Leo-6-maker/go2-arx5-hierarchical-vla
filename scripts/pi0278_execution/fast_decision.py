"""Original saved-array decision proof with explicit new authority and roots."""
import argparse,importlib.util,sys
from eval_config import bind,COMMON,ROOT,RAW,RELEASE
bind()
# Keep this goal's panel adapter when the old fast module imports 'panel'.
sys.path.insert(0,str(ROOT/'code'))
import panel
panel.scores=panel.old.scores
spec=importlib.util.spec_from_file_location('qualified_fast','/home/liuyu-nfs/Projects/go2-arx5/PI0277/R3/code/fast_decision.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);old.RELEASE=RELEASE
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('point');p.add_argument('lane',choices=['TRAIN_PHYSICS','TEST','CONFIRM']);a=p.parse_args();old.replay(a.point,a.lane)
