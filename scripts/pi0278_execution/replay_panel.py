"""Original independent full S4/decoder/native replay; no new neural calls."""
import argparse,importlib.util,sys
from eval_config import bind,ROOT
bind();sys.path.insert(0,str(ROOT/'code'))
import panel
panel.function=panel.old.function;panel.scores=panel.old.scores
spec=importlib.util.spec_from_file_location('qualified_replay','/home/liuyu-nfs/Projects/go2-arx5/PI0277/R2/code/replay_panel.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('point');p.add_argument('lane',choices=['TRAIN_PHYSICS','TEST','CONFIRM']);a=p.parse_args();old.replay(a.point,a.lane)
