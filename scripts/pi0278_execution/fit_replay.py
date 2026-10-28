"""Extend only the saved-array replay's lane selector for the parent-only panel."""
import inspect,sys
from pathlib import Path
sys.path.insert(0,'/home/liuyu-nfs/Projects/go2-arx5/PI0278/R2/code')
from eval_config import bind,ROOT
bind()
import replay_panel
old=replay_panel.old
source=inspect.getsource(old.replay)
before="if lane=='TRAIN_PHYSICS':";assert source.count(before)==1
source=source.replace(before,"if lane in ('TRAIN_PHYSICS','FIT_ONLY'):")
before="seeds=card[{'TRAIN_PHYSICS':'TRAIN4','TEST':'DEV8','CONFIRM':'CONFIRM16'}[lane]]";assert source.count(before)==1
source=source.replace(before,"seeds=[] if lane=='FIT_ONLY' else card[{'TRAIN_PHYSICS':'TRAIN4','TEST':'DEV8','CONFIRM':'CONFIRM16'}[lane]]")
exec(compile(source,__file__,'exec'),vars(old))
if __name__=='__main__':old.replay(sys.argv[1],'FIT_ONLY')
