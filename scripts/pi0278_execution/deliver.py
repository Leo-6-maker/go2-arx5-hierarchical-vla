"""Original saved-RGB/report workflow, bound only to PI0278's isolated roots."""
import importlib.util,inspect,sys
from pathlib import Path
sys.path.insert(0,'/home/liuyu-nfs/Projects/go2-arx5/PI0278/R2/code')
from eval_config import bind,ROOT
bind()
import panel
panel.function=panel.old.function
source=ROOT.parents[1]/'PI0277/R2/code/deliver_panel.py'
spec=importlib.util.spec_from_file_location('qualified_delivery',source);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
text=inspect.getsource(old.main);before="if n.value=='PI0276_':n.value='PI0277_R2_'";assert text.count(before)==1
text=text.replace(before,"if n.value=='PI0276_':n.value='PI0278_'")
exec(compile(text,str(source),'exec'),vars(old))
if __name__=='__main__':
    assert len(sys.argv)==3 and sys.argv[1].startswith('COVER93') and sys.argv[2] in ('TRAIN_PHYSICS','TEST','CONFIRM')
    old.main(*sys.argv[1:])
