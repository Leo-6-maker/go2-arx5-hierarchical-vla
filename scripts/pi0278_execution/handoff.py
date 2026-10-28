"""Existing local inference package with PI0278's confirmation verdict semantics."""
import importlib.util,inspect,sys
from pathlib import Path
sys.path.insert(0,'/home/liuyu-nfs/Projects/go2-arx5/PI0278/R2/code')
from eval_config import bind
bind()
source=Path(__file__).with_name('qualified_handoff.py')
spec=importlib.util.spec_from_file_location('qualified_handoff',source);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
text=inspect.getsource(old.handoff)
before="('LOCAL_DEV_PICKPLACE_PROTOTYPE' if qualified else 'NOT_USABLE')";assert text.count(before)==1
text=text.replace(before,"('NOT_CONFIRMED' if confirm is not None else ('DEVELOPMENT_QUALIFIED_CONFIRMATION_PENDING' if qualified else 'NOT_USABLE'))")
exec(compile(text,str(source),'exec'),vars(old))
if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1].startswith('COVER93');old.handoff(sys.argv[1])
