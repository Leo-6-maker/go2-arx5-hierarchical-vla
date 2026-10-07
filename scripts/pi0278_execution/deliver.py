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
before="if n.value=='VIDEOS.json':n.value=f'VIDEOS_{point}_{lane}.json'";assert text.count(before)==1
text=text.replace(before,"if n.value=='evidence/VIDEOS.json':n.value=f'evidence/VIDEOS_{point}_{lane}.json'")
before="    scope['main']()";assert text.count(before)==1
text=text.replace(before,"""    video_receipt=ROOT/'evidence'/f'VIDEOS_{point}_{lane}.json'
    misplaced=ROOT/f'VIDEOS_{point}_{lane}.json'
    legacy=misplaced if misplaced.exists() else ROOT/'evidence/VIDEOS.json'
    if video_receipt.exists():
        for r in read(video_receipt)['rows']:assert sha(r['path'])==r['sha256']
    elif legacy.exists() and [r['slot'] for r in read(legacy)['rows']]==slots:
        value=read(legacy)
        for r in value['rows']:
            assert sha(r['path'])==r['sha256'] and sha(valid(r['slot'])/'observations.npz')==r['source_sha256']
        value.update(status='PASS_REGISTERED_SAVED_RGB_PANEL_VIDEOS',original_legacy_receipt_sha256=sha(legacy))
        write(video_receipt,value)
    else:scope['main']()""")
exec(compile(text,str(source),'exec'),vars(old))
if __name__=='__main__':
    if sys.argv[1:]==['--check']:
        assert "if n.value=='evidence/VIDEOS.json':n.value=f'evidence/VIDEOS_{point}_{lane}.json'" in text
        assert "misplaced=ROOT/f'VIDEOS_{point}_{lane}.json'" in text
        print('PASS_VIDEO_RECEIPT_PATH_AND_EXISTING_CLIP_RECOVERY');sys.exit()
    assert len(sys.argv)==3 and sys.argv[1].startswith('COVER93') and sys.argv[2] in ('TRAIN_PHYSICS','TEST','CONFIRM')
    old.main(*sys.argv[1:])
