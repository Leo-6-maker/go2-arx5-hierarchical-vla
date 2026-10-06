"""Atomic local inference snapshots and explicitly bounded recovery retention."""
import os
from pathlib import Path
from cpu_inputs import read,sha,write

def snapshot(root,arm,step,checkpoint):
    pending=[p for p in (root/'snapshots').glob('*/READY.json') if not p.with_name('SCREEN_COMPLETE.json').exists()]
    if pending:
        write(root/'evidence'/f'{arm}{step}_SNAPSHOT_DEFERRED.json',dict(status='UNMEASURED_PENDING_PREVIOUS_SCREEN',pending=[str(p) for p in pending],step=step));return
    dest=root/'snapshots'/f'{arm}{step}';temp=dest.with_name(dest.name+'.partial');temp.mkdir(parents=True,exist_ok=False)
    names=['config.json','model.safetensors.index.json','effective_statistics.json','statistics.json','processor_config.json','embodiment_id.json','PI0277_READOUT_CONTRACT.json']
    names += sorted(p.name for p in checkpoint.glob('model-*.safetensors'))
    files=[]
    for name in names:
        source=checkpoint/name;assert source.is_file();os.link(source,temp/name)
        files.append(dict(path=name,bytes=source.stat().st_size,sha256=sha(source)))
    write(temp/'READY.json',dict(status='LOCAL_IMMUTABLE_MODEL_ONLY_SNAPSHOT',arm=arm,step=step,point=f'{arm}{step}',files=files,
        resumable=False,source_full_trainer=str(checkpoint),release='1a20200ff066ef5a003650110cd20106f9a4bf64',selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    temp.replace(dest)

def retain_two(root,arm,output):
    checkpoints=sorted(output.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[1]))
    complete=[p for p in checkpoints if (p/'PI0277_RUN.json').exists()]
    for cp in complete[:-2]:
        assert cp.resolve().is_relative_to(root.resolve()) and not cp.is_symlink()
        members=[p for p in cp.rglob('*') if p.is_file()];assert not any(p.is_symlink() for p in cp.rglob('*'))
        receipt=dict(status='RETIRING_OLDER_COMPLETE_RECOVERY_WITH_TWO_SUCCESSORS',path=str(cp),
            successors=[str(p) for p in complete[-2:]],files=[dict(path=str(p.relative_to(cp)),bytes=p.stat().st_size,links=p.stat().st_nlink) for p in members],
            snapshot_hardlinks_preserved=True)
        marker=root/'evidence/retired'/f'{arm}-{cp.name}.json';write(marker,receipt)
        for p in members:p.unlink()
        for p in sorted((p for p in cp.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):p.rmdir()
        cp.rmdir();receipt['status']='DELETED_OLDER_RECOVERY_TWO_NEWEST_PRESERVED';write(marker,receipt)
