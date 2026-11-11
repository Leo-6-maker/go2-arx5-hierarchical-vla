"""Join a sealed TRAIN base and new real DEV references, preserving TRAIN stats."""
import copy,json,os,shutil
from pathlib import Path
import numpy as np
from . import task_learning_grasp_v1 as task
from . import task_learning_dev_v1 as dev
from .task_learning_dataset_v1 import audited_episode
from .task_learning_export_v1 import write_episode
from .task_learning_native_dataset_v1 import TaskLearningNativeDataset
from .storage import digest,write_json


def member(root,name,pin):
    relative=Path(name);p=root/relative
    task.need(not relative.is_absolute()and '..'not in relative.parts
        and p.resolve()==p and p.is_file()and not p.is_symlink()and digest(p)==pin,
        'SEALED_BASE_REGULAR_MEMBER_WITH_ORIGINAL_BYTES')
    return p


def train_members(base,pin):
    task.need(base.resolve()==base and digest(base/'DATASET_MANIFEST.json')==pin,
        'EXACT_CANONICAL_TRAIN_BASE_MANIFEST')
    data=TaskLearningNativeDataset(base,'unused-before-Processor','unused-before-Processor')
    m=data.manifest
    task.need(m['bundle_kind']=='CONTINUOUS_TRAIN_BASE'
        and all(r['split']=='TRAIN'for r in m['episodes']),
        'COMPLETE_TRAIN_BASE_BEFORE_SEPARATE_DEV_REFERENCES')
    files={'statistics.json':member(base,'statistics.json',m['statistics_sha256'])}
    for row in m['episodes']:
        files[row['file']]=member(base,row['file'],row['sha256'])
        with np.load(files[row['file']],allow_pickle=False)as packet:
            paths=packet['rgb_paths'];pins=packet['rgb_sha256']
            task.need(paths.shape==pins.shape==(15,2),'ALL_REAL_CURRENT_RGB_PAIRS_RETAINED')
            for name,pin in zip(paths.flat,pins.flat):
                files[str(name)]=member(base,str(name),str(pin))
    return m,files


def dev_group(root):
    from .stable_task_dataset_v1 import load
    inp,ip=load(root/'INPUT.private.json',maximum=16*1024**2)
    lease,lp=load(root/'control/LEASE.json',maximum=1024**2)
    panel,pp=load(inp['panel'],inp['panel_sha256'],maximum=16*1024**2)
    result,rp=load(root/'RESULT.private.json',maximum=16*1024**2)
    outcomes,op=load(root/'OUTCOMES.private.json',maximum=16*1024**2)
    rows=dev.validate(panel);summary=dev.summarize(panel,outcomes)
    task.need(root.resolve()==root and Path(inp['panel']).parent==root
        and inp.get('task_learning_role')==task.DEV_REFERENCE_ROLE
        and lease.get('status')=='CLOSED'and lease.get('input_sha256')==ip['sha256']
        and lease.get('worker_exit_code')==result.get('exit_code')==0
        and inp.get('assignment_slice')==result.get('assignment_slice')==[0,4]
        and result.get('panel_sha256')==pp['sha256']
        and result.get('source_sha')==panel['source_sha']==inp['source_sha']
        and result.get('summary')==summary and summary['passes']is True
        and result.get('completed_in_group')==result.get('selected_requests')==4
        and [r['native_run']for r in outcomes]==[r['native_run']for r in rows],
        'ACTUAL_CLOSED_COMPLETE_FOUR_DEV_REFERENCES_NOT_TRAIN_OR_VLA_EVALUATION')
    episodes=[audited_episode(root/'runs'/row['native_run'],row,pp['sha256'],inp['source_sha'],
        expected_role=task.DEV_REFERENCE_ROLE)for row in rows]
    task.need(all(e['packet']['valid_window'].all()and not e['genuine_recovery_success']for e in episodes),
        'DEV_REFERENCES_ALL_REAL_NORMAL_TEACHER_DECISIONS_NO_INTERVENTION')
    return episodes,dict(input=ip,lease=lp,panel=pp,result=rp,outcomes=op)


def finalize(base,base_pin,dev_root,output):
    base,dev_root,output=Path(base),Path(dev_root),Path(output)
    task.need(output.is_absolute()and output.parent.resolve()==output.parent
        and not output.exists(),'FRESH_CANONICAL_FORMAL_DATA_BUNDLE')
    m,files=train_members(base,base_pin);episodes,proof=dev_group(dev_root)
    task.need(not({r['parent']for r in m['episodes']}&{e['row']['parent']for e in episodes})
        and not({r['id']for r in m['episodes']}&{e['row']['native_run']for e in episodes})
        and all(e['native_binary_sha256']==m['native_binary_sha256']
            and np.array_equal(e['captures'][0]['frame'],np.asarray(m['T_policy_from_target']))for e in episodes),
        'DEV_LAYOUTS_SEPARATE_WITH_SAME_QUALIFIED_NATIVE_AND_POLICY_FRAME')
    output.mkdir();(output/'episodes').mkdir();(output/'media').mkdir()
    for name,source in files.items():
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True)
        if source.stat().st_dev==output.stat().st_dev:os.link(source,target)
        else:shutil.copyfile(source,target)
        task.need(digest(source)==digest(target),'TRANSFER_PRESERVES_ALL_TRAIN_BYTES')
    result=copy.deepcopy(m)
    result.update(bundle_kind='FORMAL_TASK_LEARNING_V1',training_admitted=False,
        requires='ACTUAL_ORIGINAL_PROCESSOR_CORE_AND_HOPPER_RECONSTRUCTION',
        episodes=m['episodes']+[write_episode(e,output)for e in episodes],
        sealed_TRAIN_base_sha256=base_pin,fixed_DEV_reference_group=proof,
        DEV_references_are_BC=False,forty_condition_native_VLA_DEV='NOT_RUN_BY_TEACHER_REFERENCES')
    task.need(digest(output/'statistics.json')==m['statistics_sha256'],'TRAIN_ONLY_STATISTICS_UNCHANGED_BY_DEV')
    write_json(output/'DATASET_MANIFEST.json',result)
    return dict(manifest_sha256=digest(output/'DATASET_MANIFEST.json'),
        statistics_sha256=m['statistics_sha256'],TRAIN_episodes=len(m['episodes']),
        DEV_reference_episodes=4,DEV_optimizer_visits=0,training_admitted=False,
        new_native_integrations=0,new_GR00T_updates=0)
