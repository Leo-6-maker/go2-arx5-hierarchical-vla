"""Twelve presealed TRAIN draws at 1024, never a physical-success gate."""
import hashlib
from pathlib import Path

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0292.eval_contract import error_metrics,gate_action
from scripts.pi0294.e1 import model
from scripts.pi0294.e1.journal import Journal
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.scene_cards import digest

PLAN='plans/PV1_FIXED_TRAIN_PROBES.json'
LANES=('original_T1','original_T2','new_T1','new_T2')
ANCHORS=(0,240,560)


def parent_id(row):
    return row['scene_group'] if row['PV1_lane'].startswith('new_') else row['id']


def plan(rows):
    windows=[]
    for lane in LANES:
        parents={r['id']:r for r,_ in rows if r['PV1_lane']==lane and r['split']=='TRAIN'}
        if not parents:raise PermissionError('PV1_PROBE_REQUIRES_ONLY_PRESEALED_TRAIN_PARENTS')
        row=parents[sorted(parents)[0]]
        if row['task']!=('T1' if lane.endswith('T1') else 'T2'):
            raise PermissionError('PV1_PROBE_LANE_MUST_MATCH_ACTUAL_TASK')
        if not set(ANCHORS)<={anchor for r,anchor in rows if r['id']==row['id'] and r['PV1_lane']==lane}:
            raise PermissionError('PV1_ALL_FIXED_PROBE_ANCHORS_BEFORE_ANY_DRAW')
        parent=parent_id(row)
        for anchor in ANCHORS:
            identifier=lane+'-'+str(anchor).zfill(3)
            seed=int.from_bytes(hashlib.sha256(('PI0294/PV1/TRAIN_PROBE/'+lane+'/'+parent+'/'+str(anchor)).encode()).digest()[:8],'big')%(2**63)
            windows.append(dict(id=identifier,lane=lane,task=row['task'],parent=parent,anchor=anchor,noise_seed=seed,split='TRAIN'))
    return dict(schema='PI0294_PV1_FIXED_TRAIN_PROBES_V1',count=12,windows=windows,
        checkpoint_step=1024,DEV_or_LOCKBOX_used=False,physical_rollouts=0,
        query_plan_is_not_a_task_success_or_retraining_gate=True)


def run(root,label,instance,dataset,ledger,guard):
    if not isinstance(label,str) or not label or label in ('.','..') or Path(label).name!=label:
        raise PermissionError('PV1_FIXED_TRAIN_PROBE_LABEL_ONLY')
    root=Path(root);manifest=read(root/'RUN_MANIFEST.json');path=root/PLAN;declared=read(path)
    if (manifest['execution']['sealed_inputs'].get(PLAN)!=sha(path)
            or declared!=plan(dataset.rows) or instance.identity['global_step']!=1024):
        raise PermissionError('PV1_EXACT_PREFROZEN_1024_TRAIN_PROBE_NO_ADAPTIVE_RESELECTION')
    by_key={}
    for index,(row,anchor) in enumerate(dataset.rows):
        if row['PV1_lane'] not in LANES:continue
        key=(row['PV1_lane'],parent_id(row),anchor)
        if key in by_key:raise PermissionError('PV1_ONE_PROBE_WINDOW_PER_DECLARED_PARENT_AND_LANE')
        by_key[key]=index
    folder=root/'evidence'/label/'TRAIN_PROBES'
    if folder.is_symlink() or not folder.resolve().is_relative_to(root.resolve()):
        raise PermissionError('PV1_OWN_PROBE_EVIDENCE_ONLY')
    folder.mkdir(parents=True,exist_ok=True)
    lock=OwnerLock(folder/'SESSION.lock')
    original_training=instance.model.training;records=[];new_queries=0;reused_draws=0
    try:
        binding=dict(probe_plan_sha256=sha(path),dataset_manifest_sha256=dataset.manifest_sha,
            checkpoint_manifest_sha256=instance.identity['checkpoint_manifest_sha256'],label=label)
        binding_path=folder/'BINDING.json'
        if binding_path.is_symlink():
            raise PermissionError('PV1_NO_PROBE_BINDING_SYMLINK')
        if binding_path.exists():
            if read(binding_path)!=binding:
                raise PermissionError('PV1_SAME_PROBE_PLAN_DATA_AND_CHECKPOINT_NO_REBIND')
        elif any(p.name!='SESSION.lock' for p in folder.iterdir()):
            raise PermissionError('PV1_PRIOR_UNBOUND_PROBE_EVIDENCE_PRESERVED')
        else:write(binding_path,binding)
        # The mature generation journal reuses completed draws and fences unknown intents.
        journal=Journal(root/'reservations/probe-generations',root.name)
        instance.model.eval()
        for row in declared['windows']:
            guard();index=by_key[row['lane'],row['parent'],row['anchor']]
            state,target,images,text=dataset.raw(index);payload=model.payload(state,images,text)
            intent=journal.path('probe-'+label+'-'+row['id'])
            reused=intent.exists() and read(intent)['status']=='COMPLETE'
            physical,metadata=model.fixed_draw(root,label,row['id'],instance,payload,row['noise_seed'],
                folder/(row['id']+'.npz'),ledger,expert=target,state=state)
            new_queries+=int(not reused);reused_draws+=int(reused)
            records.append(dict(**row,errors=error_metrics(physical,target,state),
                original_gate=gate_action(physical,state,row['anchor']*.05,row['id']),
                native_expert_roundtrip_maximum_absolute_difference=metadata['native_expert_roundtrip_maximum_absolute_difference'],
                noise=metadata['noise'],inference_precision=metadata['inference_precision'],arrays_sha256=metadata['arrays_sha256']))
            write(folder/'RESULT.json',dict(status='ACTUAL_FIXED_TRAIN_PROBES_NOT_TASK_SUCCESS',
                complete=len(records)==12,records=records,probe_plan_sha256=sha(path),
                checkpoint_manifest_sha256=instance.identity['checkpoint_manifest_sha256'],
                new_queries=len(records),actual_new_queries_this_invocation=new_queries,
                reused_completed_draws=reused_draws,new_physics=0,physical_success_gate=False,
                records_sha256=digest(records)))
        return read(folder/'RESULT.json')
    finally:
        try:instance.model.train(original_training)
        finally:lock.close()
