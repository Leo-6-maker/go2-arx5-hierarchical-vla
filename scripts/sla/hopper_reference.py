"""Cold Hopper RGB16 and the matching Go2-only checkpoint format for strict Core replay."""
import argparse
from contextlib import closing
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from go2arx5_vla.sla_adapter import batch_v2, full_h40_v6, scale20k_model, scale20k_package
from go2arx5_vla.sla_adapter.multitask_data_v6 import C20_STATISTICS
from go2arx5_vla.sla_adapter.scale20k_contract import RESTORE_ORDINALS, canonical_hash
from go2arx5_vla.sla_adapter.scale20k_evaluation import bytes_sha
from scripts.pi0292.durable import OwnerLock, read
from scripts.sla.hopper_formal_outputs import load_endpoint
from scripts.sla.hopper_inputs import PLAN, PARENT
from scripts.sla.hopper_panels import FixedPanels, FORMAL_PANEL
from scripts.sla.hopper_parent_custody import digest, sync_dir, write_once
from scripts.sla.hopper_site import live_h200, pinned
from scripts.sla.replay_scale20k_core import guard_signature

SCHEMA='SLA_CORE_HOPPER_GO2_RGB16_CHECKPOINT_V1'


class TargetProvider:
    predict=scale20k_package.Provider.predict

    def __init__(self,graph,frozen,parameters,tokenizer,statistics):
        import flax.nnx as nnx
        from openpi.models import pi0
        from openpi_client.image_tools import resize_with_pad
        from scripts.sla.mfree_c20_vectorfield_v3 import extracted_fields

        self.active=nnx.State.merge(parameters['shared'],parameters['target'])
        self.frozen=frozen
        self.tokenizer=tokenizer
        self.statistics=statistics
        self.resize=resize_with_pad
        fields,_=extracted_fields(batch_v2,pi0)
        self.infer=full_h40_v6.make_sampler(graph,fields,rematerialize=False,precision='highest')


def fingerprint(array):
    value=np.asarray(array)
    if value.dtype.hasobject:
        raise ValueError('Input must have a stable numeric representation')
    return dict(dtype=str(value.dtype),shape=list(value.shape),sha256=bytes_sha(value))


def package(root,manifest_sha):
    m=pinned(dict(path=str(root/'CHECKPOINT.json'),sha256=manifest_sha))
    rows=m['arrays']
    keys=[r['key'] for r in rows]
    paths=[r['file'] for r in rows]
    if (m['schema']!=SCHEMA or m['plan']!=PLAN or m['parent']!=PARENT or m['panel']!=FORMAL_PANEL
            or m['role'] not in ('MAIN','CONTROL2000') or m['physics'] or m['hardware_allowed'] or m['native_admission']
            or m['action_sha256']!=C20_STATISTICS or len(rows)!=29 or len(set(keys))!=29 or len(set(paths))!=29
            or [r['ordinal'] for r in m['reference']]!=list(RESTORE_ORDINALS)
            or not all(any(k.startswith(p) for k in keys) for p in scale20k_package.TARGET_PREFIXES)):
        raise ValueError('Incorrect goal, population or Go2 checkpoint identity')
    if set(root.iterdir())!={root/'parameters',root/'CHECKPOINT.json'}:
        raise ValueError('Unexpected public checkpoint files')
    for r in rows:
        path=Path(r['file'])
        if (len(path.parts)!=2 or path.parts[0]!='parameters' or not path.name.endswith('.npy')
                or not r['key'].startswith(scale20k_package.TARGET_PREFIXES) or r['dtype']!='float32'
                or digest(root/path)!=r['sha256']):
            raise ValueError('Invalid inference member')
    if set(p.name for p in (root/'parameters').iterdir())!={Path(p).name for p in paths}:
        raise ValueError('Unexpected parameter file')
    return m


def core_provider(root,manifest_sha,source,target,tokenizer,*,canonical=False):
    if canonical:
        from scripts.sla.core_canonical_package import validate
        manifest=validate(root,manifest_sha)
    else:
        manifest=package(root,manifest_sha)
    return provider_from_manifest(root,manifest,source,target,tokenizer),manifest


def provider_from_manifest(root,manifest,source,target,tokenizer):
    from go2arx5_vla.sla_adapter.model_v1 import Tokenizer

    if (source['parent_manifest']['sha256']!=manifest['foundation_sha256']
            or digest(Path(tokenizer))!=manifest['tokenizer_sha256'] or source['state_sha256']!=manifest['state_sha256']):
        raise ValueError('Local base or statistics differ from the sealed endpoint')
    graphs,template,frozen,model=scale20k_model.load(source['parent_manifest'],source['parent_params'])
    if (model['frozen_fingerprint']!=manifest['frozen_model'] or
            canonical_hash(model['inventory'])!=manifest['partition_sha256']):
        raise ValueError('Bound model changed')
    parameters=scale20k_package.read_weights(root,manifest['arrays'],{k:template[k] for k in ('shared','target')})
    return TargetProvider(graphs['target'],frozen,parameters,Tokenizer(str(tokenizer)),target.statistics)


def rgb16(provider,data,panel,out,guard,expected=None):
    from go2arx5_vla.world_model.mfree_query_diagnostics_v2 import analyze
    result=[]
    if tuple(panel['restore_ordinals'])!=RESTORE_ORDINALS or panel['old_TEST_access']:
        raise ValueError('Not the fixed original RGB restore population')
    for idx,ordinal in enumerate(RESTORE_ORDINALS):
        guard()
        fixed=panel['formal']['target'][ordinal]
        index=fixed['index']
        ctx=panel['target_contexts'][str(index)]
        previous=data.reference(index)
        if data.identity(index)!=fixed['identity'] or ctx['previous_command']!=previous:
            raise ValueError('Original RGB window, identity or consumed previous command differs')
        state,_,images,prompt=data.raw(index)
        frame=data.codecs[fixed['identity']['episode_id']].T_policy_from_mount
        identity=canonical_hash(dict(window=fixed['identity'],state=fingerprint(state),prompt=prompt,
            body_wrist=[fingerprint(v) for v in images],frame=fingerprint(frame),previous=previous))
        row=dict(ordinal=ordinal,index=index,noise_id=fixed['noise_id'],input_sha256=identity)
        if expected is not None and any(expected[idx][k]!=v for k,v in row.items()):
            raise ValueError('Fixed deployment input differs before generation')
        write_once(out/f'{ordinal:03d}.intent.json',row)
        actual=provider.predict(state,images,prompt,frame,noise_id=fixed['noise_id'],previous_reference=previous)
        write_once(out/f'{ordinal:03d}.generated.json',dict(ordinal=ordinal))
        physical=actual['physical_H40']
        envelope=analyze(physical,state,np.asarray(ctx['policy_from_world']),np.asarray(ctx['reference']))
        row.update(normalized_sha256=bytes_sha(actual['normalized'][0]),decoded_sha256=bytes_sha(physical),
                   original_guard_sha256=canonical_hash(guard_signature(envelope,actual['G7_gripper_speed'])))
        write_once(out/f'{ordinal:03d}.result.json',row)
        result.append(row)
    exact=sum(a==b for a,b in zip(result,expected,strict=True)) if expected is not None else None
    return result,exact


def export(root,record,header,ancestor,rows):
    cp=Path(record['custody'])
    stage=Path(tempfile.mkdtemp(prefix='.package-',dir=root))
    (stage/'parameters').mkdir()
    arrays=[]
    for row in scale20k_package.checkpoint_parameters(header):
        name='parameters/'+row['file']
        scale20k_package.checked_copy(cp/row['file'],stage/name,row['sha256'])
        arrays.append(dict(row,file=name))
    m=dict(schema=SCHEMA,plan=PLAN,parent=PARENT,role=record['role'],phase=record['phase'],
           checkpoint_sha256=record['checkpoint'],training_source=header['binding']['source'],panel=FORMAL_PANEL,
           frozen_model=header['binding']['frozen_model'],foundation_sha256=ancestor['parent_manifest_sha256'],
           partition_sha256=ancestor['partition_sha256'],state_sha256=ancestor['target_state_sha256'],
           tokenizer_sha256=ancestor['tokenizer_sha256'],action_sha256=C20_STATISTICS,
           reference=rows,arrays=arrays,physics=False,hardware_allowed=False,native_admission=False)
    write_once(stage/'CHECKPOINT.json',m)
    sha=digest(stage/'CHECKPOINT.json')
    package(stage,sha)
    sync_dir(stage/'parameters')
    sync_dir(stage)
    dest=root/'package'
    stage.rename(dest)
    sync_dir(root)
    return dict(path=str(dest),sha256=sha)


def reference(run, mount_sha):
    from scripts.sla.hopper_endpoint_backup import checkpoint as inspect
    mount=pinned(dict(path=str(run/'MOUNT.private.json'),sha256=mount_sha))
    req=pinned(mount['reference_request'])
    site=live_h200(mount,'reference')
    view=Path(mount['custody'])
    line=Path(mount['lineage'])
    ticket=pinned(req['ticket'])
    if req['plan']!=PLAN or req['source']!=mount['source_revision'] or req['job']!=site['job']:
        raise ValueError('Reference worker identity changed')
    role,phase=ticket['role'],ticket['phase']
    event=inspect(line,role,phase,ticket['event_sha256'])['event']
    if (not event['deployment_due'] or event['checkpoint']['sha256']!=ticket['checkpoint']
            or ticket['purpose']!='reference' or event['binding']['source']!=req['training_source']):
        raise ValueError('Endpoint was not due for an independent reference')
    parent=dict(path=str(view/'parent.verified/COMPLETE.json'),sha256=PARENT)
    cfg=dict(original_parent=parent,train_assets=str(view/'assets.verified'),training_source=req['training_source'])
    record=dict(role=role,phase=phase,custody=event['checkpoint']['path'],plan=PLAN,parent=PARENT,
                arrays=140,checkpoint=ticket['checkpoint'])
    def guard():
        if time.time()>=site['deadline']-120 or (view/'STOP.json').exists() or (run/'STOP.json').exists():
            raise TimeoutError('Reference stop or allocation margin')
    out=line/'references'/f'{role}-phase{phase:05d}'
    out.parent.mkdir(exist_ok=True)
    out.mkdir()
    with closing(OwnerLock(out/'owner.lock')):
        phase_name='COMPLETE_STATE_RESTORE'
        try:
            guard()
            graphs,state,frozen=load_endpoint(cfg,record)
            phase_name='REFERENCE_DATA_AND_GENERATION'
            panels=FixedPanels(Path(cfg['train_assets']),Path(mount['eval_assets']))
            target=panels.datasets['target']
            provider=TargetProvider(graphs['target'],frozen,state['parameters'],panels.tokenizer,target.statistics)
            rows,_=rgb16(provider,target,panels.formal,out,guard)
            guard()
            header=pinned(dict(path=str(Path(record['custody'])/'COMPLETE.json'),sha256=record['checkpoint']))
            built=export(out,record,header,pinned(parent)['binding'],rows)
            result=dict(status='HOPPER_REF16_SEALED',checkpoint=record['checkpoint'],package=built,
                        attempted=16,complete=16,site=site,updates=0,native=0)
        except Exception as exc:
            common=('Endpoint and complete archived training state disagree','Frozen parent, tokenizer, statistics',
                    'Frozen model or parameter partition differs','SCALE20K_CHECKPOINT_ARRAY_',
                    'Checkpoint seal differs','Checkpoint role, configuration','Stored Adam, RNG')
            shared=phase_name=='COMPLETE_STATE_RESTORE' and any(k in str(exc) for k in common)
            result=dict(status='SHARED_INTEGRITY_FAILURE' if shared else 'REFERENCE_FAILED',
                        phase=phase_name,first_failure=f'{type(exc).__name__}: {exc}',
                        attempted=len(list(out.glob('*.intent.json'))),complete=len(list(out.glob('*.generated.json'))),
                        site=site,updates=0,native=0)
        write_once(out/'RESULT.json',result)
        write_once(run/'RESULT.private.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    p.add_argument('mount_sha256')
    args=p.parse_args()
    os.umask(0o077)
    reference(args.run,args.mount_sha256)
