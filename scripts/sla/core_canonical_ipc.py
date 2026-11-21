"""Replay the immutable Core panel through the original resident IPC worker without integration."""
from pathlib import Path
import os
import time

from scripts.sla.core_canonical_cold import INITIAL, InputAudit, pin
from scripts.sla.core_canonical_package import CANONICAL_PLAN, PROTOCOL
from scripts.sla.hopper_site import pinned


def cold_pair(spec):
    evidence = pinned(spec['canonical_config'])
    folder = Path(spec['canonical_config']['path']).parent
    arm = spec['arm']
    if (evidence['plan']!=CANONICAL_PLAN or evidence['protocol']!=PROTOCOL
            or arm not in INITIAL or evidence['receiver']!=spec['receiver']):
        raise PermissionError('Unexpected canonical capsule')
    cold = pinned(spec['cold_comparison'])
    paths = [folder/arm/mode/'RESULT.json' for mode in ('cold-A','cold-B')]
    if (cold['status']!='COLD_MATCH' or cold['arm']!=arm or cold['complete']!=[16,16]
            or cold['a']!=pin(paths[0]) or cold['b']!=pin(paths[1])):
        raise PermissionError('The two cold predictions must reconcile before IPC')
    a,b = (pinned(cold[k]) for k in ('a','b'))
    if a!=b or len(a['rows'])!=16 or a['package']!=evidence['packages'][arm]:
        raise ValueError('Cold proof or deployed endpoint changed')
    return evidence,folder,a


class Pool:
    def __init__(self,spec):
        from openpi_client.image_tools import resize_with_pad
        from go2arx5_vla.sla_adapter.adapt5k_data import AdaptTargetDataset
        from scripts.sla.hopper_reference import core_provider
        import jax

        cfg,folder,_ = cold_pair(spec)
        arm = spec['arm']
        r = pinned(spec['receiver'])
        bindings = spec['request_bindings_by_arm']
        if (spec['scope']!=PROTOCOL or spec['physics'] or spec['new_physical_requests_budget']
                or spec['new_native_tick_budget'] or spec['new_complete_generations_budget']!=16
                or spec['unified_requests_budget']!=16 or spec['new_optimizer_updates_budget']
                or spec['new_training_visits_budget'] or spec['GPU_uuid']!=r['gpu']
                or spec['neural_owner_lock']!=r['gpu_lock'] or os.environ.get('CUDA_VISIBLE_DEVICES')!=r['gpu']
                or set(bindings)!={arm} or bindings[arm]['policy_package_sha256']!=cfg['packages'][arm]
                or bindings[arm]['statistics_sha256']!=r['action']['sha256'] or jax.default_backend()!='gpu'):
            raise PermissionError('Unexpected nonintegrating canonical IPC binding')
        source = pinned(r['input'])['target_D1']
        data = AdaptTargetDataset(source['legacy'],source['captures'],resize_with_pad=resize_with_pad)
        data.bind_state(source['statistics'])
        provider,manifest = core_provider(folder/'packages'/arm,cfg['packages'][arm],r['model'],
                                          data,r['tokenizer']['path'],canonical=True)
        if (manifest['checkpoint_sha256']!=cfg['checkpoints'][arm] or
                manifest['action_sha256']!=r['action']['sha256']):
            raise ValueError('IPC provider differs from the cold package')
        self.providers = {arm:InputAudit(provider,Path(spec['registered_root']))}
        self.selected = None

    def select(self, arm):
        self.selected = self.providers[arm]

    def predict(self,*args,**kwargs):
        if self.selected is None:
            raise PermissionError('No qualified IPC arm selected')
        return self.selected.predict(*args,**kwargs)


def replay(spec,worker):
    """Use archived observations and real prior-command context. Read actions without installing."""
    import numpy as np
    from go2arx5_vla.sla_adapter.adapt5k_data import AdaptTargetDataset
    from openpi_client.image_tools import resize_with_pad
    from go2arx5_vla.world_model.hybrid_nonphysical_v3 import Receiver, wait_receipt
    from go2arx5_vla.world_model.mfree_query_diagnostics_v2 import analyze
    from go2arx5_vla.sla_adapter.dense_g7_v6 import label_margin_report
    from go2arx5_vla.sla_adapter.scale20k_contract import canonical_hash
    from scripts.pi0292.durable import write
    from scripts.sla.replay_scale20k_core import guard_signature
    from scripts.sla.hopper_reference import fingerprint
    from scripts.sla.hopper_parent_custody import digest,write_once

    _,_,cold = cold_pair(spec)
    r = pinned(spec['receiver'])
    target=pinned(r['input'])['target_D1']
    data=AdaptTargetDataset(target['legacy'],target['captures'],resize_with_pad=resize_with_pad)
    data.bind_state(target['statistics'])
    panel=pinned(r['panel'])
    arm=spec['arm']
    root=Path(spec['registered_root'])
    bus=Path(spec['ipc_root'])
    bindings=spec['request_bindings_by_arm'][arm]
    wait=lambda receipt,span:wait_receipt(receipt,alive=lambda:worker.poll() is None,
                stopped=lambda:(root/'STOP.json').exists() or time.time()>=spec['deadline_epoch'],
                timeout_s=span)
    wait(root/'READY.private.json',600)
    checks=[]
    for expected in cold['rows']:
        ordinal=expected['ordinal']
        fixed=panel['formal']['target'][ordinal]
        index=fixed['index']
        state,_,images,prompt=data.raw(index)
        frame=data.codecs[fixed['identity']['episode_id']].T_policy_from_mount
        previous=data.reference(index)
        ctx=panel['target_contexts'][str(index)]
        raw=canonical_hash(dict(window=data.identity(index),state=fingerprint(state),prompt=prompt,
            body_wrist=[fingerprint(im) for im in images],frame=fingerprint(frame),previous=previous))
        if (expected['input_sha256']!=raw or index!=expected['index'] or expected['noise_id']!=fixed['noise_id']
                or previous!=ctx['previous_command']):
            raise ValueError('IPC replay reader and cold input identity differ')
        folder=root/f'{ordinal:03d}'
        folder.mkdir()
        observation=folder/'OBSERVATION.private.npz'
        with observation.open('xb') as stream:
            np.savez(stream,state_eef10=state,body_rgb=images[0],wrist_rgb=images[1],
                     T_policy_from_mount=frame,policy_from_world=np.asarray(ctx['policy_from_world']),
                     accepted_reference_TCP=np.asarray(ctx['reference']))
        step=round(previous['query_native_s']/.002)
        receiver=Receiver(folder/'receiver',f'CANONICAL:{arm}:{ordinal}',step,previous,
                          {'kind':'ORIGINAL_CAPTURED_PREVIOUS_COMMAND','input_sha256':raw},bindings,
                          operator_stop=root/'STOP.json')
        request=receiver.request(pin(observation),prompt,fixed['noise_id'],timeout_s=120)
        write_once(folder/'IPC_INTENT.json',dict(input_sha256=raw,ordinal=ordinal,request=request['request_id']))
        write(bus/'REQUEST.json',dict(request=request,response_root=str(folder), live_path=str(receiver.root/'live.json'),
                                     branch_barriers=False))
        reply=wait(folder/'RESPONSE.json',120)
        ack=wait(folder/'DELIVERY_ACK.json',10)
        if (reply['status']!='GENERATED_NOT_INSTALLED' or reply['request_id']!=request['request_id']
                or reply['bindings']!=bindings or reply['new_generations_this_request']!=1
                or ack['status']!='ORIGINAL_RESPONSE' or ack['request_id']!=request['request_id']
                or digest(Path(reply['action']['path']))!=reply['action']['sha256']):
            raise ValueError('The original IPC worker did not return an intact generation')
        with np.load(reply['action']['path'],allow_pickle=False) as result:
            normalized=result['normalized'][0]
            physical=result['physical_H40']
        envelope=analyze(physical,state,np.asarray(ctx['policy_from_world']),np.asarray(ctx['reference']))
        seam=label_margin_report(physical,previous)
        match=dict(normalized=fingerprint(normalized)['sha256']==expected['normalized_sha256'],
                   decoded=fingerprint(physical)['sha256']==expected['decoded_sha256'],
                   guard=canonical_hash(guard_signature(envelope,seam))==expected['original_guard_sha256'],
                   inputs=pinned(pin(root/f"{fixed['noise_id']}.input.json"))==expected['inputs'])
        checks.append(dict(ordinal=ordinal,index=index,request=request['request_id'],bitwise=match,
                           response=pin(folder/'RESPONSE.json')))
    write(bus/'FINISH.json',dict(rows=len(checks)))
    tail=wait(root/'RESULT.private.json',120)
    if (len(checks)!=16 or tail['attempted_generations']!=16 or tail['completed_generations']!=16
            or tail['new_native_requests'] or not all(all(x['bitwise'].values()) for x in checks)):
        raise ValueError('The canonical production IPC result did not reproduce the 16 cold outputs')
    verdict=dict(status='IPC_MATCH',arm=arm,cold=spec['cold_comparison'],worker=pin(root/'RESULT.private.json'),
                 rows=checks,native_requests=0,updates=0)
    write_once(root/'IPC_COMPARISON.json',verdict)
    return verdict
