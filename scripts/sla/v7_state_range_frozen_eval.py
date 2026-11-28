"""Measure the sole global-TRAIN state-range change with frozen R4 on fixed TRAIN."""
import argparse
from contextlib import closing
import gc
import hashlib
import importlib.util
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import bind_imports, check_gpu, pin, read, require, sha
from scripts.pi0292.durable import OwnerLock, write


def run(root, ranges, worker, output):
    import jax
    from go2arx5_vla.sla_adapter.adapt5k_data import AdaptTargetDataset
    from go2arx5_vla.sla_adapter.codec_v1 import Quantiles
    from openpi_client.image_tools import resize_with_pad
    from scripts.sla.core_hopper_native_scope import accepted
    from scripts.sla.hopper_site import pinned
    from v7_e0_matched_forward import WORKER

    panel=root/'TRAIN-PANEL/PANEL.json'
    roster=read(pin(panel,'a52c72a1d0c14ac4823f084ab51f5dd0a67c5ad80fa0433285e39a3a3afb0afd'))
    proposal=read(pin(ranges,'3c5de8e584a5a0ca7a92d518db02014aae5db4ec05bf8182dac069d992b6df9a'))
    prior=read(pin(root/'TRAIN-PANEL-OUTPUT/ACTUAL.json','934040da0a3422899a4825369b0f924024e2cbbeab2966a3e0c47928ce84c1b9'))
    require(roster['split']=='TRAIN_DIAGNOSTIC' and roster['protected_conditions']==0
            and len(roster['frames'])==37 and prior['complete'] and prior['actual_forwards']==222
            and not proposal['changed_training_or_deployment_statistics'] and not output.exists(),
            'Requires the frozen TRAIN-only state counterfactual and complete prior output')
    baseline={(r['frame'],r['repeat']):r for r in prior['rows'] if r['model']=='R4'}
    require(len(baseline)==74,'Incomplete same-weight baseline')
    spec=read(pin(worker,WORKER));bind_imports(spec)
    identity=read(root/'R4_REFERENCE.json')
    provider_path=pin(identity['provider_source']['path'],identity['provider_source']['sha256'])
    module=importlib.util.spec_from_file_location('s1_original_r4_provider',provider_path)
    provider_module=importlib.util.module_from_spec(module);module.loader.exec_module(provider_module)
    with closing(OwnerLock(Path(spec['neural_owner_lock']))):
        gpu=check_gpu(spec['GPU_uuid'],spec['allocator_limit_bytes'],spec['minimum_host_available_bytes'])
        require(jax.default_backend()=='gpu','The actual original CUDA backend is required')
        _,_,receiver=accepted(spec)
        source=pinned(receiver['input'])['target_D1']
        target=AdaptTargetDataset(source['legacy'],source['captures'],resize_with_pad=resize_with_pad)
        target.bind_state(source['statistics'])
        weights=read(pin(Path(identity['root'])/'CHECKPOINT.json',identity['manifest_sha256']))
        model=provider_module.provider_from_manifest(Path(identity['root']),weights,receiver['model'],target,receiver['tokenizer']['path'])
        candidate=proposal['proposals']['GLOBAL_TRAIN_Q01_Q99']
        original_actions=model.statistics['actions']
        model.statistics=dict(model.statistics,state=Quantiles(candidate['q01'],candidate['q99']))
        require(model.statistics['actions'] is original_actions,'State-only treatment changed the action decoder')
        output.mkdir(mode=0o700)
        write(output/'INTENT.json',dict(scope='S1_FROZEN_R4_GLOBAL_TRAIN_STATE_RANGE_ONLY',panel_sha256=sha(panel),
            state_proposal_sha256=sha(ranges),state_treatment=candidate,model_manifest_sha256=identity['manifest_sha256'],
            action_statistics_sha256=proposal['action_statistics_sha256'],maximum_new_forwards=74,
            reused_old_forwards=74,optimizer_updates=0,native_integrations=0,
            balanced_candidate_gpu_tested=False,legacy_baseline_sha256=sha(root/'TRAIN-PANEL-OUTPUT/ACTUAL.json')))
        rows=[]
        for frame in roster['frames']:
            noise=np.asarray(jax.random.normal(jax.random.PRNGKey(frame['noise_id']),(1,40,32)))
            noise_hash=hashlib.sha256(noise.tobytes()).hexdigest()
            require(frame['observed_noise_sha256'] in (None,noise_hash),'Actual CUDA noise differs')
            with np.load(pin(frame['input']['path'],frame['input']['sha256']),allow_pickle=False) as packet:
                previous=None
                for repeat in range(2):
                    old=baseline[frame['id'],repeat]
                    require(old['noise_sha256']==noise_hash,'Reused baseline noise differs')
                    with np.load(pin(root/'TRAIN-PANEL-OUTPUT'/old['artifact'],old['action_sha256']),allow_pickle=False) as values:
                        baseline_action=values['physical_H40']
                    current=model.predict(packet['state_eef10'],[packet['body_rgb'],packet['wrist_rgb']],frame['language'],
                        packet['T_policy_from_mount'],noise_id=frame['noise_id'],previous_reference=frame['previous_command'])
                    action=current['physical_H40'];require(action.shape==(40,10) and np.isfinite(action).all(),'Invalid H40')
                    dest=output/f"{frame['id']}-R{repeat}.npz"
                    with dest.open('xb') as stream:
                        np.savez(stream,physical_H40=action,normalized=current['normalized'])
                    row=dict(frame=frame['id'],episode=frame['episode'],phase=frame['phase'],repeat=repeat,
                        old_g7=old['g7'],new_g7=current['G7_gripper_speed'],noise_sha256=noise_hash,
                        p10_xyz_rms_change_mm=float(1000*np.sqrt(np.mean((action[:10,:3]-baseline_action[:10,:3])**2))),
                        p10_jaw_mae_change_mm=float(40*np.abs(action[:10,9]-baseline_action[:10,9]).mean()),
                        old_p10_up_mm=old['p10_up_request_mm'],new_p10_up_mm=float(1000*(action[:10,2].max()-packet['state_eef10'][2])),
                        repeat_max_abs=0. if previous is None else float(np.max(np.abs(action-previous))),
                        action_sha256=sha(dest),artifact=dest.name)
                    rows.append(row);write(output/f"{frame['id']}-R{repeat}.json",row);previous=action
        require(len(rows)==74,'Incomplete new-state counterpart')
        write(output/'ACTUAL.json',dict(scope='S1_FROZEN_R4_GLOBAL_TRAIN_STATE_RANGE_ONLY',rows=rows,
            model_manifest_sha256=identity['manifest_sha256'],actual_new_forwards=74,reused_old_forwards=74,
            state_proposal_sha256=sha(ranges),panel_sha256=sha(panel),native_integrations=0,optimizer_updates=0))
        del model
        jax.clear_caches();gc.collect()


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for key in ('root','ranges','worker','output'):cli.add_argument(key,type=Path)
    run(**vars(cli.parse_args()))
