"""Same-card SFT2000 numerical check on new BODY+WRIST TRAIN inputs.

This is not one of the four native requests, nor a cross-hardware promotion.
SFT2000 retains its own original Processor, weights and BF16 math route.
"""
import json
from pathlib import Path
import numpy as np


def run(s, root, guard):
    from scripts.pi0294.numerics import prepare_environment, configure_torch, MODE
    prepare_environment(MODE)
    from scripts.pi0294.native_dataset import official_import
    official_import(s['bindings']['source'])
    import gr00t.model.gr00t_n1d7.gr00t_n1d7
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    from gr00t.data.embodiment_tags import EmbodimentTag
    import torch
    from .target_gr00t_training_v1 import _load_original_parent, gpu_identity
    from .target_gr00t_delivery_v1 import install
    from .target_gr00t_core_v1 import trim_unused_logits
    from .native_arx5_dataset_v1 import NativeCameraTransferDataset
    from .task_learning_features_v1 import inference_inputs
    from .task_learning_local_analysis_v1 import geometry
    from scripts.pi0294.local_backbone import local_model_name
    from scripts.pi0294.neural import COSMOS_REVISION
    from scripts.pi0294.native_train import tree_identity
    from scripts.pi0292.durable import write
    from .storage import digest
    configure_torch(torch, MODE)
    torch.set_num_threads(4)
    gpu_identity(torch, dict(s, execution_site='CORELAB_4080S'))
    guard()
    model = _load_original_parent(s, root, 'SFT2000_PARENT_LOAD')
    install(model, s['package']['path'], s['package']['sha256'], s['parent']['files'])
    trim_unused_logits(model)
    model.requires_grad_(False)
    model.to('cuda').eval()
    processor = Gr00tN1d7Processor.from_pretrained(Path(s['package']['path'])/'processor',
        model_name=local_model_name(root, s['bindings']), local_files_only=True,
        transformers_loading_kwargs=dict(local_files_only=True, trust_remote_code=True,
            cache_dir=s['bindings']['cosmos_hub'], revision=COSMOS_REVISION))
    processor.eval()
    data = NativeCameraTransferDataset(s['native_camera_dataset'], s['camera_audit'], processor=processor)
    write(root/'SFT2000_MODEL_LOADED.private.json', dict(model_loads=1, package=s['package'],
        Processor='ORIGINAL_SFT2000_PACKAGE', numeric_mode=MODE, model_updates=0))
    records = []
    for layout_index, layout in enumerate(('TRAIN01', 'TRAIN07', 'TRAIN13', 'TRAIN25')):
        eligible = [(i, e, a) for i, (e, a) in enumerate(data.rows)
                    if e['parent'] == 'TASK_LEARNING_UNDER_HAND_'+layout]
        episode_id = eligible[0][1]['id']
        for anchor in (0, 9):
            index = next(i for i, e, a in eligible if e['id']==episode_id and a==anchor)
            state, _, _, _ = data.raw(index)
            inputs = processor.collator([data[index]['processed']])['inputs']
            seed = 2026111100 + 100*layout_index + anchor
            outputs = []
            for repeat in (0, 1):
                guard()
                file = root/f'{layout}-anchor{anchor:02d}-repeat{repeat}.npz'
                write(file.with_suffix('.intent.json'), dict(seed=seed, repeat=repeat,
                    input_identity=tree_identity(inputs), actual_model_invocation_requested=1))
                with torch.random.fork_rng(devices=[0]), torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    torch.manual_seed(seed)
                    normalized = model.get_action(inference_inputs(inputs))['action_pred'].detach().float().cpu().numpy()
                decoded = processor.decode_action(normalized, EmbodimentTag.NEW_EMBODIMENT,
                    {'eef_9d':state[None,None,:9], 'gripper_position':state[None,None,9:]})
                physical = np.concatenate([decoded[k] for k in ('eef_9d','gripper_position')], axis=-1)[0]
                if normalized.shape != (1,40,132) or physical.shape != (40,10) or not np.isfinite(physical).all():
                    raise ValueError('ORDINARY_FULL_H40_DECODE_SEMANTICS_FAILED')
                np.savez(file, normalized=normalized, physical_H40=physical, actual_state10=state)
                outputs.append((normalized, physical))
                records.append(dict(layout=layout, episode=episode_id, anchor=anchor, repeat=repeat,
                    seed=seed, file=file.name, sha256=digest(file), raw_envelope=geometry(physical, state)))
            if not all(np.array_equal(outputs[0][i], outputs[1][i]) for i in (0,1)):
                raise ValueError('NEW_BODY_INPUT_SAME_CARD_REPLAY_NOT_BIT_EXACT')
    result = dict(schema='NATIVE_BODY_SFT2000_SAME_CARD_REPLAY_V1', package=s['package'],
        camera_audit=s['camera_audit'], dataset_sha256=data.manifest_sha,
        statistics_sha256=data.statistics_sha, GPU_uuid=s['GPU_uuid'], actual_model_loads=1,
        actual_full_model_generations=len(records), actual_Processor_decodes=len(records),
        same_card_bit_exact=True, records=records, ordinary_full_inference=True,
        numerical_mode=MODE, optimizer_updates=0, physics_integrations=0,
        original_cross_hardware_REJECT_preserved=True, original_DEV_TEST_admission=False,
        four_native_diagnostic_requests_executed=0,
        max_allocated_bytes=torch.cuda.max_memory_allocated(), max_reserved_bytes=torch.cuda.max_memory_reserved())
    write(root/'REPLAY_ACTUAL.private.json', result)
    return result
