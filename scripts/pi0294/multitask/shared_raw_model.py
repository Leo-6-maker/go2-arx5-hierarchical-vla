"""One raw model construction shared by registered training and evaluation.

Call only after the existing owner, inputs, device and capacity are admitted.
There is no launcher, registry writer, credential or remote dependency here.
"""
from pathlib import Path


def load(root,manifest,bindings,recipe,guard):
    import torch
    from transformers import AutoModel
    import gr00t.model.gr00t_n1d7.gr00t_n1d7
    from scripts.pi0292.durable import read,write
    from scripts.pi0292.train_adapter import build_processor
    from scripts.pi0294.generalization.minimal_model import unchanged_raw_parent,loader_kwargs,validate_loading
    from scripts.pi0294.generalization.minimal_adapter import ParameterContract,LEAVES
    from scripts.pi0294.multitask.book_m32_learning import LimitedContract
    from scripts.pi0294.multitask.shared_checkpoint import raw_parameter_identity
    from scripts.pi0294.local_backbone import local_model_name
    from scripts.pi0294.neural import COSMOS_REVISION
    from scripts.pi0294.e1.model import E1ResidentModel
    root=Path(root);raw=Path(manifest['raw_parent']);audit=read(root/'RAW_PARENT_AUDIT.json');guard()
    unchanged_raw_parent(raw,audit)
    model,info=AutoModel.from_pretrained(raw,**loader_kwargs(local_model_name(root,bindings),bindings['cosmos_hub'],COSMOS_REVISION))
    validate_loading(model,info)
    if getattr(model.config,'state_history_length',1)!=1:
        raise PermissionError('SHARED_ORIGINAL_SINGLE_FRAME_STATE_ENCODING')
    base=ParameterContract(model,'E1_ADAPTER_ONLY',10);contract=LimitedContract(base,2)
    write(root/'RAW_ZERO_STEP_PARAMETERS.json',raw_parameter_identity(model,contract,raw,audit,guard=guard))
    for name in LEAVES:model.get_submodule(name).retain_frozen_tables_on_cpu()
    model.to('cuda').eval();guard()
    instance=E1ResidentModel.__new__(E1ResidentModel)
    instance.root,instance.bindings,instance.model=root,bindings,model
    instance.processor=build_processor(bindings['source'],bindings['cosmos'],read(Path(bindings['dataset'])/'statistics.json'))
    instance.identity=dict(kind='SHARED4_RAW_PARENT_ZERO_STEP',global_step=0,
        pretrained_revision=recipe['pretrained_revision'],raw_parent_files_identity=recipe['raw_parent_files_identity'],
        statistics_sha256=recipe['statistics_sha256'],raw_pretraining_statistics_used=False)
    return instance,contract
