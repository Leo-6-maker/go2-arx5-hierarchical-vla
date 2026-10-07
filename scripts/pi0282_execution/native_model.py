"""Same verified Cosmos processor bytes, loaded from their local snapshot."""
import contextlib,os
from pathlib import Path


@contextlib.contextmanager
def local_processor():
    from codec import COSMOS
    import gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 as native
    snapshot=Path(os.environ.get('PI0282_COSMOS_DIR',str(COSMOS)))
    assert snapshot.name=='9ce19a195e423419c349abfc86fd07178b230561' and snapshot.is_dir()
    original=native.build_processor
    def load(name,kwargs):
        assert name=='nvidia/Cosmos-Reason2-2B'
        return original(str(snapshot),dict(kwargs,local_files_only=True))
    # Same location-only adapter as the sealed PI0280/81 native loading path.
    native.build_processor=load
    try:yield
    finally:native.build_processor=original


def load_model(path):
    import torch
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
    with local_processor():
        model,info=Gr00tN1d7.from_pretrained(path,torch_dtype=torch.bfloat16,local_files_only=True,
            output_loading_info=True,transformers_loading_kwargs={'local_files_only':True,'trust_remote_code':True})
    # Stock Gr00tPolicy also casts the complete loaded model explicitly.
    model.to(dtype=torch.bfloat16)
    return model,info


if __name__=='__main__':
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7DataCollator
    with local_processor():
        collator=Gr00tN1d7DataCollator(model_name='nvidia/Cosmos-Reason2-2B',model_type='qwen3_vl',
            transformers_loading_kwargs={'local_files_only':True,'trust_remote_code':True})
    assert collator.processor is not None
    print('PASS_NATIVE_LOCAL_PROCESSOR_CONSTRUCTOR; model_forward/optimizer/physics=0')
