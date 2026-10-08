"""Original DROID preprocessing and saved-pristine decode; no neural model exists."""
import json,os,sys
from pathlib import Path
import numpy as np
from prepare_core import write,sha

def check(root):
    root=Path(root);assert os.environ['CUDA_VISIBLE_DEVICES']==''
    sys.path.append(json.loads((root/'design/RUNTIME_CARD.json').read_text())['parent_code'])
    from retention import original_policy
    from gr00t.data.types import MessageType
    from gr00t.policy.gr00t_policy import _rec_to_dtype
    import torch
    torch.set_num_threads(4)
    card=json.loads((root/'design/RETENTION32.json').read_text());policy=original_policy(None,card['original_model_root'],card['source_tag']);records=[]
    for row in card['rows']:
        with np.load(row['input_path']) as f:
            state=f['state'];obs=dict(video={k:f[k][None].copy() for k in card['video_key_binding']},state={k:state[a:b][None,None].copy() for k,(a,b) in card['state_slices'].items()},language={policy.language_key:[[row['language']]]})
        policy.check_observation(obs);single=policy._unbatch_observation(obs)[0];step=policy._to_vla_step_data(single)
        assert not step.actions
        encoded=policy.collate_fn([policy.processor([dict(type=MessageType.EPISODE_STEP.value,content=step)])])
        inp=_rec_to_dtype(encoded,torch.bfloat16)['inputs'];assert 'action' not in inp and inp['state'].shape==(1,1,132)
        assert int(inp['embodiment_id'][0])!=10
        with np.load(row['pristine_output']) as f:
            raw=f['raw'].copy();prior={k:f['decoded_'+k].copy() for k in card['state_slices']}
        decoded=policy.processor.decode_action(raw,policy.embodiment_tag,obs['state'])
        for k,value in decoded.items():assert np.array_equal(value.astype(np.float32),prior[k]),k
        records.append(dict(id=row['id'],input_sha256=sha(row['input_path']),projector=int(inp['embodiment_id'][0]),original_decode_exact=True))
    assert len(records)==32 and policy.model is None
    write(root/'runtime/RETENTION_CPU_CHECK.json',dict(status='PASS_ORIGINAL_DROID32_INPUT_HISTORY_PROCESSOR_AND_SAVED_PRISTINE_EXACT_DECODE',records=records,model_instantiations=0,neural_queries=0,optimizer=0,physics=0))
    print('PASS_ORIGINAL_DROID32_CPU; model/forward/optimizer/physics=0')

if __name__=='__main__':check(sys.argv[1])
