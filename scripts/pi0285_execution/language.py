"""Instruction-aware native processor entry, no learned query or text substitution."""
import hashlib,json
import numpy as np
from PIL import Image
from codec import Codec,fields
from contract import INSTRUCTIONS


class LanguageCodec(Codec):
    def feature(self,base_raw,wrist_raw,actual_state,commands,instruction):
        assert instruction in INSTRUCTIONS.values()
        self.proc.train()
        step=self.step_type(images={'base_camera':[Image.fromarray(base_raw)],'hand_camera':[Image.fromarray(wrist_raw)]},states=fields(np.asarray(actual_state,dtype=np.float32).reshape(1,10)),actions=fields(np.asarray(commands,dtype=np.float32)),text=instruction,embodiment=self.tag)
        return self.proc([{'type':'episode_step','content':step}])

    def observation(self,base_raw,wrist_raw,actual_state,instruction):
        assert instruction in INSTRUCTIONS.values()
        state=np.asarray(actual_state,dtype=np.float32).reshape(1,1,10)
        obs={'state.'+k:v for k,v in fields(state).items()}
        for k,value in [('base_camera',base_raw),('hand_camera',wrist_raw)]:
            value=np.asarray(value);assert value.shape==(128,128,3) and value.dtype==np.uint8
            obs['video.'+k]=value[None,None]
        obs['annotation.human.task_description']=[instruction]
        return self.proc.process_observation(obs,self.tag)

    def batch(self,features):
        # The original GR00T collator owns padding, masks and image packing.
        return self.proc.collator(features)

    def cache_key(self,task,instruction,source_identity):
        assert task in INSTRUCTIONS and instruction==INSTRUCTIONS[task]
        return hashlib.sha256(json.dumps(dict(task=task,instruction=instruction,processor=self.identity,source=source_identity),sort_keys=True).encode()).hexdigest()


def cpu_check(root,input_path):
    import torch
    codec=LanguageCodec(root)
    with np.load(input_path) as f:base=f['base_raw'];wrist=f['wrist_raw'];state=f['actual_state']
    commands=np.tile(state,(40,1))
    features=[codec.feature(base,wrist,state,commands,INSTRUCTIONS[k]) for k in ('T1','T2')]
    single=[codec.batch([f])['inputs'] for f in features]
    mixed=codec.batch(features)['inputs'];lengths=[]
    for i,item in enumerate(single):
        token=item['input_ids'][0][item['attention_mask'][0].bool()];lengths.append(len(token))
        assert torch.equal(token,mixed['input_ids'][i][mixed['attention_mask'][i].bool()])
        assert torch.equal(item['state'][0],mixed['state'][i]) and torch.equal(item['action'][0],mixed['action'][i])
    assert lengths[0]!=lengths[1] and not torch.equal(single[0]['input_ids'],single[1]['input_ids'])
    assert codec.cache_key('T1',INSTRUCTIONS['T1'],'same-actual-source')!=codec.cache_key('T2',INSTRUCTIONS['T2'],'same-actual-source')
    return dict(status='PASS_REAL_NATIVE_VARIABLE_LANGUAGE_COLLATOR',token_lengths=lengths,padding_side=codec.proc.collator.processor.tokenizer.padding_side,native_collator_class=str(type(codec.proc.collator)),same_input_different_tokens=True,single_vs_mixed_masks_exact=True,cache_binds_language_task_processor=True,neural_queries=0,physics=0,statistics_used='PREDECESSOR_ONLY_FOR_INGESTION_CHECK_NOT_NEW_TRAIN_STATS')
