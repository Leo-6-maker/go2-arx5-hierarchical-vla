"""M4-A: replace difference-only extra objective by causal absolute-state/event grounding."""
import numpy as np
import torch
from torch.nn import functional as F
from .contact_training import supervised_loss,select,predict
from .response_training import P,first_block_pairs


def joint_terms(model,pred,logits,b,config):
    hs=[h-1 for h in config['horizons']];target=b['state'][:,[8+h for h in config['horizons']]];p=pred[:,hs];sigma=p.new_tensor(config['sigma_m'])[None,:,None];weights=p.new_tensor(config['weights'])
    valid=b['valid'][:,[7+h for h in config['horizons']]]
    if not torch.all(valid==1):raise ValueError('M4-A requires complete actual16 labels, no padding')
    position=(((p[:,:,P]-target[:,:,P])/sigma).square().mean(-1)*weights).sum(-1).mean()
    full=(((p-target)/model.normalizer.std).square().mean(-1)*weights).sum(-1).mean()
    events=(F.binary_cross_entropy_with_logits(logits[:,hs],b['events'][:,[7+h for h in config['horizons']]],reduction='none').mean(-1)*weights).sum(-1).mean()
    return position,full,events


def update(model,opt,train,fit,keys,cursor,config):
    model.zero_grad(set_to_none=True);totals=[0.]*3;base_total=joint_total=0.;pairs=0;exposures=0
    for offset in range(0,256,32):
        ids=np.arange(cursor+offset,cursor+offset+32)%len(train['state']);b=select(train,torch.tensor(ids,device=train['state'].device));base=supervised_loss(model,b,False)
        first=ids[ids<len(fit['state'])];pairs+=len(first_block_pairs(keys[first]));exposures+=len(first)
        if len(first):
            full=select(fit,torch.tensor(first,device=fit['state'].device));p,e,h,logits=predict(model,full['state'][:,:8],full['state'][:,8],full['reference'][:,:8],full['context'],full['clock'][:,:8],full['clock'][:,8],full['raw'],return_logits=True)
            terms=joint_terms(model,p,logits['event'],full,config);extra=terms[0]+.1*terms[1]+terms[2]
            for i,v in enumerate(terms):totals[i]+=float(v.detach())/8
        else:extra=base.new_zeros(())
        loss=base+extra
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite M4-A loss')
        (loss/8).backward();base_total+=float(base.detach())/8;joint_total+=float(extra.detach())/8
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
    return dict(base_loss=base_total,effect_loss=joint_total,joint_loss=joint_total,effect_terms=totals,absolute_position_loss=totals[0],normalized_full_state_loss=totals[1],event16_BCE=totals[2],pairs_used=0,cooccurring_pairs_not_supervised=pairs,absolute_sequence_exposures=exposures,gradient_norm=float(norm))
