"""TRAIN-only execution-aware loss, observing the unchanged vendor FM forward.

The native flow runs from noise at t=0 to action at t=1. Physical constraints
apply to x_t+(1-t)*v_hat, never to the flow velocity as robot velocity. The
decoder retains the existing Processor's minmax saturation and SE(3) frame.
No generated physical command is filtered, clipped, executed or certified here.
"""
import json
from pathlib import Path

from . import task_learning_features_v1 as features
from .storage import digest

SCHEMA='TASK_LEARNING_EXECUTION_LOSS_INTERNAL_ACCELERATION_V3'
DT=.05
ACCELERATION_LIMIT=.25


def rot6d_matrix(value):
    import torch
    first,second=value[...,:3],value[...,3:6]
    first=first/torch.linalg.vector_norm(first,dim=-1,keepdim=True)
    second=second-(first*second).sum(-1,keepdim=True)*first
    second=second/torch.linalg.vector_norm(second,dim=-1,keepdim=True)
    return torch.stack((first,second,torch.linalg.cross(first,second)),dim=-2)


class PhysicalDecoder:
    """Original relative EEF9 / absolute nominal-gripper minmax decode in torch."""
    def __init__(self,params,*,device):
        import torch
        self.bounds={key:{name:torch.as_tensor(params[key][name],dtype=torch.float64,device=device)
                         for name in ('min','max')}for key in ('eef_9d','gripper_position')}

    @classmethod
    def from_processor(cls,processor,*,device='cuda'):
        sap=processor.state_action_processor
        configs=sap.modality_configs['new_embodiment']['action']
        eef,grip=configs.action_configs
        features.need(configs.modality_keys==['eef_9d','gripper_position'] and sap.use_relative_action
            and not sap.use_percentiles and not configs.mean_std_embedding_keys
            and eef.rep.value=='RELATIVE' and eef.type.value=='EEF' and eef.format.value=='XYZ_ROT6D'
            and eef.state_key=='eef_9d' and grip.rep.value=='ABSOLUTE', 'EXACT_ORIGINAL_PHYSICAL_DECODE_CONFIG')
        return cls(sap.norm_params['new_embodiment']['action'],device=device)

    def unnormalize(self,value,key):
        # Vendor performs add/divide in input dtype, then uses FP64 stats.
        fraction=((value.clamp(-1.,1.)+1.)/2.).to(self.bounds[key]['min'].dtype)
        return fraction*(self.bounds[key]['max']-self.bounds[key]['min'])+self.bounds[key]['min']

    def xyz(self,normalized,state):
        relative=self.unnormalize(normalized[...,:9],'eef_9d')[...,:3]
        rotation=rot6d_matrix(state[...,3:9])
        return state[:,None,:3]+(rotation[:,None]@relative[...,None])[...,0]

    def __call__(self,normalized,state):
        import torch
        relative=self.unnormalize(normalized[...,:9],'eef_9d')
        reference=rot6d_matrix(state[...,3:9])
        xyz=state[:,None,:3]+(reference[:,None]@relative[...,:3,None])[...,0]
        rotation=reference[:,None]@rot6d_matrix(relative[...,3:9])
        grip=self.unnormalize(normalized[...,9:10],'gripper_position')
        return torch.cat((xyz,rotation[...,:2,:].reshape(*rotation.shape[:-2],6),grip),dim=-1)


def acceleration_loss(physical_xyz,expert_xyz):
    """38 interior second differences, in m/s², normalized by the original gate."""
    import torch
    actual=torch.diff(physical_xyz,n=2,dim=-2)/(DT*DT)
    expert=torch.diff(expert_xyz,n=2,dim=-2)/(DT*DT)
    match=(((actual-expert)/ACCELERATION_LIMIT)**2).sum(-1).mean()
    excess=torch.relu(torch.linalg.vector_norm(actual,dim=-1)/ACCELERATION_LIMIT-1.).square().mean()
    return .5*(match+excess),dict(acceleration_match=match,acceleration_excess=excess)


class ExecutionLoss:
    """Scoped hooks retain vendor tensors without changing forward calls or RNG."""
    def __init__(self,head,decoder,states,experts,identities,*,coefficient):
        import math
        features.need(math.isfinite(coefficient)and coefficient>=0,'FINITE_NONNEGATIVE_FROZEN_LAMBDA')
        self.head,self.decoder,self.states,self.experts,self.identities=head,decoder,states,experts,identities
        self.coefficient=coefficient;self.sample_time=head.sample_time;self.active=False
        self.record=None;self.release()
        def time(*args,**kwargs):
            value=self.sample_time(*args,**kwargs)
            if self.active:self.t=value[:,None,None]
            return value
        self.time_wrapper=time;head.sample_time=time
        self.handles=[head.action_encoder.register_forward_pre_hook(self.before_encoder),
                      head.action_decoder.register_forward_hook(self.after_decoder)]

    def begin(self,index,row):
        self.release();self.index=index;self.active=True
        features.need(row['identity']['episode_id']==self.identities[index][0]
            and row['identity']['anchor']==self.identities[index][1], 'TRAIN_AUXILIARY_ROW_IDENTITY')

    def before_encoder(self,module,args):
        if self.active:self.noisy=args[0]

    def after_decoder(self,module,args,value):
        if self.active:self.velocity=value[:,-40:]

    def auxiliary(self):
        import torch
        features.need(self.active and self.noisy is not None and self.t is not None
            and self.velocity is not None,'ACTUAL_VENDOR_FORWARD_TENSORS')
        clean=self.noisy+(1.-self.t)*self.velocity
        with torch.autocast(device_type=clean.device.type,enabled=False):
            xyz=self.decoder.xyz(clean,self.states[self.index:self.index+1])
            aux,terms=acceleration_loss(xyz,self.experts[self.index:self.index+1,:,:3])
        features.need(bool(torch.isfinite(aux)),'FINITE_EXECUTION_AUXILIARY')
        return aux,terms

    def apply(self,native):
        # λ=0 is literally the original tensor and backward graph.
        if self.coefficient==0:
            self.record=dict(coefficient=0.,original_loss_tensor_returned=True)
            return native
        aux,terms=self.auxiliary()
        self.record=dict(coefficient=self.coefficient,auxiliary=float(aux.detach()),
            **{key:float(v.detach())for key,v in terms.items()})
        return native+self.coefficient*aux.to(native.dtype)

    def release(self):
        self.noisy=self.velocity=self.t=None;self.active=False

    def close(self):
        features.need(self.head.sample_time is self.time_wrapper,'SCOPED_VENDOR_TIME_OBSERVER')
        self.head.sample_time=self.sample_time
        for handle in self.handles:handle.remove()
        self.release()


def physical_inputs(spec,cache):
    import numpy as np
    import torch
    pin=spec['physical_loss_inputs'];file=Path(pin['path'])
    features.need(digest(file)==pin['sha256'],'PINNED_TRAIN_PHYSICAL_INPUT_MANIFEST')
    manifest=json.loads(file.read_bytes());numeric=file.parent/'TRAIN1450_PHYSICAL_LOSS_INPUTS.private.npz'
    features.need(manifest['dataset']['sha256']==spec['dataset_sha256'] and manifest['feature_cache']==spec['feature_cache']
        and manifest['rows']==1450 and not manifest['DEV_or_TEST_included']
        and digest(numeric)==manifest['numeric_file_sha256'],'SAME_1450_TRAIN_ONLY_PHYSICAL_INPUTS')
    with np.load(numeric,allow_pickle=False)as z:
        identities=list(zip(z['episode'].tolist(),z['anchor'].tolist()))
        features.need(identities==[(r['episode'],r['anchor'])for r in cache['rows']], 'SAME_FEATURE_AND_PHYSICAL_ORDER')
        states=torch.as_tensor(z['actual_state10'].copy(),device='cuda',dtype=torch.float64)
        experts=torch.as_tensor(z['expert_policy_H40'].copy(),device='cuda',dtype=torch.float64)
    return states,experts,identities


def train(spec,model,contract,data,root,guard):
    from . import task_learning_core_local_v1 as base
    cache=json.loads(Path(spec['feature_cache']['path']).read_bytes())
    states,experts,identities=physical_inputs(spec,cache)
    decoder=PhysicalDecoder.from_processor(data.processor)
    objective=ExecutionLoss(model.action_head,decoder,states,experts,identities,
        coefficient=spec['recipe']['execution_loss']['coefficient'])
    try:return base.train(spec,model,contract,data,root,guard,loss_transform=objective)
    finally:objective.close()


def qualification(spec,model,contract,data,root,guard):
    """Two noncandidate λ0 updates, then TRAIN-only gradient-scale calibration."""
    import numpy as np
    import torch
    from scripts.pi0292.durable import write
    from scripts.pi0294.native_train import NativeNoiseAudit
    from . import task_learning_core_local_v1 as base
    cache=json.loads(Path(spec['feature_cache']['path']).read_bytes())
    states,experts,identities=physical_inputs(spec,cache)
    decoder=PhysicalDecoder.from_processor(data.processor)
    model.backbone.to('cpu');model.action_head.to('cuda');model.eval()
    initial={n:model.get_parameter(n).detach().clone()for n in contract.trainable_names}
    results={}
    for label in ('ORIGINAL_FM','LAMBDA_ZERO'):
        guard();output=root/label;output.mkdir();model.zero_grad(set_to_none=True)
        diagnostic=dict(spec,qualification_only=True,operation='QUALIFY_EXECUTION_LOSS_V3')
        objective=None if label=='ORIGINAL_FM'else ExecutionLoss(model.action_head,decoder,states,experts,identities,coefficient=0.)
        try:actual=base.train(diagnostic,model,contract,data,output,guard,loss_transform=objective)
        finally:
            if objective is not None:objective.close()
        results[label]=dict(actual=actual,update=json.loads((output/'UPDATE_IDENTITY.private.json').read_bytes()),
            samples=[json.loads(x)for x in(output/'SAMPLE_TRACE.jsonl').read_text().splitlines()])
    left,right=[results[k]for k in ('ORIGINAL_FM','LAMBDA_ZERO')]
    exact=left['update']==right['update'] and left['samples']==[
        {k:v for k,v in row.items()if k!='execution_loss'}for row in right['samples']]
    write(root/'LAMBDA_ZERO_ACTUAL.private.json',dict(actual_optimizer_calls=2,candidate_optimizer_calls=0,
        window_visits=64,exact_loss_gradients_parameters_optimizer_scheduler_RNG_and_noise=exact,executions=results))
    features.need(exact,'ACTUAL_LAMBDA_ZERO_COMPLETE_UPDATE_EXACT')
    with torch.no_grad():
        for name,value in initial.items():model.get_parameter(name).copy_(value)
    features.need(all(torch.equal(model.get_parameter(n),v)for n,v in initial.items()),'RESET_EXACT_SFT2000_BEFORE_CALIBRATION')
    del initial
    model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    audit=NativeNoiseAudit(model);objective=ExecutionLoss(model.action_head,decoder,states,experts,identities,coefficient=1.)
    parameters=tuple(contract.parameters());rows=[];decoded=[];finite_difference=None
    requested=spec['calibration_rows']
    features.need(len(requested)==16 and len({r['index']for r in requested})==16,'SIXTEEN_PREREGISTERED_TRAIN_WINDOWS')
    try:
        for selection in requested:
            guard();index=selection['index'];pin=cache['rows'][index];file=Path(spec['feature_cache']['path']).parent/pin['file']
            features.need(digest(file)==pin['sha256'],'UNCHANGED_CALIBRATION_FEATURE')
            row=torch.load(file,map_location='cpu',weights_only=False)
            features.need(row['identity']['episode_id']==selection['episode']and row['identity']['anchor']==selection['anchor'],
                          'PREREGISTERED_CALIBRATION_ROW')
            # Compare real original Processor and full physical10 torch decode.
            from gr00t.data.embodiment_tags import EmbodimentTag
            packet=features.action_packet(model,row,model.dtype)
            normalized=packet['action'].detach().float().cpu().numpy()
            for precision in ('float32','float64'):
                array=normalized.astype(precision)
                original=data.processor.decode_action(array,EmbodimentTag.NEW_EMBODIMENT,
                    dict(eef_9d=states[index:index+1,None,:9].cpu().numpy(),gripper_position=states[index:index+1,None,9:].cpu().numpy()))
                original=np.concatenate([original[k]for k in ('eef_9d','gripper_position')],axis=-1)
                with torch.no_grad(),torch.autocast('cuda',enabled=False):
                    physical=decoder(torch.as_tensor(array,device='cuda'),states[index:index+1]).cpu().numpy()
                delta=float(np.max(np.abs(physical-original)))
                features.need(delta<=1e-10,'ACTUAL_PROCESSOR_TORCH_PHYSICAL10_EQUIVALENCE')
                decoded.append(dict(index=index,precision=precision,maximum_absolute_error=delta))
            if finite_difference is None:
                x=packet['action'].detach().double();x[0,17,0]=0.;x.requires_grad_(True)
                def probe(v):return acceleration_loss(decoder.xyz(v,states[index:index+1]),experts[index:index+1,:,:3])[0]
                value=probe(x);gradient=torch.autograd.grad(value,x)[0]
                plus=x.detach().clone();minus=plus.clone();epsilon=1e-6
                plus[0,17,0]+=epsilon;minus[0,17,0]-=epsilon
                finite=(probe(plus)-probe(minus))/(2*epsilon)
                finite_difference=dict(autograd=float(gradient[0,17,0]),finite_difference=float(finite),epsilon=epsilon,
                    physical_XYZ_gradient_nonzero=bool(gradient[...,:3].abs().sum()>0),
                    padded_and_non_XYZ_gradient_zero=bool(torch.count_nonzero(gradient[...,3:])==0))
                features.need(torch.isclose(gradient[0,17,0],finite,atol=1e-8,rtol=1e-5).item()
                    and finite_difference['physical_XYZ_gradient_nonzero']and finite_difference['padded_and_non_XYZ_gradient_zero'],
                    'ACTUAL_PHYSICAL_AUXILIARY_FINITE_DIFFERENCE_AND_MASK')
                del x,value,gradient,plus,minus,finite
            objective.begin(index,row)
            with torch.random.fork_rng(devices=[0]),torch.autocast('cuda',dtype=torch.bfloat16):
                torch.manual_seed(selection['seed'])
                output=model.action_head(features.feature_packet(row),features.action_packet(model,row,model.dtype))
                native=features.check_loss(output);aux,terms=objective.auxiliary()
                fm_gradient=torch.autograd.grad(native,parameters,retain_graph=True,allow_unused=True)
                fm_norm=float(torch.sqrt(sum(g.detach().double().square().sum()for g in fm_gradient if g is not None)))
                del fm_gradient
                aux_gradient=torch.autograd.grad(aux,parameters,allow_unused=True)
                aux_norm=float(torch.sqrt(sum(g.detach().double().square().sum()for g in aux_gradient if g is not None)))
                del aux_gradient
            features.need(audit.verified and not audit.active and np.isfinite([fm_norm,aux_norm]).all()
                and fm_norm>0 and aux_norm>0,'ACTUAL_NONZERO_FINITE_AUXILIARY_MODEL_GRADIENT')
            rows.append(dict(selection,native_loss=float(native.detach()),auxiliary=float(aux.detach()),
                **{k:float(v.detach())for k,v in terms.items()},native_gradient_norm=fm_norm,auxiliary_gradient_norm=aux_norm,
                ratio=fm_norm/aux_norm,native_noise=audit.record))
            write(root/'CALIBRATION_PROGRESS.private.json',dict(completed=len(rows),requested=16,actual_optimizer_updates=0))
            objective.release();del native,aux,terms,output,row,packet
        coefficient=float(.1*np.median([r['ratio']for r in rows]))
        result=dict(schema=SCHEMA,status='PASS_ACTUAL_DECODE_GRADIENT_LAMBDA0_AND_TRAIN_ONLY_CALIBRATION',
            source_sha=spec['source_sha'],parent_package=spec['package'],dataset_sha256=spec['dataset_sha256'],
            statistics_sha256=spec['statistics_sha256'],physical_loss_inputs=spec['physical_loss_inputs'],
            coefficient=coefficient,formula='0.1*median(native_gradient_L2/auxiliary_gradient_L2)',
            rows=rows,decoder_comparisons=decoded,finite_difference=finite_difference,
            actual_training_forward_calls=80,noncandidate_optimizer_calls=2,candidate_optimizer_calls=0,
            calibration_optimizer_calls=0,DEV_TEST_queries=0,physics_integrations=0,
            original_Processor_minmax_saturation_preserved=True,model_precision_changed=False,
            auxiliary_physical_arithmetic='FLOAT64_CONSTANT_STATS_AND_DIFFERENCES_THEN_SCALAR_CAST_TO_NATIVE_LOSS_DTYPE',
            lambda_frozen_before_candidate_training_and_DEV=True)
        write(root/'CALIBRATION_ACTUAL.private.json',result)
        return result
    finally:objective.close();audit.close();model.zero_grad(set_to_none=True)
