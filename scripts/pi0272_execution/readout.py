"""PI0272 DIRECT: the existing head at zero latent/timestep, without Euler."""
import json
import hashlib
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
import pi0247_u_rollout_overlay as inherited


def generate(head, backbone, state, embodiment, *, execution_phase, checkpointed=False, audit=None):
    # No action/label argument: only the processed condition enters this readout.
    assert head.num_inference_timesteps == 4 and head.action_horizon == 40 and head.action_dim == 132
    vl = backbone.backbone_features
    assert execution_phase in ('training','quality')
    expected=torch.float32 if execution_phase=='training' else torch.bfloat16
    assert vl.dtype==expected,('PHASE_FEATURE_DTYPE',execution_phase,vl.dtype)
    query = torch.zeros((vl.shape[0],40,132),device=vl.device,dtype=vl.dtype)
    with inherited._module_eval(head):
        features = head.state_encoder(state.view(state.shape[0],1,-1),embodiment)
    def node(q,s,v):
        timestep = torch.zeros(v.shape[0],device=v.device,dtype=torch.long)
        action = head.action_encoder(q,timestep,embodiment)
        if head.config.add_pos_embed:
            positions = torch.arange(action.shape[1],device=v.device,dtype=torch.long)
            action = action + head.position_embedding(positions).unsqueeze(0)
        args = dict(hidden_states=torch.cat((s,action),dim=1),encoder_hidden_states=v,timestep=timestep)
        if head.config.use_alternate_vl_dit:
            args.update(image_mask=backbone.image_mask,backbone_attention_mask=backbone.backbone_attention_mask)
        return head.action_decoder(head.model(**args),embodiment)[:,-40:]
    def contexts():
        return inherited._module_eval(head,audit,'direct_forward'), inherited._module_eval(head,audit,'backward_recompute')
    if checkpointed:
        result = checkpoint(node,query,features,vl,use_reentrant=False,preserve_rng_state=True,context_fn=contexts)
    else:
        with inherited._module_eval(head,audit,'direct_reference'):
            result = node(query,features,vl)
    return result,query


def infer(head,raw_backbone,condition,*,execution_phase):
    """Condition-only inference entry, including native backbone preprocessing."""
    from transformers.feature_extraction_utils import BatchFeature
    assert set(condition)=={'state','embodiment_id'} and 'action' not in raw_backbone
    with inherited._module_eval(head):
        processed=head.process_backbone_output(BatchFeature(data=dict(raw_backbone)))
        return generate(head,processed,condition['state'],condition['embodiment_id'],execution_phase=execution_phase)[0]


def decode_gate(raw,state,processor,tag_name):
    from gr00t.data.embodiment_tags import EmbodimentTag
    from pi0228_train_calib_checkpoint_selection import _build_chunk
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat,rot6d_rows_to_matrix
    tag=EmbodimentTag(tag_name);sap=processor.state_action_processor
    groups={};start=0;values=state.detach().float().cpu().numpy()
    for key in sap.modality_configs[tag.value]['state'].modality_keys:
        width=int(sap.norm_params[tag.value]['state'][key]['dim'].item())
        groups[key]=values[...,start:start+width];start+=width
    decoded_state=sap.unapply_state(groups,tag.value)
    action=processor.decode_action(raw.detach().float().cpu().numpy(),tag,decoded_state)
    decoded=np.concatenate((np.asarray(action['eef_9d']),np.asarray(action['gripper_position'])),axis=-1)
    achieved=np.asarray(decoded_state['eef_9d'],dtype=np.float64)[0,-1]
    result=TrajectorySafetyGateV1().evaluate(_build_chunk(decoded,chunk_id='pi0272-direct-parity'),now=0.,
        achieved_position=achieved[:3],achieved_orientation=matrix_to_quat(rot6d_rows_to_matrix(achieved[3:9].reshape(1,6)))[0]).as_dict()
    return decoded,result


def install(head_cls,*,scale,offset,state_min,state_max,audit_path,parity_path,processor_getter,
            embodiment_tag,resume_after_step=0,expected_r_star='S4'):
    assert expected_r_star=='S4'  # Native flow head configuration is unchanged.
    prior=None
    if resume_after_step:
        rows=[json.loads(x) for x in audit_path.read_text().splitlines()]
        prior=json.loads(parity_path.read_text())
        assert len(rows)==resume_after_step and [x['forward_index'] for x in rows]==list(range(1,resume_after_step+1))
        assert prior['status']=='PASS_PI0272_DIRECT_READOUT_PARITY'
    else:assert not audit_path.exists() and not parity_path.exists()
    native_forward=head_cls.forward
    audit=dict(forwards=resume_after_step,parity=prior,first_update_euler_gradient_norms={},
               native_mode_snapshot=None,mode_restoration_forward_checks=resume_after_step,
               gradient_key_semantics='ONE_DIRECT_GENERATED_OUTPUT_NODE_NOT_EULER')

    def forward(self,backbone_output,action_input):
        from transformers.feature_extraction_utils import BatchFeature
        raw_backbone=BatchFeature(data=dict(backbone_output))
        expert=action_input.action;noises=[];times=[]
        randn=torch.randn;sample_time=self.sample_time;had='sample_time' in self.__dict__
        def noise(*a,**kw):
            value=randn(*a,**kw)
            if value.shape==expert.shape and value.dtype==expert.dtype and value.device==expert.device:noises.append(value)
            return value
        def time(*a,**kw):
            value=sample_time(*a,**kw);times.append(value);return value
        try:
            torch.randn=noise;self.sample_time=time
            native=native_forward(self,backbone_output,action_input)
        finally:
            torch.randn=randn
            if had:self.sample_time=sample_time
            else:del self.sample_time
        assert len(noises)==len(times)==1
        epsilon=noises[0];stock=epsilon.to(device=native['backbone_features'].device,dtype=native['backbone_features'].dtype)
        processed=BatchFeature(data={k:v for k,v in backbone_output.items() if k!='action'})
        assert 'action' not in processed
        modes=[m.training for m in self.modules()]
        if audit['native_mode_snapshot'] is None:audit['native_mode_snapshot']=modes
        assert audit['native_mode_snapshot']==modes
        cpu=torch.random.get_rng_state();cuda=torch.cuda.get_rng_state_all()
        generated,query=generate(self,processed,action_input.state,action_input.embodiment_id,execution_phase='training',checkpointed=True,audit=audit)
        assert torch.equal(cpu,torch.random.get_rng_state()) and all(torch.equal(x,y) for x,y in zip(cuda,torch.cuda.get_rng_state_all()))
        if audit['parity'] is None:
            # A separately executed non-checkpointed path, within the parity-head budget.
            with torch.no_grad():reference=infer(self,raw_backbone,dict(state=action_input.state,embodiment_id=action_input.embodiment_id),execution_phase='training')
            a,ga=decode_gate(reference,action_input.state,processor_getter(),embodiment_tag)
            b,gb=decode_gate(generated,action_input.state,processor_getter(),embodiment_tag)
            raw=float((reference.float()-generated.detach().float()).abs().max())
            xyz=float(np.max(np.abs(a[...,:3]-b[...,:3])))
            decoded=float(np.max(np.abs(a-b)))
            assert raw<=1e-4 and xyz<=1e-5 and decoded<=1e-4
            assert ga.get('decision')==gb.get('decision') and ga.get('issues')==gb.get('issues')
            assert torch.equal(cpu,torch.random.get_rng_state()) and all(torch.equal(x,y) for x,y in zip(cuda,torch.cuda.get_rng_state_all()))
            audit['parity']=dict(status='PASS_PI0272_DIRECT_READOUT_PARITY',raw_max_abs=raw,xyz_max_abs_m=xyz,
                decoded_max_abs=decoded,reference_gate=ga,candidate_gate=gb,query=inherited._tensor_identity(query),
                generated_nodes=1,Euler_steps=0,RNG_consumed=False,label_free=True,extra_parity_heads=1,
                execution_phase='training',feature_dtype=str(processed.backbone_features.dtype),query_dtype=str(query.dtype),output_dtype=str(generated.dtype))
            arrays=parity_path.with_suffix('.npz')
            with arrays.open('xb') as f:
                np.savez(f,reference=reference.detach().float().cpu().numpy(),candidate=generated.detach().float().cpu().numpy(),
                    zero_query=query.detach().float().cpu().numpy(),state=action_input.state.detach().float().cpu().numpy())
            audit['parity']['first_pass_arrays_sha256']=hashlib.sha256(arrays.read_bytes()).hexdigest()
            parity_path.write_text(json.dumps(audit['parity'],indent=2,sort_keys=True)+'\n')
        if audit['forwards']==0:
            assert generated.requires_grad
            def gradient(value):
                audit['first_update_euler_gradient_norms']['1']=float(torch.linalg.vector_norm(value.detach().float()))
                return value
            generated.register_hook(gradient)
        assert [m.training for m in self.modules()]==modes
        audit['mode_restoration_forward_checks']+=1
        state=inherited._state_eef(action_input.state,state_min,state_max)
        position,delta,details=inherited._physical_losses(generated,expert,native['action_mask'],state,scale.to(expert.device),offset.to(expert.device))
        total=native['loss']+inherited.LAMBDA_POS*position+inherited.LAMBDA_DELTA*delta
        assert torch.isfinite(total)
        audit['forwards']+=1
        row=dict(forward_index=audit['forwards'],native_loss=float(native['loss'].detach()),
            native_epsilon=inherited._tensor_identity(epsilon),native_epsilon_sha256=inherited._tensor_sha256(epsilon),
            native_tau=times[0].detach().float().cpu().reshape(-1).tolist(),stock_initial_noise=inherited._tensor_identity(stock),
            stock_initial_noise_used_for_generation=False,direct_query=inherited._tensor_identity(query),
            generated_output=inherited._tensor_identity(generated),runtime_id='PI0272_DIRECT_R1',execution_phase='training',
            feature_dtype=str(processed.backbone_features.dtype),query_dtype=str(query.dtype),output_dtype=str(generated.dtype),
            generated_nodes=1,rollout_steps=0,rollout_head_evaluations=1,checkpointed_generated_nodes=1,
            physical_position_loss=float(position.detach()),physical_delta_loss=float(delta.detach()),
            total_loss=float(total.detach()),expert_action_in_condition=False,active_action_modified=False,**details)
        inherited._append_json(audit_path,row,audit['forwards'])
        return {**native,'loss':total}
    head_cls.forward=forward
    return audit
