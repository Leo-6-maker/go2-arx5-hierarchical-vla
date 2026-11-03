"""Fixed 64-CALIB-state native W2 forks; requires an actual W1 endpoint."""
import argparse
import json
from pathlib import Path
import time


def diagnose(spec, calib, checkpoint, train_replay, output):
    import numpy as np
    import torch
    from copy import deepcopy
    from scipy.stats import spearmanr
    from go2arx5_vla.world_model.storage import load_private, write_json, dump_private, digest
    from go2arx5_vla.world_model.native import NativeAdapter
    from go2arx5_vla.world_model.contracts import compile_action, command_labels, macro_transition
    from go2arx5_vla.world_model.features import encode, SLICES
    from go2arx5_vla.world_model.model import Ensemble
    from go2arx5_vla.world_model.reward import reward
    from go2arx5_vla.world_model.replay import Replay
    from go2arx5_vla.world_model.diagnostics import ARMS, ACTION_SCALE, proposals, cluster_interval
    torch.set_num_threads(1)
    saved_model = torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved_model.get('schema')!='WM01_FULL_CHECKPOINT_V1' or saved_model['updates']!=5000:
        raise ValueError('W2 requires the fixed actual 5000-update first model')
    model = Ensemble(saved_model['model']['mean'],saved_model['model']['scale'])
    model.load_state_dict(saved_model['model'])
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    training = Replay(train_replay,split='TRAIN')
    if training.manifest_sha256!=saved_model['replay_identity']['manifest_sha256']:
        raise ValueError('Actual W1 model/TRAIN replay mismatch')
    mean_action = training.actions[:,0].mean(0)
    definition = json.loads((calib/'RESULTS.json').read_text())
    cases=[]
    for result in definition['episodes']:
        if result['split']!='CALIB':raise ValueError('W2 is fixed CALIB only')
        for name in result['branch_files']:
            cases.append((calib/result['episode_id'],name))
    if len(cases)!=64:
        raise ValueError('Retain the fixed 16x4 CALIB states; no success-based replacements')
    began,cpu=time.monotonic(),time.process_time();results=[];physics_ticks=shadow_ticks=0
    def guard():
        if time.time()>=spec['deadline_epoch']or(Path(spec['registered_root'])/'STOP.json').exists():
            raise TimeoutError('FINITE_W2_ENDPOINT')
    for index,(folder,name)in enumerate(cases):
        guard();saved=load_private(folder/name);card=saved['restore']['card'];state=saved['restore']['cached_state']
        x=encode(state,saved['restore']['controller'],saved['restore']['native'])
        base,represented=command_labels(saved['plan']['poses'][:8],saved['plan']['opening'][:8],state['tcp'])
        if not represented:raise ValueError('Fixed CALIB original action not representable; report definition failure')
        goal=torch.tensor([*card['goal_xy'],.46],dtype=torch.float32)[None]
        first_events=torch.tensor([[float(state[k])for k in ('left_object_contact','right_object_contact','native_grasp')]+[0.]],dtype=torch.float32)
        def predicted(action):
            returns=[]
            for member in model.members:
                current=torch.tensor(x)[None];events=first_events;total=action.new_zeros(1)
                for k in range(4):
                    if k==0:command=action
                    else:
                        command=torch.zeros_like(action).reshape(1,8,7)
                        command=torch.cat((command[...,:6],action.reshape(1,8,7)[:,-1:,6:].expand(1,8,1)),-1).reshape(1,56)
                    after,logits=member(current,command,goal,model.mean,model.scale)
                    next_events=torch.sigmoid(logits)
                    total=total+.99**k*reward(current,after,command,goal,events,next_events,next_events[:,-1])
                    current,events=after,next_events
                returns.append(total)
            return torch.stack(returns).mean()
        variable=torch.tensor(base.ravel()[None],dtype=torch.float32,requires_grad=True)
        predicted(variable).backward()
        gradient=variable.grad.detach().numpy().reshape(8,7)*ACTION_SCALE
        candidate,details=proposals(base,gradient,mean_action,seed=110300+index)
        rows={};predicted_returns={arm:float(predicted(torch.tensor(action.ravel()[None],dtype=torch.float32)).detach())for arm,action in candidate.items()}
        adapter=NativeAdapter(card,spec['asset_root'],executor_id=saved['restore']['executor_id'],seed=card['seed'])
        try:
            for arm in ARMS:
                adapter.restore(saved['restore']);start_ticks=adapter.engine.steps;start_shadow=adapter.shadow_ticks
                history=[];total=0.;failure=None;events=first_events
                for macro_index in range(4):
                    guard();before=adapter.snapshot();memory={k:deepcopy(getattr(adapter.engine,k))for k in saved['restore']['controller']}
                    native=adapter._get_data(adapter.engine.data)
                    action=candidate[arm]if macro_index==0 else np.c_[np.zeros((8,6)),np.full(8,candidate[arm][-1,6])]
                    plan=compile_action(action,before['tcp'],before['time_s'])
                    gate=adapter.gate(plan)
                    forecast=adapter.preflight(plan,guard)if gate['decision']!='REJECT'else None
                    if gate['decision']=='REJECT'or forecast['decision']!='PASS':
                        total-=.99**macro_index;failure='GATE_REJECT'if forecast is None else 'ORIGINAL_S1_REJECT';break
                    chunk=[]
                    for k in range(8):
                        row=adapter.advance_atom(plan,k,raw=action[k]);chunk.append(row)
                        if row['terminal']or row['truncated']:break
                    macro=macro_transition(chunk);history.append(macro);last=chunk[-1]
                    y=encode(last['state_after'],last['controller_memory_after'],last['native_after'])
                    next_events=torch.tensor([[float(last['state_after'][k])for k in ('left_object_contact','right_object_contact','native_grasp')]+[float(last['terminal'])]])
                    total+=.99**macro_index*float(reward(torch.tensor(encode(before,memory,native))[None],torch.tensor(y)[None],
                        torch.tensor(action.ravel()[None],dtype=torch.float32),goal,events,next_events,torch.tensor([float(last['terminal'])])))
                    events=next_events
                    if last['terminal']or last['truncated']:failure=last['safety_reason'];break
                ticks=adapter.engine.steps-start_ticks;shadows=adapter.shadow_ticks-start_shadow
                physics_ticks+=ticks;shadow_ticks+=shadows
                rows[arm]=dict(actual_return=total,predicted_return=predicted_returns[arm],failure=failure,
                    live_ticks=ticks,shadow_ticks=shadows,hard_safety=list(adapter.engine.safety),
                    final_object_pose=adapter.snapshot()['object_pose'],maximum_live_shadow_difference=adapter.max_shadow_error)
                dump_private(output/f'state-{index:02d}-{arm}.pkl.gz',dict(macros=history,action=candidate[arm],result=rows[arm]))
        finally:adapter.close()
        actual=[rows[a]['actual_return']for a in ARMS];prediction=[predicted_returns[a]for a in ARMS]
        feasible_delta=(candidate['gradient']-base)/ACTION_SCALE
        gradient_dot_delta=float(np.sum(gradient*feasible_delta))
        directional_fd=(actual[1]-actual[0])/max(details['matched_normalized_norm'],1e-12)
        correlation=float(spearmanr(actual,prediction).statistic)if len(set(actual))>1 and len(set(prediction))>1 else 0.
        shuffled=np.random.default_rng(110302+index).permutation(prediction)
        shuffled_correlation=float(spearmanr(actual,shuffled).statistic)if len(set(actual))>1 and len(set(prediction))>1 else 0.
        top=int(np.argmax(prediction));regret=max(actual)-actual[top]
        row=dict(index=index,scene=card['scene_group'],branch=name,arms=rows,proposal=details,
            rank_correlation=correlation,action_shuffle_rank_correlation=shuffled_correlation,
            top_choice_regret=regret,gradient_improved=actual[1]>actual[0]+1e-9,
            random_improved=actual[3]>actual[0]+1e-9,
            model_gradient_dot_feasible_delta=gradient_dot_delta,native_directional_finite_difference=directional_fd,
            directional_sign_agreement=bool(gradient_dot_delta>1e-12 and directional_fd>1e-9),
            positive_vs_negative_return=actual[1]>actual[2]+1e-9)
        write_json(output/f'state-{index:02d}.json',row);results.append(row)
    scenes=[r['scene']for r in results]
    improvement=cluster_interval([r['gradient_improved']for r in results],scenes)
    random=cluster_interval([r['random_improved']for r in results],scenes)
    rank=np.mean([r['rank_correlation']for r in results]);shuffle=np.mean([r['action_shuffle_rank_correlation']for r in results])
    violations=sum(bool(a['hard_safety'])for r in results for a in r['arms'].values())
    eligible=improvement['mean']>=.6 and improvement['mean']>random['mean']and rank>shuffle and violations==0
    result=dict(states=64,branches=320,source_sha=spec['source_sha'],checkpoint_sha256=digest(checkpoint),
        gradient_improvement=improvement,random_improvement=random,mean_rank_correlation=float(rank),
        mean_action_shuffle_rank_correlation=float(shuffle),mean_top_choice_regret=float(np.mean([r['top_choice_regret']for r in results])),
        native_directional_fd_sign_agreement_fraction=float(np.mean([r['directional_sign_agreement']for r in results])),
        positive_vs_negative_return_fraction=float(np.mean([r['positive_vs_negative_return']for r in results])),
        paired_rank_advantage=cluster_interval([r['rank_correlation']-r['action_shuffle_rank_correlation']for r in results],scenes),
        hard_safety_branches=violations,actor_BPTT_eligible=bool(eligible),
        live_physics_ticks=physics_ticks,shadow_physics_ticks=shadow_ticks,
        CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-began,actor_updates=0,
        continuation_rule='After first macro, zero task-frame deltas anchored to current actual TCP; hold last gripper',
        rejected_branches_in_denominator=True,TEST_read=False)
    write_json(output/'RESULT.json',result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('spec','calib','checkpoint','train-replay','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--run-id',required=True);parser.add_argument('--profile',required=True)
    args=parser.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(args.spec,args.run_id,args.profile,args.output,physics=True)as spec:
        print(json.dumps(diagnose(spec,args.calib,args.checkpoint,args.train_replay,args.output)),flush=True)


if __name__=='__main__':main()
