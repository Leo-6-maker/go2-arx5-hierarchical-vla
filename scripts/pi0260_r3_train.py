"""PI0260-R3 matched Motion2 training on the sealed R2 data and GR00T runtime."""
import argparse
import hashlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

ROOT = Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0260/R3')
OLD = Path('/scratch/e1192548/Projects/go2-arx5')
STAGE = Path('/scratch/e1192548/stage22_h200_runtime_deb920_20260902')
GROOT = STAGE / 'src/isaac-gr00t'
COSMOS = Path('/scratch/e1192548/stage22_h200_cosmos_v2_20260902/hf_home/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561')
G_MODEL = OLD / 'experiments/PI0253/initializer_G2000'
BASE_MODEL = ROOT / 'initializer_Base'
LAST_SAMPLE = {}


def read(path):
    return json.loads(Path(path).read_text())


def seed_for(step):
    return int.from_bytes(hashlib.sha256(f'PI0260|26003|{step}'.encode()).digest()[:8], 'big') % (2**31)


class Visits(IterableDataset):
    def __init__(self, processor, output, start=0):
        self.processor, self.output, self.cursor = processor, output, start
        self.seed = 26003
        self.rows = read(ROOT/'train_visits.json')['visits']
        assert len(self.rows) == 6000

    def reset_seed(self, seed):
        self.cursor = seed - self.seed
        assert 0 <= self.cursor < 6000

    @lru_cache(maxsize=160)
    def sequence(self, name):
        path = ROOT/'source/dataset'/name
        with np.load(path.with_suffix('.npz')) as z:
            labels = z['raw_action'].copy()
            states = z['state_v1'].copy()
        return read(path.with_suffix('.json')), states, labels

    def get_dataset_statistics(self):
        return read(ROOT/'source/dataset/statistics.json')

    def __iter__(self):
        from pi0260_r3_inputs import batch, identity
        audit = self.output/'sample_audit.jsonl'
        for row in self.rows[self.cursor:]:
            rec, states, labels = self.sequence(row['sequence'])
            index = row['index']
            position = next(i for i, item in enumerate(rec['rows']) if item['index'] == index)
            assert rec['rows'][position]['frame_sha256'] == row['frame_sha256']
            assert rec['rows'][position]['time'] == row['time_s']
            with np.load(ROOT/'source'/row['frame']) as z:
                packet = dict(body_rgb=z['body_rgb'], wrist_rgb=z['wrist_rgb'],
                              state_v1=states[position], language=rec['sequence']['instruction'])
            value = batch(self.processor, packet, labels[position])
            condition = {k:v for k,v in value['inputs'].items() if k not in ('action','action_mask')}
            LAST_SAMPLE.clear()
            LAST_SAMPLE.update(update=row['update'], condition_sha256=identity(condition))
            with audit.open('a') as stream:
                stream.write(json.dumps(dict(update=row['update'], task=row['task'],
                    command=row['command'], sequence=row['sequence'], index=index,
                    condition_sha256=LAST_SAMPLE['condition_sha256']), sort_keys=True)+'\n')
            yield value


def install(args, output):
    from gr00t.model.gr00t_n1d7 import setup
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
    from gr00t.experiment import experiment
    from transformers import TrainerCallback
    from pi0260_r3_inputs import build, identity
    from pi0260_r3_physical import bounds, install as install_physical
    import pi0247_u_rollout_overlay as rollout
    import pi0248_rotation_cpu as rotation
    from pi0252_speed_margin import peak_speed_margin

    statistics = read(ROOT/'source/dataset/statistics.json')
    lo, hi = bounds(statistics)
    state = statistics['new_embodiment']['state']['eef_9d']
    slo = torch.tensor(state['q01'], dtype=torch.float32)
    shi = slo + torch.maximum(torch.tensor(state['q99'], dtype=torch.float32)-slo,
                              torch.full_like(slo, 1e-6))
    assert lo.shape == hi.shape == (40,9) and slo.shape == shi.shape == (9,)
    install_physical()
    assert rollout.CLAMP_NORMALIZED_STATE is False
    rollout.LAMBDA_POS, rollout.LAMBDA_DELTA = .4, .1
    current = {}
    native_physical = rollout._physical_losses

    def physical(generated, expert, mask, state_eef, scale, offset):
        position, delta, details = native_physical(generated,expert,mask,state_eef,scale,offset)
        rot, rot_details = rotation.rotation_chordal_loss(generated,expert,mask,state_eef,lo[:,3:9],hi[:,3:9])
        xyz = rollout.decode_xyz(generated[...,:3].float(),state_eef.float(),scale.float(),offset.float())
        margin, peak_z, segment, speed = peak_speed_margin(xyz, torch.full((39,),.05), mask)
        current.update(rotation=rot, speed_margin=margin, peak_speed_mps=speed,
                       peak_segment=segment, peak_z=peak_z, rotation_details=rot_details)
        return position, delta, details

    rollout._physical_losses = physical
    processor_ref = {}
    audit = rollout.install(Gr00tN1d7ActionHead,
        scale=(hi[:,:3]-lo[:,:3])/2, offset=(hi[:,:3]+lo[:,:3])/2,
        state_min=slo, state_max=shi, audit_path=output/'S4_audit.jsonl',
        parity_path=output/'S4_parity.json', processor_getter=lambda:processor_ref.get('value'),
        embodiment_tag='new_embodiment', resume_after_step=args.resume,
        expected_r_star='S4')
    native_head = Gr00tN1d7ActionHead.forward

    def full_head(self, backbone_output, action_input):
        current.clear()
        result = native_head(self,backbone_output,action_input)
        assert set(('rotation','speed_margin')) <= set(current)
        extra = .1*current['rotation'] + .1*current['speed_margin']
        result['loss'] = result['loss'] + extra
        assert torch.isfinite(result['loss'])
        with (output/'full_loss.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=audit['forwards'],rotation=float(current['rotation'].detach()),
                speed_margin=float(current['speed_margin'].detach()),
                peak_speed_mps=float(current['peak_speed_mps'].detach()),
                total=float(result['loss'].detach())),sort_keys=True)+'\n')
        return result

    Gr00tN1d7ActionHead.forward = full_head

    def create_dataset(self, _save_cfg_dir):
        processor = build(statistics, COSMOS)
        self.processor = processor
        processor_ref['value'] = processor
        return Visits(processor,output,args.resume),None

    setup.Gr00tN1d7Pipeline._create_dataset = create_dataset
    setup.Gr00tN1d7Pipeline._create_collator = lambda self: (lambda rows: rows[0])

    class StopAt(TrainerCallback):
        def on_step_end(self, _args, state, control, **_kwargs):
            if state.global_step == args.limit:
                control.should_save = True
                control.should_training_stop = True
            return control

    original_trainer = experiment.Gr00tTrainer
    class MatchedTrainer(original_trainer):
        def __init__(self,*pos,**kw):
            super().__init__(*pos,**kw)
            self.add_callback(StopAt())

        def training_step(self,model,inputs,*pos,**kw):
            step = self.state.global_step+1
            assert 1 <= step <= args.limit
            condition = {k:v for k,v in inputs['inputs'].items() if k not in ('action','action_mask')}
            key = identity(condition)
            assert LAST_SAMPLE == dict(update=step,condition_sha256=key), (LAST_SAMPLE,step,key)
            devices = [torch.cuda.current_device()]
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed_for(step))
                result = super().training_step(model,inputs,*pos,**kw)
            with (output/'step_rng.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(step=step,seed=seed_for(step),condition_sha256=key,
                    mode_restored=True),sort_keys=True)+'\n')
            return result

    experiment.Gr00tTrainer = MatchedTrainer
    return audit


def run(args):
    assert os.environ.get('PBS_JOBID'), 'real training requires a PBS allocation'
    assert torch.cuda.device_count()==1 and torch.cuda.get_device_properties(0).total_memory >= 100*(1<<30)
    output = ROOT/args.arm/args.phase/'training'
    output.mkdir(parents=True,exist_ok=True)
    if args.resume:
        checkpoint=output/f'checkpoint-{args.resume}'
        assert all((checkpoint/f).is_file() for f in ('trainer_state.json','optimizer.pt','scheduler.pt','rng_state.pth'))
        assert read(checkpoint/'trainer_state.json')['global_step']==args.resume
    else:
        assert not (output/'S4_audit.jsonl').exists()
    sys.path[:0]=[str(Path(__file__).resolve().parent),
                  '/scratch/e1192548/pi0244_fixed16_10k_v1/scripts',
                  '/scratch/e1192548/pi0244_fixed16_10k_v1/src',str(GROOT)]
    import pi0228_h200_large_adapt_v1 as old
    from stage40_b0_model_control import _bind_local_cosmos_processor
    assert _bind_local_cosmos_processor(STAGE, COSMOS)['resolved_snapshot'] == str(COSMOS.resolve())
    model = BASE_MODEL if args.arm=='Base' else G_MODEL
    config_args=argparse.Namespace(stage10_root=STAGE,base_model_path=model,
        dataset_dir=OLD/'experiments/PI0227/pi0227_large_adapt_v1_inputs/train_dataset',
        modality_config=Path('/scratch/e1192548/pi0227_h200_prep_20260919/inputs/go2_arx5_modality.py'),groot_source=GROOT)
    config=old.build_config(config_args,output,steps=6000,save_steps=6000,save_limit=3)
    config.training.learning_rate=1e-5
    config.training.lr_scheduler_type='cosine'
    config.training.max_steps=6000
    config.training.save_steps=6000
    config.training.save_total_limit=3
    config.training.resume_from_checkpoint=str(output/f'checkpoint-{args.resume}') if args.resume else False
    config.data.seed=26003
    config.training.seed=26003
    config.training.remove_unused_columns=False
    audit=install(args,output)
    old.run_official(config,GROOT)
    assert audit['forwards']==args.limit,(audit['forwards'],args.limit)
    assert read(output/f'checkpoint-{args.limit}/trainer_state.json')['global_step']==args.limit
    if audit['first_update_euler_gradient_norms']:
        (output/'S4_first_update_gradients.json').write_text(json.dumps(audit['first_update_euler_gradient_norms'],sort_keys=True)+'\n')
    print(json.dumps(dict(status='R3_SEGMENT_COMPLETE',arm=args.arm,phase=args.phase,
        step=args.limit,first_update_euler_gradients=audit['first_update_euler_gradient_norms'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',required=True,choices=('Base','G'))
    parser.add_argument('--phase',required=True)
    parser.add_argument('--limit',type=int,required=True)
    parser.add_argument('--resume',type=int,default=0)
    args=parser.parse_args()
    assert 0 <= args.resume < args.limit <= 6000
    run(args)
