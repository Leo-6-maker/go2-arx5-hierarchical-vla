"""PI0263 F/X training; retains PI0261's qualified two-microbatch runtime."""
import argparse
import copy
import hashlib
import json
import os
import sys
import shutil
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import IterableDataset

ROOT = Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0263')
R3 = Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0260/R3')
STAGE = Path('/scratch/e1192548/stage22_h200_runtime_deb920_20260902')
GROOT = STAGE / 'src/isaac-gr00t'
COSMOS = Path('/scratch/e1192548/stage22_h200_cosmos_v2_20260902/hf_home/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561')
OLD = ROOT.with_name('PI0261')
DONOR = OLD / 'P/main/training/checkpoint-2000'
sys.path[:0] = [str(Path(__file__).resolve().parent), str(R3 / 'execution'),
                '/scratch/e1192548/pi0244_fixed16_10k_v1/scripts',
                '/scratch/e1192548/pi0244_fixed16_10k_v1/src', str(GROOT)]

import pi0260_r3_train as r3
from pi0260_r3_inputs import batch, identity
from pi0263_plan import seed_for as plan_seed_for, language, NONLANGUAGE


def read(path):
    return json.loads(Path(path).read_text())


def seed_for(unit):
    return plan_seed_for(unit)


class Pairs(IterableDataset):
    def __init__(self, processor, output, arm, resume):
        self.processor, self.output, self.arm = processor, output, arm
        self.seed, self.cursor = 26301, 2 * resume
        self.plan = read(ROOT / 'PAIRING_PLAN.json')
        assert len(self.plan['units']) == 2000 and len(self.plan['common_start_bank']) == 60
        self.bank = {(row['task'], row['family']): row for row in self.plan['common_start_bank']}
        registry = read(ROOT / 'registry.json')
        self.colors = {r['family_id']: r['factors']['colors'] for r in registry['families']}

    def reset_seed(self, seed):
        self.cursor = 2 * (seed - self.seed)
        assert 0 <= self.cursor < 4000 and self.cursor % 2 == 0

    @lru_cache(maxsize=160)
    def sequence(self, name):
        path = R3 / 'source/dataset' / name
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as z:
            states, labels = z['state_v1'].copy(), z['raw_action'].copy()
        return read(path.with_suffix('.json')), states, labels

    def __iter__(self):
        audit = self.output / 'sample_audit.jsonl'
        for unit in self.plan['units'][self.cursor // 2:]:
            selected = unit['selected']
            for micro, command in enumerate(unit['order']):
                item = selected[command]
                rec, states, labels = self.sequence(item['sequence'])
                j = next(j for j, row in enumerate(rec['rows']) if row['index'] == item['index'])
                assert rec['rows'][j]['frame_sha256'] == item['frame_sha256']
                assert rec['rows'][j]['active'] and rec['rows'][j]['usable']
                assert rec['sequence']['split'] == 'TRAIN' and rec['sequence']['command'] == command
                color = self.colors[unit['family']][command]
                text, template = language(self.arm, unit, rec['sequence'], color)
                with np.load(OLD / 'source' / item['frame'], allow_pickle=False) as z:
                    packet = dict(body_rgb=z['body_rgb'], wrist_rgb=z['wrist_rgb'],
                                  state_v1=states[j], language=text)
                value = batch(self.processor, packet, labels[j])
                condition = {k: v for k, v in value['inputs'].items() if k not in ('action', 'action_mask')}
                key = identity(condition)
                value['_pi0263_meta'] = dict(unit=unit['unit'], micro=micro, command=command,
                                             condition_sha256=key)
                with audit.open('a') as stream:
                    stream.write(json.dumps(dict(unit=unit['unit'], micro=micro, arm=self.arm,
                        task=unit['task'], family=unit['family'], slot=unit['slot'], command=command,
                        sequence=item['sequence'], index=item['index'], condition_sha256=key,
                        original_unit=unit['original_unit'], visit=unit['visit'], color=color, template=template,
                        language=text, nonlanguage_sha256=identity({k:value['inputs'][k] for k in NONLANGUAGE}),
                        labels_masks_sha256=identity({k:value['inputs'][k] for k in ('action','action_mask')})),
                        sort_keys=True) + '\n')
                yield value


def install(args, output):
    from gr00t.model.gr00t_n1d7 import setup
    from gr00t.experiment import experiment
    import pi0247_u_rollout_overlay as rollout

    # Reuse the exact R3 full-S4 physical loss, decoder parity, and frozen model scope.
    audit = r3.install(SimpleNamespace(resume=2 * args.resume, limit=args.limit), output)
    old_dataset = setup.Gr00tN1d7Pipeline._create_dataset

    def create_dataset(self, cfg_dir):
        visits, _ = old_dataset(self, cfg_dir)
        return Pairs(visits.processor, output, args.arm, args.resume), None

    setup.Gr00tN1d7Pipeline._create_dataset = create_dataset
    base_trainer = experiment.Gr00tTrainer
    native_training_step = base_trainer.__bases__[0].training_step
    current = {}
    original_append = rollout._append_json

    def capture(path, row, count):
        original_append(path, row, count)
        if path == output / 'S4_audit.jsonl':
            current['s4'] = row

    rollout._append_json = capture

    class PairTrainer(base_trainer):
        def _save_checkpoint(self, *pos, **kw):
            # Full Trainer write overlaps the current resume point, plus 10 GiB reserve.
            required = read(ROOT / 'DONOR_LOCAL_READBACK.json')['complete_trainer_bytes'] + 10 * 2**30
            assert shutil.disk_usage('/scratch/e1192548').free >= required, 'CHECKPOINT_WRITE_HEADROOM'
            return super()._save_checkpoint(*pos, **kw)

        def save_model(self, output_dir=None, *pos, **kw):
            # The final bare save duplicates the complete checkpoint model on scratch.
            if output_dir is None and self.state.global_step == args.limit:
                assert (output / f'checkpoint-{args.limit}' / 'trainer_state.json').is_file()
                return
            return super().save_model(output_dir, *pos, **kw)

        def training_step(self, model, inputs, *pos, **kw):
            if not hasattr(self, '_pi_head_hook'):
                heads_path = ROOT / 'raw_training_heads.jsonl'
                self._pi_heads = sum(1 for _ in heads_path.open()) if heads_path.exists() else 0
                def record_head(_module, _inputs):
                    self._pi_heads += 1
                    assert self._pi_heads <= 100000, 'RAW_TRAINING_HEAD_CAP'
                    rollout._append_json(heads_path, dict(ordinal=self._pi_heads, arm=args.arm,
                        phase=args.phase, job=os.environ['PBS_JOBID'],
                        unit=self.state.global_step+1, rollout_phase=audit.get('phase'),
                        grad_enabled=torch.is_grad_enabled()), self._pi_heads)
                self._pi_head_hook = model.action_head.model.register_forward_pre_hook(record_head)
                native_clip = self.accelerator.clip_grad_norm_
                def record_clip(*clip_args, **clip_kw):
                    norm = native_clip(*clip_args, **clip_kw)
                    rollout._append_json(output / 'update_gradient_norms.jsonl',
                        dict(unit=self.state.global_step+1, accumulated_norm=float(norm.detach()),
                             microbatches=2, clip_max_norm=1.), self.state.global_step+1)
                    return norm
                self.accelerator.clip_grad_norm_ = record_clip
                from transformers import TrainerCallback
                class Accounting(TrainerCallback):
                    def on_step_end(_self, _args, state, control, **_kw):
                        path = ROOT / 'raw_completed_updates.jsonl'
                        n = sum(1 for _ in path.open()) + 1 if path.exists() else 1
                        assert n <= 4288, 'RAW_COMPLETED_UPDATE_CAP'
                        rollout._append_json(path, dict(ordinal=n, arm=args.arm, phase=args.phase,
                            job=os.environ['PBS_JOBID'], unit=state.global_step), n)
                self.add_callback(Accounting())
            micro_global = getattr(self, '_pi_micro', 2 * args.resume) + 1
            unit = (micro_global + 1) // 2
            micro = (micro_global - 1) % 2
            assert self.args.gradient_accumulation_steps == 2
            assert self.state.global_step + 1 == unit <= args.limit
            meta = inputs.pop('_pi0263_meta')
            assert set(meta) == {'unit', 'micro', 'command', 'condition_sha256'}
            assert (meta['unit'], meta['micro']) == (unit, micro)
            condition = {k: v for k, v in inputs['inputs'].items() if k not in ('action', 'action_mask')}
            key = identity(condition)
            assert meta['condition_sha256'] == key and '_pi0263_meta' not in inputs
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(seed_for(unit))
                # Force Trainer's native 1/2 scaling for each supervised microbatch.
                loss = native_training_step(self, model, inputs, None)
            row = current.pop('s4')
            assert row['forward_index'] == micro_global
            packet = (row['native_epsilon'], row['stock_initial_noise'], row['native_tau'])
            if micro == 0:
                self._pi_first_packet = packet
            else:
                assert self._pi_first_packet == packet, ('PAIRED_RANDOM_PACKET_MISMATCH', unit)
                del self._pi_first_packet
            with (output / 'step_rng.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(unit=unit, micro=micro, command=meta['command'],
                    condition_sha256=key, seed=seed_for(unit), native_epsilon=row['native_epsilon'],
                    stock_initial_noise=row['stock_initial_noise'], native_tau=row['native_tau'],
                    loss=float(loss.detach()), mode_restored=True), sort_keys=True) + '\n')
            self._pi_micro = micro_global
            return loss

    experiment.Gr00tTrainer = PairTrainer
    return audit


def run(args):
    assert os.environ.get('PBS_JOBID') and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.device_count() == 1
    output = ROOT / args.arm / args.phase / 'training'
    output.mkdir(parents=True, exist_ok=True)
    assert read(ROOT / 'DONOR_LOCAL_READBACK.json')['status'] == 'PI0263_FIXED_P2000_LOCAL_READBACK_PASS'
    assert read(ROOT / 'PAIRING_PLAN.json')['schema'] == 'PI0263_COUNTERBALANCED_PLAN_V1'
    assert read(ROOT / 'PROCESSED_PLAN_AUDIT.json')['status'] == 'PASS_FINITE_ACTUAL_COLOR_INPUT_LABEL_TEMPLATE_RNG_CHECKS'
    if args.phase == 'main':
        assert read(ROOT / args.arm / 'canary_compare.json')['status'] == 'PASS_PI0263_REAL_TWO_MICROBATCH_EXACT_RESUME_CANARY'
    if args.resume:
        checkpoint = output / f'checkpoint-{args.resume}'
        assert all((checkpoint / name).is_file() for name in
                   ('trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth'))
        assert read(checkpoint / 'trainer_state.json')['global_step'] == args.resume
        assert sum(1 for _ in (output / 'S4_audit.jsonl').open()) == 2 * args.resume
        assert sum(1 for _ in (output / 'sample_audit.jsonl').open()) == 2 * args.resume
    else:
        assert not (output / 'S4_audit.jsonl').exists()
    import pi0228_h200_large_adapt_v1 as old
    from stage40_b0_model_control import _bind_local_cosmos_processor
    assert _bind_local_cosmos_processor(STAGE, COSMOS)['resolved_snapshot'] == str(COSMOS.resolve())
    config_args = argparse.Namespace(stage10_root=STAGE, base_model_path=DONOR,
        dataset_dir=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0227/pi0227_large_adapt_v1_inputs/train_dataset'),
        modality_config=Path('/scratch/e1192548/pi0227_h200_prep_20260919/inputs/go2_arx5_modality.py'),
        groot_source=GROOT)
    save_steps = 1000 if args.phase == 'main' else (4 if args.phase == 'split' else 8)
    config = old.build_config(config_args, output, steps=2000, save_steps=save_steps, save_limit=2)
    config.training.learning_rate = 1e-5
    config.training.lr_scheduler_type = 'cosine'
    config.training.gradient_accumulation_steps = 2
    config.training.max_steps = 2000
    config.training.save_steps = save_steps
    config.training.save_total_limit = 2
    config.training.resume_from_checkpoint = bool(args.resume)
    config.data.seed = config.training.seed = 26301
    config.training.remove_unused_columns = False
    assert config.training.warmup_steps == 10
    assert config.training.max_grad_norm == 1. and config.training.weight_decay == 1e-5
    audit = install(args, output)
    old.run_official(config, GROOT)
    assert audit['forwards'] == 2 * args.limit
    assert read(output / f'checkpoint-{args.limit}' / 'trainer_state.json')['global_step'] == args.limit
    if audit['first_update_euler_gradient_norms'] and not args.resume:
        (output / 'S4_first_update_gradients.json').write_text(
            json.dumps(audit['first_update_euler_gradient_norms'], sort_keys=True) + '\n')
    print(json.dumps(dict(status='PI0263_SEGMENT_COMPLETE', arm=args.arm, phase=args.phase,
                          step=args.limit, microbatches=audit['forwards'])), flush=True)


def self_check():
    x = torch.tensor(0.3, requires_grad=True)
    losses = [(x - target) ** 2 for target in (1., -0.4)]
    for loss in losses:
        (loss / 2).backward(retain_graph=True)
    split = x.grad.detach().clone()
    x.grad = None
    (sum(losses) / 2).backward()
    assert torch.allclose(split, x.grad, rtol=0, atol=1e-7)
    assert [((2 * step - 1 + 1) // 2, (2 * step + 1) // 2) for step in range(1, 9)] == [(s, s) for s in range(1, 9)]
    def packet(unit):
        with torch.random.fork_rng():
            torch.manual_seed(seed_for(unit))
            return torch.randn(1, 40, 132), torch.rand(1)

    for unit in range(1, 9):
        a, b = packet(unit), packet(unit)
        assert all(torch.equal(left, right) for left, right in zip(a, b))

    def toy(start=0, saved=None, interrupt_at=None):
        weight = torch.nn.Parameter(torch.tensor(0.3))
        optimizer = torch.optim.AdamW([weight], lr=1e-5, betas=(.9, .999), eps=1e-8, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
        if saved is not None:
            weight.data.copy_(saved['weight'])
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
        checkpoint = saved
        for unit in range(start + 1, 9):
            optimizer.zero_grad()
            for micro, target in enumerate((1., -.4)):
                ((weight - target) ** 2 / 2).backward()
                if interrupt_at == (unit, micro):
                    raise RuntimeError('DISCARDED_MID_UNIT')
            optimizer.step()
            scheduler.step()
            assert optimizer.state[weight]['step'].item() == unit and scheduler.last_epoch == unit
            if unit == 4:
                checkpoint = dict(weight=weight.detach().clone(),
                                  optimizer=copy.deepcopy(optimizer.state_dict()),
                                  scheduler=copy.deepcopy(scheduler.state_dict()))
        return weight.detach().clone(), optimizer.state_dict(), scheduler.state_dict(), checkpoint

    continuous = toy()
    _, _, _, checkpoint = toy()
    try:
        toy(start=4, saved=checkpoint, interrupt_at=(5, 0))
    except RuntimeError as error:
        assert str(error) == 'DISCARDED_MID_UNIT'
    else:
        raise AssertionError('missing midunit interruption')
    resumed = toy(start=4, saved=checkpoint)
    assert torch.equal(continuous[0], resumed[0])
    assert continuous[2] == resumed[2]
    print('PASS_PI0263_CPU_HALF_GRADIENT_ONE_STEP_CURSOR_PACKET_AND_RESUME_CHECK')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('F', 'X'))
    parser.add_argument('--phase', choices=('continuous', 'split', 'main'))
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        assert args.arm and args.phase and 0 <= args.resume < args.limit <= 2000
        run(args)
