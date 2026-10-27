"""Cold recipe/session contract around the original Trainer; no runtime entry."""
from scripts.pi0294.pv1 import mix1024 as mature_recipe
from scripts.pi0294.e1.trainer import learning_rate_multiplier
from . import minimal_study as study
from .minimal_dataset import PANEL_SHA, CURSOR_SHA

SCHEMA = 'ARX5_MINIMAL_MATCHED_RAW_PARENT_512_THEN_CONDITIONAL1024_V1'
ARMS = ('E1_ADAPTER_ONLY', 'E2_SHARED_ACTION_HEAD')
RAW_REVISION = '2fc962b973bccdd5d8ce4f67cc63b264d6886495'
RAW_PARENT_FILES_ID = '0e493187af087d156f6352d46ce14e049ff18e7c1cb1eabc8671357ab4b96e71'
MODE = 'MATCHED_ACTION_HEAD_AND_BACKBONE_EVAL_WITH_AUTOGRAD_V1'


def recipe(original, arm, *, raw_parent_manifest_sha256):
    study.need(arm in ARMS and raw_parent_manifest_sha256 == RAW_PARENT_FILES_ID,
               'DISTINCT_ARM_AND_EXACT_RAW_PARENT_IDENTITY')
    r = mature_recipe.recipe(original, 'C50')
    for field in ('new_expert_fraction', 'entire_branch_new_fraction'):
        r.pop(field, None)
    r.update(schema=SCHEMA, route='RAW_PARENT_MINIMAL_ARX5', arm=arm,
             initial_parent_checkpoint_manifest_sha256=raw_parent_manifest_sha256,
             pretrained_repo='nvidia/GR00T-N1.7-3B', pretrained_revision=RAW_REVISION,
             initial_optimizer='FRESH_RAW_PRETRAINED_INDEPENDENT_PER_ARM',
             trainable_modules=['TARGET_CATEGORY10_STATE_ACTION_ENCODER_DECODER'] +
                 ([] if arm == ARMS[0] else ['SHARED_DIT_VLLN_VL_SELF_ATTENTION']),
             shared_position_embedding_frozen=True, target_category=10,
             module_mode=MODE, state_dropout_prob=0.,
             sampling='ARX5_MATCHED_TWELVE_PARENT_ABSOLUTE_CURSOR_V1',
             panel_sha256=PANEL_SHA, full_cursor_sha256=CURSOR_SHA,
             pilot_terminal_steps=512, scheduler_horizon_steps=1024, max_steps=1024,
             checkpoints=[512, 1024], development_candidate_steps=[512],
             conditional_second_endpoint_steps=1024, first_phase_sample_visits=16384,
             sample_visits=32768, flow_noise='ARM_INDEPENDENT_STEP_AND_MICROBATCH_SUBSTREAM_V1',
             statistics_sha256=study.R0, statistics_fit=False)
    return r


def validate(value, original):
    study.need(value == recipe(original, value['arm'],
                               raw_parent_manifest_sha256=value['initial_parent_checkpoint_manifest_sha256']),
               'EXACT_MINIMAL_MATCHED_RECIPE_NO_OLD_ARM_OR_NEW_NUMERICS')
    return value


def trainer_arguments(value, original):
    validate(value, original)
    # Reuse the exact original optimizer/numerical/data-loader arguments.
    args = mature_recipe.trainer_arguments(mature_recipe.recipe(original, 'C50'), original)
    study.need(args['max_steps'] == 1024 and args['gradient_accumulation_steps'] == 16
               and args['per_device_train_batch_size'] == 2, 'MATCHED_1024_SCHEDULE_BATCH32')
    return args


def step_boundary(step, cursor, *, pending, unknown, session_stop):
    study.need(type(session_stop) is int and session_stop in (512, 1024)
               and type(step) is int and 0 <= step <= session_stop
               and type(cursor) is int and cursor == step * 32 and not pending and not unknown,
               'KNOWN_COMMITTED_STEP_IN_FIXED_PILOT_OR_CONTINUATION')
    return dict(should_training_stop=step == session_stop,
                should_save=step in (512, 1024), full_resume_required=True)


def lr_schedule(value):
    study.need(value.get('schema') == SCHEMA and value.get('max_steps') == 1024
               and value.get('scheduler_horizon_steps') == 1024 and value.get('warmup_updates') == 64,
               'FIXED_1024_SCHEDULE_NOT_RESTARTED_AT512')
    return [value['learning_rate'] * learning_rate_multiplier(i, 1024, 64) for i in range(1025)]


def resume_state(saved, complete, trainer_state, value, sampler, member_hashes, *, session_stop):
    step = complete.get('global_step')
    files = complete.get('files', {})
    required = {'config.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json',
                'SAMPLER_STATE.json', 'statistics.json', 'ADAPTER_PARAMETER_CONTRACT.json',
                'FROZEN_PARAMETER_IDENTITY.json', 'MODEL_PARAMETER_STATE_IDENTITY.json'}
    study.need(session_stop in (512, 1024) and type(step) is int and 0 < step < session_stop
               and (session_stop == 512 or step >= 512)
               and complete.get('training_identity') == value['arm']
               and complete.get('full_resume') is True and required <= set(files)
               and any(n.endswith('.safetensors') for n in files)
               and any(n.startswith('processor/') for n in files)
               and files == member_hashes and files['statistics.json'] == study.R0
               and trainer_state.get('global_step') == step and saved.get('recipe') == value
               and saved.get('sampler') == sampler.state_dict(step * 32)
               and saved.get('manifest_sha256') == sampler.dataset.manifest_sha
               and saved.get('statistics_sha256') == study.R0,
               'OWN_COMPLETE_MODEL_OPTIMIZER_SCHEDULER_RNG_SAMPLER_NO_RESET')
    return step


def checkpoint_members(checkpoint, names):
    """Strict full-resume members cannot escape the sealed checkpoint root."""
    from pathlib import Path
    from scripts.pi0294.native_dataset import sha
    root = Path(checkpoint).resolve()
    result = {}
    for name in names:
        p = root/name
        study.need(isinstance(name, str) and not Path(name).is_absolute()
                   and '..' not in Path(name).parts and not p.is_symlink()
                   and p.is_file() and p.resolve().is_relative_to(root), 'OWN_REGULAR_CHECKPOINT_MEMBERS_ONLY')
        result[name] = sha(p)
    return result


def make_trainer_class():
    """Original data-loader/loss/optimizer/Trainer; new explicit terminal saves.

    An existing native owner MUST supply model/request/resource admission,
    pending/commit bookkeeping and its step-end update callback. The boundary
    observer is appended AFTER it, and only reads the owner's exact effects.
    This factory is not a launcher or a substitute for that owner integration.
    """
    from pathlib import Path
    import shutil
    from gr00t.experiment.trainer import Gr00tTrainer
    from transformers import TrainerCallback
    from scripts.pi0294.mp1.trainer import make_trainer_class as mature
    from scripts.pi0294.e1.trainer import check_trace
    from scripts.pi0292.durable import read, write
    from scripts.pi0294.native_dataset import sha
    from .minimal_adapter import frozen_fingerprint, matched_flow_rng, parameter_state_fingerprint

    class Boundary(TrainerCallback):
        def __init__(self, trainer):
            self.trainer = trainer

        def on_step_end(self, args, state, control, **kwargs):
            trainer = self.trainer
            effects = trainer.known_effects()
            study.need(effects['optimizer_updates'] == state.global_step,
                       'OWNER_COMMIT_MUST_PRECEDE_TERMINAL_OBSERVER')
            boundary = step_boundary(state.global_step, trainer.consumed,
                pending=effects['pending_updates'], unknown=effects['unknown_effects'],
                session_stop=trainer.session_stop)
            if boundary['should_save']:
                control.should_save = True
            if boundary['should_training_stop']:
                control.should_training_stop = True
            return control

    class Trainer(mature()):
        def __init__(self, *args, parameter_contract, session_stop, original_recipe, known_effects, **kwargs):
            r = kwargs['recipe']
            validate(r, original_recipe)
            study.need(r.get('schema') == SCHEMA and r.get('arm') == parameter_contract.arm
                       and kwargs['args'].max_steps == 1024 and session_stop in (512, 1024)
                       and parameter_contract.model is kwargs['model'] and callable(known_effects),
                       'EXPLICIT_MINIMAL_TRAINER_IDENTITY')
            parameter_contract.validate()
            self.parameter_contract, self.session_stop = parameter_contract, session_stop
            self.known_effects = known_effects
            self.frozen_baseline = frozen_fingerprint(parameter_contract.model)
            kwargs['callbacks'] = list(kwargs.get('callbacks', [])) + [Boundary(self)]
            super().__init__(*args, **kwargs)

        def create_optimizer(self):
            self.parameter_contract.validate()
            result = super().create_optimizer()
            actual = [p for g in self.optimizer.param_groups for p in g['params']]
            expected = self.parameter_contract.parameters()
            study.need(len(actual) == len(expected) and {id(p) for p in actual} == {id(p) for p in expected},
                       'OPTIMIZER_ONLY_THE_EXACT_DECLARED_PARAMETERS')
            return result

        def training_step(self, model, inputs, num_items_in_batch=None):
            self.parameter_contract.validate()
            batch = self.args.per_device_train_batch_size
            step = self.state.global_step
            microbatch = (self.consumed - step * 32) // batch
            study.need(0 <= step < self.session_stop
                       and 0 <= microbatch < self.args.gradient_accumulation_steps
                       and self.consumed == step * 32 + microbatch * batch,
                       'NO_EXTRA_MICROBATCH_OR_OPTIMIZER_STEP_AFTER_SESSION_TERMINAL')
            action = inputs['inputs']['action']
            devices = [action.device.index] if action.device.type == 'cuda' else []
            with matched_flow_rng(self.recipe['seed'], step=step, microbatch=microbatch,
                                  cuda_devices=devices):
                return super().training_step(model, inputs, num_items_in_batch)

        def train(self, resume_from_checkpoint=None, **kwargs):
            if resume_from_checkpoint:
                cp = Path(resume_from_checkpoint)
                complete = read(cp/'CHECKPOINT_COMPLETE.json')
                members = checkpoint_members(cp, complete['files'])
                sampler = self.train_dataset.make_sampler(self.train_dataset)
                resume_state(read(cp/'SAMPLER_STATE.json'), complete, read(cp/'trainer_state.json'),
                             self.recipe, sampler, members, session_stop=self.session_stop)
                study.need(read(cp/'ADAPTER_PARAMETER_CONTRACT.json') == self.parameter_contract.describe(),
                           'NO_WRONG_CATEGORY_OR_PARAMETER_MASK_RESTORE')
                study.need(read(cp/'FROZEN_PARAMETER_IDENTITY.json') == self.frozen_baseline,
                           'FROZEN_PARENT_BASELINE_UNCHANGED_ON_RESTORE')
            else:
                study.need(self.session_stop == 512, 'CONTINUATION_MUST_RESTORE_OWN512_OR_LATER_FULL_STATE')
            return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            result = super()._load_from_checkpoint(resume_from_checkpoint, model=model)
            self.parameter_contract.validate()
            actual = parameter_state_fingerprint(model if model is not None else self.model)
            expected = read(Path(resume_from_checkpoint)/'MODEL_PARAMETER_STATE_IDENTITY.json')
            study.need(actual == expected, 'EVERY_RESTORED_PARAMETER_AND_BUFFER_MUST_MATCH_FULL_STATE')
            self.parameter_contract.enforce_modes()
            return result

        def _save_checkpoint(self, model, trial):
            step = self.state.global_step
            study.need(0 < step <= self.session_stop and self.consumed == step * 32,
                       'SAVE_ONLY_EXACT_COMMITTED_CURSOR')
            self.parameter_contract.validate()
            study.need(frozen_fingerprint(self.model) == self.frozen_baseline, 'FROZEN_PARAMETERS_OR_BUFFERS_CHANGED')
            # Bypass only the historical256 inference-only save override. The
            # original official Trainer writes model/optimizer/scheduler/RNG.
            self.args.save_only_model = False
            Gr00tTrainer._save_checkpoint(self, model, trial)
            cp = Path(self.args.output_dir)/('checkpoint-' + str(step))
            self.train_dataset.initialize()
            self.train_dataset.processor.save_pretrained(cp/'processor')
            study.need(sha(self.train_dataset.root/'statistics.json') == study.R0, 'R0_UNCHANGED')
            shutil.copyfile(self.train_dataset.root/'statistics.json', cp/'statistics.json')
            trace = check_trace(self.train_dataset.trace_root, self.train_dataset, self.start_cursor, self.consumed)
            write(cp/'SAMPLER_STATE.json', dict(sampler=self.sampler.state_dict(self.consumed),
                  recipe=self.recipe, manifest_sha256=self.train_dataset.manifest_sha,
                  statistics_sha256=study.R0, worker_trace=trace))
            write(cp/'ADAPTER_PARAMETER_CONTRACT.json', self.parameter_contract.describe())
            write(cp/'FROZEN_PARAMETER_IDENTITY.json', self.frozen_baseline)
            write(cp/'MODEL_PARAMETER_STATE_IDENTITY.json', parameter_state_fingerprint(self.model))
            files = {str(p.relative_to(cp)): sha(p) for p in sorted(cp.rglob('*')) if p.is_file()}
            required = {'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'config.json', 'trainer_state.json',
                        'SAMPLER_STATE.json', 'statistics.json', 'ADAPTER_PARAMETER_CONTRACT.json',
                        'FROZEN_PARAMETER_IDENTITY.json', 'MODEL_PARAMETER_STATE_IDENTITY.json'}
            study.need(required <= set(files) and any(n.endswith('.safetensors') for n in files)
                       and any(n.startswith('processor/') for n in files), 'FULL_MINIMAL_CHECKPOINT_MEMBERS_REQUIRED')
            write(cp/'CHECKPOINT_COMPLETE.json', dict(global_step=step, files=files, full_resume=True,
                  training_identity=self.recipe['arm'], session_terminal=step == self.session_stop,
                  scheduler_horizon=1024, parameter_state_format='WRAPPED_CATEGORY_TABLES_FULL_RESUME_ONLY',
                  inference_delivery_requires_official_merge=True))
    return Trainer
