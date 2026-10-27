"""Cold recipe/reader/Trainer wiring. Framework fixtures are not native proof."""
from copy import deepcopy
import json
import base64
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from scripts.pi0294.generalization import minimal_study as study
from scripts.pi0294.generalization import minimal_training as training
from scripts.pi0294.generalization.minimal_dataset import Sampler, PANEL_SHA
from scripts.pi0294.test_pv1_recipe import old_recipe


def actual_panel():
    p = Path(__file__).resolve().parents[1]/'reports/PI0294/design/parallel-science-20261022'
    return json.loads((p/'ARX5_MINIMAL_ADAPTATION_TWELVE_PARENT_AND_CURSOR_METADATA_FROZEN_20261027.json').read_text())['panel']


def data_fixture():
    panel = actual_panel()
    return SimpleNamespace(panel=panel, rows=[(r, a) for r in panel['parents'] for a in range(601)],
                           manifest_sha='CAPTURED_PARENT_LIST_READER_FIXTURE_NOT_NATIVE_BANK')


def recipe(arm=training.ARMS[0]):
    return training.recipe(old_recipe(), arm, raw_parent_manifest_sha256=training.RAW_PARENT_FILES_ID)


def test_raw_parent_and_trainable_range_are_explicit_and_no_CORR_alias():
    a, b = recipe(), recipe(training.ARMS[1])
    different = {k for k in a if a[k] != b[k]}
    assert different == {'arm', 'trainable_modules'}
    assert a['pretrained_revision'] == training.RAW_REVISION
    assert a['initial_parent_checkpoint_manifest_sha256'] == training.RAW_PARENT_FILES_ID
    assert a['panel_sha256'] == PANEL_SHA and 'new_expert_fraction' not in a
    for wrong in ['94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38', '0' * 64]:
        with pytest.raises(ValueError, match='RAW_PARENT_IDENTITY'):
            training.recipe(old_recipe(), training.ARMS[0], raw_parent_manifest_sha256=wrong)


@pytest.mark.parametrize('field,value', [('max_steps', 512), ('warmup_updates', 32),
                                      ('statistics_fit', True), ('shared_position_embedding_frozen', False)])
def test_scientific_recipe_drift_rejected(field, value):
    r = recipe(); r[field] = value
    with pytest.raises(ValueError, match='EXACT_MINIMAL'):
        training.validate(r, old_recipe())


def test_first512_keeps_original1024_schedule_and_optimizer_arguments():
    r = recipe(); args = training.trainer_arguments(r, old_recipe()); rates = training.lr_schedule(r)
    assert args['max_steps'] == 1024 and args['gradient_accumulation_steps'] == 16
    assert rates[0] == 0 and rates[64] == 1e-5 and rates[-1] == pytest.approx(1e-6)
    assert rates[512] != pytest.approx(rates[-1], abs=1e-9)
    assert training.step_boundary(512, 16384, pending=[], unknown=[], session_stop=512)['should_training_stop']
    assert not training.step_boundary(512, 16384, pending=[], unknown=[], session_stop=1024)['should_training_stop']


@pytest.mark.parametrize('pending,unknown', [([{}], []), ([], ['UNKNOWN'])])
def test_uncommitted_or_unknown_effects_cannot_be_terminal(pending, unknown):
    with pytest.raises(ValueError, match='KNOWN_COMMITTED'):
        training.step_boundary(512, 16384, pending=pending, unknown=unknown, session_stop=512)


def test_actual_panel_sampler_reorders_rows_but_keeps_exact_cursor_identity():
    data = data_fixture(); full = Sampler(data)
    reversed_data = data_fixture(); reversed_data.rows.reverse(); other = Sampler(reversed_data)
    for cursor in [0, 1, 31, 32, 16383, 16384, 32767]:
        assert data.rows[full.index_at(cursor)] == reversed_data.rows[other.index_at(cursor)]
    resumed = Sampler(data, start=16384)
    assert list(full)[16384:] == list(resumed)
    assert full.state_dict(16384) == resumed.state_dict(16384)
    assert full.state_dict(16384)['stop'] == 32768


def resume_packet(step=512):
    sampler = Sampler(data_fixture())
    names = ['config.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json',
             'SAMPLER_STATE.json', 'statistics.json', 'ADAPTER_PARAMETER_CONTRACT.json',
             'FROZEN_PARAMETER_IDENTITY.json', 'MODEL_PARAMETER_STATE_IDENTITY.json',
             'model.safetensors', 'processor/config.json']
    files = {n: study.digest({'CPU_FIXTURE_MEMBER': n}) for n in names}
    files['statistics.json'] = study.R0
    r = recipe()
    complete = dict(global_step=step, training_identity=r['arm'], full_resume=True, files=files)
    saved = dict(recipe=r, sampler=sampler.state_dict(step * 32), manifest_sha256=sampler.dataset.manifest_sha,
                 statistics_sha256=study.R0)
    return saved, complete, dict(global_step=step), r, sampler, files


def test_own512_full_state_accepted_only_for_continuation_not_replayed_pilot():
    packet = resume_packet()
    assert training.resume_state(*packet, session_stop=1024) == 512
    with pytest.raises(ValueError, match='OWN_COMPLETE'):
        training.resume_state(*packet, session_stop=512)


@pytest.mark.parametrize('change', ['other_arm', 'missing_optimizer', 'changed_bytes', 'changed_R0', 'reset_cursor'])
def test_resume_rejects_other_arm_missing_state_or_reset(change):
    saved, complete, state, r, sampler, files = resume_packet()
    if change == 'other_arm': complete['training_identity'] = training.ARMS[1]
    elif change == 'missing_optimizer': files = {k: v for k, v in files.items() if k != 'optimizer.pt'}
    elif change == 'changed_bytes': files = dict(files, **{'scheduler.pt': 'CHANGED_CPU_FIXTURE'})
    elif change == 'changed_R0': saved['statistics_sha256'] = 'CHANGED_CPU_FIXTURE'
    elif change == 'reset_cursor': saved['sampler']['cursor'] = 0
    with pytest.raises(ValueError, match='OWN_COMPLETE'):
        training.resume_state(saved, complete, state, r, sampler, files, session_stop=1024)


def test_training_step_delegates_through_matched_rng_and_enforces512_boundary(monkeypatch, tmp_path):
    torch = pytest.importorskip('torch')
    from scripts.pi0294.mp1 import trainer as mature

    # Only the unavailable official framework layer is a CPU test fixture.
    # Production factory still resolves the original Gr00tTrainer/mature chain.
    class CPUFramework:
        pass

    class CPUBase:
        def __init__(self, *args, **kwargs):
            self.args, self.recipe, self.model = kwargs['args'], kwargs['recipe'], kwargs['model']
            self.state = SimpleNamespace(global_step=0); self.consumed = 0
            self.callbacks = kwargs['callbacks']

        def training_step(self, model, inputs, num_items_in_batch=None):
            self.consumed += 2
            return torch.randn(4), torch.distributions.Beta(1.5, 1.).sample((2,))

        def create_optimizer(self):
            self.optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad])
            return self.optimizer

        def train(self, resume_from_checkpoint=None, **kwargs):
            return {'CPU_FRAMEWORK_FIXTURE': True, 'resume': resume_from_checkpoint}

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            # Fixture can omit a tensor load, just like a permissive framework
            # loader could do. The new post-load identity must detect that.
            return None

    fake = ModuleType('gr00t.experiment.trainer'); fake.Gr00tTrainer = CPUFramework
    monkeypatch.setitem(sys.modules, 'gr00t.experiment.trainer', fake)
    fake_transformers = ModuleType('transformers'); fake_transformers.TrainerCallback = object
    monkeypatch.setitem(sys.modules, 'transformers', fake_transformers)
    monkeypatch.setattr(mature, 'make_trainer_class', lambda: CPUBase)
    cls = training.make_trainer_class()

    class Contract:
        def __init__(self, model, arm): self.model, self.arm = model, arm
        def validate(self): return None
        def parameters(self): return list(self.model.parameters())
        def enforce_modes(self): self.model.eval()

    def instance(arm):
        model = torch.nn.Linear(4, 4); c = Contract(model, arm)
        return cls(model=model, parameter_contract=c, session_stop=512, recipe=recipe(arm),
                   original_recipe=old_recipe(), known_effects=lambda: effects,
                   args=SimpleNamespace(max_steps=1024, per_device_train_batch_size=2, gradient_accumulation_steps=16))

    effects = dict(optimizer_updates=0, pending_updates=[], unknown_effects=[])
    a, b = instance(training.ARMS[0]), instance(training.ARMS[1])
    a.create_optimizer(); b.create_optimizer()
    inputs = {'inputs': {'action': torch.zeros(2, 40, 132)}}
    x = a.training_step(a.model, inputs); torch.rand(19); y = b.training_step(b.model, inputs)
    assert all(torch.equal(u, v) for u, v in zip(x, y))
    a.state.global_step = 512; a.consumed = 16384
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    with pytest.raises(ValueError, match='OWNER_COMMIT'):
        a.callbacks[-1].on_step_end(a.args, a.state, control)
    effects['optimizer_updates'] = 512
    a.callbacks[-1].on_step_end(a.args, a.state, control)
    assert control.should_training_stop and control.should_save
    with pytest.raises(ValueError, match='NO_EXTRA_MICROBATCH'):
        a.training_step(a.model, inputs)
    a.session_stop = 1024
    with pytest.raises(ValueError, match='CONTINUATION_MUST_RESTORE'):
        a.train()
    from scripts.pi0294.generalization.minimal_adapter import parameter_state_fingerprint
    expected = parameter_state_fingerprint(a.model)
    (tmp_path/'MODEL_PARAMETER_STATE_IDENTITY.json').write_text(json.dumps(expected))
    a._load_from_checkpoint(tmp_path)
    with torch.no_grad(): next(a.model.parameters()).add_(1)
    with pytest.raises(ValueError, match='EVERY_RESTORED'):
        a._load_from_checkpoint(tmp_path)


def test_checkpoint_member_paths_and_symlinks_cannot_escape(tmp_path):
    (tmp_path/'model.safetensors').write_bytes(b'CPU_TEST_ONLY')
    assert set(training.checkpoint_members(tmp_path, ['model.safetensors'])) == {'model.safetensors'}
    (tmp_path/'link').symlink_to(tmp_path/'model.safetensors')
    for name in ['../outside', str(tmp_path/'model.safetensors'), 'link']:
        with pytest.raises(ValueError, match='OWN_REGULAR'):
            training.checkpoint_members(tmp_path, [name])


def test_actual_original_reader_metadata_joins_fixed_panel_without_processor_or_RGB(tmp_path):
    path = os.environ.get('ARX5_ORIGINAL_METADATA_CAPTURE')
    if not path:
        pytest.skip('Actual private original metadata receipt not provided; no payload qualification')
    from scripts.pi0294.generalization.minimal_dataset import Dataset, ORIGINAL
    transport = json.loads(Path(path).read_text()); assert transport['returncode'] == 0
    captured = json.loads(transport['stdout'])
    for name in ('DATASET_MANIFEST.json', 'statistics.json'):
        (tmp_path/name).write_bytes(base64.b64decode(captured['files'][name]['base64']))
    panel = actual_panel()
    reader = Dataset(tmp_path, source='NOT_IMPORTED_IN_METADATA_CPU_TEST', cosmos='NOT_LOADED',
                     split='TRAIN', for_optimization=True, panel=panel)
    assert len(reader) == 7212 and reader.original_manifest_sha == ORIGINAL
    assert reader.statistics_sha == study.R0 and reader.processor is None
    assert {r['id'] for r, _ in reader.rows} == {r['id'] for r in panel['parents']}
    sampler = reader.make_sampler(reader)
    expected = list(study.paired_cursors(panel, stop=32))
    for cursor, index in list(sampler)[:32]:
        identity = reader.identity(index)
        assert identity['episode_id'] == expected[cursor]['parent']
        assert identity['anchor'] == expected[cursor]['anchor']
        assert identity['split'] == 'TRAIN'
