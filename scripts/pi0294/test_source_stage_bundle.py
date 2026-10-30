"""Real private pinned assets, no MuJoCo import or physical integration."""
from copy import deepcopy
import base64
import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.pi0294.multitask import source_stage_bundle as bundle
from scripts.pi0294.multitask import night_source_stage as stage
from scripts.pi0294.multitask.libero_book_cards import cards as book_cards
from scripts.pi0294.multitask.robocasa_cabinet_cards import cards as cabinet_cards
from scripts.pi0294.multitask.robocasa_cabinet_assets import XML_PATHS


@pytest.fixture(scope='module')
def actual():
    directory = os.environ.get('NIGHT_SOURCE_PRIVATE_CAPTURE_DIR')
    if not directory:
        pytest.skip('Explicit existing private pinned member capture required')
    root = Path(directory)
    original = json.loads((root/'NIGHT_SOURCE_NATIVE_STATIC_PREPARED.private.json').read_bytes())['payload']
    p = deepcopy(original)
    p.update(task_kind='ALL_PINNED', cards=dict(BOOK=book_cards(), ROBOCASA_CABINET=cabinet_cards()))
    cabinet = root/'robocasa-pinned-minimum-source30'
    p['robocasa_definition_files'] = {
        name: base64.b64encode((cabinet/'robocasa/models/assets'/path).read_bytes()).decode()
        for name, path in XML_PATHS.items()}
    p['robocasa_texture_files'] = {}
    for name, filename in [('wood', 'official-wood-tiles.png'), ('metal', 'official-bright_metal.png')]:
        b = (cabinet/filename).read_bytes()
        p['robocasa_texture_files'][name] = dict(base64=base64.b64encode(b).decode(),
            bytes=len(b), sha256=hashlib.sha256(b).hexdigest())
    p['qualification'] = dict(p['qualification'],
        fixed_task_cards_sha256=hashlib.sha256(bundle.raw(p['cards'])).hexdigest(),
        independent_raw_auditors=['scripts/pi0294/multitask/night_book_audit.py',
                                 'scripts/pi0294/multitask/night_cabinet_audit.py'])
    return original, p


def test_actual_complete_two_and_four_task_members_and_plan(actual):
    original, mixed = actual
    old = bundle.unpack(original)
    new = bundle.unpack(mixed)
    assert len(old['definitions']) == 2 and len(old['binaries']) == 4
    assert len(new['definitions']) == 5 and len(new['binaries']) == 6
    assert new['total_binary_bytes'] == 18345876
    assert new['plan'] == [dict(task_kind=k, side=s) for k, s in
        [('BOOK', 'left'), ('BOOK', 'right'), ('ROBOCASA_CABINET', 'right'), ('ROBOCASA_CABINET', 'left')]]
    assert set(new['robocasa_definition_members']) == set(bundle.CABINET_XML)
    assert set(new['robocasa_texture_members']) == {'wood', 'metal'}


@pytest.mark.parametrize('fault', ['missing_texture', 'changed_texture', 'changed_XML', 'short_task', 'book_only'])
def test_incomplete_or_changed_mixed_payload_rejected_before_staging(actual, fault):
    p = deepcopy(actual[1])
    if fault == 'missing_texture':
        del p['robocasa_texture_files']['metal']
    elif fault == 'changed_texture':
        row = p['robocasa_texture_files']['wood']
        b = bytearray(base64.b64decode(row['base64'])); b[-1] ^= 1
        row.update(base64=base64.b64encode(b).decode(), sha256=hashlib.sha256(b).hexdigest())
    elif fault == 'changed_XML':
        row = p['robocasa_definition_files']['slab']
        p['robocasa_definition_files']['slab'] = base64.b64encode(base64.b64decode(row)+b'\n').decode()
    elif fault == 'short_task':
        p['cards']['ROBOCASA_CABINET']['left']['task_duration_s'] = 31.
        p['qualification']['fixed_task_cards_sha256'] = hashlib.sha256(bundle.raw(p['cards'])).hexdigest()
    else:
        p.update(task_kind='BOOK', cards=book_cards())
        p['qualification']['fixed_task_cards_sha256'] = bundle.CARD_SHA['BOOK']
    with pytest.raises((ValueError, PermissionError)):
        bundle.unpack(p)


def test_manifest_cannot_silently_drop_two_conditions_and_native_programs_compile():
    tasks = bundle.plan('ALL_PINNED')
    m = dict(task_kind='ALL_PINNED', predeclared_tasks=tasks, attempts_this_chain_max=4)
    assert bundle.checked_manifest(m) == ('ALL_PINNED', tasks)
    for field, value in [('attempts_this_chain_max', 2), ('predeclared_tasks', ['left', 'right'])]:
        with pytest.raises(PermissionError):
            bundle.checked_manifest(dict(m, **{field: value}))
    assert bundle.checked_manifest(dict(predeclared_tasks=['left', 'right'], attempts_this_chain_max=2))[0] == 'BOOK'
    for name in ('PROGRAM', 'READBACK', 'LAUNCH'):
        compile(getattr(stage, name), name, 'exec')


def test_single_book_manifest_requires_the_actual_static_route_revision():
    m=dict(task_kind='BOOK',condition_side='left',book_route_revision=bundle.BOOK_ROUTE_VERSION,
           predeclared_tasks=['left'],attempts_this_chain_max=1)
    assert bundle.checked_manifest(m)==('BOOK',[dict(task_kind='BOOK',side='left')])
    for changes in [dict(condition_side='right'),dict(book_route_revision='UNKNOWN'),
                    dict(attempts_this_chain_max=2),dict(predeclared_tasks=['left','right'])]:
        with pytest.raises(PermissionError):bundle.checked_manifest(dict(m,**changes))


@pytest.mark.parametrize('revision',[bundle.BOOK_ROUTE_VERSION,bundle.BOOK_MARGIN_VERSION,bundle.BOOK_DESCENT_VERSION,bundle.BOOK_RELEASE_VERSION])
def test_actual_static_book_card_assets_and_single_condition_are_bound_together(actual,revision):
    from scripts.pi0294.multitask.libero_book_teacher import route_card
    p=deepcopy(actual[0]);p.update(task_kind='BOOK',condition_side='left',
        book_route_revision=revision,cards=dict(left=route_card(revision),right=book_cards()['right']))
    pin={bundle.BOOK_ROUTE_VERSION:bundle.BOOK_ROUTE_CARD_SHA,bundle.BOOK_MARGIN_VERSION:bundle.BOOK_MARGIN_CARD_SHA,
         bundle.BOOK_DESCENT_VERSION:bundle.BOOK_DESCENT_CARD_SHA,bundle.BOOK_RELEASE_VERSION:bundle.BOOK_RELEASE_CARD_SHA}[revision]
    p['qualification'].update(condition_side='left',book_route_revision=revision,fixed_task_cards_sha256=pin)
    assert bundle.unpack(p)['plan']==[dict(task_kind='BOOK',side='left')]
    for changes in [dict(condition_side=None),dict(book_route_revision='UNKNOWN')]:
        with pytest.raises(PermissionError):bundle.unpack(dict(p,**changes))
    p['cards']['left']['book_route']['front_cross_book_x_m']-=.005
    with pytest.raises(PermissionError):bundle.unpack(p)
