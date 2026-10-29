"""Static task inspection only; no simulator or third-party framework imports."""
import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.pi0294.multitask.open_source_tasks import inspect_libero_book_task


# Synthetic grammar fixture. The actual upstream three-file check is separate.
SYNTHETIC = b'''(define (problem example) (:domain robosuite)
(:language place the book in the left region)
(:regions (left_region (:target fixture_1)))
(:fixtures fixture_1 - desk_caddy)
(:objects book_1 - black_book distractor_1 - mug)
(:init (On book_1 table))
(:goal (And (In book_1 fixture_1_left_region))))'''


def inspect(body):
    return inspect_libero_book_task(body, expected_sha256=hashlib.sha256(body).hexdigest())


def test_static_result_preserves_original_predicate_and_never_admits():
    result = inspect(SYNTHETIC)
    assert result['upstream_goal'] == dict(predicate='In', object='book_1', region='fixture_1_left_region')
    assert result['original_initial_state'] == [['On', 'book_1', 'table']]
    grouped = inspect(SYNTHETIC.replace(b'book_1 - black_book', b'book_1 book_2 - black_book'))
    assert grouped['objects']['book_1'] == grouped['objects']['book_2'] == 'black_book'
    assert all(result[k] is False for k in ('task_executed', 'action_conversion_qualified',
        'demonstration_admitted', 'execution_authority'))


@pytest.mark.parametrize('body', [SYNTHETIC[:-1], SYNTHETIC+b' extra', SYNTHETIC+SYNTHETIC,
    SYNTHETIC.replace(b'(:domain robosuite)', b'(:domain robosuite) (:domain robosuite)'),
    SYNTHETIC.replace(b'(In book_1', b'(In undeclared_book'),
    SYNTHETIC.replace(b'fixture_1_left_region', b'fixture_1_missing_region'),
    SYNTHETIC.replace(b'(In book_1', b'(On book_1')])
def test_malformed_or_semantically_different_task_is_rejected(body):
    with pytest.raises(ValueError):
        inspect(body)


def test_byte_tamper_is_not_hidden_by_equivalent_grammar():
    with pytest.raises(ValueError, match='EXACT_PINNED'):
        inspect_libero_book_task(SYNTHETIC+b'\n', expected_sha256=hashlib.sha256(SYNTHETIC).hexdigest())


def test_three_actual_pinned_upstream_files_and_matched_language_pair():
    path = os.environ.get('OPEN_SOURCE_PRIMARY_READBACK')
    if not path:
        pytest.skip('Explicit primary-source metadata capture required')
    capture = json.loads(Path(path).read_bytes())
    bodies = [capture['files'][4]['result']['content'].encode(),
              capture['extra_files'][0]['result']['content'].encode(),
              capture['extra_files'][1]['result']['content'].encode()]
    pins = ['7a46821e87185877e6bf9f2128f56842cb2bb58e3822f0b0a9650d4b78250ae4',
            '4a4ce354be7b6123b80a2d28a21d055cfd775c2841b94e205c32ae83a2902e67',
            'b17612a9a5fb0314c5aadb297cf8936d8e36ae6d97840e2d7fd8622399321323']
    results = [inspect_libero_book_task(b, expected_sha256=pin) for b, pin in zip(bodies, pins)]
    left, right, shelf = results
    assert left['original_regions'] == right['original_regions']
    assert left['original_initial_state'] == right['original_initial_state']
    assert left['upstream_goal']['region'] != right['upstream_goal']['region']
    assert shelf['upstream_goal'] == dict(predicate='In', object='black_book_1',
        region='wooden_two_layer_shelf_1_top_region')
