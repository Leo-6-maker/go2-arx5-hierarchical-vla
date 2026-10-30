from copy import deepcopy
import hashlib

import numpy as np
import pytest

from scripts.pi0294.multitask.libero_book_cards import cards
from scripts.pi0294.multitask.libero_book_teacher import (
    RELEASE_REVISION, route_card, seed_card, checked_route_card, attempt_operation, SEED_OFFSETS_M)
from scripts.pi0294.multitask.source_stage_bundle import raw, BOOK_RELEASE_CARD_SHA, BOOK_SEED_CARD_SHA, checked_manifest


def test_base_card_unchanged_and_four_explicit_distinct_initial_conditions():
    base=route_card(RELEASE_REVISION)
    assert hashlib.sha256(raw(dict(left=base,right=cards()['right']))).hexdigest()==BOOK_RELEASE_CARD_SHA
    for i,offset in enumerate(SEED_OFFSETS_M,1):
        s=seed_card(i);assert checked_route_card(s)
        delta=np.array(s['initial_book_native_world'])-np.array(base['initial_book_native_world'])
        np.testing.assert_allclose(delta[:2,3],offset,atol=1e-16)
        for key in ('initial_tcp','caddy_world','book','book_route','independent_demo_conditions','instruction','task_duration_s','warmup_s'):
            assert s[key]==base[key]
        assert hashlib.sha256(raw(dict(left=s,right=cards()['right']))).hexdigest()==BOOK_SEED_CARD_SHA[i]
        assert attempt_operation(s,'left').endswith(f'_SEED{i:03}')
        bad=deepcopy(s);bad['initial_book_native_world'][0][3]+=.001
        with pytest.raises(PermissionError):checked_route_card(bad)


@pytest.mark.parametrize('index',[True,False,0,5,1.,'1'])
def test_undeclared_or_mistyped_seed_rejected(index):
    with pytest.raises(ValueError):seed_card(index)


def test_seed_manifest_cannot_change_task_or_version():
    m=dict(task_kind='BOOK',condition_side='left',book_route_revision=RELEASE_REVISION,
        book_seed_index=1,predeclared_tasks=['left'],attempts_this_chain_max=1)
    assert checked_manifest(m)[1]==[dict(task_kind='BOOK',side='left')]
    for k,v in [('book_seed_index',True),('book_seed_index',5),('book_route_revision','UNKNOWN')]:
        bad=dict(m,**{k:v})
        with pytest.raises(PermissionError):checked_manifest(bad)
