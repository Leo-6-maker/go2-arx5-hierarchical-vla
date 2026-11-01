from copy import deepcopy
import hashlib
import numpy as np
import pytest
from scripts.pi0294.multitask.book_geometry_cards import geometry_card,conditions
from scripts.pi0294.multitask.libero_book_teacher import route_card,RELEASE_REVISION,checked_route_card,attempt_operation
from scripts.pi0294.multitask.libero_book_cards import cards
from scripts.pi0294.multitask.source_stage_bundle import raw,BOOK_GEOMETRY_CARD_SHA,checked_manifest
from scripts.pi0294.multitask.source_export import source_identity


def test_new_train_identities_preserve_asset_home_goal_score_and_old_seed_population():
    original=route_card(RELEASE_REVISION);ids=set()
    for i in range(1,7):
        c=geometry_card(i);checked_route_card(c);ids.add(c['id'])
        assert 'book_seed_index'not in c and c['study_split']=='TRAIN'
        for key in('initial_tcp','caddy_world','instruction','task_duration_s','warmup_s','source_port'):
            assert c[key]==original[key]
        assert c['book']['target_book_world']==original['book']['target_book_world']
        assert c['book_route']==original['book_route']
        if i<=2:
            assert c['book']==original['book']
            assert not np.array_equal(c['initial_book_native_world'],original['initial_book_native_world'])
        else:assert c['initial_book_native_world']==original['initial_book_native_world']
        assert hashlib.sha256(raw(dict(left=c,right=cards()['right']))).hexdigest()==BOOK_GEOMETRY_CARD_SHA[i]
        assert 'GEOMETRY'in attempt_operation(c,'left')
        row=source_identity(c,dict(native_LIBERO_goal_final=True));assert row['episode_id']==f'BOOK_GEOMETRY_TRAIN{i:03}'
    assert len(ids)==6 and len(conditions())==6


def test_geometry_identity_cannot_alias_old_seeds_or_confirmation_or_change_score():
    c=geometry_card(1);c['book_seed_index']=1
    with pytest.raises(PermissionError):checked_route_card(c)
    for value in(True,0,7):
        with pytest.raises(ValueError):geometry_card(value)
    c=geometry_card(2);c['task_duration_s']=20
    with pytest.raises(PermissionError):checked_route_card(c)
    m=dict(task_kind='BOOK',condition_side='left',book_route_revision=RELEASE_REVISION,
        book_geometry_index=1,predeclared_tasks=['left'],attempts_this_chain_max=1)
    checked_manifest(m);m['book_seed_index']=1
    with pytest.raises(PermissionError):checked_manifest(m)
