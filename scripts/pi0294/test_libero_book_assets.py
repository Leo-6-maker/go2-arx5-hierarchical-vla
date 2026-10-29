"""Pinned real XML and the original native predicate; no simulation."""
import os
from pathlib import Path
import numpy as np
import pytest
from scripts.pi0294.multitask import libero_book_assets as a


def test_native_goal_uses_object_origin_with_strict_bounds_and_original_z_extension():
    kwargs=dict(site_position=[0,0,0],site_rotation=np.eye(3),half_size=[.1,.2,.3])
    assert a.native_site_contains(**kwargs,object_body_origin=[0,0,-.305])
    assert not a.native_site_contains(**kwargs,object_body_origin=[0,0,-.31])
    assert not a.native_site_contains(**kwargs,object_body_origin=[.1,0,0])
    assert a.native_site_contains(**kwargs,object_body_origin=[.099,0,0])
    # This predicate cannot admit a demonstration or assert dynamics support.
    assert a.native_site_contains(**kwargs,object_body_origin=[0,0,0])


def test_preserves_native_rotated_aabb_approximation_without_silently_fixing_it():
    c=2**-.5;r=np.array([[c,-c,0],[c,c,0],[0,0,1]])
    # Original abs(R @ half_size) differs from abs(R) @ half_size.
    assert not a.native_site_contains(site_position=[0,0,0],site_rotation=r,
        half_size=[1,2,1],object_body_origin=[1,0,0])
    assert a.native_site_contains(site_position=[0,0,0],site_rotation=r,
        half_size=[1,2,1],object_body_origin=[.1,0,0])


def test_actual_minimum_xmls_keep_all_original_collision_geometry_and_goal_sites():
    root=os.environ.get('NIGHT_LIBERO_ASSET_CAPTURE_DIR')
    if not root:pytest.skip('actual pinned private upstream XML capture required')
    root=Path(root)/'libero/libero/assets/turbosquid_objects'
    book=(root/'black_book/black_book.xml').read_bytes();caddy=(root/'desk_caddy/desk_caddy.xml').read_bytes()
    b=a.pinned_definition('black_book',book);c=a.pinned_definition('desk_caddy',caddy)
    assert b['collision_geoms'][0]['size']=='0.01434 0.05519 0.06720'
    assert len(c['collision_geoms'])==17
    assert c['containment_sites']['left_contain_region']['position']==[.00631,.14551,.06558]
    assert c['containment_sites']['right_contain_region']['position']==[.00631,-.14339,.06558]
    assert set(b['asset_files'])=={'black_book_texture.png','visual/black_book_vis.msh'}
    assert not b['has_executed_physics']and not c['training_admitted']
    with pytest.raises(ValueError):a.pinned_definition('black_book',book.replace(b'0.01434',b'0.01500'))


def test_invalid_transforms_and_unknown_asset_definitions_fail():
    with pytest.raises(ValueError):a.pinned_definition('other',b'<mujoco/>')
    with pytest.raises(ValueError):a.native_site_contains(site_position=[0,0,0],site_rotation=np.zeros((3,3)),half_size=[1,1,1],object_body_origin=[0,0,0])
    with pytest.raises(ValueError):a.native_site_contains(site_position=[0,0,0],site_rotation=np.eye(3),half_size=[1,-1,1],object_body_origin=[0,0,0])
