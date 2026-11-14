import json
import os
from contextlib import ExitStack

import pytest

from go2arx5_vla.world_model import corelab_world_owner_v3 as world
from go2arx5_vla.world_model import pilot1_storage
from go2arx5_vla.world_model.corelab_expert_semantics_v3 import ROLE
from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model.task_learning_grasp_v1 import PHASE
from scripts.pi0292.durable import OwnerLock


@pytest.mark.parametrize('count', [2,3])
def test_independent_worlds_can_hold_their_locks_beside_legacy_owner(tmp_path, monkeypatch,count):
    campaign = tmp_path / 'campaign'
    locks = campaign / 'control/corelab-world-owners-v3'
    locks.mkdir(parents=True)
    pool = tmp_path / 'worlds'
    pool.mkdir()
    worlds = [dict(root=str(pool / f'world-{i}'), cpus=[i + 4], ROS_DOMAIN_ID=192 + i) for i in range(count)]
    for slot in worlds:
        (pool / slot['root']).mkdir()
    binding = dict(schema=world.SCHEMA, plan_revision=world.PLAN, worlds=worlds,
                   pool_root=str(pool), campaign_root=str(campaign), source_sha='a' * 40, reserved_cpus=[0, 1, 2, 3])
    config = tmp_path / 'binding.json'
    config.write_text(json.dumps(binding))
    pin = dict(path=str(config), sha256=digest(config))
    specs, entries = [], []
    for slot in worlds:
        root, domain = slot['root'], slot['ROS_DOMAIN_ID']
        name = root.rsplit('/', 1)[1]
        specs.append(dict(registered_root=root, campaign_root=str(campaign), source_sha='a' * 40,
                          corelab_world_binding_v3=pin, task_learning_role=ROLE, phase=PHASE,
                          v3_plan_revision=world.PLAN, ROS_DOMAIN_ID=domain, cpus=slot['cpus'],
                          native_environment=dict(ROS_DOMAIN_ID=str(domain)), role=name, owner='owner',
                          **{key: f'{root}/{key}' for key in ('logs', 'tmp', 'cache', 'data')}))
        entries.append(dict(path=root, run_uuid=name, role=name, source_sha='a' * 40, owner='owner',
                            ROS_DOMAIN_ID=domain, corelab_world_binding_sha256=pin['sha256'],
                            corelab_world_owner=str(locks / f'ros-{domain}.lock')))
    monkeypatch.setattr(os, 'sched_getaffinity', lambda pid: set(range(8)))
    monkeypatch.setattr(pilot1_storage, 'local_path', lambda spec, root: root)
    registry = dict(entries=entries)
    legacy = OwnerLock(campaign / 'OWNER.lock')
    try:
        with ExitStack() as stack:
            for spec in specs:stack.enter_context(world.world_owner(spec,registry))
            with pytest.raises(BlockingIOError):
                with world.world_owner(specs[0], registry):
                    pytest.fail('same domain admitted twice')
    finally:
        legacy.close()
    with pytest.raises(PermissionError):
        with world.world_owner(dict(specs[0], task_learning_role='SLA_STUDENT'), registry):
            pytest.fail('foreign role admitted')
    worlds[1]['cpus'] = worlds[0]['cpus']
    with pytest.raises(PermissionError):
        world.validate_layout(binding)
