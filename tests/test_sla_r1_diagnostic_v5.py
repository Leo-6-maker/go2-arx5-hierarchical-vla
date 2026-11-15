import copy
import json
from pathlib import Path

import pytest

from go2arx5_vla.world_model import sla_r1_diagnostic_v5 as r1, task_learning_grasp_v1 as task
from go2arx5_vla.world_model.storage import digest


def panel():
    d, _ = r1.declaration()
    path = Path(r1.__file__).resolve().parents[3]/r1.DECLARATION
    return dict(schema=r1.SCHEMA, task_learning_role=task.SLA_ROLE,
        preregistration=dict(path=str(path), sha256=digest(path)), B1_global_step=500,
        population='TRAIN8', protocol=task.protocol(task.SLA_ROLE),
        policies={m:{} for m in r1.METHODS}, assignments=[dict(row, method=m,
            native_run=f'fixed-{m[:2].lower()}-{row["layout"]}-r{row["restart"]}',
            purpose=r1.SCHEMA, command=task.COMMAND) for m in r1.METHODS for row in d['population']])


def test_r0_r1_full_fixed_order_no_layout_or_seed_change():
    p = panel()
    assert len(r1.validate_panel(p)) == 16
    for change in ('seed', 'order', 'restart', 'population'):
        bad = copy.deepcopy(p)
        if change == 'seed':bad['assignments'][8]['query_seeds'][0] += 1
        if change == 'order':bad['assignments'].reverse()
        if change == 'restart':bad['assignments'][0]['restart'] = 8
        if change == 'population':bad['population'] = 'DEV16'
        with pytest.raises(ValueError):r1.validate_panel(bad)


def test_qualified_provider_change_is_rejected_but_export_change_is_separate(tmp_path):
    source = Path(r1.__file__).resolve().parents[3]
    import shutil
    root = tmp_path/'src/go2arx5_vla/sla_adapter';root.mkdir(parents=True)
    for p in (source/'src/go2arx5_vla/sla_adapter').glob('*.py'):shutil.copyfile(p,root/p.name)
    assert r1.inference_identity(source,tmp_path)
    p = root/'provider_v3.py'
    p.write_text(p.read_text()+'\n# export-only comment\n')
    assert r1.inference_identity(source,tmp_path)
    p.write_text(p.read_text().replace('num_steps=10','num_steps=9'))
    with pytest.raises(ValueError,match='PROVIDER_CLASS'):r1.inference_identity(source,tmp_path)
