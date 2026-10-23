"""Test only the private-parser identity boundary locally; actual code on Core."""
import importlib.util
import os
from pathlib import Path
import pytest
from . import target_time_wire_cpu as unit

def test_changed_private_parser_rejects_before_any_import_or_execution(tmp_path,monkeypatch):
    p=tmp_path/'ee_trajectory.py';p.write_text('raise RuntimeError("must not execute unpinned parser")')
    monkeypatch.setattr(importlib.util,'spec_from_file_location',lambda *a,**k:pytest.fail('NO_UNPINNED_MODULE_IMPORT'))
    with pytest.raises(PermissionError,match='EXACT_FROZEN_PRIVATE_PARSER_BYTES'):unit.load_frozen_parser(p)

def test_actual_private_mujoco_parser_world_h40_receiver_contract():
    path=os.environ.get('PI0294_FROZEN_TARGET_PARSER')
    if not path:pytest.skip('actual private frozen MuJoCo parser requires registered Core runtime')
    r=unit.qualify_parser_file(Path(path))
    assert r['world_pose_samples_checked']==41 and r['invalid_cases_rejected']==11
    assert r['status']=='CPU_ACTUAL_FROZEN_TARGET_TIME_WIRE_PASS'
    assert not r['target_task_native_qualified'] and not r['current_native_admission']
