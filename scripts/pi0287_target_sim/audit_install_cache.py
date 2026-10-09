"""Witness an install/readout cache gap using actual native canonicalization on CPU."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[2]
    private = Path(os.environ['PI0287_WBC_ROOT'])
    source = private / 'go1_gym/envs/roboduet/wbc_env.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WBCEnv')
    getter = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_arm_observations')
    refresh = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_update_ee_task_space_error')
    consumed = {n.attr for n in ast.walk(getter) if isinstance(n, ast.Attribute) and
                isinstance(n.value, ast.Name) and n.value.id == 'self'}
    refreshed = {n.attr for n in ast.walk(refresh) if isinstance(n, ast.Attribute) and
                 isinstance(n.value, ast.Name) and n.value.id == 'self'}
    fields = ('ee_pos_err', 'ee_rot_err_axis_angle', 'arm_target_pos_body', 'arm_target_quat_body')
    assert set(fields) <= consumed and set(fields) <= refreshed

    # Reuse the existing actual Gamma/TimeLaw CPU setup; no WBC policy is constructed.
    spec = importlib.util.spec_from_file_location('cache_audit_setup', repo / 'tests/target_sim/test_index_bridge.py')
    tests = importlib.util.module_from_spec(spec); spec.loader.exec_module(tests)
    tests.NativeBridgeTests.setUpClass()
    test = tests.NativeBridgeTests(); test.setUp()
    torch = test.torch
    before = {}
    for field in fields:
        value = torch.full((2, 4 if field.endswith('quat_body') else 3), .25)
        setattr(test.wbc, field, value)
        before[field] = value.clone()
    called = []
    test.wbc._update_ee_task_space_error = lambda: called.append('native_refresh')
    chunk = test.chunk(eef_position=__import__('numpy').tile([2., 3., 4.], (40, 1)))
    receipt = test.bridge.submit(0, chunk, issue_timestamp=10., install_timestamp=10.)
    assert receipt['status'] == 'INSTALLED'
    torch.testing.assert_close(test.wbc.arm_goal_pos_world[0], torch.tensor([2., 3., 4.]))
    unchanged = {field: torch.equal(before[field], getattr(test.wbc, field)) for field in fields}
    assert all(unchanged.values()) and not called
    result = dict(status='CPU_REPRODUCED_INSTALL_READOUT_CACHE_GAP',
        native_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        bridge_source_sha256=hashlib.sha256((repo/'src/go2arx5_vla/target_sim/bridge.py').read_bytes()).hexdigest(),
        witness='Actual native Gamma/TimeLaw with synthetic CPU cache arrays; no actor inference',
        installed_world_position=test.wbc.arm_goal_pos_world[0].tolist(),
        native_actor_consumes_cached_fields=list(fields), cached_fields_unchanged=unchanged,
        native_refresh_called=False,
        actual_first_fixture_actor_caches='UNOBSERVED_NOT_RECORDED',
        explains_actual_failure='NOT_PROVEN', original_physical_failures_preserved=True,
        implementation_gap='Bridge install and nominal_reset do not refresh native task-error/body-target caches before first actor readout',
        proposed_repair='Refresh native derived target state before inference; verify actual cache parity without altering progress/gains',
        runtime_repair_applied=False, extra_GPU_run_allowed=False,
        new_physics=0, WBC_forwards=0, GR00T_forwards=0, optimizer_updates=0,
        selected='NONE', deployment='NONE')
    output = repo / 'reports/PI0287/execution/INSTALL_CACHE_CPU_AUDIT.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'explains_actual_failure', 'new_physics', 'WBC_forwards')}))


if __name__ == '__main__':
    main()
