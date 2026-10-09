"""Read actual private executor/runtime identity; no policy forward or simulation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import types


def inference_package(root):
    # Inference modules share a package with PPO's wandb-dependent training runner.
    # Load their real source modules without executing the unused training __init__.
    name = 'go1_gym_learn.ppo_cse_automatic'
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(Path(root) / 'go1_gym_learn/ppo_cse_automatic')]
        sys.modules[name] = package


def file_identity(path):
    path = Path(path)
    if not path.is_file():
        return {"path": str(path), "status": "MISSING"}
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def inspect(config):
    root = Path(config['wbc_root'])
    sys.path.insert(0, str(root))
    import isaacgym  # Native import must precede torch.
    import torch
    inference_package(root)
    from go1_gym.envs.config import (build_roboduet_config, RoboDuetRuntimeOptions,
                                     apply_config_snapshot, recompute_observation_dims)
    from scripts.load_policy import _validate_checkpoint_layout
    run = Path(config['controller_run'])
    parameters = run / 'parameters.pkl'
    files = [file_identity(parameters)]
    missing = []
    snapshot = pickle.loads(parameters.read_bytes())['Cfg'] if parameters.exists() else {}
    cfg = build_roboduet_config(options=RoboDuetRuntimeOptions(num_envs=1, robot='go2_x5'))
    if snapshot:
        apply_config_snapshot(cfg, snapshot, drop_unknown=True)
        recompute_observation_dims(cfg)
    layouts = {}
    # Same checkpoint-era layout rule as delivered loader; never pad/truncate inputs.
    recorded = snapshot.get('arm', {}).get('arm_num_observations')
    if recorded == int(cfg.arm.arm_num_observations) + 16 and cfg.wbc.goal_reaching.enabled:
        cfg.arm.checkpoint_observation_layout = 'goal_reaching_extended_v1'
        cfg.arm.arm_num_observations = int(recorded)
        cfg.arm.arm_num_obs_history = int(recorded) * cfg.arm.arm_num_observation_history
    for role in ('arm', 'dog'):
        p = run / ('checkpoints_' + role) / ('ac_weights_' + config['checkpoint_id'] + '.pt')
        files.append(file_identity(p))
        if not p.is_file():
            missing.append(str(p))
        else:
            state = torch.load(p, map_location='cpu', weights_only=True)
            layouts[role] = _validate_checkpoint_layout(state, role, cfg)
    asset = Path(cfg.asset.file.replace('{MINI_GYM_ROOT_DIR}', str(root)))
    sources = ['go1_gym/envs/config/wbc.py', 'go1_gym/envs/roboduet/legged_robot.py',
               'go1_gym/envs/roboduet/wbc_env.py', 'modules/trajectory.py',
               'scripts/load_policy.py', 'scripts/play_by_key_stage2_traj.py']
    files += [file_identity(root / p) for p in sources] + [file_identity(asset)]
    files += [file_identity(p) for p in sorted((asset.parent.parent / 'meshes').rglob('*')) if p.is_file()]
    if not asset.is_file() or 'go2_x5' not in str(asset):
        missing.append('checkpoint-bound Go2-X5 asset')
    if not parameters.exists():
        missing.append('complete parameters.pkl / observation and command layout')
    if getattr(cfg.asset, 'fix_base_link', True):
        missing.append('free-floating controller configuration')
    if not getattr(cfg.wbc.goal_reaching, 'target_mode', '') == 'trajectory':
        missing.append('Stage2 trajectory configuration')
    return {
        'status': 'BLOCKED_EXECUTOR_ASSET' if missing else 'COMPLETE_HISTORICAL_EXECUTOR_IDENTITY_CPU',
        'controller_class': config['controller_class'], 'missing': missing,
        'not_new_WBCv1_stage2_delivery': config['controller_class'] == 'HISTORICAL_EXECUTOR_BASELINE',
        'runtime': {'python': sys.version, 'python_executable': sys.executable,
                    'isaacgym_file': isaacgym.__file__, 'torch': torch.__version__,
                    'cuda': torch.version.cuda, 'sim_dt_s': cfg.sim.dt,
                    'decimation': cfg.control.decimation,
                    'actor_dt_s': cfg.sim.dt * cfg.control.decimation},
        'controller': {'action_mode': cfg.arm.action_mode, 'tcp_link': cfg.asset.ee_body_name,
                       'tcp_local_position_m': list(cfg.arm.ik.ee_local_pos), 'layouts': layouts,
                       'native_safety': {k: getattr(cfg.wbc.rewards, k) for k in
                           ('use_terminal_body_height', 'use_terminal_roll', 'use_terminal_pitch')},
                       'max_gamma_points': cfg.wbc.goal_reaching.trajectory.max_gamma_points,
                       'max_tl_points': cfg.wbc.goal_reaching.trajectory.max_tl_points},
        'files': files, 'neural_GR00T_forwards': 0, 'WBC_forwards': 0,
        'physics_integrations': 0, 'selected': 'NONE', 'deployment': 'NONE'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = inspect(json.loads(Path(args.config).read_text()))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'missing', 'controller', 'runtime')}))


if __name__ == '__main__':
    main()
