"""Declared EEF10 interface using the unchanged official native processor."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

BASE = Path('/home/liuyu-nfs/Projects/go2-arx5')
ROOT = BASE / 'PI0282/data'
PRISTINE = Path('/var/cache/users/liuyu-nfs/go2-arx5-vla-stage10/assets/GR00T-N1.7-3B')
OFFICIAL = BASE / 'PI0280/official'
COSMOS = BASE / 'PI0250/runtime/hf_home/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561'
TAG = 'new_embodiment'
INSTRUCTION = 'Stack the red cube on the green cube and release it.'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def native_imports():
    official=Path(os.environ.get('PI0282_OFFICIAL_DIR', str(OFFICIAL)))
    sys.path.insert(0, str(official))
    import gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 as native_module
    assert Path(native_module.__file__).resolve().is_relative_to(official.resolve())
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import VLAStepData
    return Gr00tN1d7Processor, EmbodimentTag.NEW_EMBODIMENT, VLAStepData


def modality():
    return {
        'video': {'delta_indices': [0], 'modality_keys': ['base_camera', 'hand_camera']},
        'state': {'delta_indices': [0], 'modality_keys': ['eef_9d', 'gripper_position']},
        'action': {'delta_indices': list(range(40)), 'modality_keys': ['eef_9d', 'gripper_position'],
                   'action_configs': [
                       {'rep': 'RELATIVE', 'type': 'EEF', 'format': 'XYZ_ROT6D', 'state_key': 'eef_9d'},
                       {'rep': 'ABSOLUTE', 'type': 'NON_EEF', 'format': 'DEFAULT', 'state_key': 'gripper_position'}]},
        'language': {'delta_indices': [0], 'modality_keys': ['annotation.human.task_description']}}


def fields(array):
    import numpy as np
    x = np.asarray(array)
    assert x.shape[-1] == 10 and np.isfinite(x).all()
    return {'eef_9d': x[..., :9], 'gripper_position': x[..., 9:10]}


class Codec:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        cls, self.tag, self.step_type = native_imports()
        kwargs = json.loads((self.root / 'processor_config.json').read_text())['processor_kwargs']
        kwargs['statistics'] = json.loads((self.root / 'statistics.json').read_text())
        kwargs['embodiment_id_mapping'] = json.loads((self.root / 'embodiment_id.json').read_text())
        kwargs['model_name'] = os.environ.get('PI0282_COSMOS_DIR', str(COSMOS))
        kwargs['transformers_loading_kwargs'] = {'local_files_only': True, 'trust_remote_code': True}
        self.proc = cls(**kwargs)
        self.identity = {n: sha(self.root / n) for n in ('processor_config.json', 'statistics.json', 'embodiment_id.json')}

    def encode_actions(self, commands, actual_state):
        import numpy as np
        command = np.asarray(commands, dtype=np.float32)
        state = np.asarray(actual_state, dtype=np.float32).reshape(1, 10)
        assert command.shape == (40, 10)
        a = self.proc.state_action_processor.apply_action(fields(command), TAG, fields(state))
        result = np.zeros((40, 132), dtype=np.float32)
        result[:, :10] = np.concatenate([a['eef_9d'], a['gripper_position']], axis=-1)
        return result

    def decode_actions(self, normalized, actual_state):
        import numpy as np
        x = np.asarray(normalized, dtype=np.float32)
        assert x.shape[-1] == 132
        state = np.asarray(actual_state, dtype=np.float32).reshape(1, 10)
        a = self.proc.state_action_processor.unapply_action(fields(x[..., :10]), TAG, fields(state))
        return np.concatenate([a['eef_9d'], a['gripper_position']], axis=-1).astype(np.float32)

    def action_audit(self, commands, actual_state):
        import numpy as np
        from gr00t.data.types import ActionType, ActionFormat
        from gr00t.data.utils import normalize_values_minmax
        p = self.proc.state_action_processor
        relative = p._convert_to_relative_action(commands[:, :9], actual_state[:9], ActionType.EEF, ActionFormat.XYZ_ROT6D)
        raw = np.concatenate([normalize_values_minmax(relative, p.norm_params[TAG]['action']['eef_9d']),
                              normalize_values_minmax(commands[:, 9:10], p.norm_params[TAG]['action']['gripper_position'])],axis=-1)
        return {'native_preclip_out_of_range_count':int((np.abs(raw)>1).sum()),
                'active_values':int(raw.size),'native_preclip_max_abs':float(np.abs(raw).max()),
                'native_preclip_fraction':float((np.abs(raw)>1).mean()),'native_inverse_clipping':True}

    def observation(self, base_raw, wrist_raw, actual_state):
        import numpy as np
        state = np.asarray(actual_state, dtype=np.float32).reshape(1, 1, 10)
        obs = {'state.' + k: v for k, v in fields(state).items()}
        for k, value in [('base_camera', base_raw), ('hand_camera', wrist_raw)]:
            value = np.asarray(value)
            assert value.shape == (128, 128, 3) and value.dtype == np.uint8
            obs['video.' + k] = value[None, None]
        obs['annotation.human.task_description'] = [INSTRUCTION]
        return self.proc.process_observation(obs, self.tag)

    def feature(self, base_raw, wrist_raw, actual_state, commands):
        import numpy as np
        from PIL import Image
        self.proc.train()
        content = self.step_type(images={'base_camera': [Image.fromarray(base_raw)],
                                         'hand_camera': [Image.fromarray(wrist_raw)]},
                                 states=fields(np.asarray(actual_state, dtype=np.float32).reshape(1, 10)),
                                 actions=fields(np.asarray(commands, dtype=np.float32)),
                                 text=INSTRUCTION, embodiment=self.tag)
        return self.proc([{'type': 'episode_step', 'content': content}])

    def encode_item(self, base_raw, wrist_raw, actual_state, commands):
        feature = self.feature(base_raw, wrist_raw, actual_state, commands)
        return dict(self.proc.collator([feature])['inputs'])


def load_codec(root=ROOT):
    return Codec(root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--stdio', action='store_true')
    args = parser.parse_args()
    assert args.stdio
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    import numpy as np
    codec = load_codec(args.root)
    print(json.dumps({'ready': True, 'status': 'READY_CPU_CODEC', 'identity': codec.identity}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get('close'):
            break
        source = Path(request['input'])
        target = Path(request['output'])
        assert source.resolve().is_relative_to(Path(args.root).parent.resolve())
        assert target.resolve().is_relative_to(Path(args.root).parent.resolve())
        assert not target.exists(), 'Preserve original first output'
        with np.load(source) as data:
            normalized = codec.encode_actions(data['commands'], data['actual_state'])
            decoded = codec.decode_actions(normalized, data['actual_state'])
            commands = data['commands'].copy()
            state = data['actual_state'].copy()
        from cpu_inputs import physical_errors
        np.savez(target, normalized=normalized, decoded=decoded, commands=commands, actual_state=state,
                 identity_json=np.asarray(json.dumps(codec.identity,sort_keys=True)),
                 source_sha256=np.asarray(sha(source)),
                 clip_audit_json=np.asarray(json.dumps(codec.action_audit(commands,state),sort_keys=True)),
                 roundtrip_json=np.asarray(json.dumps(physical_errors(commands,decoded),sort_keys=True)))
        print(json.dumps({'status': 'OK', 'output': str(target), 'sha256': sha(target), 'identity': codec.identity}), flush=True)


if __name__ == '__main__':
    main()
