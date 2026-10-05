"""New isolated CPU process: actual existing state/action processor, no models."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                  MKL_NUM_THREADS='4', OPENBLAS_CORETYPE='Prescott', HF_HUB_OFFLINE='1',
                  TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import resource
import sys
import numpy as np

BASE = Path('/home/liuyu-nfs/Projects/go2-arx5')
DATA = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0260/R2')
sys.path[:0] = [str(DATA/'qa'), str(DATA/'runtime'), '/home/liuyu-nfs/Projects/isaac-gr00t']
import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
resource.setrlimit(resource.RLIMIT_AS, (16*2**30, 16*2**30))

def forbidden(*args, **kwargs):
    raise RuntimeError('NO_NEURAL_QUERY_OR_OPTIMIZER_IN_PREPARATION')

torch.nn.Module.__call__ = forbidden
torch.optim.Optimizer.__init__ = forbidden
import pi0260_r2_codec as codec

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main(output):
    assert not torch.cuda.is_available()
    stats_path = BASE/'PI0274/contracts/ROT/source/dataset/statistics.json'
    expected = '0ea96eaf9ba6f2bfb5c16681542723e9ee0e82281dd07c4e33ed9189799580e9'
    assert sha(stats_path) == expected
    statistics = json.loads(stats_path.read_text())
    proc = codec.processor(statistics)
    rows = []
    # Synthetic format probes, not new scene data or task supervision.
    for opening in (0., .25, .5, .75, 1.):
        state = np.array([.01, 0, .02, 1, 0, 0, 0, 1, 0, opening], np.float32)
        label = np.tile(state, (40, 1))
        packed_state, packed_action, mask = codec.encode(proc, state, label)
        decoded = codec.decode(proc, packed_action, state)
        assert packed_action.shape == mask.shape == (40, 132)
        assert np.all(mask[:, :10] == 1) and not np.any(mask[:, 10:])
        assert not np.any(packed_action[:, 10:]) and np.isfinite(packed_action).all()
        error = float(np.max(abs(decoded[:, 9] - label[:, 9])))
        assert error <= 1e-5
        rows.append(dict(raw_opening=opening, encoded_state=float(packed_state[0, 9]),
                         encoded_action=float(packed_action[0, 9]), decoded_opening=float(decoded[0, 9]),
                         gripper_roundtrip_max=error))
    assert sha(stats_path) == expected
    result = dict(status='PASS_ACTUAL_OLD_STATE_ACTION_PROCESSOR_GRIPPER_RANGE_PROBE',
                  evidence_domain='SYNTHETIC_CPU_FORMAT_PROBES_NOT_TASK_DATA', rows=rows,
                  encoded_action_peak=max(abs(r['encoded_action']) for r in rows),
                  normalization='unclamped q01/q99 affine; denominator floor 1e-6',
                  gripper_mask_active=True, unchanged_old_statistics_sha256=expected,
                  source_hashes={str(Path(codec.__file__)): sha(codec.__file__),
                                 str(Path(inspect.getfile(type(proc)))): sha(inspect.getfile(type(proc))),
                                 str(Path(__file__)): sha(__file__)},
                  conclusion='old range roundtrips but produces extreme normalized magnitudes; freeze new task TRAIN-only scale before learning',
                  full_VL_processor_and_real_object_episode='NOT_RUN', gradients='NOT_QUERIED',
                  model_queries=0, optimizer_updates=0, physics_steps=0, renders=0, teacher_calls=0,
                  launch_allowed=False, selected_checkpoint='NONE', deployment_selected_checkpoint='NONE')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(result, sort_keys=True), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
