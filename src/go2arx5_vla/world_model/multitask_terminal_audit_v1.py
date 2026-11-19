"""Read the last actual packet, including partial atoms, with the original Score."""
from pathlib import Path

from .multitask_score_v6 import Score, Task
from .multitask_state_v6 import TwoObjectState, packets
from .native_replay import forbid_integration
from .storage import digest

START = 5703


def replay(spec, journal, terminal_tick, cached_score):
    import mujoco

    model_path = spec['model_path']
    if digest(model_path) != spec['frozen_files'][model_path]:
        raise ValueError('TERMINAL_SCORE_REQUIRES_THE_ORIGINAL_SCENE')
    if terminal_tick < START:
        return dict(task_started=False, terminal_tick=terminal_tick, score=None, cached_score=cached_score)
    task = Task.parse(spec['task_instruction_id'])
    scorer = None
    initial_state = None
    with forbid_integration(mujoco):
        reader = TwoObjectState(mujoco.MjModel.from_binary_path(model_path), task.color)
        for packet in packets(Path(journal)):
            step = packet['step']
            if step < START:
                continue
            sample = reader.update(packet)
            if scorer is None:
                if step != START:
                    raise ValueError('REPLAY_INITIAL_TASK_PACKET_MISSING')
                initial_state = {key: packet[key].tolist() for key in ('qpos', 'qvel', 'ctrl')}
                initial_state.update(TCP=sample['tcp'].tolist(), base=sample['base_pose'].tolist(),
                                     objects={color: obj['pose'].tolist() for color, obj in sample['objects'].items()})
                scorer = Score(task.instruction, sample, spec['multitask_geometry'])
            scorer.update(sample)
    if scorer is None or scorer.last != terminal_tick or scorer.gaps:
        raise ValueError('REPLAY_TERMINAL_TASK_PACKET_OR_2MS_CONTINUITY_MISSING')
    score = scorer.result()
    return dict(task_started=True, terminal_tick=terminal_tick, final_partial_atom_ticks=(terminal_tick-START) % 25,
                scored_native_states=score['states'], score=score, cached_score=cached_score,
                differs_from_cache=score != cached_score, initial_state=initial_state,
                selected_reference='UNKNOWN', new_native_ticks=0)
