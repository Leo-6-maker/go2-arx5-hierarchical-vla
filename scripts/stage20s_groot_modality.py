"""Stage20S SIM_VISUAL_REACH_V0 official NEW_EMBODIMENT modality.

This module is imported only by the node-local official GR00T runtime.  The
sample period is an explicit simulation test value; it is not a production
Go2+ARX5 action-rate decision.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig


SIM_ACTION_SAMPLE_PERIOD_S = 0.05
SIM_ACTION_SAMPLE_PERIOD_PROVENANCE = "TEST_ONLY_SIM_VISUAL_REACH_V0"
SIM_ACTION_HORIZON = 40

SIM_VISUAL_REACH_MODALITY_CONFIG = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["external_rgb", "wrist_rgb"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["eef_9d", "gripper_position"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(SIM_ACTION_HORIZON)),
        modality_keys=["eef_9d", "gripper_position"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="eef_9d",
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="gripper_position",
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.language.task_description"],
    ),
}

register_modality_config(SIM_VISUAL_REACH_MODALITY_CONFIG, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
