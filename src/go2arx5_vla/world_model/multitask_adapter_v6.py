"""Two passive object roles on the existing paused mobile executor."""
from .multitask_contact_v6 import preview_permitted
from .multitask_score_v6 import Task
from .multitask_state_v6 import TwoObjectState, parse_state, read_state
from .target_sequential import TargetSequentialAdapter


class MultitaskAdapter(TargetSequentialAdapter):
    native_packet_parser = staticmethod(parse_state)
    read_native_state = staticmethod(read_state)

    def __init__(self, spec, output):
        super().__init__(spec, output)
        self.task = Task.parse(spec['task_instruction_id'])

    def bind_simulator_environment(self, environment):
        return dict(environment, MFREE_V6_SELECTED_COLOR=self.task.color)

    def make_task_state(self, model):
        return TwoObjectState(model, self.task.color)

    def native_preview_contact(self, names, distance, stage):
        a, b = names
        return preview_permitted(a, b, distance, stage, self.task.color)
