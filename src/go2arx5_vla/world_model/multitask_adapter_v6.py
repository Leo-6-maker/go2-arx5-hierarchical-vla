"""Two passive object roles on the existing paused mobile executor."""
from pathlib import Path

from .multitask_contact_v6 import preview_permitted
from .multitask_score_v6 import Task
from .multitask_state_v6 import TwoObjectState, parse_state, read_state
from .target_sequential import TargetSequentialAdapter, owned_simulator_pid


class MultitaskAdapter(TargetSequentialAdapter):
    native_packet_parser = staticmethod(parse_state)
    read_native_state = staticmethod(read_state)

    def __init__(self, spec, output):
        super().__init__(spec, output)
        self.task = Task.parse(spec['task_instruction_id'])

    def bind_simulator_environment(self, environment):
        return dict(environment, MFREE_V6_SELECTED_COLOR=self.task.color)

    def start(self):
        super().start()
        wrapper = next(process for name,process in self.processes if name=='sim')
        pid = owned_simulator_pid(wrapper.pid,self.spec['native_binary'])
        maps = (Path('/proc')/str(pid)/'maps').read_text()
        for library in ('outer_library','observer_library','mobile_safety_library','trace_library'):
            if self.spec[library] not in maps:
                raise PermissionError('ACTUAL_MULTITASK_ORIGINAL_CHAIN_MEMBER_NOT_LOADED:'+library)

    def make_task_state(self, model):
        return TwoObjectState(model, self.task.color)

    def native_preview_contact(self, names, distance, stage):
        a, b = names
        return preview_permitted(a, b, distance, stage, self.task.color)
