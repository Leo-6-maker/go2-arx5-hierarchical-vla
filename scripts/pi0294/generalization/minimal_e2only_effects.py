"""Allocation-local E2-only effects; imports no model, Torch or CUDA."""
from copy import deepcopy
from . import minimal_e2only_contract as c,minimal_execution as e
from .minimal_effects import Effects

SCHEMA='ARX5_E2_ONLY_ALLOCATION_EFFECTS_V1'
SUMMARY='ARX5_E2_ONLY_ACTUAL_EFFECT_SUMMARY_V1'
COMPLETE='all_E2_only_sessions_runtime_complete'


class E2OnlyEffects(Effects):
    schema=SCHEMA;summary_schema=SUMMARY;all_complete_field=COMPLETE
    def plan(self):return c.sessions()
    def load_limits(self):return(2,)
    def training_sessions(self):return[s['id']for s in self.plan()]
    def qualification_query_floor(self):return 0
    def initial_metadata(self):return dict(closed_predecessor=deepcopy(c.binding()))
    def check(self,v):
        super().check(v)
        e.need(v.get('closed_predecessor')==c.binding(),'E2ONLY_EFFECTS_NO_REUSED122_OR_E1_SESSION')
        return v
    def summary(self):
        return dict(super().summary(),closed_predecessor=c.binding(),E1_new_training_or_inference=0)


