"""Own0/25/3 E1 inference and512/37/2 E2; old effects are input lineage."""
from copy import deepcopy
from .minimal_effects import Effects
from . import minimal_saved512_contract as c,minimal_execution as e

SCHEMA='ARX5_SAVED512_TWO_SESSION_ALLOCATION_EFFECTS_V1'
SUMMARY='ARX5_SAVED512_TWO_SESSION_ACTUAL_EFFECT_SUMMARY_V1'


class Saved512Effects(Effects):
    schema=SCHEMA;summary_schema=SUMMARY;all_complete_field='all_saved512_sessions_runtime_complete'
    def __init__(self,root,allocation,*,request_path,guard):
        super().__init__(root,allocation,request_path=request_path,guard=guard)
        e.need(self.adapter.VERSION==c.VERSION and(self.request['optimizer_update_limit'],self.request['query_limit'],self.request['model_load_limit'])==(512,62,5),
            'SAVED512_EXPLICIT_NEW_REQUEST_AND_OWN_EFFECT_LIMITS')
        from .minimal_saved512_inputs import Inputs
        import json
        # The zero-effect journal must exist before the original bind/Q->R wait.
        # Readonly saved-input checks grant no model or optimizer permission.
        self.saved=Inputs(root,json.loads((self.root/'RUN_MANIFEST.json').read_bytes()),guard=lambda:None)
        self.saved.guard=guard;self.saved.prior.guard=guard
        self.binding=self.saved.binding
    def plan(self):return deepcopy(c.sessions())
    def load_limits(self):return(3,2)
    def training_sessions(self):return[c.sessions()[1]['id']]
    def qualification_query_floor(self):return 0
    def initial_metadata(self):return dict(audited_saved512=deepcopy(self.binding))
    def check(self,v):
        super().check(v);e.need(v.get('audited_saved512')==self.binding,'SAVED512_EFFECTS_EXACT_SAVED_ORIGIN')
        return v
    def summary(self):
        v=super().summary();v.update(audited_saved512=deepcopy(self.binding),retained_old_native=deepcopy(c.OLD_PAID),
            E1_new_optimizer_calls=0,carried_queries_added_to_new_links=0,scientific_endpoints_independently_audited=False)
        return v
