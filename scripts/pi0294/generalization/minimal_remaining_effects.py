"""Three remaining sessions in the SAME original allocation effect journal.

The admitted request/profile and worker selector still have to select this
explicit type. Importing it never registers, submits, resumes or initializes.
The saved prefix is an immutable input, never copied into new effect counters
or query-intent links. Original transaction/query/pending-update methods stay.
"""
import json,os,stat
from copy import deepcopy
from pathlib import Path
from . import minimal_remaining_contract as r,minimal_execution as e
from .minimal_effects import Effects

SCHEMA='ARX5_MINIMAL_REMAINING_THREE_SESSION_ALLOCATION_EFFECTS_V1'
SUMMARY='ARX5_MINIMAL_ACTUAL_SAVED_PREFIX_REMAINDER_EFFECT_SUMMARY_V1'


class RemainingEffects(Effects):
    schema=SCHEMA
    summary_schema=SUMMARY
    all_complete_field='all_remaining_sessions_runtime_complete'

    def __init__(self,root,allocation,*,request_path,guard):
        # The original constructor calls the explicit request adapter. It
        # deliberately refuses this new version until its full central/native
        # request branch is installed; this class is not a compatibility alias.
        super().__init__(root,allocation,request_path=request_path,guard=guard)
        e.need(self.adapter.VERSION==r.VERSION
            and self.request['platform_training_version']==r.VERSION
            and self.request['profile']==r.PROFILE
            and (self.request['optimizer_update_limit'],self.request['query_limit'],self.request['model_load_limit'])==(1026,76,7),
            'REMAINDER_EFFECTS_EXACT_NEW_TYPED_REQUEST_AND_OWN_LIMITS')
        manifest=json.loads((self.root/'RUN_MANIFEST.json').read_bytes());bodies=[]
        for name,pin in((r.PREFIX,r.RESULT),(r.CLOSURE,r.CLOSED)):
            path=self.root/name;s=path.lstat()
            e.need(not path.is_symlink()and stat.S_ISREG(s.st_mode)and s.st_uid==os.getuid()
                and stat.S_IMODE(s.st_mode)==0o400,'REMAINDER_EFFECTS_OWN_IMMUTABLE_AUDITED_PREFIX')
            body=path.read_bytes();e.need(e.sha(body)==pin==manifest['execution']['sealed_inputs'].get(name),
                'REMAINDER_EFFECTS_REAL_PREFIX_BYTES_AND_MANIFEST_SEAL');bodies.append(body)
        self.prefix_binding=r.saved_prefix(*bodies)

    def plan(self):
        rows=deepcopy(r.sessions())
        for spec in rows:spec['query_limit']=spec['own_query_limit']
        return rows

    def load_limits(self):return tuple(s['own_model_load_limit']for s in r.sessions())
    def training_sessions(self):return [s['id']for s in self.plan()]
    def qualification_query_floor(self):return 0
    def initial_metadata(self):return dict(audited_saved_prefix=deepcopy(self.prefix_binding))

    def check(self,value):
        super().check(value)
        e.need(value.get('audited_saved_prefix')==self.prefix_binding,
            'REMAINDER_EFFECTS_OLD_PROOF_IMMUTABLE_NOT_NEW_QUERY_LINKS')
        for spec in r.sessions():
            row=value['sessions'][spec['id']]
            e.need(row['status']!='COMPLETE'or row['model_loads']==spec['own_model_load_limit'],
                'REMAINDER_EFFECTS_ALL_REAL_RAW_RESTORE_AND_EXPORT_LOADS')
        return value

    def logical_session_counters(self,identifier):
        value=self.guard();row=value['sessions'][identifier]
        return r.logical_counters(identifier,{k:row[k]for k in r.KEYS})

    def summary(self):
        out=super().summary()
        out.update(retained_old_native=deepcopy(r.OLD_PAID),
            audited_prefix_result_sha256=r.RESULT,audited_prefix_closure_sha256=r.CLOSED,
            carried_saved_E2_zero_queries=15,carried_queries_added_to_new_links=0,
            all_original_study_sessions_runtime_complete=out[self.all_complete_field],
            scientific_endpoints_independently_audited=False)
        return out
