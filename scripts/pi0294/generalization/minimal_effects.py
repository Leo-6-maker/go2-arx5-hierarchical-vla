"""Five finite sessions in the original allocation-local effect journal.

No global budget, submission, model call or native permission is implemented.
The original fixed_draw Journal, file lock, pre-effect query debit and pending
optimizer commit remain authoritative. Qualified and formal steps are distinct.
"""
from copy import deepcopy
import json,math,os,re,time
from pathlib import Path

from scripts.pi0292.durable import read
from scripts.pi0294.dec1.a40_native_ledger import Ledger as OriginalEffects
from scripts.pi0294.pv1.pbs_owner import publish_release_record as once
from scripts.pi0294.pv1.recipe import commit_update as original_commit
from . import minimal_execution as execution,minimal_training as training

SCHEMA='ARX5_MINIMAL_FIVE_SESSION_ALLOCATION_EFFECTS_V1'
SUMMARY='ARX5_MINIMAL_ACTUAL_FIVE_SESSION_EFFECT_SUMMARY_V1'
LOADS=(1,3,3,2,2)
need=execution.need


class Effects(OriginalEffects):
    schema=SCHEMA
    summary_schema=SUMMARY
    all_complete_field='all_five_sessions_runtime_complete'

    def plan(self):return execution.native_sessions()
    def load_limits(self):return LOADS
    def training_sessions(self):return [s['id']for s in self.plan()[1:]]
    def qualification_query_floor(self):return 15
    def initial_metadata(self):return {}

    def __init__(self,root,allocation,*,request_path,guard):
        from . import minimal_execution_adapter as adapters
        self.root=Path(root);self.allocation=allocation
        self.path=self.root/'A40_NATIVE_EFFECT_LEDGER.json'
        self.request_path=Path(request_path);body=self.request_path.read_bytes()
        self.request=json.loads(body);self.adapter=adapters.request(self.request)
        need(self.request_path.resolve().is_relative_to(self.root.resolve())
            and not self.request_path.is_symlink()and callable(guard)
            and allocation['request_sha256']==execution.sha(body)
            and allocation['request_id']==self.request['request_id']
            and allocation['source_sha']==self.request['source_sha'],
            'EFFECTS_EXACT_CURRENT_REQUEST_AND_EXTERNAL_NATIVE_GUARD')
        self.native_guard=guard

    def initialize(self):
        need(not self.path.exists()and not self.path.is_symlink(),'EFFECTS_NO_INITIALIZE_OR_RESTART_REPLAY')
        r=self.request
        value=dict(schema=self.schema,platform_training_version=self.adapter.VERSION,
            run_uuid=self.root.name,job_id=self.allocation['job_id'],request_id=r['request_id'],
            request_sha256=self.allocation['request_sha256'],source_sha=r['source_sha'],
            optimizer_updates=0,generation_queries=0,model_loads=0,CPU_s=0.,
            pending_updates=[],unknown_effects=[],query_intent_links=[],training_runs={},
            active_session=None,sessions={row['id']:dict(status='NOT_STARTED',
                optimizer_updates=0,generation_queries=0,model_loads=0)
                for row in self.plan()},
            global_budget_is_only_external_reservation=True)
        value.update(self.initial_metadata())
        self.check(value);once(self.path,value);return value

    def check(self,v):
        r=self.request;plan=self.plan();ids=[x['id']for x in plan]
        need(v.get('schema')==self.schema and v.get('platform_training_version')==self.adapter.VERSION
            and v.get('run_uuid')==self.root.name and v.get('job_id')==self.allocation['job_id']
            and v.get('request_id')==r['request_id']and v.get('request_sha256')==self.allocation['request_sha256']
            and v.get('source_sha')==r['source_sha']and v.get('unknown_effects')==[]
            and set(v.get('sessions',{}))==set(ids)
            and type(v.get('CPU_s'))in(int,float)and math.isfinite(v['CPU_s'])
            and 0<=v['CPU_s']<=r['CPU_upper_s']and v.get('global_budget_is_only_external_reservation')is True,
            'EFFECTS_TYPED_OWN_BOUND_ALLOCATION_NO_UNKNOWN_CLEAR')
        statuses=[];totals=dict(optimizer_updates=0,generation_queries=0,model_loads=0)
        for spec,loads in zip(plan,self.load_limits()):
            row=v['sessions'][spec['id']];statuses.append(row.get('status'))
            need(row.get('status')in('NOT_STARTED','ACTIVE','COMPLETE'),'EFFECTS_SESSION_STATUS')
            for key,upper in [('optimizer_updates',spec['optimizer_limit']),
                              ('generation_queries',spec['query_limit']),('model_loads',loads)]:
                need(type(row.get(key))is int and 0<=row[key]<=upper,'EFFECTS_FINITE_SESSION_'+key)
                totals[key]+=row[key]
            if row['status']=='NOT_STARTED':
                need(all(row[k]==0 for k in totals),'EFFECTS_NO_FUTURE_SESSION_EFFECT')
            if row['status']=='COMPLETE':
                need(row['optimizer_updates']==spec['optimizer_limit']
                    and row['generation_queries']==spec['query_limit']and row['model_loads']>0
                    and type(row.get('terminal_receipt_sha256'))is str
                    and re.fullmatch('[0-9a-f]{64}',row['terminal_receipt_sha256']),
                    'EFFECTS_COMPLETE_SESSION_EXACT_COUNTERS_AND_RECEIPT')
            run=v['training_runs'].get(spec['id'])
            if run is not None:
                need(spec['arm']in training.ARMS and run['last_native_step']==row['optimizer_updates']
                    and re.fullmatch('[0-9a-f]{64}',run['recipe_sha256']),
                    'EFFECTS_OWN_SESSION_RECIPE_AND_STEP')
            elif row['optimizer_updates']:need(False,'EFFECTS_COMMITTED_STEP_HAS_NO_ORIGINAL_RUN')
        active=[x for x in ids if v['sessions'][x]['status']=='ACTIVE']
        need(len(active)<=1 and v['active_session']==(active[0]if active else None),
            'EFFECTS_ONE_SEQUENTIAL_ACTIVE_SESSION')
        n=statuses.count('COMPLETE')
        need(statuses[:n]==['COMPLETE']*n and all(x=='NOT_STARTED'for x in statuses[n+len(active):])
            and(not active or statuses[n]=='ACTIVE'),'EFFECTS_NO_SKIPPED_OR_REORDERED_SESSION')
        need(all(v.get(k)==x for k,x in totals.items())
            and len(v.get('query_intent_links',[]))==v['generation_queries']
            and set(v.get('training_runs',{}))<=set(self.training_sessions()),'EFFECTS_AGGREGATE_COUNTS_INCLUDE_QUALIFICATION')
        pending=v.get('pending_updates');need(type(pending)is list and len(pending)<=1,'EFFECTS_ONE_PENDING_UPDATE')
        if pending:
            need(bool(active),'EFFECTS_PENDING_WITHOUT_ACTIVE_SESSION')
            sid=active[0];spec=next(x for x in plan if x['id']==sid);row=v['sessions'][sid]
            need(spec['optimizer_limit']>0 and row['optimizer_updates']<spec['optimizer_limit']
                and pending==[dict(arm=spec['arm'],run_key=sid,local_step_before=row['optimizer_updates'],request_id=r['request_id'])],
                'EFFECTS_EXACT_PENDING_NATIVE_STEP_NEVER_ANOTHER_ARM')
        return v

    def guard(self):
        self.native_guard();return self.check(read(self.path))

    def start_session(self,identifier):
        self.guard()
        with self.transaction()as v:
            need(v['active_session']is None and v['pending_updates']==[]and self.linked_query_intents(v)==[],
                'EFFECTS_NO_ACTIVE_OR_UNRESOLVED_PRIOR_SESSION')
            ids=[x['id']for x in self.plan()]
            need(identifier in ids and v['sessions'][identifier]['status']=='NOT_STARTED'
                and all(v['sessions'][s]['status']=='COMPLETE'for s in ids[:ids.index(identifier)]),
                'EFFECTS_NEXT_ORIGINAL_SESSION_ONCE_ONLY')
            v['sessions'][identifier]['status']='ACTIVE';v['active_session']=identifier

    def charge(self,key,count):
        self.guard()
        with self.transaction()as v:
            sid=v['active_session'];need(sid is not None,'EFFECTS_NO_LOAD_OR_QUERY_OUTSIDE_SESSION')
            if key=='model_loads':need(self.linked_query_intents(v)==[],'EFFECTS_UNKNOWN_QUERY_BEFORE_MODEL_LOAD')
            self._charge_value(v,key,count)
            v['sessions'][sid][key]+=count

    def known_effects(self,identifier):
        v=self.guard();need(v['active_session']==identifier,'EFFECTS_TRAINER_MUST_BIND_OWN_SESSION')
        return dict(optimizer_updates=v['sessions'][identifier]['optimizer_updates'],
                    pending_updates=deepcopy(v['pending_updates']),unknown_effects=deepcopy(v['unknown_effects']))

    def begin_update(self,identifier,local_step,recipe):
        self.guard()
        original=read(self.root/'plans/FROZEN_SOURCE_PAIR_RECIPE.json');training.validate(recipe,original)
        spec=next(x for x in self.plan()if x['id']==identifier)
        need(recipe['arm']==spec['arm']and spec['optimizer_limit']>0,'EFFECTS_EXACT_SESSION_TRAINING_RECIPE')
        with self.transaction()as v:
            row=v['sessions'][identifier]
            need(v['active_session']==identifier and type(local_step)is int and row['optimizer_updates']==local_step
                and 0<=local_step<spec['optimizer_limit']and v['pending_updates']==[]
                and self.linked_query_intents(v)==[]and row['model_loads']>0,
                'EFFECTS_EXACT_KNOWN_COMMITTED_SESSION_BOUNDARY')
            if not spec['formal_candidate']:
                need(row['generation_queries']==self.qualification_query_floor(),'EFFECTS_MATCHED_ZERO_STEP_DRAWS_BEFORE_QUALIFICATION')
            else:
                need(row['generation_queries']==0,'EFFECTS_FORMAL_PROBES_ONLY_AFTER_COMMITTED512')
            expected=dict(recipe_sha256=execution.sha(execution.raw(recipe)),last_native_step=local_step)
            if identifier not in v['training_runs']:
                need(local_step==0,'EFFECTS_FRESH_FORMAL_OR_QUALIFICATION_OPTIMIZER_START_ZERO')
                v['training_runs'][identifier]=expected
            need(v['training_runs'][identifier]==expected,'EFFECTS_NO_SESSION_RECIPE_OR_STEP_RESET')
            v['pending_updates']=[dict(arm=spec['arm'],run_key=identifier,local_step_before=local_step,
                                       request_id=self.request['request_id'])]

    def commit_update(self,identifier,local_step_after):
        spec=next(x for x in self.plan()if x['id']==identifier)
        with self.transaction()as v:
            need(v['active_session']==identifier,'EFFECTS_COMMIT_ACTIVE_SESSION_ONLY')
            new=original_commit(v,arm=spec['arm'],local_step_after=local_step_after,request_id=self.request['request_id'])
            new['sessions'][identifier]['optimizer_updates']=local_step_after
            v.clear();v.update(new)

    def complete_session(self,identifier,receipt_path):
        self.guard();path=Path(receipt_path)
        need(not path.is_symlink()and path.is_file()and path.resolve().is_relative_to(self.root.resolve())
            and path.stat().st_uid==os.getuid()and path.stat().st_mode&0o777==0o400,
            'EFFECTS_OWN_IMMUTABLE_RUNTIME_TERMINAL_RECEIPT')
        body=path.read_bytes();proof=json.loads(body)
        with self.transaction()as v:
            need(v['active_session']==identifier and v['pending_updates']==[]and self.linked_query_intents(v)==[],
                'EFFECTS_NO_UNRESOLVED_SESSION_EFFECTS_AT_TERMINAL')
            spec=next(x for x in self.plan()if x['id']==identifier);row=v['sessions'][identifier]
            need(proof.get('schema')=='ARX5_MINIMAL_RUNTIME_SESSION_TERMINAL_V1'
                and proof.get('session_id')==identifier and proof.get('arm')==spec['arm']
                and proof.get('request_sha256')==self.allocation['request_sha256']
                and proof.get('raw_parent_files_identity')==training.RAW_PARENT_FILES_ID
                and proof.get('statistics_sha256')==execution.R0
                and proof.get('optimizer_updates')==row['optimizer_updates']==spec['optimizer_limit']
                and proof.get('query_attempts')==row['generation_queries']==spec['query_limit']
                and proof.get('cursor')==spec['sample_visits']
                and proof.get('status')=='COMPLETE_SAVED_REQUIRES_INDEPENDENT_OWNER_AUDIT',
                'EFFECTS_ORIGINAL_REAL_SESSION_TERMINAL_COUNTERS_NOT_GLOBAL_SUCCESS')
            row.update(status='COMPLETE',terminal_receipt_sha256=execution.sha(body),
                       terminal_receipt_relative=str(path.relative_to(self.root)))
            v['active_session']=None

    def summary(self):
        v=self.check(read(self.path));unknown=self.linked_query_intents(v)
        return dict(schema=self.summary_schema,job_id=self.allocation['job_id'],request_id=self.request['request_id'],
            request_sha256=self.allocation['request_sha256'],source_sha=self.request['source_sha'],
            optimizer_updates_committed=v['optimizer_updates'],query_attempts=v['generation_queries'],
            model_load_attempts=v['model_loads'],pending_updates=v['pending_updates'],unknown_query_intents=unknown,
            sessions=deepcopy(v['sessions']),active_session=v['active_session'],CPU_s=v['CPU_s'],
            **{self.all_complete_field:all(x['status']=='COMPLETE'for x in v['sessions'].values())},
            independently_audited=False,global_budget_is_only_external_reservation=True,
            new_physics=0,new_episodes=0,finished_epoch=time.time())
