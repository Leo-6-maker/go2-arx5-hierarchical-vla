"""Bounded native effect journal backed by one centrally reserved A40 job.

These are local allocation effects, NOT a second authoritative global budget.
Only native terminal settlement merges their deltas into the existing owner.
"""
from contextlib import contextmanager
import fcntl
import math
import os
from pathlib import Path
import time

from scripts.pi0292.durable import read,write
from .a40_allocation import VERSION


def fail(message):raise PermissionError('VAP1_A40_EFFECTS_'+message)


class Ledger:
    def __init__(self,root,allocation):
        self.root=Path(root);self.allocation=allocation
        self.path=self.root/'A40_NATIVE_EFFECT_LEDGER.json'
        self.request=read(self.root/'pbs'/('alloc-'+str(allocation['submit_count']).zfill(2))/'REQUEST.json')

    def initialize(self):
        r=self.request;start=r['expected_native_start_step']
        if self.path.exists():raise FileExistsError('A40_NATIVE_EFFECT_LEDGER_NO_BLIND_RESTART')
        recipes=read(self.root/'plans/PRIMARY_LEARNING_RECIPES.json')['recipes'];recipe=recipes[r['training_block']['arm']]
        from scripts.pi0294.scene_cards import digest
        value=dict(schema='VAP1_A40_LOCAL_NATIVE_EFFECT_LEDGER_V1',platform_training_version=VERSION,
            run_uuid=self.root.name,job_id=self.allocation['job_id'],request_sha256=self.allocation['request_sha256'],
            source_sha=r['source_sha'],arm=r['training_block']['arm'],start_native_step=start,
            optimizer_updates=start,generation_queries=0,model_loads=0,CPU_s=0.,pending_updates=[],unknown_effects=[],
            query_intent_links=[],training_runs=({r['training_block']['arm']:dict(recipe_sha256=digest(recipe),last_native_step=start)} if start else {}),
            global_budget_is_only_external_reservation=True)
        from scripts.pi0294.pv1.pbs_owner import publish_release_record
        self.check(value);publish_release_record(self.path,value)
        return value

    def check(self,v):
        r=self.request;arm=r['training_block']['arm'];start=r['expected_native_start_step']
        from .segmented_training import enabled
        segmented=enabled(r)
        from . import b_heartbeat_recovery as b
        imported_b=b.enabled(r) and start==1024
        if imported_b:b.lineage(r)
        if (v.get('schema')!='VAP1_A40_LOCAL_NATIVE_EFFECT_LEDGER_V1'
                or v.get('platform_training_version')!=VERSION or v.get('job_id')!=self.allocation['job_id']
                or v.get('request_sha256')!=self.allocation['request_sha256'] or v.get('source_sha')!=r['source_sha']
                or v.get('arm')!=arm or v.get('start_native_step')!=start or v.get('unknown_effects')!=[]
                or type(v['optimizer_updates']) is not int or not start<=v['optimizer_updates']<=4096
                or type(v['generation_queries']) is not int or not 0<=v['generation_queries']<=29
                or type(v['model_loads']) is not int or not 0<=v['model_loads']<=4
                or type(v['CPU_s']) not in (int,float) or not math.isfinite(v['CPU_s'])
                or not 0<=v['CPU_s']<=r['CPU_upper_s'] or type(v['pending_updates']) is not list or len(v['pending_updates'])>1
                or set(v['training_runs'])- {arm}
                or v['training_runs'].get(arm,{}).get('last_native_step',0)!=v['optimizer_updates']
                or len(v['query_intent_links'])!=v['generation_queries']):fail('BOUND_EXACT_ALLOCATION_EFFECT_COUNTERS')
        if v['pending_updates'] and (v['pending_updates'][0].get('request_id')!=r['request_id']
                or v['pending_updates'][0].get('arm')!=arm or v['pending_updates'][0].get('run_key')!=arm
                or v['pending_updates'][0].get('local_step_before')!=v['optimizer_updates']):
            fail('EXACT_SINGLE_PENDING_UPDATE_IDENTITY')
        if (v['optimizer_updates']>start or v['pending_updates']) and v['generation_queries']<17:
            fail('NO_UPDATE_BEFORE_SEVENTEEN_COMPATIBILITY_DRAWS')
        if (not(segmented and (start==2048 or imported_b)) and (v['optimizer_updates']>1024 or v['pending_updates'] and v['pending_updates'][0]['local_step_before']>=1024) and v['generation_queries']!=29):
            fail('NO_UPDATE_1025_BEFORE_ORIGINAL_TWELVE_PROBES')
        if segmented and (start==2048 or imported_b) and (v['optimizer_updates']>start or v['pending_updates']):
            from .long_training import probe_receipt
            probe_receipt(self.root,arm)
        return v

    @contextmanager
    def transaction(self):
        if self.path.is_symlink() or self.path.stat().st_uid!=os.getuid():fail('OWN_MUTABLE_EFFECT_LEDGER')
        with self.path.with_suffix('.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            value=self.check(read(self.path));yield value;self.check(value);write(self.path,value);self.path.chmod(0o600)

    def charge(self,key,count):
        if key not in ('generation_queries','model_loads') or count!=1:fail('ONE_REAL_QUERY_OR_MODEL_LOAD_ONLY')
        with self.transaction() as v:
            self._charge_value(v,key,count)

    def _charge_value(self,v,key,count):
        """Same pre-effect debit for an explicitly typed session subclass.

        Caller holds the original transaction lock. This does not dispatch a
        query and never converts an unresolved intent into permission to retry.
        """
        if key not in ('generation_queries','model_loads') or count!=1:fail('ONE_REAL_QUERY_OR_MODEL_LOAD_ONLY')
        if key=='generation_queries':
            folder=self.root/'reservations/probe-generations';pending=[]
            for path in folder.glob('*.json'):
                row=read(path)
                if row['status']!='COMPLETE':pending.append((path,row))
            if len(pending)!=1 or pending[0][1]['run_uuid']!=self.root.name:
                fail('ONE_DURABLE_GENERATION_INTENT_NO_UNRESOLVED_PRIOR_QUERY')
            path,row=pending[0]
            if any(link['message_id']==row['message_id'] for link in v['query_intent_links']):
                fail('UNKNOWN_QUERY_CANNOT_BE_CHARGED_AND_SENT_AGAIN')
            v['query_intent_links'].append(dict(index=v['generation_queries'],message_id=row['message_id'],
                intent_relative=str(path.relative_to(self.root)),payload_sha256=row['payload_sha256']))
        v[key]+=1

    def cpu(self,cost):
        from .alternate_accounting import finite
        finite(cost)
        with self.transaction() as v:v['CPU_s']+=cost

    def guard(self):
        from .a40_training_context import owner_guard
        owner_guard(self.root,self.allocation);return self.check(read(self.path))

    def summary(self):
        v=self.check(read(self.path));r=self.request
        unknown=self.linked_query_intents(v)
        arm=r['training_block']['arm'];recipe=read(self.root/'plans/PRIMARY_LEARNING_RECIPES.json')['recipes'][arm]
        from scripts.pi0294.scene_cards import digest
        return dict(schema='VAP1_A40_NATIVE_SOURCE_EFFECT_SUMMARY_V1',job_id=self.allocation['job_id'],
            request_id=r['request_id'],request_sha256=self.allocation['request_sha256'],source_sha=r['source_sha'],arm=arm,
            start_native_step=r['expected_native_start_step'],optimizer_updates_committed=v['optimizer_updates']-r['expected_native_start_step'],
            last_native_step=v['optimizer_updates'],query_attempts=v['generation_queries'],model_load_attempts=v['model_loads'],
            pending_updates=v['pending_updates'],unknown_query_intents=unknown,recipe_sha256=digest(recipe),
            effect_journal_complete=True,new_physics=0,new_episodes=0,finished_epoch=time.time())

    def linked_query_intents(self,v):
        """Inspect original durable query links, including unresolved calls."""
        unknown=[]
        folder=self.root/'reservations/probe-generations'
        expected={link['intent_relative'] for link in v['query_intent_links']}
        actual={str(p.relative_to(self.root)) for p in folder.glob('*.json')}
        if actual!=expected:fail('UNLINKED_QUERY_INTENTS_REQUIRE_RECONCILIATION')
        for link in v['query_intent_links']:
            path=self.root/link['intent_relative']
            if path.is_symlink() or path.stat().st_uid!=os.getuid():fail('OWN_QUERY_INTENT_BYTES')
            row=read(path)
            if (row['run_uuid']!=self.root.name or row['message_id']!=link['message_id']
                    or row['payload_sha256']!=link['payload_sha256'] or row['status'] not in ('INTENT','COMPLETE')):
                fail('ORIGINAL_QUERY_INTENT_LINK_CHANGED')
            if row['status']!='COMPLETE':unknown.append(link['index'])
            else:
                from scripts.pi0294.e1.journal import digest
                if digest(row['result'])!=row.get('result_sha256'):fail('COMPLETED_QUERY_RESULT_CHANGED')
        return unknown
