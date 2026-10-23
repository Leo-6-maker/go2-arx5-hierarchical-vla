"""Tiny full-state fixtures exercise import integrity, never native evidence."""
from contextlib import ExitStack,contextmanager
from copy import deepcopy
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from . import b_recovery_intake as intake,b_heartbeat_recovery as b,segment_intake as segment
from .test_segment_intake import fixture,put


def bound():
    return dict(root='/registered/closed/b1',run_manifest_sha256='8c549bbca0fed64e05112444e9e460ec66fa4bc487a0983cd8d1ddf20772669d',
        lease_sha256='28a7252b7b05a9d7cdf7fdd7c1ff168f3ee303d5e97392e361b9562c2564e524',
        control_terminal_sha256=b.CONTROL_CLOSURE_SHA,terminal_sha256=b.CLOSURE_SHA,job_id=b.FAILED_JOB,
        request_id=b.FAILED_REQUEST,request_sha256=b.REQUEST_SHA,source_sha=b.FAILED_SOURCE,
        checkpoint1024_sha256=b.SEED_SHA,probe_result_sha256=b.PROBE_SHA)

@contextmanager
def failed_fixture():
    with fixture('B_COVERAGE') as (s,t,old,recipe,data),ExitStack() as stack:
        c=segment.read(s/'TERMINAL_RECONCILIATION.json')
        c.update(status='CLOSED_B1_HEARTBEAT_FAILURE_1829_B1024_PRESERVED',latest_complete_step=1024,
            committed_updates_without_new_complete_state=805,prior_complete_state_preserved=True,retired_or_deleted_files=0,
            settlement=dict(updates=7720,unknown_effects=[]))
        c['effects'].update(last_native_step=1829,optimizer_updates_committed=1829)
        old['terminal_sha256']=segment.sha(put(s/'TERMINAL_RECONCILIATION.json',c))
        lease=segment.read(s/'lease.json');lease['terminal_receipt_sha256']=old['terminal_sha256']
        old['lease_sha256']=segment.sha(put(s/'lease.json',lease))
        v={k:x for k,x in old.items() if k not in ('arm','checkpoint2048_sha256')}
        v['control_terminal_sha256']='d'*64
        pins=dict(FAILED_JOB=old['job_id'],FAILED_REQUEST=old['request_id'],FAILED_SOURCE=old['source_sha'],
            REQUEST_SHA=old['request_sha256'],SEED_SHA=old['checkpoint1024_sha256'],PROBE_SHA=old['probe_result_sha256'],
            RECIPE=segment.digest(recipe),CLOSURE_SHA=old['terminal_sha256'])
        stack.enter_context(patch.multiple(b,**pins))
        # Only literal production source pins differ in this tiny fixture.
        # The actual binding validator is separately checked without mocks.
        stack.enter_context(patch.object(intake,'valid_binding',return_value=v))
        yield s,t,v,recipe,data

class BRecoveryIntakeTests(unittest.TestCase):
    def test_exact_failed_B_binding_not_other_parent_or_success(self):
        v=bound();self.assertEqual(intake.valid_binding(v),v)
        for key in v:
            x=dict(v);x[key]='WRONG' if key!='root' else 'relative'
            with self.subTest(key=key),self.assertRaises(PermissionError):intake.valid_binding(x)
        with self.assertRaises(PermissionError):intake.valid_binding(dict(v,extra=True))

    def test_mature_full_optimizer_rng_sampler_and_original_probe_audit(self):
        with failed_fixture() as (s,t,v,recipe,data):
            before={p:segment.sha(p) for p in s.rglob('*') if p.is_file()}
            proof=intake.audit_closed_source(v,recipe=recipe,dataset=data)
            self.assertEqual(proof['global_step'],1024);self.assertEqual(proof['actual_consumed_cursor'],32768)
            self.assertEqual(proof['paid_failed_step_preserved'],1829);self.assertEqual(proof['new_updates'],0)
            self.assertEqual(proof['new_probes'],0);self.assertEqual(len(proof['probe_files']),13)
            self.assertEqual(before,{p:segment.sha(p) for p in s.rglob('*') if p.is_file()})

    def test_corrupt_complete_members_or_probes_unknown_and_active_reject(self):
        for mode in ('optimizer','RNG','sampler','probe','unknown','active','recipe'):
            with self.subTest(mode=mode),failed_fixture() as (s,t,v,recipe,data):
                if mode in ('optimizer','RNG','sampler'):
                    n={'optimizer':'optimizer.pt','RNG':'rng_state.pth','sampler':'SAMPLER_STATE.json'}[mode]
                    p=s/'training/B_COVERAGE/checkpoint-1024'/n;p.chmod(0o600);p.write_text('CHANGED')
                elif mode=='probe':
                    p=s/'evidence/B_COVERAGE-1024/TRAIN_PROBES/probe-0.npz';p.chmod(0o600);p.write_bytes(b'CHANGED')
                elif mode=='unknown':
                    c=segment.read(s/'TERMINAL_RECONCILIATION.json');c['effects']['pending_updates']=[{}]
                    put(s/'TERMINAL_RECONCILIATION.json',c)
                elif mode=='active':
                    lease=segment.read(s/'lease.json');lease['status']='ACTIVE';put(s/'lease.json',lease)
                else:recipe['arm']='A_CONTROL'
                with self.assertRaises((PermissionError,ValueError)):intake.audit_closed_source(v,recipe=recipe,dataset=data)

    def test_preserved_old1024_and_new2048_origins_are_explicit_for_followup(self):
        with fixture('B_COVERAGE') as (s,t,v,recipe,data):
            oldjob='88.stdct-mgmt-02';oldsource='e'*40
            cp=s/'training/B_COVERAGE/checkpoint-1024/CHECKPOINT_COMPLETE.json'
            c=segment.read(cp);c.update(origin_job_id=oldjob,origin_source_sha=oldsource)
            v['checkpoint1024_sha256']=segment.sha(put(cp,c))
            rp=s/'evidence/B_COVERAGE-1024/TRAIN_PROBES/RESULT.json'
            r=segment.read(rp);r['checkpoint_manifest_sha256']=v['checkpoint1024_sha256'];v['probe_result_sha256']=segment.sha(put(rp,r))
            v['B_heartbeat_recovery_seed']=dict(authority_release=b.RELEASE,grant_sha256=b.GRANT_SHA,
                job_id=oldjob,source_sha=oldsource,checkpoint1024_sha256=v['checkpoint1024_sha256'],probe_result_sha256=v['probe_result_sha256'])
            m=segment.read(s/'RUN_MANIFEST.json');m['authority'].update(dict(zip(b.SCOPE_KEYS,(b.RELEASE,b.GRANT_SHA))))
            v['run_manifest_sha256']=segment.sha(put(s/'RUN_MANIFEST.json',m))
            c=segment.read(s/'TERMINAL_RECONCILIATION.json');c['effects'].update(start_native_step=1024,optimizer_updates_committed=1024,query_attempts=17)
            v['terminal_sha256']=segment.sha(put(s/'TERMINAL_RECONCILIATION.json',c))
            lease=segment.read(s/'lease.json');lease['terminal_receipt_sha256']=v['terminal_sha256'];v['lease_sha256']=segment.sha(put(s/'lease.json',lease))
            output=segment.read(s/'OUTPUT_BINDING.private.json');output['run_manifest_sha256']=v['run_manifest_sha256'];put(s/'OUTPUT_BINDING.private.json',output)
            with patch.multiple(b,FAILED_JOB=oldjob,FAILED_SOURCE=oldsource,SEED_SHA=v['checkpoint1024_sha256'],PROBE_SHA=v['probe_result_sha256']):
                proof=segment.audit_closed_source(v,recipe=recipe,dataset=data)
                self.assertEqual(proof['training_lineage_key'],b.LINEAGE)
                v['B_heartbeat_recovery_seed']['job_id']=v['job_id']
                with self.assertRaises(PermissionError):segment.audit_closed_source(v,recipe=recipe,dataset=data)

if __name__=='__main__':unittest.main()
