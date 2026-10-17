from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from .mp1 import pre_model_qualification as qualification
from .mp1.n1_limits import RELEASE
from .native_dataset import sha


class PreModelQualificationContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'source-pair-01';self.directory=self.root/'pbs/alloc-06'
        self.directory.mkdir(parents=True)
        (self.root/'RUN_MANIFEST.json').write_text(json.dumps(dict(code=dict(vla_commit='a'*40))))
        self.allocation=dict(job_id='123.hopper-m-02',hostname='fixture',allocation_start_epoch=1000,allocation_end_epoch=2000)
        rows=[]
        for i in range(81):
            clock=1000+7.5*i
            rows.append(dict(monotonic_s=7.5*i,proof=dict(clock_sampled_after_read=True,watchdog_s=300,
                heartbeat_written=False,heartbeat=dict(run_uuid=self.root.name,observed_epoch=clock-10),
                server_revalidation=True,revalidated_clock_epoch=clock,cached_clock_epoch=clock)))
        self.value=dict(schema='MP1_N1_NATIVE_ENDURANCE_AND_TRANSPORT_V1',
            status='PASS_NATIVE_PREMODEL_HEARTBEAT_AND_REAL_TRANSPORT',authority_release=RELEASE,
            run_uuid=self.root.name,job_id=self.allocation['job_id'],hostname='fixture',source_sha='a'*40,
            manifest_sha256=sha(self.root/'RUN_MANIFEST.json'),models_loaded=0,new_model_queries=0,new_physics_integrations=0,
            production_heartbeat_or_science_fault_injection=False,
            heartbeat=dict(clock_source='CLOCK_REALTIME_AFTER_EACH_READ_AND_CLOCK_MONOTONIC',rows=rows),
            transport=dict(codec='PRODUCTION_JSON_LINE_STDIO_AND_SEALED_OBSERVATION_V1',scope='ISOLATED_FIXTURE_NO_MODEL_NO_PHYSICS',
                watchdogs_s=[20,60,900],duplicate_fixture_installs=0,physical_installs=0,unknown_effects_resent=0,
                cycles=[dict(sequence=i,query_uuid='fixture-'+str(i),RGB_payload_bytes=100000,maximum_RPC_s=.2,
                    fixture_install_count=1,ACK_matches_request=True,request_sha256=f'{i:064x}',response_sha256=f'{i+1:064x}') for i in range(80)]),
            isolated_faults={name:dict(status='PASS',unknown_effects_resent=0,failed_closed=True) for name in qualification.FAULTS})
        self.path=self.directory/qualification.FILENAME

    def put(self,value=None):
        if self.path.exists():self.path.chmod(0o600)
        self.path.write_text(json.dumps(self.value if value is None else value)+'\n');self.path.chmod(0o400)

    def test_missing_native_result_is_not_a_cpu_test_or_previous_smoke(self):
        with self.assertRaisesRegex(PermissionError,'NOT_RUN'):qualification.require(self.root,self.directory,self.allocation)
        self.put()
        result=qualification.require(self.root,self.directory,self.allocation)
        self.assertEqual(result['heartbeat_elapsed_s'],600)
        self.assertEqual(result['transport_cycles'],80)

    def test_wrong_job_host_or_preloaded_model_rejected(self):
        for key,value in [('job_id','previous.hopper-m-02'),('hostname','login-node'),('models_loaded',1),('source_sha','b'*40)]:
            with self.subTest(key=key):
                changed=dict(self.value,**{key:value});self.put(changed)
                with self.assertRaises(PermissionError):qualification.require(self.root,self.directory,self.allocation)

    def test_short_smoke_late_reply_or_duplicate_install_cannot_qualify(self):
        for change in ['short','few','repeat','resend']:
            value=deepcopy(self.value)
            if change=='short':value['heartbeat']['rows']=value['heartbeat']['rows'][:-1]
            if change=='few':value['transport']['cycles']=value['transport']['cycles'][:-1]
            if change=='repeat':value['transport']['cycles'][0]['fixture_install_count']=2
            if change=='resend':value['isolated_faults']['late_response']['unknown_effects_resent']=1
            self.put(value)
            with self.subTest(change=change),self.assertRaises(PermissionError):qualification.require(self.root,self.directory,self.allocation)

    def test_true_future_expired_wrong_run_and_clock_before_read_fail_closed(self):
        for change in ['future','expired','wrong_run','before_read']:
            value=deepcopy(self.value);proof=value['heartbeat']['rows'][20]['proof']
            if change=='future':proof['heartbeat']['observed_epoch']=proof['revalidated_clock_epoch']+.001
            if change=='expired':proof['heartbeat']['observed_epoch']=proof['revalidated_clock_epoch']-300.001
            if change=='wrong_run':proof['heartbeat']['run_uuid']='OTHER'
            if change=='before_read':proof['clock_sampled_after_read']=False
            self.put(value)
            with self.subTest(change=change),self.assertRaises(PermissionError):qualification.require(self.root,self.directory,self.allocation)


if __name__=='__main__':unittest.main()
