import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.pi0294.fd1 import sha
from scripts.pi0294.vap1 import attempts, plan


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():path.unlink()
    path.write_text(json.dumps(value)+'\n');path.chmod(0o400);return sha(path)


class AttemptContracts(unittest.TestCase):
    def test_one_legal_audited_negative_does_not_silently_create_a_100percent_success_gate(self):
        groups=[plan.candidate(*slot,0) for slot in plan.slots()]
        rows=[dict(id=c['id'],task_success=True,admitted=True,evidence_qualified=True,
            native_faults=0,unknown_effects=[],R0_PASS_to_R1_FAIL=False) for c in plan.cards(groups)[:24]]
        row=rows[1];row.update(task_success=False,admitted=False,evidence_qualified=False,
            status='PRESERVED_NEGATIVE_NOT_SUCCESS_DATA',source_raw_manifest_sha256='a'*64)
        self.assertEqual(plan.bank_gate(rows,groups)['status'],'NOT_READY')
        row['attempt_evidence']=dict(status=attempts.STATUS,partial_negative_audit_sha256='b'*64,
            original_capture_audit_sha256='c'*64,original_raw_manifest_sha256='a'*64,complete_success_data=False,
            admitted=False,rejected_prefix_live_ticks=0,raw_and_video_rehashed=True,new_integrations=0,new_constructors=0)
        gate=plan.bank_gate(rows,groups)
        self.assertEqual(gate['status'],'PASS_VAP1_DATA_BANK_GATE');self.assertEqual(gate['accounted'],24)
        self.assertEqual(gate['successes'],dict(T1=12,T2=11));self.assertEqual(gate['paired_groups'],11)
        for key,value in (('raw_and_video_rehashed',False),('rejected_prefix_live_ticks',1),('admitted',True)):
            bad=copy.deepcopy(rows);bad[1]['attempt_evidence'][key]=value
            self.assertEqual(plan.bank_gate(bad,groups)['status'],'NOT_READY')
        rows[1]['native_faults']=1
        self.assertEqual(plan.bank_gate(rows,groups)['status'],'NOT_READY')

    def test_rehashes_actual_saved_negative_and_preserves_original_failure_and_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            child=Path(tmp)/'capture';raw=child/'episodes/negative';raw.mkdir(parents=True)
            put(child/'lease.json',dict(status='CLOSED'))
            put(child/'OUTPUT_BINDING.private.json',dict(lease=str(child/'lease.json')))
            put(child/'RESOURCE_LEDGER.json',dict(status='TERMINAL_CAPTURE',unknown_effects=[]))
            put(child/'RUN_MANIFEST.json',dict(authority=dict(protocol_release=plan.AUTH)))
            put(raw/'RESULT.json',dict(fixture=False,model_calls=0,physics_steps=5600,error='PermissionError: PRE_EXECUTION_SAFETY_REJECT'))
            put(raw/'S1_RESULT.json',dict(live_native_failures=0,actions_modified=False,maximum_live_shadow_difference=0.))
            members=[dict(path=n,bytes=(raw/n).stat().st_size,sha256=sha(raw/n)) for n in ('RESULT.json','S1_RESULT.json')]
            manifest=put(raw/'FILE_MANIFEST.json',members)
            witness=dict(status='PASS_INDEPENDENT_PARTIAL_NEGATIVE_WITNESS',
                valid_scientific_negative_not_engineering_replacement=True,manifest_sha256=manifest,
                id='negative',complete=False,admitted=False,task_success=False,rejected_prefix_live_ticks=0,physics_steps=5600)
            audit=put(raw/'PARTIAL_NEGATIVE_AUDIT.json',witness)
            movie=child/'evidence/audits/negative/actual.mp4';movie.parent.mkdir(parents=True);movie.write_bytes(b'CPU fixture not video')
            original=dict(id='negative',audit_status=witness['status'],raw_manifest_sha256=manifest,native_faults=0,
                task_success=False,audit_sha256=audit,video=dict(relative_path=movie.relative_to(child).as_posix(),
                    bytes=movie.stat().st_size,sha256=sha(movie)))
            put(movie.parent/'CAPTURE_AUDIT.json',original)
            case=dict(source_episode=str(raw),capture_receipt=original)
            rows=[dict(id='negative',status='PRESERVED_NEGATIVE_NOT_SUCCESS_DATA',evidence_qualified=False,
                source_raw_manifest_sha256=manifest,admitted=False,task_success=False,native_faults=0,unknown_effects=[])]
            before=copy.deepcopy(rows);counts=[];qualified=attempts.qualify(rows,dict(negative=case),lambda:counts.append(1))
            self.assertEqual(rows,before);self.assertFalse(qualified[0]['evidence_qualified'])
            self.assertFalse(qualified[0]['admitted']);self.assertFalse(qualified[0]['task_success'])
            self.assertEqual(qualified[0]['attempt_evidence']['status'],attempts.STATUS);self.assertEqual(len(counts),3)
            bad=copy.deepcopy(witness);bad['rejected_prefix_live_ticks']=1;put(raw/'PARTIAL_NEGATIVE_AUDIT.json',bad)
            with self.assertRaises((PermissionError,ValueError)):
                attempts.qualify(rows,dict(negative=case),lambda:None)


if __name__=='__main__':unittest.main()
