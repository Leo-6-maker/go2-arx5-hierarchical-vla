"""Exact reservation credit is arithmetic only, with no invented PBS identity."""
from pathlib import Path
import unittest
from . import b_reservation_reconciliation as r
GRANT=(Path(__file__).resolve().parents[3]/'reports/PI0294/design/parallel-science-20261022/B1_HEARTBEAT_RECOVERY_AUTHORIZATION_20261023.json').read_bytes()
def fixture():
 return dict(actual_Hopper_ledger_sha256=r.COMPLETED_GLOBAL_SHA,actual_DEC1_ledger_sha256=r.DEC1_SHA),dict(reservation_scope='EXACT_B_RECOVERY_RECORDED_PREQSUB',request_id=r.REQUEST,request_sha256=r.REQUEST_SHA,original_reservation_intent_sha256=r.INTENT_SHA,job_id=None,submission_already_charged=True,work_id='TRAIN:B_COVERAGE:SEGMENT:2048',kind='TRAIN',resources=dict(H200_s=22560,Hopper_CPU_s=54000,optimizer_updates=1024,neural_queries=29,Hopper_storage_bytes=70000000000))
class RecordedPreqsub(unittest.TestCase):
 def test_exact_recorded_request_has_no_job_id(self):
  s,row=fixture();self.assertIs(r.validate_funded_preqsub(s,row,b_recovery_bytes=GRANT),row);self.assertIsNone(row['job_id'])
 def test_wrong_actual_ledger_bytes_refused(self):
  for k in ('actual_Hopper_ledger_sha256','actual_DEC1_ledger_sha256'):
   s,row=fixture();s[k]='0'*64
   with self.subTest(k=k),self.assertRaises(PermissionError):r.validate_funded_preqsub(s,row,b_recovery_bytes=GRANT)
 def test_wrong_request_or_invented_job_refused(self):
  for k,v in (('request_id','new-request'),('request_sha256','0'*64),('original_reservation_intent_sha256','0'*64),('job_id','not-a-real-job'),('submission_already_charged',False),('work_id','TRAIN:A_CONTROL:LONG')):
   s,row=fixture();row[k]=v
   with self.subTest(k=k),self.assertRaises(PermissionError):r.validate_funded_preqsub(s,row,b_recovery_bytes=GRANT)
 def test_no_partial_envelope_or_scope_expansion(self):
  for k,v in (('H200_s',22559),('Hopper_CPU_s',53999),('optimizer_updates',1023),('neural_queries',28),('Hopper_storage_bytes',0),('PBS_submissions',1)):
   s,row=fixture();row['resources'][k]=v
   with self.subTest(k=k),self.assertRaises(PermissionError):r.validate_funded_preqsub(s,row,b_recovery_bytes=GRANT)
  s,row=fixture()
  with self.assertRaises(PermissionError):r.validate_funded_preqsub(s,row,b_recovery_bytes=None)
if __name__=='__main__':unittest.main()
