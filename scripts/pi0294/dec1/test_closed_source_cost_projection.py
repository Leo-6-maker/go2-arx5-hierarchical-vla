"""Synthetic proposal fixtures; no native adoption, IO mutation or B submission."""
from copy import deepcopy
import json,unittest
from pathlib import Path
import pytest
from scripts.pi0294.dec1.test_source_cpu_closeout import packet
from scripts.pi0294.dec1 import source_cpu_closeout as close
from scripts.pi0294.dec1 import closed_source_cost_projection as u
class Tests(unittest.TestCase):
 def fixture(self):
  mp=pytest.MonkeyPatch();self.addCleanup(mp.undo);f=packet.__wrapped__(mp);plan=close.propose(**f['bind']());p,c=plan['proposed_parent'],plan['proposed_Core'];p['GPU_reserved_s']=0
  d=deepcopy(f['Core']);d.update(CPU_s=10000.,GPU_allocated_s=70000,GPU_reserved_s=0,CPU_reserved_s=0,
   optimizer_updates=5891,neural_queries=97,new_submissions=['preserved-A'],optimizer_updates_reserved=0,neural_queries_reserved=0)
  receipt=dict(status='SOURCE_DUAL_FINANCIAL_TERMINAL_BOUND_SETTLED_CORE_LEASE_CLOSED_COMPONENT_UNKNOWN',
   Core=dict(status='CORE_FINANCIAL_BOUND_CHARGED_LEASE_CLOSED_SCIENCE_UNKNOWN',invoice_sha256=plan['invoice_sha256']),
   LOCAL_readback=dict(status='CONFIRMED_CURRENT_PROPOSED_BYTES_WITH_RECEIPT'),invoice=plan['invoice'])
  boundary=dict(schema='VAP1_A4096_CLOSED_FOR_INACTIVE_DEC1_SOURCE_COST_PROJECTION_V1',job_id='1477527.stdct-mgmt-02',step=4096,native_state='F',
   Exit_status=0,owner_naturally_exited=True,GPU_lease='CLOSED',CPU_lease='CLOSED',full_state_audit_passed=True,once_settlement_confirmed=True)
  return p,c,d,receipt,boundary
 def call(self,p,c,d,r,b):return u.propose(u.encode(p),u.encode(c),u.encode(d),dual_receipt_bytes=u.encode(r),A_boundary_bytes=u.encode(b))
 def test_closed_Core_costs_project_once_inside_DEC1_without_recharging_global_or_GPU_CPU(self):
  p,c,d,r,b=self.fixture();before=deepcopy((p,c,d));v=self.call(p,c,d,r,b);after=v['proposed_Hopper_DEC1'];a=r['invoice']['financial_charged']
  self.assertEqual(after['episode_starts'],d['episode_starts']+1);self.assertEqual(after['physics_ticks'],d['physics_ticks']+16490)
  self.assertEqual(after['constructors'],d['constructors']+6);self.assertEqual(after['CPU_s'],d['CPU_s'])
  self.assertEqual(after['GPU_allocated_s'],d['GPU_allocated_s']);self.assertEqual(after['new_submissions'],d['new_submissions'])
  self.assertFalse(v['new_global_charge']);self.assertEqual((p,c,d),before)
  self.assertEqual(v['total_closed_Core_output_upper_bytes'],a['CoreLab_output_bytes'])
  with self.assertRaises(PermissionError):self.call(p,c,after,r,b)
 def test_active_A_or_missing_success_close_evidence_refuses_before_projection(self):
  for kind in ('step','Q','exit1','owner','GPU_lease','CPU_lease','audit','settlement','parent_GPU_hold','native_GPU_hold','native_CPU_hold'):
   with self.subTest(kind=kind):
    p,c,d,r,b=self.fixture()
    if kind=='step':b['step']=2048
    if kind=='Q':b['native_state']='Q'
    if kind=='exit1':b['Exit_status']=1
    if kind=='owner':b['owner_naturally_exited']=False
    if kind in ('GPU_lease','CPU_lease'):b[kind]='ACTIVE'
    if kind=='audit':b['full_state_audit_passed']=False
    if kind=='settlement':b['once_settlement_confirmed']=False
    if kind=='parent_GPU_hold':p['GPU_reserved_s']=22560
    if kind=='native_GPU_hold':d['GPU_reserved_s']=22560
    if kind=='native_CPU_hold':d['CPU_reserved_s']=54000
    with self.assertRaises(PermissionError):self.call(p,c,d,r,b)
 def test_unclosed_source_unknown_write_or_mismatched_invoice_never_catches_up(self):
  for kind in ('Core_row','parent_row','unknown_write','remaining','mismatch','Core_receipt','LOCAL_receipt','clock'):
   with self.subTest(kind=kind):
    p,c,d,r,b=self.fixture();key=r['invoice']['request_id']
    if kind=='Core_row':c['source_cpu_reservations'][key]['status']='RUNNING'
    if kind=='parent_row':p['source_cpu_reservations'][key]['status']='RESERVED'
    if kind=='unknown_write':p['unknown_effects']=['unreconciled']
    if kind=='remaining':p['source_cpu_reservations'][key]['remaining']['CoreLab_CPU_s']=1
    if kind=='mismatch':r['invoice']['financial_charged']['episode_starts']+=1
    if kind=='Core_receipt':r['Core']['status']='UNKNOWN'
    if kind=='LOCAL_receipt':r['LOCAL_readback']['status']='UNKNOWN'
    if kind=='clock':d['original_T0_epoch']+=1
    with self.assertRaises(PermissionError):self.call(p,c,d,r,b)
if __name__=='__main__':unittest.main()
