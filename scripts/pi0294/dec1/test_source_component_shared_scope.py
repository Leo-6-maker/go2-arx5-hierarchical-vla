"""Exact source-file identity regression; no native worker or authority proof."""
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.pv1 import scope
from scripts.pi0294.dec1 import source_cpu_intake as intake


def binding():
 return dict(frozen_numerical_files=dict(intake.ORIGINAL_13,**{
  p:r['original_sha256'] for p,r in intake.GLUE_2.items()}),
  frozen_original_13_files=deepcopy(intake.ORIGINAL_13),authorized_interface_glue_files=deepcopy(intake.GLUE_2))


def test_original_fifteen_legacy_roles_still_reject_the_two_new_glue_bytes():
 code=Path(__file__).resolve().parents[3]
 for role in (scope.EVAL_ROLE,scope.FINAL_EVAL_ROLE,scope.CAPTURE_ROLE):
  with pytest.raises(ValueError,match='FROZEN_ORIGINAL_NUMERICAL'):
   scope.validate_numerical_files(code,binding(),role)


def test_new_role_requires_exact_original_pins_and_explicit_glue_provenance():
 code=Path(__file__).resolve().parents[3];m=binding();stage=dict(original_pin_report_sha256=intake.PIN_REPORT_SHA256)
 scope.validate_numerical_files(code,m,scope.SOURCE_COMPONENT_ROLE,stage)
 m['authorized_interface_glue_files']['scripts/pi0294/episode.py']['explicit_glue_publication']='f'*40
 with pytest.raises(PermissionError,match='DELTA_PROVENANCE'):
  scope.validate_numerical_files(code,m,scope.SOURCE_COMPONENT_ROLE,stage)


def test_component_registry_is_unique_and_old_role_profile_unchanged():
 assert scope.registry_role(scope.SOURCE_COMPONENT_ROLE,'one')!=scope.registry_role(scope.SOURCE_COMPONENT_ROLE,'two')
 assert scope.CPU_PROFILES[scope.SOURCE_COMPONENT_ROLE]==scope.CPU_PROFILES[scope.EVAL_ROLE]=='4080-simulation'
 with pytest.raises(PermissionError):scope.registry_role(scope.SOURCE_COMPONENT_ROLE,'../escape')


def test_unknown_role_does_not_acquire_the_component_exception():
 with pytest.raises(PermissionError):scope.validate_numerical_files('.',binding(),'FAKE_COMPONENT',{})


def test_only_component_metadata_closeout_survives_worker_and_hard_deadlines():
 a=dict(T0_epoch=1,stop_new_physics_epoch=10,hard_closeout_epoch=20);l=dict(deadline_epoch=9)
 for now in (9,10,20,21):
  assert scope.lease_time_allowed(scope.SOURCE_COMPONENT_ROLE,True,now,a,l)
  assert not scope.lease_time_allowed(scope.SOURCE_COMPONENT_ROLE,False,now,a,l)
  assert not scope.lease_time_allowed(scope.EVAL_ROLE,True,now,a,l)
 assert not scope.lease_time_allowed(scope.SOURCE_COMPONENT_ROLE,True,0,a,l)


def test_actual_child_ledger_can_preserve_expired_overrun_costs_but_cannot_advance(tmp_path,monkeypatch):
 from scripts.pi0294.test_pv1_core_ledger import fixture
 from scripts.pi0294.pv1 import core_ledger
 from scripts.pi0292.durable import read,write
 root,old=fixture(tmp_path);m=read(root/'RUN_MANIFEST.json')
 m['scope']=scope.SOURCE_COMPONENT_ROLE;write(root/'RUN_MANIFEST.json',m)
 meter=core_ledger.Ledger(root)
 with pytest.raises(RuntimeError):
  with meter.steps('shadow_integrations',200,'unknown-prefix'):raise RuntimeError('ACTUAL_UNKNOWN')
 meter.cpu(meter.limits['CPU_s']+1)
 monkeypatch.setattr(core_ledger.time,'time',lambda:meter.a['hard_closeout_epoch']+10)
 value=meter.guard(closeout=True)
 assert value['Core_CPU_s']==meter.limits['CPU_s']+1
 assert value['reservations']['unknown-prefix']['maximum']==200
 with pytest.raises(TimeoutError):meter.charge('engine_starts',1)
 with pytest.raises(TimeoutError):old.guard(closeout=True)
