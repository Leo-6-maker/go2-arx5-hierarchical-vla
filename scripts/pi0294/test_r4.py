"""Zero-compute authority and clock checks."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import r4


class R4AuthorityTests(unittest.TestCase):
    def test_no_clock_reset_and_adoption_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/r4.RUN;root.mkdir()
            (root/'AUTHORITY_ADOPTION.json').write_text('{}')
            authority=dict(authorizes_execution=True,T0_epoch=r4.T0,new_work_cutoff_epoch=r4.CUTOFF,
                deadline_epoch=r4.DEADLINE,AUG_FREEZE_epoch=r4.AUG_FREEZE,limits={})
            manifest=dict(run_uuid=r4.RUN,pi='PI0294',owner='EXISTING_LOCAL_PC',
                authority=dict(protocol_ref=r4.AUTHORITY,go_comment=r4.GO_COMMENT),limits={})
            with patch.object(r4,'sha',return_value=r4.AUTHORITY_SHA):
                for mutation in ('clock','permission','run'):
                    a,m=copy.deepcopy(authority),copy.deepcopy(manifest)
                    if mutation=='clock':a['deadline_epoch']+=1
                    elif mutation=='permission':a['authorizes_execution']=False
                    else:m['run_uuid']='other'
                    with patch.object(r4,'read',side_effect=[a,m]):
                        with self.assertRaises(PermissionError):r4.validate_entry(root,profile='4080-simulation',now=r4.T0+1)
                with patch.object(r4,'read',side_effect=[authority,manifest]):
                    with self.assertRaises(PermissionError):r4.validate_entry(root,profile='4080-simulation',now=r4.CUTOFF)
            self.assertEqual(r4.DEADLINE-r4.T0,12*3600)
            self.assertEqual(r4.AUG_FREEZE-r4.T0,3*3600)

    def test_collection_union_and_fixed_counts(self):
        m=dict(run_uuid=r4.RUN)
        a=dict(T0_epoch=r4.T0,cutoff_epoch=r4.CUTOFF,deadline_epoch=r4.DEADLINE,
            lease_expires_epoch=r4.DEADLINE,allowed_cpus=list(range(8)),root='unused',
            limits=dict(physics_starts=160,physics_steps=2560000,rgb_pairs=102560,cpu_s=80*3600,output_bytes=72*1024**3))
        p=dict(kind='R4_CONTACT_PILOT',episodes=[{}]*4)
        budget=dict(Core_CPU_s=930,CPU_reservations={})
        with patch.object(r4,'read',return_value=budget):r4.validate_cpu_admission(m,a,p,2,r4.T0+1)
        with patch.object(r4,'read',return_value=budget):
            with self.assertRaises(PermissionError):r4.validate_cpu_admission(m,a,dict(p,episodes=[{}]*3),2,r4.T0+1)
        budget['CPU_reservations']['other']=dict(cpus=[0],cpu_s_cap=100,kind='CPU',pid=-1)
        with patch.object(r4,'read',return_value=budget):
            with self.assertRaises(PermissionError):r4.validate_cpu_admission(m,a,p,2,r4.T0+1)


if __name__=='__main__':unittest.main()
