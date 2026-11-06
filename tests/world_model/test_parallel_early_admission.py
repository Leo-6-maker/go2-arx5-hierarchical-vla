import json,tempfile,unittest
from pathlib import Path
from scripts.wm02.pilot1_parallel_worker import close_proven_unstarted_children
from go2arx5_vla.world_model.storage import digest


class EarlyAdmission(unittest.TestCase):
    def fixture(self,root):
        spec=root/'batch.json';slots=[];leases=[]
        for i in range(2):
            r=root/str(i);r.mkdir();lp=r/'lease.json';sp=r/'spec.json'
            sp.write_text(json.dumps(dict(registered_root=str(r),lease=str(lp),parallel_batch_spec=str(spec))))
            lp.write_text(json.dumps(dict(status='ACTIVE',run_uuid=r.name,spec_sha256=digest(sp))))
            slots.append(dict(spec=str(sp),spec_sha256=digest(sp)));leases.append(lp)
        s=dict(slots=slots);spec.write_text(json.dumps(s));return spec,s,leases

    def test_actual_prelaunch_failure_closes_only_matching_own_unstarted_leases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,s,leases=self.fixture(root)
            self.assertEqual(close_proven_unstarted_children(p,s,root),['0','1'])
            for lp in leases:
                v=json.loads(lp.read_text());self.assertEqual(v['status'],'CLOSED');self.assertFalse(v['worker_started'])

    def test_possible_child_launch_or_partial_native_output_never_gets_inferred_closed(self):
        for marker in ['batch-01/INHERITED_OWNER.private.json','0/native-01']:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);p,s,leases=self.fixture(root);m=root/marker
                if marker.endswith('.json'):m.parent.mkdir();m.write_text('{}')
                else:m.mkdir()
                self.assertEqual(close_proven_unstarted_children(p,s,root),[])
                self.assertTrue(all(json.loads(lp.read_text())['status']=='ACTIVE'for lp in leases))


if __name__=='__main__':unittest.main()
