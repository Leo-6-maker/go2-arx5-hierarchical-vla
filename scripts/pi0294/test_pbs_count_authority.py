"""Direct count-only authority plus captured full-ledger resource regression."""
import base64,gzip,json
from copy import deepcopy
from pathlib import Path
import unittest
from scripts.pi0294.generalization import pbs_count_authority as p

REPO=Path(__file__).resolve().parents[2]
CARD=REPO/'reports/PI0294/design/parallel-science-20261022/GPU_PBS_COUNT_UNLIMITED_DIRECT_AUTHORITY_20261029.json'
class CountAuthority(unittest.TestCase):
    def before(self):
        return dict(revision=7,used=dict(PBS_submissions=41,H200_s=1234),
            PBS_submissions=[dict(status='NATIVE_TERMINAL')for _ in range(41)],
            pending_updates=[],unknown_effects=[],other_caps=dict(GPU_s=532800,CPU_s=954000),
            package=dict(original_resources=dict(PBS_submissions=2),remaining_resources=dict(PBS_submissions=1)))
    def adopt(self,g=None):
        return p.propose(json.dumps(g or self.before()).encode(),CARD.read_bytes(),now_epoch=p.CLOCK[0]+100)
    def test_only_one_new_marker_and_revision_no_budget_or_history_rewrite(self):
        g=self.before();v=self.adopt(g)
        self.assertEqual(set(v)-set(g),{p.FIELD})
        self.assertTrue(all(v[k]==value for k,value in g.items()if k!='revision'))
        self.assertIsNone(v[p.FIELD]['PBS_submission_count_limit'])
        self.assertEqual(v[p.FIELD]['new_resource_costs'],{})
    def test_large_real_count_does_not_hit_a_new_hidden_count_ceiling(self):
        v=self.adopt()
        for count in(44,45,100,10000):
            v['PBS_submissions']=[dict(status='NATIVE_TERMINAL')for _ in range(count)]
            v['used']['PBS_submissions']=count
            self.assertGreater(p.legacy_vector_bound(v,44),count+3)
            self.assertEqual(p.history_upper(v,43),max(43,count))
            self.assertEqual(v['other_caps'],self.before()['other_caps'])
    def test_without_actual_adoption_old_count_semantics_remain(self):
        self.assertEqual(p.legacy_vector_bound(self.before(),44),44)
        self.assertEqual(p.history_upper(self.before(),43),43)
    def test_tampered_grant_or_marker_rejected(self):
        with self.assertRaises(PermissionError):p.grant(CARD.read_bytes()+b' ')
        for key,value in [('PBS_submission_count_limit',100),('other_caps_changed',True),('new_GPU_permission',True),('new_resource_costs',dict(H200_s=1))]:
            v=self.adopt();v[p.FIELD][key]=value
            with self.subTest(key=key),self.assertRaises(PermissionError):p.validate(v)
    def test_active_unknown_changed_count_and_late_adoption_rejected(self):
        for change in('active','unknown','pending'):
            g=self.before()
            if change=='active':g['PBS_submissions'][-1]['status']='SUBMITTED_PENDING_NATIVE'
            elif change=='unknown':g['unknown_effects']=['UNKNOWN_PRESERVED']
            else:g['pending_updates']=['UNFINISHED']
            with self.subTest(change=change),self.assertRaises(PermissionError):self.adopt(g)
        with self.assertRaises(PermissionError):p.propose(json.dumps(self.before()).encode(),CARD.read_bytes(),now_epoch=p.CLOCK[1])
        v=self.adopt();v['used']['PBS_submissions']+=1
        with self.assertRaises(PermissionError):p.validate(v)
    def test_exact_current_closed_full_history_and_all_other_projected_dimensions(self):
        path=REPO.parent/'private/minimal-adaptation-20261027/E2ONLY_F1507480_TERMINAL01_INDEPENDENT_CLOSED_READBACK.private.json'
        if not path.exists():self.skipTest('current private closed native capture unavailable')
        from scripts.pi0294.generalization import minimal_scope as scope,minimal_execution as e
        from scripts.pi0294.pv1.neural_ledger import check
        proof=json.loads(path.read_bytes());gb=gzip.decompress(base64.b64decode(proof['GLOBAL_gzip_base64']));g=json.loads(gb)
        local=json.loads((path.parent/'D25_SAME64_WIRE01_PREPARED.private.json').read_bytes());lb=gzip.decompress(base64.b64decode(local['LOCAL_gzip_base64']))
        before=scope.first512_forecast(gb,lb);v=p.propose(gb,CARD.read_bytes(),now_epoch=proof['observed_epoch']);check(v)
        after=scope.first512_forecast(e.raw(v),lb)
        self.assertEqual(after['cap_changes_required'],{})
        self.assertIsNone(after['dimensions']['PBS_submissions']['cap'])
        for key,row in before['dimensions'].items():
            with self.subTest(dimension=key):
                self.assertEqual(after['dimensions'][key]['projected'],row['projected'])
                if key!='PBS_submissions':self.assertEqual(after['dimensions'][key]['cap'],row['cap'])
        self.assertEqual(v['PBS_submissions'],g['PBS_submissions'])
        self.assertEqual(v['reserved'],g['reserved'])
if __name__=='__main__':unittest.main()
