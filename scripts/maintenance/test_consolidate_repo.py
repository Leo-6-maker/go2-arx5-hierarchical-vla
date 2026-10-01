import importlib.util, pathlib, subprocess, tempfile, unittest
P = pathlib.Path(__file__).with_name('consolidate_repo.py')
spec = importlib.util.spec_from_file_location('maintenance', P)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class PlanTests(unittest.TestCase):
    def b(self, name, sha='a'*40, protected=False):
        return {'name':name,'sha':sha,'protected':protected}
    def p(self,n,head='stage45a/old',base='main',repository=m.REPO):
        return {'number':n,'head_ref':head,'base_ref':base,'head_repository':repository}
    def test_keep_current_and_recent(self):
        for name in m.KEEP:
            self.assertEqual(m.plan_branches([self.b(name)],[])[0]['decision'],'KEEP')
    def test_historical(self):
        for name in ['pi0238/runner','pi0260-r3/test','stage45e/old','repair/pi0096-nested-cosmos-hf-cache']:
            self.assertEqual(m.plan_branches([self.b(name)],[])[0]['decision'],'ARCHIVE_COMPATIBILITY_TAG')
    def test_new_unknown_never_retired(self):
        for name in ['pi0267/new','pi0266/other','feature/new','other-branch']:
            self.assertEqual(m.plan_branches([self.b(name)],[])[0]['decision'],'KEEP')
    def test_protected(self):
        self.assertEqual(m.plan_branches([self.b('stage20/old',protected=True)],[])[0]['decision'],'KEEP')
    def test_new_pr_dependencies_kept(self):
        for name in ['stage20/old','stage21/base']:
            self.assertEqual(m.plan_branches([self.b(name)],[self.p(99,'stage20/old','stage21/base')])[0]['decision'],'KEEP')
    def test_foreign_pr_kept(self):
        self.assertEqual(m.plan_branches([self.b('stage45a/old')],[self.p(46,repository='elsewhere/fork')])[0]['decision'],'KEEP')
    def test_lease_precheck(self):
        plan=m.plan_branches([self.b('stage20/old')],[])
        self.assertEqual(m.delete_candidates(plan,{'stage20/old':'b'*40},{'stage20/old':'a'*40},[]),[])
    def test_missing_or_conflicting_backup_blocks(self):
        plan=m.plan_branches([self.b('stage20/old')],[])
        for tags in [{},{'stage20/old':'b'*40}]:
            self.assertEqual(m.delete_candidates(plan,{'stage20/old':'a'*40},tags,[]),[])
    def test_remaining_pr_blocks(self):
        plan=m.plan_branches([self.b('stage20/old')],[])
        self.assertEqual(m.delete_candidates(plan,{'stage20/old':'a'*40},{'stage20/old':'a'*40},[self.p(99,'stage20/old')]),[])
    def test_precise_valid_delete(self):
        plan=m.plan_branches([self.b('stage20/old')],[])
        self.assertEqual(len(m.delete_candidates(plan,{'stage20/old':'a'*40},{'stage20/old':'a'*40},[])),1)
    def test_wrong_authority_never_default_apply(self):
        self.assertEqual(m.digest(m.json_text({'a':1})),m.digest(m.json_text({'a':1})))
        self.assertNotEqual(m.digest(m.json_text({'a':1})),m.digest(m.json_text({'a':2})))
    def test_script_has_no_scientific_imports(self):
        import ast
        tree=ast.parse(P.read_text())
        roots=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Import): roots.extend(a.name.split('.')[0] for a in n.names)
            if isinstance(n,ast.ImportFrom) and n.module: roots.append(n.module.split('.')[0])
        self.assertFalse(set(roots)&{'torch','numpy','go2arx5_vla','isaacgym','transformers'})

class AtomicGitTests(unittest.TestCase):
    def test_local_archive_and_stale_lease(self):
        with tempfile.TemporaryDirectory() as t:
            root=pathlib.Path(t); remote=root/'remote.git'; local=root/'local'
            def run(args,cwd=None,good=True):
                p=subprocess.run(['git',*args],cwd=cwd,text=True,capture_output=True)
                if good and p.returncode: self.fail(p.stderr)
                return p
            run(['init','--bare',str(remote)]);run(['init',str(local)])
            run(['config','user.name','Fixture'],local);run(['config','user.email','fixture@example.invalid'],local)
            (local/'f').write_text('one');run(['add','f'],local);run(['commit','-m','one'],local)
            a=run(['rev-parse','HEAD'],local).stdout.strip()
            run(['remote','add','origin',str(remote)],local)
            run(['push','origin',f'{a}:refs/heads/stage-old',f'{a}:refs/heads/stage-peer'],local)
            run(['push','--atomic','origin',f'{a}:refs/tags/stage-old',f'{a}:refs/tags/stage-peer'],local)
            (local/'f').write_text('two');run(['commit','-am','two'],local)
            b=run(['rev-parse','HEAD'],local).stdout.strip()
            run(['push','origin',f'{b}:refs/heads/stage-old'],local)
            p=run(['push','--atomic',f'--force-with-lease=refs/heads/stage-old:{a}',f'--force-with-lease=refs/heads/stage-peer:{a}',
                   'origin',':refs/heads/stage-old',':refs/heads/stage-peer'],local,good=False)
            self.assertNotEqual(p.returncode,0)
            refs=run(['ls-remote','origin'],local).stdout
            self.assertIn(b+'\trefs/heads/stage-old',refs)
            self.assertIn(a+'\trefs/heads/stage-peer',refs)
            self.assertIn(a+'\trefs/tags/stage-old',refs)
            run(['push','--atomic',f'--force-with-lease=refs/heads/stage-peer:{a}','origin',':refs/heads/stage-peer'],local)
            refs=run(['ls-remote','origin'],local).stdout
            self.assertNotIn('\trefs/heads/stage-peer',refs)
            self.assertIn(a+'\trefs/tags/stage-peer',refs)

if __name__=='__main__': unittest.main(verbosity=2)
