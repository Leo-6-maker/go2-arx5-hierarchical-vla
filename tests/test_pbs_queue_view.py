import json
from types import SimpleNamespace
import unittest

from scripts.operations.pbs_queue_view import owned_queue


class QueueViewTests(unittest.TestCase):
    def test_both_sites_use_plain_selection_then_individual_json(self):
        for prefix in ('/opt/pbs/bin','/cm/shared/apps/pbspro/current/bin'):
            with self.subTest(prefix=prefix):
                calls=[]
                def query(argv,**kwargs):
                    calls.append(argv)
                    if argv[0].endswith('/qselect'):
                        return SimpleNamespace(returncode=0,stdout='1471903.stdct-mgmt-02\n')
                    return SimpleNamespace(returncode=0,stdout=json.dumps(dict(Jobs={
                        '1471903.stdct-mgmt-02':dict(Job_Owner='e1192548@head',job_state='Q')})))
                jobs=owned_queue('e1192548',prefix=prefix,query=query)
                self.assertEqual(set(jobs),{'1471903.stdct-mgmt-02'})
                self.assertEqual(calls[0],[prefix+'/qselect','-u','e1192548'])
                self.assertEqual(calls[1],[prefix+'/qstat','-f','-F','json','1471903.stdct-mgmt-02'])

    def test_empty_selection_does_not_parse_empty_user_view_as_json(self):
        calls=[]
        def query(argv,**kwargs):
            calls.append(argv);return SimpleNamespace(returncode=0,stdout='\n')
        self.assertEqual(owned_queue('e1192548',prefix='/opt/pbs/bin',query=query),{})
        self.assertEqual(len(calls),1)

    def test_foreign_duplicate_and_failed_selection_remain_rejected(self):
        for selection,code,owner in (('one',1,'e1192548'),('one one',0,'e1192548'),('one',0,'foreign')):
            def query(argv,**kwargs):
                if argv[0].endswith('/qselect'):
                    return SimpleNamespace(returncode=code,stdout=selection)
                return SimpleNamespace(returncode=0,stdout=json.dumps(dict(Jobs={'one':dict(Job_Owner=owner+'@head',job_state='Q')})))
            with self.assertRaises((PermissionError,ValueError)):
                owned_queue('e1192548',prefix='/opt/pbs/bin',query=query)

    def test_unknown_site_cannot_invoke_any_command(self):
        def query(*args,**kwargs):raise AssertionError('must not invoke')
        with self.assertRaises(PermissionError):owned_queue('e1192548',prefix='/unknown/bin',query=query)
