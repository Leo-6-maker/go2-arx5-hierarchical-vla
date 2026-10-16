"""CPU-only MP1 contracts; synthetic rows never qualify scientific data."""
from collections import Counter
from types import SimpleNamespace
import unittest
from pathlib import Path
import subprocess
import tempfile

from .mp1.dataset import SourceSampler


def dataset(tasks=('T1',)):
    rows=[]
    for task in ['T1','T2']:
        for episode in range(96 if task=='T1' else 30):
            rows.extend((dict(id=task+'-'+str(episode),task=task,split='TRAIN'),a) for a in range(3))
    original=len(rows)
    for task in tasks:
        for episode in range(2):
            rows.extend((dict(id='C-'+task+'-'+str(episode),task=task,split='TRAIN',source_student_episode='student-'+str(episode)),a) for a in range(5))
    return SimpleNamespace(rows=rows,original_rows=original,manifest_sha='fixture-only')


class MP1Tests(unittest.TestCase):
    def test_ROS_setup_optional_variables_do_not_disable_later_guards(self):
        definition=Path('infrastructure/containers/vanda_wbc_jazzy.def').read_text()
        prefix=definition.split('%test\n',1)[1].split('    /opt/wbc-python/bin/python',1)[0]
        with tempfile.TemporaryDirectory() as d:
            setup=Path(d)/'setup.sh';setup.write_text('test -z "$AMENT_TRACE_SETUP_FILES"\nexport ROS_DISTRO=jazzy\n')
            script=prefix.replace('/opt/ros/jazzy/setup.sh',str(setup))
            good=subprocess.run(['/bin/sh','-c',script+'test "$ROS_DISTRO" = jazzy'],capture_output=True)
            self.assertEqual(good.returncode,0)
            unset=subprocess.run(['/bin/sh','-c',script+'echo "$MP1_POST_SETUP_UNDEFINED"'],capture_output=True)
            self.assertNotEqual(unset.returncode,0)
            setup.write_text('false\n')
            failed=subprocess.run(['/bin/sh','-c',script+'echo SHOULD_NOT_EXECUTE'],capture_output=True)
            self.assertNotEqual(failed.returncode,0);self.assertNotIn(b'SHOULD_NOT_EXECUTE',failed.stdout)

    def test_task_balance_precedes_T1_only_correction(self):
        d=dataset();s=SourceSampler(d,stop=64);seen=Counter()
        for cursor,index in s:
            seen[d.rows[index][0]['task']]+=1
            seen['correction' if index>=d.original_rows else 'original']+=1
        self.assertEqual(seen,Counter(T1=32,T2=32,correction=16,original=48))
        self.assertEqual(s.state_dict(64)['overall_correction_fraction'],.25)
        self.assertEqual({t:len(v) for t,v in s.original.items()},dict(T1=96,T2=30))

    def test_both_qualified_tasks_receive_half_within_task(self):
        d=dataset(('T1','T2'));s=SourceSampler(d,stop=64)
        self.assertEqual(sum(i>=d.original_rows for c,i in s),32)
        self.assertEqual(s.state_dict(64)['overall_correction_fraction'],.5)

    def test_BC_and_exact_resume_order(self):
        for tasks in [(),('T1',),('T1','T2')]:
            d=dataset(tasks);full=list(SourceSampler(d,stop=96))
            self.assertEqual(full[32:],list(SourceSampler(d,start=32,stop=96)))
            self.assertEqual(SourceSampler(d,stop=96).seed,20261017)
            if not tasks:self.assertFalse(any(i>=d.original_rows for c,i in full))

    def test_DEV_or_missing_original_episode_is_rejected(self):
        d=dataset();d.rows[0][0]['split']='DEV'
        with self.assertRaises(ValueError):SourceSampler(d)
        d=dataset();d.rows=d.rows[3:];d.original_rows-=3
        with self.assertRaises(ValueError):SourceSampler(d)

    def test_out_of_scope_cursor_and_single_correction_episode_fail(self):
        d=dataset();s=SourceSampler(d,stop=32)
        for cursor in [-1,32]:
            with self.assertRaises(IndexError):s.index_at(cursor)
        d.rows=d.rows[:-5]
        with self.assertRaises(ValueError):SourceSampler(d)
        d=dataset()
        for row,a in d.rows[d.original_rows:]:row['source_student_episode']='same-parent'
        with self.assertRaises(ValueError):SourceSampler(d)


if __name__=='__main__':unittest.main()
