"""Byte fixtures only: no target constructor, policy or integration is called."""
import unittest,tempfile,struct,json
from pathlib import Path
from go2arx5_vla.world_model.init_trace_reader import RECORD,read_trace,read_snapshots,audit_trace

def fixture(root,*,stop=False,legacy_publish=False):
    path=Path(root)/'trace';rows=[];snapshots=[]
    def add(kind,token=0,values=(),t=0.,pos=0,vel=0,ref=None):
        n=len(values);data=list(values)+[0.]*(96-n)
        if ref:data[:3]=ref
        rows.append(RECORD.pack(len(rows),(len(rows)+1)*1000,t,kind,n,0,token,pos,vel,token,*data))
    for token in(1,2):
        add(2,token);add(3,token,[0.]*48);add(4,token,ref=(token-1,token,1024));snapshots.append(struct.pack('<5Q',token,token-1,token,4,1024)+struct.pack('<1024f',*([3.25]*1024)));add(5,token,[0.]*12)
        for channel in(0,1):add(6,token,()if legacy_publish else[channel])
        for channel in(0,1):add(7,token,[channel])
        add(8,0,pos=token,vel=token)
    add(9,values=[0.]*90);add(10,values=[0.]*86);add(11,1,[0.]*86);add(12,1,[0.]*86,t=.002)
    if stop:add(13,1,t=.002);add(14,1,t=.002)
    path.write_bytes(b'WM02IT01'+struct.pack('<II',840,65536)+b''.join(rows));Path(str(path)+'.snapshots').write_bytes(b''.join(snapshots));Path(str(path)+'.summary').write_text(json.dumps(dict(events=len(rows),dropped=0,stop_latched=stop)))
    return path

class InitEvidenceAudit(unittest.TestCase):
    def test_full_payload_reference_survives_reader_and_no_stop_is_not_tested(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=fixture(tmp);rows=read_trace(p);s=read_snapshots(str(p)+'.snapshots',keep_values=True);self.assertEqual(s[(1,0)]['values'],[3.25]*1024)
            self.assertEqual(next(r for r in rows if r['kind']==4)['payload_reference']['original_count'],1024)
            a=audit_trace(p,max_native_ticks=50);self.assertEqual(a['status'],'RECORDED_TRACE_CHECKS_PASS');self.assertEqual(a['STOP_check'],'NOT_TESTED');self.assertIsNone(a['integrations_after_first_STOP']);self.assertTrue(a['target_qualification'].startswith('NOT_ESTABLISHED'));self.assertEqual(a['new_effects_of_audit']['target_integrations'],0)
    def test_zero_drop_does_not_hide_missing_payload_or_legacy_channel(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=fixture(tmp,legacy_publish=True);Path(str(p)+'.snapshots').write_bytes(b'');a=audit_trace(p)
            self.assertEqual(a['writer']['dropped'],0);self.assertEqual(a['status'],'INSUFFICIENT_RECORDED_EVIDENCE');self.assertEqual(a['full_payloads_not_retained'],2);self.assertIn('LEGACY_PUBLICATION_CHANNEL_NOT_RECORDED',a['missing'])
    def test_mixed_versions_and_actual_extra_step_tokens_fail(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=fixture(tmp,stop=True);raw=bytearray(p.read_bytes());rows=read_trace(p);apply=next(r['ticket']for r in rows if r['kind']==8);off=16+apply*840;values=list(RECORD.unpack(raw[off:off+840]));values[8]=0;raw[off:off+840]=RECORD.pack(*values)
            n=len(rows)
            for kind,t in((11,.002),(12,.004)):
                raw+=RECORD.pack(n,(n+1)*1000,t,kind,86,0,2,2,2,2,*([0.]*96));n+=1
            p.write_bytes(raw);Path(str(p)+'.summary').write_text(json.dumps(dict(events=n,dropped=0,stop_latched=True)));a=audit_trace(p)
            self.assertEqual(a['status'],'RECORDED_TRACE_CHECKS_FAIL');self.assertEqual(a['integrations_after_first_STOP'],1);self.assertIn('ACTUAL_POSITION_VELOCITY_VERSION_MISMATCH',a['errors'])
    def test_snapshot_reference_identity_and_truncation_are_strict(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=fixture(tmp);s=Path(str(p)+'.snapshots');blob=bytearray(s.read_bytes());struct.pack_into('<Q',blob,16,99);s.write_bytes(blob)
            with self.assertRaises(ValueError):audit_trace(p)
            s.write_bytes(blob[:-1])
            with self.assertRaises(ValueError):read_snapshots(s)
    def test_empty_loader_trace_cannot_pass_and_nonfinite_values_rejected(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'trace';p.write_bytes(b'WM02IT01'+struct.pack('<II',840,65536));Path(str(p)+'.summary').write_text(json.dumps(dict(events=0,dropped=0,stop_latched=False)));a=audit_trace(p);self.assertEqual(a['status'],'INSUFFICIENT_RECORDED_EVIDENCE');self.assertIn('NO_NATIVE_STEP_RECORDS',a['missing'])
            p.write_bytes(p.read_bytes()+RECORD.pack(0,1,0.,3,1,0,1,0,0,1,float('nan'),*([0.]*95)))
            with self.assertRaises(ValueError):read_trace(p)

if __name__=='__main__':unittest.main()
