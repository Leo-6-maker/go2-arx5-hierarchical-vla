"""Mature-probe callback ordering with fake Torch/model modules only."""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from scripts.operations import vanda_inference_probe as probe
from scripts.pi0294.e1.model import E1ResidentModel
from scripts.pi0294.resident import ResidentModel
import scripts.pi0294.numerics
from scripts.pi0294.mp1.probe_compare import PRECISION
from .alternate_accounting import VERSION


class ProbeAdapterTests(unittest.TestCase):
    def run_fixture(self,*,new=True,stop_before_query=False):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name)
        def put(n,v):(root/n).write_text(json.dumps(v))
        put('identity.json',dict(files={}));put('statistics.json',{})
        put('observation.json',dict(payload=dict(fixture='SYNTHETIC_NOT_REAL_TRAIN_DATA')))
        put('lease.json',dict(status='ACTIVE'));put('ALLOCATION_ADMISSION.json',dict(allocation_end_epoch=time.time()+1800))
        m=dict(output_root=str(root),checkpoint=str(root),identity=str(root/'identity.json'),cosmos_files={},
            bindings=dict(source='CPU_FIXTURE_ONLY',cosmos='CPU_FIXTURE_ONLY',cosmos_hub='CPU_FIXTURE_ONLY'),
            observation=str(root/'observation.json'),lease=str(root/'lease.json'),run_uuid='fixture')
        if new:m.update(dec1_qualification_version=VERSION,dec1_request_id='fixed-fixture-request')
        trace=[]
        class Model:
            action_head=SimpleNamespace(num_inference_timesteps=4)
            def to(self,*a):return self
            def eval(self):return self
            def requires_grad_(self,v):assert v is False
        def load(*args,**kw):
            trace.append('LOAD_ACTUAL_FIXTURE');return Model(),{}
        def generate(instance,payload,seed):
            trace.append(('GENERATE',seed))
            return dict(physical_H40=np.zeros((40,10)),normalized=np.zeros((1,40,132),dtype=np.float32),
                normalized_outside_unit_components=0,noise=dict(fixture=True),inference_precision=PRECISION)
        def events(kind,payload):
            trace.append((kind,payload.get('index')))
            if kind=='QUERY_BEGIN' and stop_before_query:raise PermissionError('CPU_FIXTURE_STALE_OWNER')
        cuda=SimpleNamespace(device_count=lambda:1,get_device_name=lambda i:'NVIDIA A40',synchronize=lambda:None,
            reset_peak_memory_stats=lambda:None,max_memory_allocated=lambda:10,max_memory_reserved=lambda:20)
        torch=ModuleType('torch');torch.cuda=cuda;torch.__version__='CPU_FAKE_NO_GPU'
        transformers=ModuleType('transformers');transformers.AutoModel=SimpleNamespace(from_pretrained=load)
        modules={'torch':torch,'transformers':transformers}
        for name in ('gr00t','gr00t.model','gr00t.model.gr00t_n1d7','gr00t.model.gr00t_n1d7.gr00t_n1d7'):
            module=ModuleType(name);module.__path__=[];modules[name]=module
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules,modules))
            stack.enter_context(patch.dict('os.environ',{'PBS_JOBID':'fixture.stdct-mgmt-02'}))
            for target,value in [('scripts.operations.vanda_inference_probe.validate',m),
                    ('scripts.pi0294.numerics.configure_torch',dict(mode='CPU_FAKE')),
                    ('scripts.pi0292.train_adapter.official_import',None),
                    ('scripts.pi0292.train_adapter.build_processor',object()),
                    ('scripts.pi0294.local_backbone.local_model_name','CPU_FIXTURE_ONLY')]:
                stack.enter_context(patch(target,return_value=value))
            stack.enter_context(patch.object(probe.os,'sched_setaffinity'))
            stack.enter_context(patch.object(probe.resource,'setrlimit'))
            stack.enter_context(patch.object(probe.signal,'signal'))
            stack.enter_context(patch.object(probe.signal,'alarm'))
            stack.enter_context(patch.object(E1ResidentModel,'generate',generate))
            stack.enter_context(patch.object(ResidentModel,'generate',generate))
            if stop_before_query:
                with self.assertRaises(PermissionError):probe.probe(root/'manifest',events=events)
                result=json.loads((root/'RESULT.private.json').read_text())
            else:result=probe.probe(root/'manifest',**({'events':events} if new else {}))
        return root,result,trace

    def test_all_seventeen_calls_are_bracketed_by_durable_observer_callbacks(self):
        root,result,trace=self.run_fixture()
        self.assertEqual(trace[:3],[('MODEL_LOAD_BEGIN',None),'LOAD_ACTUAL_FIXTURE',('MODEL_LOAD_COMPLETE',None)])
        for i in range(17):
            start=3+i*3
            self.assertEqual(trace[start:start+3],[('QUERY_BEGIN',i),('GENERATE',29400+i),('QUERY_COMPLETE',i)])
        self.assertEqual(trace[-1],('FINAL',None));self.assertEqual(len(result['queries']),17)
        with np.load(root/'query-00.npz',allow_pickle=False) as z:
            self.assertEqual(str(z['request_id'].item()),'fixed-fixture-request')
            self.assertEqual(json.loads(str(z['inference_precision_json'].item())),PRECISION)

    def test_stale_callback_prevents_generation_not_just_result_publication(self):
        root,result,trace=self.run_fixture(stop_before_query=True)
        self.assertFalse(any(isinstance(x,tuple) and x[0]=='GENERATE' for x in trace))
        self.assertEqual(result['status'],'ENGINEERING_INCOMPLETE')
        self.assertFalse((root/'query-00.npz').exists())

    def test_legacy_probe_keeps_original_file_fields_and_no_new_callback_requirement(self):
        root,result,trace=self.run_fixture(new=False)
        self.assertEqual(len(result['queries']),17)
        self.assertFalse(any(isinstance(x,tuple) and x[0]=='QUERY_BEGIN' for x in trace))
        with np.load(root/'query-00.npz',allow_pickle=False) as z:
            self.assertEqual(set(z.files),{'physical_H40','normalized'})


if __name__=='__main__':unittest.main()
