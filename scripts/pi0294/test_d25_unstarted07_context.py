"""Actual unstarted06 capture; proposed07 source is an explicit CPU fixture."""
import base64,gzip,json,os
from pathlib import Path
from copy import deepcopy
import pytest
from scripts.pi0294.pv1 import mix_accounting as a,mix_core_accounting as c
from scripts.pi0294.pv1 import mix_core_d25_unstarted07 as n,mix_core_d25_admission_repair as repair
from scripts.pi0294.pv1 import mix_core_context as context,mix_core_d25_cold_source as cold,mix_core_d25_local_successor as paid

@pytest.fixture(scope='module')
def captured():
 d=os.environ.get('D25_NODE_LOCAL05_PRIVATE_CAPTURE_DIR')
 if not d:pytest.skip('actual unstarted06 captures required')
 root=Path(d);D=root/'MAINTENANCE30_CONTROL_IO';read=lambda k:json.loads((D/(k+'.private.json')).read_bytes())
 p=read('NODE_LOCAL06_RUNTIME_ASSEMBLY_FINAL_PREPARED');gb,lb=(gzip.decompress(base64.b64decode(p[k]))for k in('proposed_GLOBAL_gzip_base64','proposed_LOCAL_gzip_base64'));g,l=map(json.loads,(gb,lb))
 closed=read('NL06_COLD_CLOSED_ACTUAL');now=closed['observed_epoch']+1;source=read('D25_SOURCE06_COPY_PREPARED');seal=deepcopy(source['seal']);seal['source_sha']='f'*40
 seal['files']['scripts/pi0294/pv1/mix_core_context.py']=a.sha(Path(context.__file__).read_bytes());seal['files']['scripts/pi0294/pv1/mix_core_d25_unstarted07.py']=a.sha(Path(n.__file__).read_bytes())
 args=dict(authority_bytes=(D/'DIRECT_AUTHORITY.json').read_bytes(),closed_readback=closed,source_seal_bytes=a.raw(seal),frozen=source['frozen'],source_readback=dict(status='ACTUAL_COMPLETE_COLD_SOURCE_SHA_READBACK',source_sha=seal['source_sha'],source_seal_sha256=a.sha(a.raw(seal)),file_sha256=seal['files'],uid=2009,hostname='corelab-4080s',model_calls=0,physics_ticks=0,observed_epoch=now,CPU_fixture_only=True),now_epoch=now)
 return root,p,g,l,args,n.prepare(gb,lb,**args)

def test_actual06_request_selects_d25_at_the_model_entry(captured):
 _,_,g,l,_,_=captured
 assert context.d25_request_marker(l,n.RID)==l[paid.FIELD]
 assert context.d25_request_marker(l,repair.FAILED)==l[paid.FIELD]
 assert context.d25_request_marker(l,'00000000-0000-4000-8000-000000000000')is None
 assert g[c.FIELD][n.RID]['status']==c.STATES[0]

def test_one_unstarted_source_rebind_keeps_request_costs_and_real_assembler(captured):
 from scripts.pi0294.pv1 import mix_core_assembly as assembler,mix_core_delegation as delegation
 root,p,oldg,oldl,args,plan=captured;g,l=plan['assembly_global'],plan['assembly_LOCAL'];m=g[n.FIELD]
 owner=deepcopy(p['owner']);om=owner['manifest'];owner_root=str(Path(owner['root']).with_name('v5-cd-d25-eval-owner-07'));om.update(run_uuid=Path(owner_root).name,output_root=owner_root);om['code'].update(vla_commit=m['source_sha'],source_seal_sha256=m['source_seal_sha256']);om['execution']['sealed_inputs'][cold.BINDING]=n.digest(m)
 capture=base64.b64decode(p['assembly']['rows']['hopper']['files'][context.CAPTURE]);template=json.loads((root.parent/'science-next/wdr1-target-identity/v4a-core-eval-preparation-01/CORE_PAIRED_PEER_TEMPLATES_7cc75180.private.json').read_bytes());remote=deepcopy(owner['config']['remote_binding']);remote['remote_seal_sha256']=m['source_seal_sha256']
 packet=assembler.assemble(template,global_bytes=a.raw(g),local_bytes=a.raw(l),training_capture_bytes=capture,arm='D25',source_sha=m['source_sha'],source_seal_bytes=args['source_seal_bytes'],owner_root=owner_root,owner_manifest_bytes=a.raw(om),canonical_LOCAL_ledger=owner['config']['canonical_LOCAL_ledger'],remote_binding=remote,run_name=n.RUN,request_id=n.RID,nonce=m['submission_nonce'],GPU_uuid=oldg[c.FIELD][n.RID]['GPU_uuid'],now_epoch=args['now_epoch'],desktop_grant_bytes=base64.b64decode(owner['desktop_grant_base64']),global_accounting_mode=delegation.MODE)
 result=n.finalize(plan,packet);v,w=result['proposed_global'],result['proposed_LOCAL'];c.checked(v);c.paired(v,w)
 assert set(v[c.FIELD])==set(oldg[c.FIELD])and v[c.FIELD][repair.FAILED]==oldg[c.FIELD][repair.FAILED]
 assert v['used']==oldg['used']and v['reserved']==oldg['reserved']and v['CPU_s']==oldg['CPU_s']
 assert v[a.PACKAGE]==oldg[a.PACKAGE]and w[a.PACKAGE]==oldl[a.PACKAGE]
 assert v[c.FIELD][n.RID]['status']==c.STATES[0]and v[c.FIELD][n.RID]['native_handle']is None
 assert cold.effective_identity(v,v[paid.FIELD])==dict(source_sha='f'*40,run_uuid=n.RUN)
 assert context.d25_request_marker(w,n.RID)==w[paid.FIELD]
 assert result['new_native_requests']==0 and not result['native_admission']
 bad=deepcopy(packet);bad['record']['query_limit']-=1
 with pytest.raises(PermissionError):n.finalize(plan,bad)
