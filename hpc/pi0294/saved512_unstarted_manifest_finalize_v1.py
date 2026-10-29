"""Complete one unstarted cold manifest identity through original onceIO.

No source capsule, science, data, GLOBAL/LOCAL, GPU request, model or optimizer
is edited/executed. Every old metadata byte is archived before its exact CAS.
"""
import copy,hashlib,json,os,resource,sys,time,traceback
from pathlib import Path
OLD='ARX5_MINIMAL_REAL_SAVED_PREFIX_REMAINDER_V1'
NEW='ARX5_MINIMAL_SAVED_E1_512_FRESH_E2_512_V1'
SOURCE='769c59ff7912ff16bdeb6f99bfde37f759a6fd37'
MANIFEST='0de0347bda459b1420208126315f6867fb60eab284ba276f19955b7aeb114c9d'
AUDIT='ec94fe6bbea487ab986af513828de709696e2f6107cfe4de06bfd72b2ce3b633'
RAW_RESULT='13e11cc12c1a27df0724bb7be8711a39cf7dae60cfbab886b65497e41065b935'
raw=lambda v:(json.dumps(v,indent=2,allow_nan=False)+'\n').encode()
sha=lambda b:hashlib.sha256(b).hexdigest()


def proposal(manifest,binding,lease,audit):
    m,b,l,a=map(copy.deepcopy,(manifest,binding,lease,audit))
    if not(m.get('platform_training_version')==OLD and m['code']['vla_commit']==SOURCE
        and m['run_uuid']=='arx5-minimal-saved-e1-inference-e2-fresh512-03'
        and m['execution']['profile']=='vanda-arx5-minimal-saved-e1-inference-fresh-e2-fixed512-cold03'
        and m['execution']['agent_liveness_required']is False
        and b['run_manifest_sha256']==l['manifest_sha256']==a['run_manifest_sha256']==sha(raw(m))
        and l['status']=='ACTIVE'and l['run_uuid']==m['run_uuid']
        and a['status']=='PASS_ACTUAL_MINIMAL_FIRST512_COLD_INPUTS_NO_MODEL'and a['source_sha']==SOURCE
        and a['actual_twelve_reader_rows']==7212 and a['new_GPU_PBS_model_query_update_physics']==0
        and not{'RUN_MANIFEST.json','OUTPUT_BINDING.private.json','control/COLD_INPUT_AUDIT.private.json'}&set(m['execution']['sealed_inputs'])):
        raise PermissionError('SAVED512_ONLY_EXACT_UNSTARTED_OLD_TOP_TYPE_AND_REAL_COLD_INPUT_PROOF')
    before=sha(raw(m));m['platform_training_version']=NEW;after=sha(raw(m))
    b['run_manifest_sha256']=after;l['manifest_sha256']=after
    a.update(run_manifest_sha256=after,original_actual_cold_data_audit_sha256=sha(raw(a)),
        cold_top_type_completion=dict(before=before,after=after,only_changed_manifest_key='platform_training_version',
            reused_actual_readers_source_inputs_unchanged=True,new_model_query_update_permission=False))
    return m,b,l,a


def main():
    P=json.loads(sys.stdin.read());out=Path(P['stage']);root=Path(P['root']);began=time.process_time();applied=[]
    def put(path,body):
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
        with os.fdopen(fd,'wb')as f:f.write(body);f.flush();os.fsync(f.fileno())
    try:
        resource.setrlimit(resource.RLIMIT_CPU,(59,60));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        assert os.getuid()==22182 and not os.environ.get('PBS_JOBID')and os.environ['CUDA_VISIBLE_DEVICES']==''
        oldstage=Path(P['cold_stage']);z=json.loads((oldstage/'OS_EXIT.json').read_bytes());result=(oldstage/'RESULT.json').read_bytes()
        assert sha(result)==RAW_RESULT and z['returncode']==0 and not z['timeout']and z['foreground_wait_completed']
        for name,a,b in(('LAUNCHED.json','wrapper_pid','wrapper_procstart'),('CHILD_STARTED.json','pid','procstart')):
            v=json.loads((oldstage/name).read_bytes());p=Path('/proc',str(v[a]),'stat');f=p.read_text().rsplit(')',1)[1].split()if p.exists()else None
            assert not(f and f[0]not in('Z','X')and int(f[19])==int(v[b]))
        assert not(root/'control/resident-owner').exists()and not(root/'control/RESIDENT_WRAPPER_PROCESS.private.json').exists()and list((root/'pbs').iterdir())==[]
        mp=root/'RUN_MANIFEST.json';bp=root/'OUTPUT_BINDING.private.json';ap=root/'control/COLD_INPUT_AUDIT.private.json'
        mb=mp.read_bytes();bb=bp.read_bytes();ab=ap.read_bytes();m,b,a=map(json.loads,(mb,bb,ab));lp=Path(b['lease']);lb=lp.read_bytes();l=json.loads(lb)
        assert sha(mb)==MANIFEST and sha(ab)==AUDIT and sha(lb)==P['before_lease_sha256']
        code=root/m['code']['capsule_relative'];seal=json.loads((root/'SOURCE_SEAL.json').read_bytes())
        assert seal['source_sha']==SOURCE and sha((root/'SOURCE_SEAL.json').read_bytes())==m['code']['source_seal_sha256']
        for name,pin in seal['files'].items():assert sha((code/name).read_bytes())==pin
        for name,pin in m['execution']['sealed_inputs'].items():assert sha((root/name).read_bytes())==pin
        gp=root.parent.parent/'HOPPER_RESOURCE_LEDGER.json';gb=gp.read_bytes();g=json.loads(gb)
        assert sha(gb)==P['GLOBAL_sha256']and g['used']['PBS_submissions']==39 and g['pending_updates']==g['unknown_effects']==[]and not any(x.get('request_id')==P['request_id']for x in g['PBS_submissions'])
        new=proposal(m,b,l,a);assert {k for k in m if m[k]!=new[0][k]}=={'platform_training_version'}
        sys.path[:0]=[str(code),str(code/'src')]
        from scripts.pi0294.generalization import minimal_execution_adapter as adapters,minimal_saved512_execution as execution
        assert adapters.manifest(new[0])is execution
        from scripts.pi0294.dec1.source_cpu_once_io import commit
        plan=[]
        for label,path,before,value in zip(('MANIFEST','BINDING','LEASE','AUDIT'),(mp,bp,lp,ap),(mb,bb,lb,ab),new):
            archive=out/('BEFORE_'+label+'.json');put(archive,before)
            plan.append(dict(label=label,path=str(path),before_sha256=sha(before),after_sha256=sha(raw(value)),archive=str(archive),key='ARX5-SAVED512-COLD03-TOP-TYPE-'+label))
        put(out/'EXACT_METADATA_PLAN.json',raw(plan))
        for item,value in zip(plan,new):
            path=Path(item['path']);key=item['key'];receipt=commit(path,expected_before_sha256=item['before_sha256'],proposed_bytes=raw(value),request_id=key,
                intent_path=path.with_name(key+'.INTENT.json'),receipt_path=path.with_name(key+'.RECEIPT.json'),lock_path=path.with_name(path.name+'.cold-type.owner.lock'))
            assert path.read_bytes()==raw(value);applied.append(dict(label=item['label'],receipt=receipt))
        assert gp.read_bytes()==gb and adapters.manifest(json.loads(mp.read_bytes()))is execution
        assert all(sha((root/name).read_bytes())==pin for name,pin in new[0]['execution']['sealed_inputs'].items())
        put(out/'RESULT.json',raw(dict(status='ACTUAL_UNSTARTED_COLD03_TOP_TYPE_AND_ALL_BOUND_METADATA_ONCE_COMPLETED',source_sha=SOURCE,
            original_full_cold_result_sha256=RAW_RESULT,original_manifest_sha256=MANIFEST,manifest_sha256=sha(mp.read_bytes()),
            audit_sha256=sha(ap.read_bytes()),binding_sha256=sha(bp.read_bytes()),lease_sha256=sha(lp.read_bytes()),
            new_manifest_top_version=NEW,source_and_all28_inputs_unchanged=True,original_1391_source_and_7212_reader_qualification_reused=True,
            old_bytes_archived=True,GLOBAL_sha256=sha(gb),new_GPU_PBS_model_queries_updates=0,applied=applied,CPU_actual_s=time.process_time()-began,observed_epoch=time.time())))
    except BaseException as error:
        put(out/'FAILED_OR_UNKNOWN.json',raw(dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()[-12000:],applied=applied,never_automatically_replay=True)));raise

if __name__=='__main__':main()
