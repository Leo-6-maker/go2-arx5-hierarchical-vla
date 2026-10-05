"""One fixed new endpoint, registered field and inherited original panels."""
import fcntl,importlib.util,os,subprocess,sys
from pathlib import Path
from quality_gate import BASE,ROOT,RAW,read,sha,admitted
source=BASE/'PI0273/code/quality_evaluate.py';spec=importlib.util.spec_from_file_location('pi0273_closed_evaluation_driver',source);previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
complete=previous.complete
def native_environment():
    python,env=previous.native_environment();env['PYTHONPATH']=str(ROOT/'code')+':'+env['PYTHONPATH'];return python,env
def main():
    point='EQUIL5120';admitted(point)
    source_seal=read(ROOT/'design/QUALITY_FINAL_EXECUTION_FILESET.json');assert source_seal['status']=='FROZEN_PI0274_FINAL_DRIVER_BEFORE_EQUIL_QUERIES'
    for row in source_seal['files']:assert sha(row['path'])==row['sha256']
    modern=BASE/'PI0250/runtime/venv/bin/python';env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(ROOT/'code'),OPENBLAS_CORETYPE='Prescott');env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None)
    def run(python,script,args,environment,log):
        admitted(point)
        with (ROOT/'evidence'/log).open('ab') as stream:subprocess.run([str(python),'-u',str(ROOT/'code'/script),*args],env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)
    # Field owns the same lock internally and is completed before the endpoint lanes.
    with (ROOT/'runtime/quality-EQUIL-driver.lock').open('a') as driver:
        fcntl.flock(driver,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (ROOT/'evidence/FIELD_COMPLETE_EQUIL5120.json').exists():run(modern,'field.py',['--point',point],env,'quality-field-'+point+'.log')
        with (ROOT/'runtime/quality.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            card=ROOT/'design/RUNTIME_OFFLINE_CARD.json';offline=read(card)
            records=[RAW/'offline'/point/domain/r['uid']/'record.json' for domain,rows in [('fit',offline['fit_rows']),('factor',offline['factor_rows']),('legacy',offline['legacy_rows'])] for r in rows];assert len(records)==968
            if not complete(records,point,card):run(modern,'quality_offline.py',['--point',point],env,'quality-offline-'+point+'.log')
            assert complete(records,point,card)
            card=ROOT/'design/NEW_FIT192.json';records=[RAW/'newfit'/point/r['input_id']/'record.json' for r in read(card)['rows']];assert len(records)==192
            if not complete(records,point,card):run(modern,'newfit.py',['--point',point],env,'quality-newfit-'+point+'.log')
            assert complete(records,point,card)
            card=ROOT/'design/SUPPORT_CARD.json';records=[RAW/'late'/point/r['input_id']/f"j{n['j']}"/'record.json' for r in read(card)['late'] for n in r['noise']];assert len(records)==592
            if not complete(records,point,card):run(modern,'late_final.py',['--point',point],env,'quality-late-'+point+'.log')
            assert complete(records,point,card)
            native,native_env=native_environment();card=ROOT/'design/RUNTIME_EVALUATION_CARD.json'
            families=[f for f in read(BASE/'PI0267/design/SCENE_CARD.json')['families'] if f['split']=='NEW_SIM_EVAL'];assert len(families)==8
            for f in families:
                fid=f['family_id'];records=[RAW/'virtual'/point/f'{fid}-{task}-c{c}-{backend}'/'result.json' for task in ('M1','M3') for c in (0,1) for backend in ('E0','E1')]
                if not complete(records,point,card):run(native,'quality_virtual.py',['--point',point,'--family',fid],native_env,'quality-main-'+point+'-'+fid+'.log')
                assert complete(records,point,card)
            component=read(ROOT/'design/RUNTIME_COMPONENT_CARD.json')
            for fid in dict.fromkeys(r['family'] for r in component['rows']):
                records=[RAW/'component'/point/r['case']/'result.json' for r in component['rows'] if r['family']==fid];assert len(records)==16
                if not complete(records,point,card,True):run(native,'quality_components.py',['--point',point,'--family',fid],native_env,'quality-component-'+point+'-'+fid+'.log')
                assert complete(records,point,card,True)
        from quality_worker import write
        write(ROOT/'evidence/QUALITY_QUERY_LANES_COMPLETE.json',dict(status='ALL_FIXED_NEW_QUALITY_QUERY_LANES_COMPLETE',new_offline_rows=968,new_early_rows=192,new_late_rows=592,new_main=64,new_component=32,
            BROAD_original_outputs_reused=True,EQUIL_final_only=True,CPU_video_archive_and_verdict_still_required=True,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
        print('ALL_FIXED_EQUIL_ENDPOINT_QUERY_LANES_COMPLETE',flush=True)
if __name__=='__main__':main()
