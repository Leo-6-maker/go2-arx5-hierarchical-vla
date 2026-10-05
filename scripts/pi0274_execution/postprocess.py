"""Finite saved-output closeout, automatically continuing through CPU/video/readback."""
import fcntl,os,subprocess
from quality_gate import BASE,ROOT,RAW,read,sha,admitted
from quality_worker import write

def main():
    admitted();assert read(ROOT/'evidence/QUALITY_QUERY_LANES_COMPLETE.json')['status']=='ALL_FIXED_NEW_QUALITY_QUERY_LANES_COMPLETE'
    seal=read(ROOT/'design/POSTPROCESS_EXECUTION_FILESET_R1.json')
    for r in seal['files']:assert sha(r['path'])==r['sha256']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',OPENBLAS_CORETYPE='Prescott',PYTHONPATH=str(ROOT/'code'))
    env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None)
    python=BASE/'PI0250/runtime/venv/bin/python'
    def run(script,*args):
        with (ROOT/'evidence'/('postprocess-'+script+'.log')).open('ab') as log:
            subprocess.run([str(python),'-u',str(ROOT/'code'/script),*args],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (ROOT/'runtime/postprocess.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run('query_inventory.py')
        for point in ('BROAD5120','EQUIL5120'):
            if not (RAW/'delivery'/f'FIELD_CPU_{point}.json').exists():run('field_cpu.py','--point',point)
            run('field_report.py','--point',point)
        run('report.py');run('transitions.py');run('onset_final.py');run('compare.py');run('video.py');run('research_return.py')
        run('archive_results.py','predictions');run('final_audit.py');run('archive_results.py','summary')
        write(ROOT/'evidence/POSTPROCESS_ARCHIVES_COMPLETE.json',dict(status='ALL_FIXED_CPU_VIDEO_AND_HF_READBACKS_COMPLETE',
            artifact_audit_sha256=sha(ROOT/'evidence/FINAL_ARTIFACT_AUDIT.json'),summary_receipt_sha256=sha(ROOT/'evidence/RESULT_ARCHIVE_summary.json'),
            report_sha256=sha(RAW/'delivery/FINAL_RESEARCH_RETURN.md'),terminal_GitHub_publication='STILL_REQUIRED',
            selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
        print('ALL_FIXED_CPU_VIDEO_AND_HF_READBACKS_COMPLETE',flush=True)
if __name__=='__main__':main()
