"""Reuse the qualified final-weight downloader and tensor-scope audit; no model calls."""
import ast,importlib.util,json,os
from pathlib import Path
from run import ROOT,sha,write


def main():
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='0',TRANSFORMERS_OFFLINE='0')
    source=ROOT.parent/'PI0273/code/materialize.py'
    spec=importlib.util.spec_from_file_location('pi0276_materialize_helper',source)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    mapping={'PASS_PI0273_BOTH_ACTUAL_5120_MATCHED_TRAINING':'PASS_PI0276_ACTUAL8192_TASK_TRAINING',
        5120:8192,'5120':'8192','checkpoint-5120.archive_receipt.json':'checkpoint-8192.archive_receipt.json',
        'BROAD':'TASK','TARGET':'TASK','ROT':'TASK','contracts/ROT/source/dataset/statistics.json':'contracts/TASK/source/dataset/statistics.json',
        'PI0273_FIXED_MODEL_ONLY_R1':'PI0276_FIXED_MODEL_ONLY_V1',
        'PASS_PI0273_FIXED_MODE_LABELLED_MODEL_ONLY_VIEW':'PASS_PI0276_FIXED_FINAL_TASK_VIEW',
        '/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0273/models':'/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0276/models'}
    for n in ast.walk(node):
        if isinstance(n,ast.Constant) and n.value in mapping:n.value=mapping[n.value]
    assert not any(isinstance(n,ast.Constant) and isinstance(n.value,str) and 'contracts/ROT' in n.value for n in ast.walk(node))
    scope=vars(helper).copy();scope['ROOT']=ROOT
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),scope)
    scope['main']('TASK8192')
    donor=helper.read(ROOT.parent/'PI0270/model_bindings/AGG4096.json')
    for name,digest in donor['model_files_sha256'].items():assert sha(Path(donor['view'])/name)==digest
    mode=ROOT/'design/TASK_READOUT_R1.json';stats=ROOT/'contracts/TASK/source/dataset/statistics.json'
    base=dict(donor,schema='PI0276_NEW_CONTRACT_BASE_POLICY_V1',point='BASE0',effective_contract='TASK',
        statistics_sha256=sha(stats),readout_contract_sha256=sha(mode),historical_AGG_policy=False)
    write(ROOT/'model_bindings/BASE0.json',base)
    for step in (4096,8192):
        rec=helper.read(ROOT/f'evidence/TASK/checkpoint-{step}.full_readback.json')
        assert rec['status']=='PASS_FULL_FILE_READBACK' and rec['complete_trainer'] and rec['step']==step
    write(ROOT/'evidence/FINAL_POLICY_ADMISSION.json',dict(status='PASS_PI0276_FINAL_TASK_AND_BASE_POLICY_IDENTITIES',
        training_audit_sha256=sha(ROOT/'evidence/MAIN_TRAINING_AUDIT.json'),
        scope_sha256=sha(ROOT/'evidence/TASK8192/FROZEN_SCOPE.json'),
        complete_trainer_readbacks={str(step):sha(ROOT/f'evidence/TASK/checkpoint-{step}.full_readback.json') for step in (4096,8192)},
        bindings={point:sha(ROOT/f'model_bindings/{point}.json') for point in ('BASE0','TASK8192')},
        helper_source_sha256=sha(source),model_queries=0,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    from evaluate import freeze_noise
    freeze_noise()
    paths=list((ROOT/'code').glob('*.py'))+list((ROOT/'authority').glob('*'))
    paths+=[p for p in (ROOT/'design').glob('*.json') if p.name!='EVALUATION_FILESET.json']+list((ROOT/'model_bindings').glob('*.json'))
    paths+=[stats,ROOT/'evidence/FINAL_POLICY_ADMISSION.json',source]
    write(ROOT/'design/EVALUATION_FILESET.json',dict(files=[dict(path=str(p),sha256=sha(p)) for p in sorted(set(paths)) if p.is_file()],
        queries_before_seal=0,TEST_episodes_before_seal=0))


if __name__=='__main__':main()
