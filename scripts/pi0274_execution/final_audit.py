"""Actual full new scope; archived predecessor results remain distinct from new work."""
import ast,importlib.util
from pathlib import Path
from quality_gate import BASE,ROOT,RAW,read,sha
from quality_worker import write
source=BASE/'PI0273/code/final_audit.py'
spec=importlib.util.spec_from_file_location('pi0273_closed_final_audit',source)
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)

def with_field_predictions(cpu):
    assert cpu['status']=='PASS_PI0274_ALL_NEW_S4_DECODE_AND_TRAJECTORIES';cpu=dict(cpu);cpu['predictions']=list(cpu['predictions'])
    for point in ('BROAD5120','EQUIL5120'):
        proof=read(RAW/'delivery'/f'FIELD_CPU_{point}.json');assert proof['status']=='PASS_ALL2496_SAVED_FIELD_EULER_DECODE_NOISE_METRIC_GATE' and len(proof['checks'])==2496
        cpu['predictions']+=proof['checks']
    return cpu

def final_write(path,value):
    if path.name=='FINAL_ARTIFACT_AUDIT.json':
        data=read(ROOT/'evidence/RESULT_ARCHIVE_data.json');assert data['status']=='PASS_PUBLIC_FIXED_REVISION_FILES_AND_ALL_MEMBERS' and data['revision']=='a786d91e501d241eb8af29d9666f6b8fd3c7a6be'
        assert data['tree_sha256']=='427e854ff3cd4e284fe20809738c790add6da8b8ac0e050aeea604c41b321520'
        admission=read(ROOT/'evidence/DATA_ADMISSION.json');assert admission['proposals']==1248 and admission['training_required']==672 and not admission['training_failures']
        exposure=read(ROOT/'evidence/EXPOSURE_AND_ACTUAL_SEEN.json');assert exposure['supplemental_unique_physical_poses']==672
        old=read(ROOT/'evidence/A0_ONSET_ANALYSIS.json');assert old['episodes']==128 and old['status']=='PASS_ALL128_SAVED_ONSETS_AND_OBSERVED_VISIBILITY'
        erratum=read(ROOT/'evidence/A0_VISIBILITY_SOURCE_BINDING.json');assert sha(erratum['actual_path'])==erratum['sha256'] and erratum['numerical_tables_unchanged']
        onset=read(ROOT/'evidence/EQUIL_ONSET_ANALYSIS.json');assert onset['episodes']==64
        for proof in (old,onset):
            for name,digest in proof['tables'].items():assert sha(RAW/'delivery'/name)==digest
        comparison=read(RAW/'delivery/INTERVENTION_COMPARISON.json');assert comparison['status']=='COMPLETE_FIXED_LOCAL_FIELD_AND_ORIGINAL_START_COMPARISON'
        for name,digest in comparison['tables'].items():assert sha(RAW/'delivery'/name)==digest
        fields=[]
        for point in ('BROAD5120','EQUIL5120'):
            proof=read(RAW/'delivery'/f'FIELD_{point}.json');assert proof['observed_rows']==proof['registered_rows']==2496 and len(proof['matrices'])==192
            assert sha(RAW/'delivery'/f'FIELD_{point}.csv')==proof['table_sha256']
            for r in proof['source_checks']:
                assert sha(r['record'])==r['sha256'] and sha(r['tensor'])==r['sha256_tensor']
            fields.append(dict(point=point,analysis_sha256=sha(RAW/'delivery'/f'FIELD_{point}.json'),CPU_sha256=sha(RAW/'delivery'/f'FIELD_CPU_{point}.json'),summary=proof['summary']))
        verdict=read(RAW/'delivery/PRIMARY_VERDICT.json');baseline=verdict['original_ROT2048_legacy_baseline']
        assert baseline['REJECT']==0 and baseline['median_xyz_rmse_m']==.015121491176463085 and baseline['median_so3_rmse_rad']==.046877545849429575
        inventory=read(ROOT/'evidence/QUALITY_OUTPUT_INVENTORY.json');assert inventory['new_teacher_attempts']==1248 and inventory['new_data_frame_pairs']==1248
        assert inventory['attempts']<=14336 and inventory['heads']<=57344 and inventory['starts']<=104 and inventory['total_new_frame_attempts']<=9216
        value.update(formal_updates=dict(EQUIL=5120),new_teacher_labels=1248,new_data_frame_pairs=1248,new_frame_attempts=inventory['total_new_frame_attempts'],
            data_archive=data,fields=fields,original_BROAD_control='REUSED_NO_RETRAIN_OR_OLD_PANEL_QUERY',old_A0_episodes=128,new_onset_episodes=64,
            comparison_sha256=sha(RAW/'delivery/INTERVENTION_COMPARISON.json'),execution_complete_not_quality_PASS=True)
    write(path,value)

def prepared():
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main');edits=[]
    class Registered(ast.NodeTransformer):
        def visit_Assign(self,n):
            if len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='cpu':
                self.generic_visit(n);n.value=ast.Call(func=ast.Name(id='with_field_predictions',ctx=ast.Load()),args=[n.value],keywords=[]);edits.append('all_field_and_nonfield_CPU')
            return self.generic_visit(n)
        def visit_Tuple(self,n):
            if all(isinstance(e,ast.Constant) for e in n.elts) and ast.literal_eval(n)==('BROAD5120','TARGET5120'):n.elts=[ast.Constant('EQUIL5120')];edits.append('new_video_only')
            return self.generic_visit(n)
        def visit_Constant(self,n):
            mapping={128:64,64:32,4:2,10272:5136,10528:5264,196608:98304,
                'PASS_PI0273_ALL_NEW_S4_DECODE_AND_TRAJECTORIES':'PASS_PI0274_ALL_NEW_S4_DECODE_AND_TRAJECTORIES',
                'PASS_PI0273_ALL_FIXED_RESULTS_TRAINERS_AND_BUDGETS':'PASS_PI0274_ALL_FIXED_RESULTS_TRAINERS_DATA_AND_BUDGETS'}
            return ast.copy_location(ast.Constant(mapping[n.value]),n) if n.value in mapping else n
    node=Registered().visit(node);assert edits==['all_field_and_nonfield_CPU','new_video_only']
    scope=vars(previous).copy();scope.update(ROOT=ROOT,RAW=RAW,write=final_write,with_field_predictions=with_field_predictions)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),scope);return scope['main']
if __name__=='__main__':prepared()()
