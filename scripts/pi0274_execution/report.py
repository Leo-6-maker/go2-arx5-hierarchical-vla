"""Qualified CPU arithmetic with closed AGG/BROAD tables and one new endpoint."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import argparse,ast,importlib.util
from quality_gate import BASE,ROOT,RAW,read
from reuse import table
source=BASE/'PI0273/code/report.py'
spec=importlib.util.spec_from_file_location('pi0273_closed_report',source)
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
previous.ROOT=ROOT;previous.RAW=RAW;previous.POINTS=('AGG4096','BROAD5120','EQUIL5120');previous.table=table
helpers,paired,condition_means,METRICS=previous.helpers,previous.paired,previous.condition_means,previous.METRICS

def consumed_pairs(fitting):
    # Actual new packet identities, never the old slot's provenance-only source index.
    rows=[fitting['base'][v['source_index']] if v['stream']=='BASE' else v for v in fitting['streams']['EQUIL']]
    assert len(rows)==10240
    return {'EQUIL':{(r['condition_sha256'],r['labels_masks_sha256']) for r in rows}}

def prepared():
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main');edits=[]
    replacements={
        'offline':"offline=[r for r in table('PI0273','OFFLINE2904.csv') if r['point'] in ('AGG4096','BROAD5120')]",
        'virtual':"virtual=[r for r in table('PI0273','VIRTUAL192.csv') if r['point'] in ('AGG4096','BROAD5120')]",
        'components':"components=[r for r in table('PI0273','COMPONENT96.csv') if r['point'] in ('AGG4096','BROAD5120')]",
        'old_support':'old_support=None',
        'early':"early=[r for r in table('PI0273','EARLY_CONTEXT768.csv') if r['point'] in ('AGG4096','BROAD5120','S4FIT1024')]",
        'late':"late=[r for r in table('PI0273','LATE_CONTEXT2368.csv') if r['point'] in ('AGG4096','BROAD5120','S4FIT1024')]",
        'consumed':'consumed=consumed_pairs(fitting)'}
    class Registered(ast.NodeTransformer):
        def visit_Assign(self,n):
            if len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in replacements:
                key=n.targets[0].id;edits.append(key);return ast.copy_location(ast.parse(replacements[key]).body[0],n)
            return self.generic_visit(n)
        def visit_For(self,n):
            if isinstance(n.target,ast.Name) and n.target.id=='point' and isinstance(n.iter,ast.Subscript):
                assert isinstance(n.iter.value,ast.Name) and n.iter.value.id=='POINTS';n.iter=ast.parse("('EQUIL5120',)",mode='eval').body;edits.append('new_point_only')
            if isinstance(n.target,ast.Name) and n.target.id=='row' and isinstance(n.iter,ast.Name) and n.iter.id=='late':
                n.iter=ast.parse("[r for r in late if r['point']=='EQUIL5120']",mode='eval').body;edits.append('closed_late_metrics_reused')
            return self.generic_visit(n)
        def visit_Constant(self,n):
            mapping={'TARGET5120':'EQUIL5120','PI0273_SAVED_ARRAYS_NO_LEARNED_CALL_OR_OPTIMIZER':'PI0274_SAVED_ARRAYS_NO_LEARNED_CALL_OR_OPTIMIZER',
                'PASS_PI0273_ALL_NEW_S4_DECODE_AND_TRAJECTORIES':'PASS_PI0274_ALL_NEW_S4_DECODE_AND_TRAJECTORIES',
                'PASS_PI0273_ALL_FIXED_TABLES_CPU_REPLAY_AND_QUALIFICATIONS':'PASS_PI0274_ALL_FIXED_TABLES_CPU_REPLAY_AND_QUALIFICATIONS'}
            if n.value in mapping:return ast.copy_location(ast.Constant(mapping[n.value]),n)
            return n
    node=Registered().visit(node)
    assert all(edits.count(k)==1 for k in replacements) and edits.count('new_point_only')==3 and edits.count('closed_late_metrics_reused')==1
    scope=vars(previous).copy();scope.update(ROOT=ROOT,RAW=RAW,table=table,consumed_pairs=consumed_pairs)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),scope)
    return scope['main']

def self_check():
    prepared();previous.self_check()
    fitting={'base':[dict(condition_sha256='old',labels_masks_sha256='old-label')],
        'streams':{'EQUIL':[dict(stream='SUPPLEMENT',source_index=0,condition_sha256='new',labels_masks_sha256='new-label')]*10240}}
    assert consumed_pairs(fitting)=={'EQUIL':{('new','new-label')}}
    print('PASS_NEW_PACKET_SEEN_FLAGS_AND_CLOSED_ORIGINAL_CPU_BOUNDARIES',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-check',action='store_true');self_check() if p.parse_args().self_check else prepared()()
