"""Registered data replacement on the unchanged qualified PI0273 Trainer."""
import argparse,ast,hashlib,importlib.util,json,sys
from functools import lru_cache
from pathlib import Path

ROOT=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0274');CONTROL=ROOT.with_name('PI0273')
spec=importlib.util.spec_from_file_location('pi0273_qualified_train',CONTROL/'code/train.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
old,paired,np,torch,read,sha,read_lines=previous.old,previous.paired,previous.np,previous.torch,previous.read,previous.sha,previous.read_lines
RELEASE='495b17078cb57fdbd47d8257d33202c57beb058e'
STATS_SHA='0ea96eaf9ba6f2bfb5c16681542723e9ee0e82281dd07c4e33ed9189799580e9'
def seed_for(unit):return int.from_bytes(hashlib.sha256(f'PI0273:TRAIN:27303:{unit}'.encode()).digest()[:8],'big')%2**31
previous.ROOT=ROOT;previous.seed_for=seed_for

def comparable(ok,key,unit):
    if ok:return
    path=ROOT/'evidence/CONTROL_NOT_COMPARABLE.json'
    if not path.exists():path.write_text(json.dumps(dict(status='CONTROL_NOT_COMPARABLE',key=key,unit=unit))+'\n')
    raise RuntimeError(('CONTROL_NOT_COMPARABLE',key,unit))

class Pairs(previous.Pairs):
    def __init__(self,processor,output,arm,resume):
        super().__init__(processor,output,arm,resume)
        assert arm=='EQUIL'
        p=CONTROL/'BROAD/main/training/sample_audit.jsonl'
        assert sha(p)=='c939cfdd80d0ccf62e77fcb3e07b9d0e492eb57f8bd28bfb828554fa8b44d624'
        self.closed=read_lines(p);assert len(self.closed)==10240
    @lru_cache(maxsize=672)
    def supplement(self,pose_id,digest):
        folder=ROOT/'source/data';p=folder/pose_id/'sample.npz'
        assert not p.is_symlink() and p.resolve().is_relative_to(folder.resolve()) and sha(p)==digest
        with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in ('body_rgb','wrist_rgb','state_v1','raw_action')}
    def encode_visit(self,visit):
        if visit['stream']=='BASE':return self.encode(visit['source_index'])
        sample=self.supplement(visit['pose_id'],visit['sample_sha256']);packet={k:sample[k] for k in ('body_rgb','wrist_rgb','state_v1')};packet['language']=visit['language']
        value=old.batch(self.processor,packet,sample['raw_action']);condition={k:v for k,v in value['inputs'].items() if k not in ('action','action_mask')}
        assert old.identity(condition)==visit['condition_sha256'] and old.identity({k:value['inputs'][k] for k in ('action','action_mask')})==visit['labels_masks_sha256']
        return value
    def record(self,visit):
        if visit['stream']=='BASE':return super().record(visit)
        source=self.card['base'][visit['source_index']];record=dict(visit)
        record.update(arm='EQUIL',bucket='GOAL_LOCAL',source_slot='SUPPLEMENT_GOAL_LOCAL',source_sample_sha256=visit['sample_sha256'],source_unit=source['unit'],source_micro=source['micro'],old_source_index_provenance_only=True)
        return record
    def __iter__(self):
        for visit in self.card['streams']['EQUIL'][self.cursor:]:
            record=self.record(visit);control=self.closed[2*(visit['unit']-1)+visit['micro']]
            for k in ('unit','micro','seed','stream','family','task','command','language'):comparable(record[k]==control[k],'CONTEXT_'+k,visit['unit'])
            if visit['stream']=='BASE':
                for k in ('source_index','condition_sha256','labels_masks_sha256','nonlanguage_sha256','source_sample_sha256'):comparable(record[k]==control[k],'BASE_'+k,visit['unit'])
            assert visit['seed']==seed_for(visit['unit'])
            value=self.encode_visit(visit);value['_pi0264_meta']=dict(unit=visit['unit'],micro=visit['micro'],command=record['command'],condition_sha256=record['condition_sha256'])
            with (self.output/'sample_audit.jsonl').open('a') as f:f.write(json.dumps(record,sort_keys=True)+'\n')
            yield value

def adapted():
    """Only roots, arm allowlists, metadata and literal budget caps may differ."""
    source=(CONTROL/'code/train.py').read_text();tree=ast.parse(source)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('hooks','run','equal','compare')]
    edits=[]
    class Binding(ast.NodeTransformer):
        def visit_Constant(self,node):
            replacements={196608:98304,10528:5264,'TARGET':'EQUIL',
                'PASS_PI0273_FULL_BASE_ENCODING_AND_BOTH_10240_SCHEDULES':'PASS_PI0274_NEW_INPUTS_AND_REUSED8192_BASE',
                'PASS_PI0273_ACTUAL_FROZEN_HELPERS_AND_HOOKS':'PASS_PI0274_ACTUAL_FROZEN_HELPERS_AND_HOOKS',
                'PASS_PI0273_REAL_EXACT_RESUME_CANARY':'PASS_PI0274_REAL_EXACT_RESUME_CANARY',
                'PI0273_READOUT_CONTRACT.json':'PI0274_READOUT_CONTRACT.json','pi0273_provenance':'pi0274_provenance',
                'pi0273_provenance/CPU_ENCODER_KERNEL.json':'pi0274_provenance/CPU_ENCODER_KERNEL.json',
                'EXECUTION_FILESET.json':'EXECUTION_FILESET_R1.json','RUNTIME_BINDING_':'RUNTIME_BINDING_R1_',
                'PI0273_SEGMENT_COMPLETE':'PI0274_SEGMENT_COMPLETE'}
            if node.value in replacements:edits.append(str(node.value));return ast.copy_location(ast.Constant(replacements[node.value]),node)
            return node
        def visit_For(self,node):
            if isinstance(node.iter,ast.Tuple) and all(isinstance(e,ast.Constant) for e in node.iter.elts) and ast.literal_eval(node.iter)==('BROAD','TARGET'):
                node.iter=ast.Tuple(elts=[ast.Constant('EQUIL')],ctx=ast.Load());edits.append('single_canary')
            return self.generic_visit(node)
        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='other':
                edits.append('closed_BROAD_random_packets');return ast.copy_location(ast.parse("other=CONTROL/'BROAD/main/training/step_rng.jsonl'").body[0],node)
            return self.generic_visit(node)
        def visit_Assert(self,node):
            if isinstance(node.msg,ast.Tuple) and node.msg.elts and isinstance(node.msg.elts[0],ast.Constant) and node.msg.elts[0].value=='MATCHED_PACKET':
                edits.append('explicit_comparability_receipt');return ast.copy_location(ast.parse("comparable(row[k]==expected[k],'ACTUAL_RANDOM_PACKET_'+k,row['unit'])").body[0],node)
            return self.generic_visit(node)
    nodes=[Binding().visit(n) for n in nodes];assert len(nodes)==4 and edits.count('single_canary')==1 and edits.count('closed_BROAD_random_packets')==1
    scope=vars(previous).copy();scope.update(ROOT=ROOT,CONTROL=CONTROL,RELEASE=RELEASE,STATS_SHA=STATS_SHA,Pairs=Pairs,seed_for=seed_for,comparable=comparable)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(CONTROL/'code/train.py'),'exec'),scope)
    return scope,edits
runtime,ADAPTATIONS=adapted();hooks,run,compare=runtime['hooks'],runtime['run'],runtime['compare']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('EQUIL',),required=True);p.add_argument('--phase',choices=('continuous','split','main'));p.add_argument('--limit',type=int);p.add_argument('--resume',type=int,default=0);p.add_argument('--compare',action='store_true');a=p.parse_args()
    if a.compare:compare(a.arm)
    else:assert a.phase and 0<=a.resume<a.limit<=5120;run(a)
