"""Literal new field plus EQUIL endpoint accounting, with preserved historical reuse."""
import ast,importlib.util
from quality_gate import BASE,ROOT,RAW,read,sha
from quality_worker import write
source=BASE/'PI0273/code/query_inventory.py'
spec=importlib.util.spec_from_file_location('pi0273_closed_query_inventory',source)
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
previous.ROOT=ROOT
field_loop=ast.parse("""
for point in ('BROAD5120','EQUIL5120'):
    assert read(ROOT/'evidence'/('FIELD_COMPLETE_'+point+'.json'))['status']=='PASS_ALL2496_REGISTERED_FIELD_OUTPUTS'
    for row in read(ROOT/'design/POSE_CARD.json')['rows']:
        for j in (0,1):
            path=RAW/'field'/point/row['id']/f'j{j}'/'record.json';rec=read(path)
            assert rec['point']==point and rec['row']==row and rec['j']==j and rec['pose_card_sha256']==sha(ROOT/'design/POSE_CARD.json')
            assert rec['status']=='VALID' and rec['tensor_sha256']==sha(rec['tensor'])
            prediction(rec['tensor'],point);records.append(dict(path=str(path),sha256=sha(path)))
""").body[0]

def final_write(path,value):
    if path.name=='QUALITY_OUTPUT_INVENTORY.json':
        teachers=previous.journal('teacher_attempts',1280);frames=previous.journal('data_frame_attempts',1248)
        assert len(teachers)==len(frames)==1248
        assert {r['proposal'] for r in teachers}=={r['proposal'] for r in frames}=={r['id'] for r in read(ROOT/'design/POSE_CARD.json')['rows']}
        assert value['frame_attempts']+len(frames)<=9216 and value['starts']<=104
        value.update(new_teacher_attempts=len(teachers),new_data_frame_pairs=len(frames),total_new_frame_attempts=value['frame_attempts']+len(frames),
            field_outputs=4992,new_endpoint_panels=1,BROAD_original_queries_repeated=0,new_WBC=0,new_dynamic_physics=0,new_hardware=0)
        value['journals'].update({n:sha(ROOT/'evidence'/f'{n}.jsonl') for n in ('teacher_attempts','data_frame_attempts')})
    write(path,value)

def prepared():
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main');edits=[]
    class Registered(ast.NodeTransformer):
        def visit_Assign(self,n):
            if len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='actual':
                edits.append('field_outputs');self.generic_visit(n)
                lanes=[e for e in ast.walk(n) if isinstance(e,ast.Tuple) and all(isinstance(x,ast.Constant) for x in e.elts) and ast.literal_eval(e)==('offline','newfit','late','virtual','component')]
                assert len(lanes)==1;lanes[0].elts.append(ast.Constant('field'));return [field_loop,n]
            return self.generic_visit(n)
        def visit_Assert(self,n):
            if any(isinstance(x,ast.Name) and x.id=='records' for x in ast.walk(n)):
                edits.append('record_denominator');return ast.copy_location(ast.parse('assert len(records)==968+192+592+64+32+4992').body[0],n)
            return self.generic_visit(n)
        def visit_Constant(self,n):
            mapping={18432:14336,73728:57344,208:104,15360:9216,192:96,17584:13784}
            return ast.copy_location(ast.Constant(mapping[n.value]),n) if n.value in mapping else n
    node=Registered().visit(node);assert edits==['record_denominator','field_outputs']
    scope=vars(previous).copy();scope.update(ROOT=ROOT,RAW=RAW,POINTS=('EQUIL5120',),write=final_write)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),scope);return scope['main']
if __name__=='__main__':prepared()()
