"""Closed evaluation code, with only registered roots, points and budgets adapted."""
import ast,hashlib,sys,types
from pathlib import Path
BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0274';OLD=BASE/'PI0273'
NAMES=('quality_worker','quality_virtual','quality_components','quality_offline','newfit','late')
def load(name):
    assert name in NAMES
    path=OLD/'code'/(name+'.py');tree=ast.parse(path.read_text());edits=[]
    class Registered(ast.NodeTransformer):
        def visit_Constant(self,node):
            mapping={'PI0273':'PI0274','/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0273':'/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0274','TARGET5120':'EQUIL5120',18432:14336,73728:57344,208:104,15360:9216}
            if node.value in mapping:edits.append(str(node.value));return ast.copy_location(ast.Constant(mapping[node.value]),node)
            return node
        def visit_Assert(self,node):
            # New field noise is explicitly frozen; all old banks remain unchanged.
            if name=='quality_worker' and isinstance(node.test,ast.Call) and isinstance(node.test.func,ast.Name) and node.test.func.id=='any':
                expression=ast.unparse(node.test) if hasattr(ast,'unparse') else ''
                if 'noise_path' in expression:
                    tuples=[n for n in ast.walk(node.test) if isinstance(n,ast.Tuple)]
                    assert len(tuples)==1;tuples[0].elts.append(ast.parse("raw_root/'noise-field'",mode='eval').body);edits.append('registered_field_noise_root')
            return self.generic_visit(node)
        def visit_Tuple(self,node):
            values=[n.value for n in node.elts if isinstance(n,ast.Constant)]
            if name=='quality_worker' and values==['RUNTIME_EVALUATION_CARD.json','RUNTIME_OFFLINE_CARD.json','RUNTIME_COMPONENT_CARD.json','NEWFIT_QUERY_CARD.json','LATE_QUERY_CARD.json']:
                node.elts.append(ast.Constant('FIELD_QUERY_CARD.json'));edits.append('registered_field_noise_card')
            return self.generic_visit(node)
        def visit_Assign(self,node):
            if name=='quality_worker' and any(isinstance(t,ast.Name) and t.id=='output' for t in node.targets):
                self.generic_visit(node);edits.append('closed_BROAD_field_only')
                guard=ast.parse("if point=='BROAD5120':\n assert output.resolve().is_relative_to(Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0274/field'))").body[0]
                return [node,ast.copy_location(guard,node)]
            if name=='quality_virtual' and any(isinstance(t,ast.Name) and t.id=='factors' for t in node.targets) and isinstance(node.value,ast.Constant) and node.value.value==0:
                edits.append('include_data_render_journal');return ast.copy_location(ast.parse('factors=count_data_frames()').body[0],node)
            return self.generic_visit(node)
    tree=Registered().visit(tree)
    if name=='quality_worker':assert edits.count('registered_field_noise_root')==edits.count('registered_field_noise_card')==edits.count('closed_BROAD_field_only')==1
    if name=='quality_virtual':assert edits.count('include_data_render_journal')==1
    module=types.ModuleType('pi0274_reused_'+name);module.__file__=str(path)
    module.__dict__['count_data_frames']=lambda:sum(1 for _ in (ROOT/'evidence/data_frame_attempts.jsonl').open())
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),module.__dict__)
    module._source_binding=dict(name=name,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),registered_adaptations=edits)
    return module
