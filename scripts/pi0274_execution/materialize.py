"""New final model-only view via the closed fixed-revision scope checker."""
import argparse,ast,types
from pathlib import Path
ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0274');OLD=ROOT.parent/'PI0273';source=OLD/'code/materialize.py'
tree=ast.parse(source.read_text());changes=[]
class Bind(ast.NodeTransformer):
    def visit_Constant(self,n):
        mapping={'PI0273':'PI0274','TARGET':'EQUIL','/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0273/models':'/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0274/models',
                 'PASS_PI0273_BOTH_ACTUAL_5120_MATCHED_TRAINING':'PASS_PI0274_ACTUAL5120_MATCHED_BROAD_TRAINING','PI0273_FIXED_MODEL_ONLY_R1':'PI0274_FIXED_MODEL_ONLY_R1','PASS_PI0273_FIXED_MODE_LABELLED_MODEL_ONLY_VIEW':'PASS_PI0274_FIXED_MODE_LABELLED_MODEL_ONLY_VIEW'}
        if n.value in mapping:changes.append(str(n.value));return ast.copy_location(ast.Constant(mapping[n.value]),n)
        return n
tree=Bind().visit(tree);assert 'PASS_PI0273_BOTH_ACTUAL_5120_MATCHED_TRAINING' in changes
runtime=types.ModuleType('pi0274_reused_materialization');runtime.__file__=str(source);exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),vars(runtime))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',choices=('EQUIL5120',),required=True);runtime.main(p.parse_args().point)
