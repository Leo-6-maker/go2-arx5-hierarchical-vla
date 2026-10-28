import ast,inspect,json,sys,torch,hashlib
from pathlib import Path
sys.path.insert(0,'/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage13/extracted/WBCTrackingShare')
import modules.trajectory as trajectory
import wbc_repair
native=trajectory.update_s_batch
wbc_repair.install()
namespace={}
exec('from modules.trajectory import update_s_batch',namespace)
assert namespace['update_s_batch'] is wbc_repair.update_s_batch
assert native is not namespace['update_s_batch']
p=Path('/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage13/extracted/WBCTrackingShare/go1_gym/envs/roboduet/wbc_env.py')
node=next(n for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='_update_trajectory_progress')
assert any(isinstance(n,ast.ImportFrom) and n.module=='modules.trajectory' and 'update_s_batch' in [v.name for v in n.names] for n in ast.walk(node))
old=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0281/code/wbc_repair.py')
oldnode=next(n for n in ast.parse(old.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='update_s_batch')
newnode=next(n for n in ast.parse(Path(wbc_repair.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='update_s_batch')
newnode.body=newnode.body[2:]
assert ast.dump(oldnode,include_attributes=False)==ast.dump(newnode,include_attributes=False)
print(json.dumps(dict(status='PASS_ACTUAL_LOCAL_IMPORT_BINDING_AND_UNCHANGED_RULE_AST',consumer_file=str(p),consumer_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),native_function=native.__module__,resolved_function=namespace['update_s_batch'].__module__,new_models=0,new_physics=0)))
