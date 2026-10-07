"""CPU-only exact delivered projection and saved-pose comparison."""
import ast, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
ROOT = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0281')
OLD = ROOT.parent/'PI0280'
WBC = Path('/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage13/extracted/WBCTrackingShare')
sys.path.insert(0, str(WBC))
import modules.trajectory as trajectory
from wbc_repair import update_s_batch as repair


def batch(p, q, anchor):
    gamma, law = trajectory.to_canonical(torch.tensor(p-anchor, dtype=torch.float32),
        trajectory.quat_to_mat(torch.tensor(q, dtype=torch.float32)), .05)
    b = trajectory.TrajectoryBatch(1, 1536, 512, device='cpu')
    b.load(torch.tensor([0]), [gamma], [law])
    b.gamma_p += torch.tensor(anchor, dtype=torch.float32)
    return b


def replay(b, poses):
    sn = torch.zeros(1); sr = torch.zeros(1); rows = []
    for p, R, t in poses:
        p = torch.tensor([p], dtype=torch.float32); R = torch.tensor([R], dtype=torch.float32)
        oldn, oldr = sn.clone(), sr.clone()
        sn, dn = trajectory.update_s_batch(sn, p, R, b)
        sr, dr = repair(sr, p, R, b)
        rows.append(dict(time=t, native_previous=float(oldn[0]), native=float(sn[0]),
                         repair_previous=float(oldr[0]), repair=float(sr[0]), native_dlat=float(dn[0]), repair_dlat=float(dr[0])))
    return rows


def main():
    a = np.load(OLD/'wbc/vertical/reference.npz')
    b = batch(a['position'], a['quaternion'], a['anchor'])
    saved = json.loads((OLD/'wbc/vertical/rows.json').read_text())
    saved_rows = replay(b, [(r['native_TCP'], r['rotation'], r['time']) for r in saved])
    cases = {}
    for name in ('hold', 'monotonic', 'return'):
        t = np.arange(201)*.05; p = np.zeros((201, 3)); q = np.tile([0., 0., 0., 1.], (201, 1))
        if name == 'monotonic': p[:, 2] = .025*(1-np.cos(np.pi*np.minimum(t, 5)/5))
        if name == 'return': p[:, 2] = .025*(1-np.cos(2*np.pi*t/10))
        bc = batch(p, q, np.zeros(3))
        # Perfect raw reference poses isolate projection semantics from physics.
        rows = replay(bc, [(pos, np.eye(3), float(ti)) for pos, ti in zip(p, t)])
        cases[name] = dict(L=float(bc.L[0]), rows=rows)
    replacement = batch(a['position'], a['quaternion'], a['anchor'])
    reset = replay(replacement, [(a['position'][0], trajectory.quat_to_mat(torch.tensor(a['quaternion'][0])).numpy(), 0.)])
    bc = batch(np.c_[np.zeros(201),np.zeros(201),.025*(1-np.cos(np.pi*np.minimum(np.arange(201)*.05,5)/5))],
               np.tile([0.,0.,0.,1.],(201,1)),np.zeros(3))
    previous=torch.tensor([.02]); point=torch.tensor([[0.,0.,.025]])
    boundary=[]
    for label, point in [('single_segment_existing_progress', point), ('same_pose_dwell', point), ('backward_pose_no_regression',torch.tensor([[0.,0.,.005]]))]:
        n,d=repair(previous,point,torch.eye(3)[None],bc)
        assert n[0]>=previous[0]
        boundary.append(dict(case=label,previous=float(previous[0]),new=float(n[0]),dlat=float(d[0])))
        previous=n
    binding = json.loads((OLD/'evidence/WBC_BINDING.json').read_text())
    expected = next(r['sha256'] for r in binding['files'] if r['path'] == trajectory.__file__)
    actual = hashlib.sha256(Path(trajectory.__file__).read_bytes()).hexdigest()
    assert actual == expected
    # Execute the exact delivered packing function without importing a simulator.
    shim = ROOT.parent/'PI0252/runtime/PI0226_pytorch3d_shim'
    sys.path.insert(0, str(shim))
    import pytorch3d.transforms as pt3d
    math_path = WBC/'go1_gym/utils/math_utils.py'
    tree = ast.parse(math_path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'quat_xyzw_to_rot6d')
    namespace = dict(pt3d=pt3d)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(math_path), 'exec'), namespace)
    raw = Rotation.from_euler('xyz', [.31,-.47,.83])
    base = Rotation.from_euler('xyz', [-.2,.4,.6]); heading = Rotation.from_euler('z', .6)
    errors = {}
    cb=batch(np.zeros((201,3)),np.tile(raw.as_quat(),(201,1)),np.array([.2,-.1,.3]))
    _,preview_p,preview_q,_=cb.sample_preview(torch.zeros(1),.15,9)
    errors['raw_to_anchored_canonical_matrix']=float(np.max(abs(cb.R_at(torch.zeros(1)).numpy()[0]-raw.as_matrix())))
    errors['canonical_to_preview_quaternion']=float(np.max(abs(Rotation.from_quat(preview_q.numpy()[0]).as_matrix()-raw.as_matrix())))
    assert errors['raw_to_anchored_canonical_matrix']<1e-6 and errors['canonical_to_preview_quaternion']<1e-6
    for frame, rot in [('world', raw), ('heading', heading.inv()*raw), ('body', base.inv()*raw)]:
        qt = torch.tensor(rot.as_quat()[None], dtype=torch.float32)
        packed = namespace['quat_xyzw_to_rot6d'](qt).numpy()[0]
        errors[frame] = float(np.max(abs(packed-rot.as_matrix()[:, :2].reshape(6))))
        assert errors[frame] < 1e-6
    assert np.max(abs(trajectory.quat_to_mat(torch.tensor(raw.as_quat()[None], dtype=torch.float32)).numpy()[0]-raw.as_matrix())) < 1e-6
    evidence = dict(status='CPU_COMPLETE_NO_PHYSICS', runtime=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,device='cpu'), original_module=dict(file=trajectory.__file__, sha256=actual),
      orientation=dict(status='PASS_CPU_ALGEBRA_SOURCE_CHAIN_ONLY', packing='interleaved first two matrix columns; delivered shim unchanged',
        math_file=str(math_path),math_sha256=hashlib.sha256(math_path.read_bytes()).hexdigest(), nonidentity_max_errors=errors,
        anchoring='Exact native external anchor translates gamma_p only; gamma_quat remains world xyzw.',
        same_TCP='Native link6 quaternion; TCP is link6 plus fixed xyz offset [0.1424,0,0.0001057], identity rotation. URDF FK applies same offset.',
        actor_error='Native sign-corrected quaternion vector part = sin(theta/2) proxy, preserved.',
        limitation='Native actor observation and actual rigid-body orientation equality require the new instrumented physics fixture; no CPU claim of closed-loop orientation accuracy.'),
      saved_return=dict(L=float(b.L[0]), rows=saved_rows, position_max_FK_cache_error=max(r['FK_cache_error_m'] for r in saved),
         limitation='Saved rotation is URDF FK; old native rigid-body quaternion and actor observations were not saved. This is a positional replay with an FK orientation proxy, not exact old observation replay.'),
      ideal_cases=cases, boundary_cases=boundary, reference_replacement_reset=dict(old_progress=float(b.L[0]),native_lifecycle_new_progress=0., rows=reset,
        limitation='CPU reproduces canonical replacement at native zero progress; actual environment lifecycle reset is verified by source, not physics.'),
      new_model_queries=0, actor_forwards=0, physics_steps=0, training_updates=0)
    out=ROOT/'wbc/CPU_PROJECTION.json';out.parent.mkdir(parents=True, exist_ok=True);out.write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps(dict(status=evidence['status'], sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
      saved_last=saved_rows[-1], ideal_summaries={k:dict(L=v['L'], native_last=v['rows'][-1]['native'], repair_last=v['rows'][-1]['repair'],
       native_max_step=max(r['native']-r['native_previous'] for r in v['rows']), repair_max_step=max(r['repair']-r['repair_previous'] for r in v['rows'])) for k,v in cases.items()})))

if __name__ == '__main__': main()
