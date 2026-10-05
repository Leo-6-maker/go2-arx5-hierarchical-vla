"""Finite-scale world-left pose maps; no gradient or executed episode implied."""
import numpy as np
from scipy.spatial.transform import Rotation
D=np.array([.005,.005,.005,.05,.05,.05]);VARIANTS=['px','py','pz','rx','ry','rz']
def scaled(p,r,gp,gr):return np.r_[np.asarray(p)-gp,Rotation.from_matrix(np.asarray(r)@gr.T).as_rotvec()]/D
def matrix(values):
    assert set(values)=={'center'}|{v+s for v in VARIANTS for s in ('+','-')}
    b=values['center'];A=np.column_stack([(values[v+'+']-values[v+'-'])/2 for v in VARIANTS]);residual=np.column_stack([values[v+'+']+values[v+'-']-2*b for v in VARIANTS])
    return dict(b=b.tolist(),A=A.tolist(),symmetry_residual=residual.tolist(),spectral_radius=float(max(abs(np.linalg.eigvals(A)))),singular_values=np.linalg.svd(A,compute_uv=False).tolist(),position_from_rotation=A[:3,3:].tolist(),rotation_from_position=A[3:,:3].tolist(),finite_scale_not_analytic_Jacobian=True,not_WBC_stability_certificate=True)
def self_check():
    b=np.array([.2,-.1,.3,-.2,.1,.4]);A=np.diag([.1,.2,.3,.4,.5,.6]);values={'center':b}
    for i,v in enumerate(VARIANTS):values[v+'+']=b+A[:,i];values[v+'-']=b-A[:,i]
    got=matrix(values);assert np.allclose(got['A'],A,rtol=0,atol=1e-15) and np.allclose(got['b'],b) and abs(got['spectral_radius']-.6)<1e-15
    gr=Rotation.from_euler('xyz',[.3,-.2,.7]).as_matrix();gp=np.array([.4,.1,.5]);axis=np.array([.05,0.,0.]);z=scaled(gp,Rotation.from_rotvec(axis).as_matrix()@gr,gp,gr)
    assert np.allclose(z,[0,0,0,1,0,0],rtol=0,atol=1e-14)
    try:matrix({'center':b})
    except AssertionError:pass
    else:raise AssertionError('INCOMPLETE_MATRIX_ACCEPTED')
if __name__=='__main__':self_check();print('PASS_FINITE_SCALE_WORLD_LEFT_MAP_FIXTURES_NO_QUERY')
