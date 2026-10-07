"""CPU-only native Panda/DROID frame comparison against genuine source states."""
import ast,hashlib,json,sys,urllib.request,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from bind_pristine import ROOT,digest
sys.path.insert(0,str(ROOT.parent/'PI0250/runtime/reference-317a/src'))
from go2arx5_vla.wbc.native.kinematics import rpy_to_matrix,axis_angle_to_matrix,homogeneous
from gr00t.data.state_action.droid_frame import DROID_EEF_ROTATION_CORRECT

def main():
    origin=ROOT/'assets/droid_semantics';origin.mkdir(exist_ok=True)
    pinned=origin/'SOURCE.json'
    if pinned.exists():identity=json.loads(pinned.read_text())
    else:
        with urllib.request.urlopen('https://api.github.com/repos/droid-dataset/droid/commits/main',timeout=30) as response:revision=json.load(response)['sha']
        identity=dict(repo='droid-dataset/droid',revision=revision,files=[])
        for name in ('droid/misc/transformations.py','droid/franka/robot.py'):
            url='https://raw.githubusercontent.com/droid-dataset/droid/'+revision+'/'+name
            with urllib.request.urlopen(url,timeout=30) as response:content=response.read()
            target=origin/Path(name).name;target.write_bytes(content);identity['files'].append(dict(url=url,path=str(target),sha256=digest(target)))
        pinned.write_text(json.dumps(identity,indent=2)+'\n')
    assert 'as_euler("xyz"' in (origin/'transformations.py').read_text()
    assert '1 - (self._gripper.get_state().width / self._max_gripper_width)' in (origin/'robot.py').read_text()
    source=ROOT.parent/'PI0265/code/pi0258_geometry.py'
    text=source.read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='full_fk')
    scope=dict(np=np,rpy_to_matrix=rpy_to_matrix,axis_angle_to_matrix=axis_angle_to_matrix,homogeneous=homogeneous)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope);fk=scope['full_fk']
    path=ROOT.parent/'PI0276/ManiSkill/mani_skill/assets/robots/panda/panda_v3.urdf';urdf=ET.parse(path).getroot()
    rows=[];transforms=[]
    for episode in (1,2):
        parquet=ROOT/'assets/droid_sample/data/chunk-000'/('episode_%06d.parquet'%episode)
        state=np.stack(pd.read_parquet(parquet)['observation.state'])
        for index,row in enumerate(state):
            poses=fk(urdf,dict(zip(['panda_joint%d'%i for i in range(1,8)],row[10:17])),np.eye(4))
            a,b=row[3:6],row[6:9];encoded=np.stack([a,b,np.cross(a,b)])@DROID_EEF_ROTATION_CORRECT.T
            # Invert the fixed official converter, then DROID's actual xyz Euler API.
            rotation=Rotation.from_euler('xyz',Rotation.from_matrix(encoded).as_euler('XYZ')).as_matrix()
            observed=homogeneous(rotation,row[:3]);T=np.linalg.inv(poses['panda_hand_tcp'])@observed
            transforms.append(T);rows.append(dict(episode=episode,frame=index,gripper_native=float(row[9])))
    transforms=np.asarray(transforms);fixed=transforms[0]
    xyz=np.linalg.norm(transforms[:,:3,3]-fixed[:3,3],axis=1)
    angles=Rotation.from_matrix(transforms[:,:3,:3]@fixed[:3,:3].T).magnitude()
    # Source-only geometry consistency, not model quality or fitted statistics.
    consistent=bool(xyz.max()<.001 and angles.max()<.001)
    output=dict(status='PANDA_DROID_STATIC_TCP_CONSISTENT' if consistent else 'BLOCKED_SOURCE_TCP_EQUIVALENCE_UNRESOLVED',
        official_frame_file_sha256=digest(ROOT/'official/gr00t/data/state_action/droid_frame.py'),URDF_sha256=digest(path),FK_source_sha256=digest(source),
        rows=len(rows),droid_source=identity,orientation_binding='DROID xyz Euler -> official XYZ converter + fixed correction; inverse in opposite order',
        first_sample_tool_transform=fixed.tolist(),max_tool_translation_variation_m=float(xyz.max()),max_tool_rotation_variation_rad=float(angles.max()),
        gripper_range=[min(x['gripper_native'] for x in rows),max(x['gripper_native'] for x in rows)],
        geometry_tolerance_m_rad=.001,model_queries=0,physics_steps=0,training_updates=0)
    old=ROOT/'evidence/SOURCE_TCP_CONTRACT.json'
    if old.exists():old.with_name('SOURCE_TCP_CONTRACT_prior_'+digest(old)[:12]+'.json').write_bytes(old.read_bytes())
    np.savez_compressed(ROOT/'evidence/PANDA_DROID_TCP_ARRAYS.npz',tool_transform=transforms,xyz_variation=xyz,rotation_variation=angles)
    (ROOT/'evidence/SOURCE_TCP_CONTRACT.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output))

if __name__=='__main__':main()
