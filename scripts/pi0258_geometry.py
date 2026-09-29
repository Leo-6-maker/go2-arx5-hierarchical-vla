"""Source URDF geometry for isolated kinematic rendering and collision checks."""
from pathlib import Path
import ast
import itertools
import sys
import xml.etree.ElementTree as ET
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from go2arx5_vla.wbc.native.kinematics import rpy_to_matrix,axis_angle_to_matrix,homogeneous
def full_fk(urdf,positions,root):
    elements=list(urdf.findall('joint'));children={e.find('child').attrib['link'] for e in elements}
    parents={e.find('parent').attrib['link'] for e in elements};base=next(iter(parents-children))
    poses={base:root.copy()}
    while elements:
        count=len(elements)
        for e in elements[:]:
            parent=e.find('parent').attrib['link']
            if parent not in poses:continue
            o=e.find('origin');xyz=np.zeros(3) if o is None else np.fromstring(o.attrib.get('xyz','0 0 0'),sep=' ')
            rpy=np.zeros(3) if o is None else np.fromstring(o.attrib.get('rpy','0 0 0'),sep=' ')
            a=e.find('axis');axis=np.fromstring('1 0 0' if a is None else a.attrib['xyz'],sep=' ')
            t=homogeneous(rpy_to_matrix(rpy),xyz);q=positions.get(e.attrib['name'],0.)
            if e.attrib['type'] in ('revolute','continuous'):t=t@homogeneous(axis_angle_to_matrix(axis,q),np.zeros(3))
            elif e.attrib['type']=='prismatic':t=t@homogeneous(np.eye(3),axis*q)
            poses[e.find('child').attrib['link']]=poses[parent]@t;elements.remove(e)
        assert len(elements)<count,'disconnected URDF'
    return poses


def defaults(config):
    tree=ast.parse(Path(config).read_text())
    for n in ast.walk(tree):
        if isinstance(n,ast.Dict):
            for k,v in zip(n.keys,n.values):
                if isinstance(k,ast.Constant) and k.value=='init_state.default_joint_angles':return ast.literal_eval(v)
    raise ValueError('source default joints absent')


class Collision:
    def __init__(self,urdf_path):
        sys.path.insert(0,'/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0258/runtime/collision_deps')
        import fcl
        import trimesh
        self.fcl=fcl;self.urdf=ET.parse(urdf_path).getroot();self.parts=[]
        components={l.attrib['name']:l.attrib['name'] for l in self.urdf.findall('link')}
        def root(n):
            while components[n]!=n:n=components[n]
            return n
        joints=self.urdf.findall('joint')
        for j in joints:
            if j.attrib['type']=='fixed':components[root(j.find('child').attrib['link'])]=root(j.find('parent').attrib['link'])
        adjacent={frozenset((root(j.find('parent').attrib['link']),root(j.find('child').attrib['link']))) for j in joints}
        for link in self.urdf.findall('link'):
            for index,c in enumerate(link.findall('collision')):
                o=c.find('origin');xyz=np.fromstring(o.attrib.get('xyz','0 0 0') if o is not None else '0 0 0',sep=' ')
                rpy=np.fromstring(o.attrib.get('rpy','0 0 0') if o is not None else '0 0 0',sep=' ')
                g=list(c.find('geometry'))[0]
                if g.tag=='box':shape=fcl.Box(*np.fromstring(g.attrib['size'],sep=' '))
                elif g.tag=='cylinder':shape=fcl.Cylinder(float(g.attrib['radius']),float(g.attrib['length']))
                elif g.tag=='sphere':shape=fcl.Sphere(float(g.attrib['radius']))
                elif g.tag=='mesh':
                    mesh=trimesh.load(str((Path(urdf_path).parent/g.attrib['filename']).resolve()),force='mesh',process=False)
                    scale=np.fromstring(g.attrib.get('scale','1 1 1'),sep=' ')
                    shape=fcl.BVHModel();shape.beginModel(len(mesh.vertices),len(mesh.faces));shape.addSubModel(np.asarray(mesh.vertices)*scale,np.asarray(mesh.faces));shape.endModel()
                else:raise ValueError('unimplemented collision geometry '+g.tag)
                self.parts.append(dict(link=link.attrib['name'],index=index,local=homogeneous(rpy_to_matrix(rpy),xyz),shape=shape))
        self.pairs=[];self.exemptions=[]
        for a,b in itertools.combinations(range(len(self.parts)),2):
            x,y=self.parts[a]['link'],self.parts[b]['link'];components_pair=frozenset((root(x),root(y)))
            if root(x)==root(y) or components_pair in adjacent:self.exemptions.append([x,y,'SAME_FIXED_COMPONENT_OR_DIRECT_URDF_ADJACENCY'])
            else:self.pairs.append((a,b))

    def check(self,poses,obstacle=None,half_extents=None):
        f=self.fcl;objects=[]
        for part in self.parts:
            t=poses[part['link']]@part['local'];objects.append(f.CollisionObject(part['shape'],f.Transform(t[:3,:3],t[:3,3])))
        request=f.CollisionRequest(num_max_contacts=8,enable_contact=True);hits=[]
        def pair(a,b,label):
            result=f.CollisionResult();f.collide(a,b,request,result)
            if result.is_collision:
                depths=[float(c.penetration_depth) for c in result.contacts]
                depth=max(depths,default=0.)
                if depth>1e-6:hits.append(dict(pair=label,max_penetration_m=depth))
        for a,b in self.pairs:pair(objects[a],objects[b],[self.parts[a]['link'],self.parts[b]['link']])
        ground=f.CollisionObject(f.Box(20,20,1),f.Transform(np.array([0.,0.,-.5])))
        for part,obj in zip(self.parts,objects):pair(obj,ground,[part['link'],'ground'])
        if obstacle is not None:
            obj=f.CollisionObject(f.Box(*(2*np.asarray(half_extents))),f.Transform(np.asarray(obstacle,dtype=float)))
            for part,body in zip(self.parts,objects):pair(body,obj,[part['link'],'grey_obstacle'])
        return hits


def collision_check_fixture():
    import json
    from pi0255_motion3 import write
    r=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0258')
    reg=json.loads((r/'design/registry.json').read_text());collision=Collision(reg['urdf']['path']);f=collision.fcl
    a=f.CollisionObject(f.Box(1,1,1),f.Transform(np.array([0.,0.,0.])))
    for x,expected in [(2.,False),(.25,True)]:
        b=f.CollisionObject(f.Box(1,1,1),f.Transform(np.array([x,0.,0.])));result=f.CollisionResult()
        f.collide(a,b,f.CollisionRequest(enable_contact=True),result);assert result.is_collision==expected
    source=json.loads(Path('/home/liuyu-nfs/Projects/go2-arx5/PI0256/evidence/runtime_readback.json').read_text())
    positions=defaults(next(x['path'] for x in source['files'] if x['name']=='native_config'))
    poses=full_fk(collision.urdf,positions,homogeneous(np.eye(3),np.array(reg['root_pose'][:3])))
    hits=collision.check(poses)
    receipt=dict(schema='PI0258_COLLISION_PREFLIGHT_V1',fixture='PASS_SEPARATED_AND_PENETRATING_BOXES',
        geometry_count=len(collision.parts),nonadjacent_pairs=len(collision.pairs),exemptions=collision.exemptions,
        nominal_root_default_joints_contacts=hits,tolerance_m=1e-6,physics_steps=0,
        fcl_version='0.7.0.8',trimesh_version='4.8.3')
    write(r/'evidence/collision_fixture.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='exemptions'}))


if __name__=='__main__':collision_check_fixture()

