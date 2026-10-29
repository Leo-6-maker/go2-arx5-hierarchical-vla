"""Pinned LIBERO book/caddy geometry and its original site predicate.

No renderer, simulator, network, robot changes or data admission. The native
site goal is intentionally separate from project contact/release/hold checks.
"""
import hashlib
from pathlib import PurePosixPath
import xml.etree.ElementTree as ET
import numpy as np

COMMIT='8f1084e3132a39270c3a13ebe37270a43ece2a01'
XML_BLOBS=dict(black_book='73481d4f87d7480cc75add0fe7cce147e5ea2921',
    desk_caddy='bb59530d8629878f8974efaed4abf79d31743ee3')
SITE_OBJECT_BLOB='88eceb274d13b4774307d9d7f69f1674d52adda4'


def vector(text,length):
    a=np.fromstring(text,sep=' ')
    if a.shape!=(length,)or not np.isfinite(a).all():
        raise ValueError('FINITE_EXACT_UPSTREAM_GEOMETRY_VECTOR')
    return a


def quaternion_matrix(text):
    q=vector(text,4);norm=np.linalg.norm(q)
    if norm<.99 or norm>1.01:raise ValueError('UPSTREAM_WXYZ_UNIT_QUATERNION')
    w,x,y,z=q/norm
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def pinned_definition(name,body):
    if name not in XML_BLOBS or type(body)is not bytes or not 0<len(body)<=32768:
        raise ValueError('ONLY_THE_TWO_PINNED_MINIMUM_OBJECT_XMLS')
    blob=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
    if blob!=XML_BLOBS[name]or b'<!'in body:
        raise ValueError('EXACT_ORIGINAL_GIT_BLOB_WITHOUT_EXTERNAL_ENTITIES')
    root=ET.fromstring(body)
    if root.tag!='mujoco'or root.get('model')!=name:
        raise ValueError('EXACT_ORIGINAL_OBJECT_MODEL')
    obj=root.find("worldbody/body/body[@name='object']")
    if obj is None:raise ValueError('ORIGINAL_OBJECT_FRAME_REQUIRED')
    refs=[]
    for a in root.findall('asset/*'):
        if 'file'in a.attrib:
            p=PurePosixPath(a.attrib['file'])
            if p.is_absolute()or '..'in p.parts:raise ValueError('RELATIVE_MINIMUM_ASSET_CLOSURE')
            refs.append(str(p))
    collision=[dict(g.attrib)for g in obj.findall('geom')if g.get('group')=='0']
    if not collision or any(g.get('type')!='box'for g in collision):
        raise ValueError('ORIGINAL_PASSIVE_COLLISION_BOXES_ONLY')
    for g in collision:
        if np.any(vector(g['size'],3)<=0):raise ValueError('POSITIVE_ORIGINAL_COLLISION_EXTENTS')
        vector(g.get('pos','0 0 0'),3);quaternion_matrix(g.get('quat','1 0 0 0'))
    sites={}
    for site in obj.findall('site'):
        if site.get('name','').endswith('_contain_region'):
            sites[site.attrib['name']]=dict(position=vector(site.get('pos','0 0 0'),3).tolist(),
                rotation=quaternion_matrix(site.get('quat','1 0 0 0')).tolist(),
                half_size=vector(site.attrib['size'],3).tolist())
    if name=='black_book'and len(collision)!=1 or name=='desk_caddy'and not{
            'left_contain_region','right_contain_region'}<=set(sites):
        raise ValueError('ORIGINAL_SINGLE_BOOK_AND_DISTINCT_CADDY_GOAL_SITES')
    return dict(name=name,commit=COMMIT,git_blob_sha=blob,collision_geoms=collision,
        containment_sites=sites,asset_files=refs,body_origin_is_native_goal_position=True,
        has_executed_physics=False,training_admitted=False)


def native_site_contains(*,site_position,site_rotation,half_size,object_body_origin):
    """Match pinned SiteObject.in_box exactly, including its AABB approximation.

    SiteObjectState.check_contact always returns True for these sites. Thus
    LIBERO In(book, caddy_site) reduces to this position predicate, without
    checking the whole book, physical support, gripper release or withdrawal.
    """
    p,o,h=map(lambda x:np.asarray(x,dtype=float),(site_position,object_body_origin,half_size))
    r=np.asarray(site_rotation,dtype=float)
    if (p.shape!=(3,)or o.shape!=(3,)or h.shape!=(3,)or r.shape!=(3,3)
            or not all(np.isfinite(x).all()for x in(p,o,h,r))or np.any(h<=0)
            or not np.allclose(r.T@r,np.eye(3),atol=1e-8,rtol=0)
            or abs(np.linalg.det(r)-1)>1e-8):
        raise ValueError('ACTUAL_FINITE_BODY_ORIGIN_AND_SITE_TRANSFORM')
    extent=np.abs(r@h)
    lower=p-extent;upper=p+extent;lower[2]-=.01
    return bool(np.all(o>lower)and np.all(o<upper))
