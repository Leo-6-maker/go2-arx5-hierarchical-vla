"""Frozen six-package geometry; only a planning clone/target ghost is changed."""
import itertools
import numpy as np

def rodrigues(axis,angle):
    a=np.asarray(axis,dtype=float);a=a/np.linalg.norm(a);x,y,z=a
    k=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+np.sin(angle)*k+(1-np.cos(angle))*(k@k)

def pad_binding(engine,state,closed=False):
    pads=[]
    for i,g in enumerate(engine.pad):
        p=state['tcp'][:3,:3].T@(engine.data.geom_xpos[g]-state['tcp'][:3,3])
        p[1]-=(1 if i==0 else -1)*state['finger_q'][i]
        if closed:p[1]+=(1 if i==0 else -1)*.002
        axes=state['tcp'][:3,:3].T@engine.data.geom_xmat[g].reshape(3,3)
        pads.append(dict(center=p,axes=axes,size=engine.model.geom_size[g].copy()))
    return pads

def packages(engine,state,task,card):
    obj=np.asarray(state['object_pose']);out=[];z=np.array([0.,0.,1.])
    if task=='T1':
        b=np.mean([p['center'] for p in pad_binding(engine,state)],axis=0)
        for axis in (0,1):
            c=obj[:3,axis].copy()
            if c@state['tcp'][:3,1]<0:c=-c
            for degrees in (0,30,-30):
                a=rodrigues(c,np.deg2rad(degrees))@(-z);r=np.column_stack([a,c,np.cross(a,c)])
                contact=np.eye(4);contact[:3,:3]=r;contact[:3,3]=obj[:3,3]-r@b
                approach=contact.copy();approach[:3,3]+=.08*z
                out.append(dict(index=len(out),task=task,axis=axis,degrees=degrees,approach=approach,contact=contact,closing=c))
    else:
        from .scene import push_geometry
        d=np.asarray(card['goal_xy'])-np.asarray(card['source_xy']);d=d/np.linalg.norm(d);world_d=np.r_[d,0.]
        r0=np.asarray(push_geometry(engine,d)['rotation']);pads=pad_binding(engine,state,closed=True)
        cube_support=float(np.abs(world_d@obj[:3,:3])@np.full(3,.02))
        for degrees in (0,30,-30,60,-60,180):
            r=rodrigues(world_d,np.deg2rad(degrees))@r0
            front=max(float(world_d@(r@p['center'])+np.abs(world_d@r@p['axes'])@p['size']) for p in pads)
            bottom=min(float((r@p['center'])[2]-np.abs((r@p['axes'])[2])@p['size']) for p in pads)
            contact=np.eye(4);contact[:3,:3]=r;contact[:2,3]=obj[:2,3]-d*(cube_support+front)+.001*d;contact[2,3]=.444-bottom
            approach=contact.copy();approach[:2,3]-=.026*d;approach[2,3]+=.08
            out.append(dict(index=len(out),task=task,degrees=degrees,approach=approach,contact=contact,direction=d,
                front_support=front,bottom_support=bottom))
    return out

def alignment(engine,state,package):
    """Actual pad enclosure in the selected frame; no changed task score."""
    from .component_teacher import pad_corners
    pads=pad_corners(engine);obj=np.asarray(state['object_pose']);r=np.asarray(package['contact'])[:3,:3]
    if package['task']=='T2':
        from .component_teacher import geometry
        return geometry(engine,state,'T2',package['direction'])
    corners=np.asarray(list(itertools.product((-.02,.02),repeat=3)))@obj[:3,:3].T+obj[:3,3]
    local_p=(pads-obj[:3,3])@r;local_o=(corners-obj[:3,3])@r
    lo=local_p.min(axis=1);hi=local_p.max(axis=1);omin=local_o.min(axis=0);omax=local_o.max(axis=0)
    negative,positive=np.argsort(local_p.mean(axis=1)[:,1])
    enclosed=bool(hi[negative,1]<=omin[1] and lo[positive,1]>=omax[1] and
        np.all(lo[:,[0,2]]<=omax[[0,2]]) and np.all(hi[:,[0,2]]>=omin[[0,2]]))
    c=r[:,1];horizontal=np.cross([0.,0.,1.],c);horizontal/=np.linalg.norm(horizontal)
    pc=pads@c;oc=corners@c;order=np.argsort(pc.mean(axis=1))
    encloses=bool(pc[order[0]].max()<=oc.min()-.004 and pc[order[1]].min()>=oc.max()+.004)
    ph=pads@horizontal;oh=corners@horizontal
    overlaps=bool(np.all(ph.min(axis=1)<=oh.max()) and np.all(ph.max(axis=1)>=oh.min()))
    above=bool(np.all(pads[:,:,2].min(axis=1)>=corners[:,2].max()+.004))
    slab=bool(np.all(pads[:,:,2].min(axis=1)<=corners[:,2].max()+.08))
    result=dict(precontact_geometry_ready=bool(encloses and overlaps and above and slab),
        actual_gap_m=float(pc[order[1]].min()-pc[order[0]].max()),encloses_selected_closing_axis=encloses,
        overlaps_transverse_axis=overlaps,pads_above_actual_object=above,within_precontact_slab=slab)
    result.update(actual_contact_alignment=enclosed,actual_pad_corners_world=pads.tolist(),
        actual_object_world=obj.tolist(),selected_candidate=int(package['index']),selected_frame=r.tolist())
    return result
