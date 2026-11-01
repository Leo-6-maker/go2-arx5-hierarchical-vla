"""Six prospective TRAIN variants, disjoint from old position-only identities.

These are candidate cards, not static qualification or successful episodes.
The book asset, target, robot home, controller,32s task and score stay fixed.
"""
import math
import numpy as np

VERSION='BOOK_TRAIN_YAW3_OR_GRASP2MM_V1'
VARIANTS=(('YAW_MINUS3',-3.,0.,0.),('YAW_PLUS3',3.,0.,0.),
          ('GRASP_X_MINUS2MM',0.,-.002,0.),('GRASP_X_PLUS2MM',0.,.002,0.),
          ('GRASP_Y_MINUS2MM',0.,0.,-.002),('GRASP_Y_PLUS2MM',0.,0.,.002))


def geometry_card(index):
    from .libero_book_teacher import route_card,RELEASE_REVISION
    if type(index)is not int or not 1<=index<=len(VARIANTS):raise ValueError('DECLARED_NEW_BOOK_TRAIN_GEOMETRY')
    label,yaw,dx,dy=VARIANTS[index-1];card=route_card(RELEASE_REVISION)
    angle=math.radians(yaw);rotation=np.array([[math.cos(angle),-math.sin(angle),0.],
        [math.sin(angle),math.cos(angle),0.],[0.,0.,1.]])
    native=np.asarray(card['initial_book_native_world']).copy();native[:3,:3]=rotation@native[:3,:3]
    card['initial_book_native_world']=native.tolist()
    grasp=np.asarray(card['book']['object_to_tcp_grasp']).copy();grasp[:2,3]+=[dx,dy]
    card['book']['object_to_tcp_grasp']=grasp.tolist()
    card.update(id='BOOK_GEOMETRY_TRAIN_'+label,study_split='TRAIN',book_geometry_index=index,
        source_geometry_set=VERSION,initial_world_yaw_delta_deg=yaw,grasp_translation_object_m=[dx,dy,0.],
        old_position_population_or_confirmation=False,candidate_static_or_native_qualification=False)
    return card


def conditions():
    return [dict(id=geometry_card(i)['id'],geometry_index=i,split='TRAIN',xy_offset_m=[0.,0.])for i in range(1,7)]
