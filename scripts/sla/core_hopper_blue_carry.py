"""One TRAIN-only F2 blue-right expert change, routing the lifted cube around the red cube."""
import numpy as np

from go2arx5_vla.world_model.multitask_teacher_v6 import MultitaskTeacher, TRANSFER_CLEARANCE_M

CELL='CHV1-TRAIN-F2-PLACE-blue-right'
PILOT=CELL+'-02'
PRIOR=[CELL+'-00',CELL+'-01']


def planar(sample,geometry):
    centers=[]
    extents=[]
    for color in ('blue','red'):
        pose=np.asarray(sample['objects'][color]['pose'])
        centers.append(pose[:2,3])
        extents.append((abs(pose[:3,:3])@geometry['object_half_sizes'][color])[:2])
    return centers[0],centers[1],extents[0]+extents[1]


def remaining_clearance(start,turn_x,zone,obstacle,extent):
    """Conservative lower bounds along the two axis-aligned legs including tracking error at the turn."""
    first=(min(abs(start[0]-obstacle[0]),abs(turn_x-obstacle[0]))-extent[0]
           if (start[0]-obstacle[0])*(turn_x-obstacle[0])>0 else -extent[0])
    second=abs(zone[1]-obstacle[1])-extent[1]
    return float(first),float(second)


class BlueRightCarry(MultitaskTeacher):
    def __init__(self,initial,condition,geometry,instruction,model):
        if (condition['family'],condition['mode'],instruction)!=('F2','M_FREE','PLACE-blue-right'):
            raise ValueError('Carry repair is bound to its diagnosed training cell')
        super().__init__(initial,condition,geometry,instruction)
        self.turn_x=None
        self.final_leg=False

    def carry_goal(self,sample,zone):
        obj,obstacle,extent=planar(sample,self.geometry)
        if self.turn_x is None:
            self.turn_x=float(obj[0])
        side=np.sign(zone[1]-obstacle[1])
        if side*(obj[1]-obstacle[1]) >= extent[1]+TRANSFER_CLEARANCE_M:
            self.final_leg=True
        if not self.final_leg:
            waypoint=np.array([self.turn_x,zone[1]])
            goal,_=super().carry_goal(sample,waypoint)
            return goal,False
        return super().carry_goal(sample,zone)
