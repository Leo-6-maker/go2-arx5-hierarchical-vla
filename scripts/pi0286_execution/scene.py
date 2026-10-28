"""Same scene for both instructions. Native objects and dynamics are inherited."""
import numpy as np
import sapien,torch
from scipy.spatial.transform import Rotation
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


def footprint(xy,yaw):
    corners=np.asarray([[-.02,-.02],[-.02,.02],[.02,-.02],[.02,.02]])
    rotation=np.asarray([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    return corners@rotation.T+xy


def layout(seed):
    rng=np.random.Generator(np.random.PCG64(seed))
    for proposal in range(1000):
        zone=rng.uniform(-.12,.12,2);red=rng.uniform(-.17,.17,2);green=rng.uniform(-.17,.17,2)
        yaw=rng.uniform(-np.pi/4,np.pi/4,2)
        redcorners=footprint(red,yaw[0]);greencorners=footprint(green,yaw[1])
        if max(abs(redcorners).max(),abs(greencorners).max())>.2:continue
        if np.linalg.norm(red-zone)<.15 or np.linalg.norm(red-green)<.12:continue
        # Conservative separating axis: whole green footprint outside the zone +2cm.
        if not any(greencorners[:,axis].min()>zone[axis]+.10 or greencorners[:,axis].max()<zone[axis]-.10 for axis in (0,1)):continue
        return dict(layout=seed,proposal=proposal,red_xy=red.tolist(),green_xy=green.tolist(),zone_xy=zone.tolist(),red_yaw=float(yaw[0]),green_yaw=float(yaw[1]))
    raise ValueError(('GEOMETRY_PROPOSAL_EXHAUSTED',seed))


@register_env('PI0285PickPlaceStack-v1',max_episode_steps=660)
class CommonScene(StackCubeEnv):
    def __init__(self,*args,layout_id,**kwargs):
        self.layout_spec=layout(layout_id)
        super().__init__(*args,**kwargs)

    def _load_scene(self,options):
        super()._load_scene(options)
        builder=self.scene.create_actor_builder()
        material=sapien.render.RenderMaterial(base_color=[0.02,0.08,1.,1.])
        builder.add_box_visual(half_size=[.08,.08,.00002],material=material)
        builder.initial_pose=sapien.Pose(p=[*self.layout_spec['zone_xy'],.00003])
        self.blue_zone=builder.build_static(name='PI0285-blue-visual-only')

    def _initialize_episode(self,env_idx,options):
        # Preserve the original robot initialization distribution and native table.
        super()._initialize_episode(env_idx,options)
        v=self.layout_spec
        for actor,key in ((self.cubeA,'red'),(self.cubeB,'green')):
            yaw=v[key+'_yaw'];q=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
            actor.set_pose(Pose.create_from_pq(p=torch.tensor([[*v[key+'_xy'],.02]],device=self.device),q=torch.tensor([q],device=self.device)))


def self_check():
    for seed in list(range(285000,285008))+list(range(285100,285164))+list(range(285200,285216)):
        a=layout(seed);assert a==layout(seed)
        assert np.linalg.norm(np.asarray(a['red_xy'])-a['zone_xy'])>=.15
        assert np.linalg.norm(np.asarray(a['red_xy'])-a['green_xy'])>=.12
        for key in ('red','green'):assert abs(footprint(a[key+'_xy'],a[key+'_yaw'])).max()<=.2
    print('PASS_GEOMETRY_ONLY_NO_RENDER_OR_PHYSICS')

if __name__=='__main__':self_check()
