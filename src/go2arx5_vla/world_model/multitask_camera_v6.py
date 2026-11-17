"""Explicit two-object camera ABI; retain original body/wrist image recipe."""
from pathlib import Path
from .native_camera_transfer_v1 import FixedBodyRenderer
from .native_body_camera_capture_v1 import BodyBaseCameraSession
from .storage import digest


class TwoObjectBodyRenderer(FixedBodyRenderer):
    MODEL_ABI=(41,38,20,0,0,0)

    def __init__(self,model_path):
        super().__init__(model_path)
        self.model.body('task_blue_cube');self.model.geom('task_blue_cube_geom')

    def visible_masks(self,packet):
        import mujoco
        self.render(packet,audit_masks=False);renderer=self.renderer;m=self.model;result={}
        targets={'red':m.geom('task_cube_geom').id,'blue':m.geom('task_blue_cube_geom').id,
            'left_region':m.geom('task_goal_0').id,'right_region':m.geom('task_goal_1').id}
        try:
            renderer.enable_segmentation_rendering()
            for camera in('body_rgb','wrist_rgb'):
                renderer.update_scene(self.data,camera=camera);seg=renderer.render().copy()
                result[camera]={name:int(((seg[:,:,0]==identifier)&(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))).sum())for name,identifier in targets.items()}
        finally:renderer.disable_segmentation_rendering()
        return result


class TwoObjectCameraSession(BodyBaseCameraSession):
    def acquire(self,model_path,folder):
        from scripts.pi0294.task_native_worker import FrameRecorder
        if self.busy or self.closed or Path(model_path)!=self.model_path:raise PermissionError('EXCLUSIVE_TWO_OBJECT_CAMERA_SESSION_REQUIRED')
        if self.native is None:
            if digest(self.model_path)!=self.model_sha:raise ValueError('TWO_OBJECT_CAMERA_MODEL_BYTES_CHANGED')
            self.native=FrameRecorder(self.model_path,folder);self.native.reset_each_packet=True;self.render_copy=TwoObjectBodyRenderer(self.model_path)
        else:self.native.folder=folder
        self.busy=True;return self.native,self.render_copy
