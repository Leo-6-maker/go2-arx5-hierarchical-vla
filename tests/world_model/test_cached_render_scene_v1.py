"""Real pinned MuJoCo visualization, without GL or physics integration."""
import numpy as np


def test_cached_capture_restores_fresh_plane_placement_without_context_rebuild():
    import mujoco
    from go2arx5_vla.world_model.native_collection_performance_v1 import reset_render_scene
    model=mujoco.MjModel.from_xml_string('''<mujoco>
      <asset><material name="floor" texrepeat="5 5"/></asset><worldbody>
      <geom type="plane" size="0 0 .1" material="floor"/>
      <camera name="external" pos=".8 .4 1"/>
      </worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FIXED;camera.fixedcamid=0
    option=mujoco.MjvOption()
    class Renderer:
        def __init__(self):
            self._scene=mujoco.MjvScene(model,maxgeom=100)
            self._mjr_context=object()
        @property
        def scene(self):return self._scene
    renderer=Renderer();context=renderer._mjr_context
    def update():
        mujoco.mjv_updateScene(model,data,option,None,camera,mujoco.mjtCatBit.mjCAT_ALL,renderer.scene)
        return renderer.scene.geoms[0].pos.copy()
    fresh=update();retained=update()
    assert not np.array_equal(retained,fresh), 'Exercise actual previous-camera plane history'
    reset_render_scene(renderer,model)
    np.testing.assert_array_equal(update(),fresh)
    assert renderer._mjr_context is context
    assert renderer.scene.maxgeom==100 and data.time==0
