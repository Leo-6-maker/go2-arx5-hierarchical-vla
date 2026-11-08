import numpy as np
import pytest
from go2arx5_vla.world_model.task_learning_visual_probe_v1 import red_centroid,leave_layout_out


def test_centroid_uses_largest_real_red_component_and_requires_area():
    image=np.zeros((256,256,3),dtype=np.uint8)
    image[10:14,20:24,0]=255;image[100:110,120:130]=[200,40,30]
    c=red_centroid(image)
    assert c['valid'] and c['pixels']==100
    assert np.allclose(c['centroid_xy'],np.array([124.5,104.5])/255.)
    image[100:110,120:130]=0
    assert red_centroid(image)['valid']is False


def data():
    rng=np.random.default_rng(7)
    centers=rng.normal(size=(10,4));x=np.repeat(centers,2,axis=0)+rng.normal(scale=.001,size=(20,4))
    y=x@np.array([[.001,0],[0,.001],[.0005,0],[0,.0005]])
    return x,y,np.repeat([f'QUAL{i:02d}'for i in range(1,11)],2)


def test_whole_layout_holdout_can_detect_available_visual_information():
    x,y,g=data();r=leave_layout_out(x,y,g)
    assert r['information_probe_passes']
    assert r['XY_RMSE_m']<r['baseline_XY_RMSE_m']*.01
    assert all(v['restart_count']==2 for v in r['per_layout'])
    assert r['GR00T_visual_conditioning_proven']is False
    assert r['recovery_observability_proven']is False
    order=np.random.default_rng(8).permutation(20)
    shuffled=leave_layout_out(x[order],y[order],g[order])
    assert np.isclose(r['XY_RMSE_m'],shuffled['XY_RMSE_m'])


def test_constant_images_cannot_pass_by_using_held_out_layout_identity():
    x,y,g=data();r=leave_layout_out(np.zeros_like(x),y,g)
    assert not r['information_probe_passes']
    assert r['XY_RMSE_m']==r['baseline_XY_RMSE_m']


def test_missing_or_incorrectly_grouped_restarts_do_not_pass():
    x,y,g=data()
    assert leave_layout_out(x[:-1],y[:-1],g[:-1])['observed_requests']==19
    with pytest.raises(ValueError):leave_layout_out(x[:17],y[:17],g[:17])
    g[1]='EXTRA'
    with pytest.raises(ValueError):leave_layout_out(x,y,g)
