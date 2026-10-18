"""Closed-form boxes only; no native engine or hidden physics fixture."""
from types import SimpleNamespace
import unittest

import numpy as np
from scripts.pi0294.fd2_geometry_audit import box_gap_and_footprint


class SavedGeometryTests(unittest.TestCase):
    def engine(self):
        model=SimpleNamespace(geom_size=np.array([[.3,.3,.02],[.02,.01,.015],[.02,.01,.015]]))
        data=SimpleNamespace(geom_xpos=np.array([[.45,0.,.42],[.45,-.02,.459],[.45,.02,.467]]),
            geom_xmat=np.tile(np.eye(3).ravel(),(3,1)))
        return SimpleNamespace(model=model,data=data,geoms={'table':0,'left_pad':1,'right_pad':2})

    def test_actual_table_top_and_box_vertices_not_nominal_reference(self):
        gap,inside,top=box_gap_and_footprint(self.engine())
        np.testing.assert_allclose(gap,[.004,.012],atol=1e-15)
        self.assertAlmostEqual(top,.44);self.assertEqual(inside,[True,True])

    def test_pad_outside_finite_table_is_not_an_exact_surface_gap_certificate(self):
        e=self.engine();e.data.geom_xpos[1,0]=2.
        _,inside,_=box_gap_and_footprint(e);self.assertFalse(inside[0])

    def test_tilted_pad_support_is_vertex_based(self):
        e=self.engine();angle=.2
        rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        e.data.geom_xmat[1]=rotation.ravel();gap,_,_=box_gap_and_footprint(e)
        self.assertAlmostEqual(gap[0],.459-.44-(.02*np.sin(angle)+.015*np.cos(angle)))


if __name__=='__main__':unittest.main()
