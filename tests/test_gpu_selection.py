import unittest
from scripts.pi0294.night10.gpu_selection import Device,choose_device


class GPUSelectionTest(unittest.TestCase):
    def choose(self,rows,**kw):
        return choose_device(rows,available_RAM_bytes=40,required_RAM_bytes=36,**kw)

    def test_busy_gpu0_selects_qualified_gpu1_and_switches_back_at_next_boundary(self):
        def rows(a,b):return [Device(0,'A',a,24576,True,True,True,120),
                              Device(1,'B',b,15360,True,True,False,120)]
        self.assertEqual(self.choose(rows(2600,15899)).index,1)
        self.assertEqual(self.choose(rows(30000,100)).index,0)

    def test_unqualified_gpu_only_available_to_native_probe(self):
        d=Device(1,'B',15899,15360,False,False,False,120)
        self.assertIsNone(self.choose([d]))
        self.assertEqual(self.choose([d],qualification=True),d)

    def test_capacity_stability_sharing_and_ram_each_block(self):
        base=dict(index=0,uuid='A',free_MiB=25000,entry_MiB=24576,qualified=True,
                  shared_authorized=True,other_compute_present=True,stable_seconds=120)
        for change in [dict(free_MiB=20000),dict(stable_seconds=119),dict(shared_authorized=False)]:
            self.assertIsNone(self.choose([Device(**(base|change))]))
        self.assertIsNone(choose_device([Device(**base)],available_RAM_bytes=35,required_RAM_bytes=36))


if __name__=='__main__':unittest.main()
