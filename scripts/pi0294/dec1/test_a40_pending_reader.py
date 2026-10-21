"""Pending data is reserved, not falsely reported closed or free."""
import copy
import unittest
from .a40_future import pending_source_reservations
from .a40_allocation import AUTHORITY,T0


class PendingReaderTests(unittest.TestCase):
    def test_pending_output_and_transfer_are_not_erased(self):
        row=dict(source_data_profile='ORIGINAL126_C1',authority_release=AUTHORITY,
            original_T0_epoch=T0,unknown_effects=[],new_PBS=0,new_models=0,
            costs_inside_original_PV1_and_DEC1=True,output_reserved_bytes=24*1024**3,
            transfer_reserved_bytes=8*1024**3)
        h=dict(external_CPU_source_data_reservations=dict(reader=row),engineering_CPU_upper_settlements=dict(reader={}))
        d=dict(source_data_reservations=dict(reader=copy.deepcopy(row)))
        self.assertEqual(pending_source_reservations(h,d),dict(output_bytes=24*1024**3,transfer_bytes=8*1024**3))
        for mutation in ('missing','profile','unpaid','unknown'):
            a,b=copy.deepcopy(h),copy.deepcopy(d)
            if mutation=='missing':a['external_CPU_source_data_reservations']={}
            elif mutation=='profile':a['external_CPU_source_data_reservations']['reader']['source_data_profile']='other'
            elif mutation=='unpaid':a['engineering_CPU_upper_settlements']={}
            else:a['external_CPU_source_data_reservations']['reader']['unknown_effects']=['effect']
            with self.subTest(mutation=mutation),self.assertRaises(PermissionError):pending_source_reservations(a,b)

    def test_empty_reservations_keep_old_contract(self):
        self.assertEqual(pending_source_reservations({},{}),dict(output_bytes=0,transfer_bytes=0))


if __name__=='__main__':unittest.main()
