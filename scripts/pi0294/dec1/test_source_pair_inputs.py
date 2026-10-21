import base64
from copy import deepcopy
import hashlib
import unittest

from .source_pair_inputs import prepare_B,ARM_INPUTS


def fixture():
    a={'shared-recipe.json':b'original recipe'}
    a.update({name:('original A '+name).encode() for name in ARM_INPUTS})
    b=dict(a,**{name:('original B '+name).encode() for name in ARM_INPUTS})
    def row(values):
        return dict(small_inputs_base64={k:base64.b64encode(v).decode() for k,v in values.items()},
            small_input_sha256={k:hashlib.sha256(v).hexdigest() for k,v in values.items()},
            metadata_windows=160930,metadata_composite_manifest_sha256='a'*64,metadata_sampler_support_sha256='b'*64)
    pair=dict(A_CONTROL=row(a),B_COVERAGE=row(b))
    current=dict(arm='A_CONTROL',source='c'*40,small=dict(pair['A_CONTROL']['small_inputs_base64']),
        source_files={'fixture.py':'unchanged sealed code'})
    current['small']['actual-closed-Vanda-bindings.json']='immutable same closed inputs'
    return current,pair


class SourcePairColdInputs(unittest.TestCase):
    def test_only_two_canonical_arm_overrides_no_original_mutation(self):
        current,pair=fixture();before=deepcopy((current,pair))
        out=prepare_B(current,pair,source_sha='c'*40)
        self.assertEqual((current,pair),before);self.assertEqual(out['arm'],'B_COVERAGE')
        self.assertEqual({k for k in current['small'] if current['small'][k]!=out['small'][k]},ARM_INPUTS)
        self.assertEqual(out['source_files'],current['source_files'])
        self.assertEqual(out['small']['actual-closed-Vanda-bindings.json'],current['small']['actual-closed-Vanda-bindings.json'])
        self.assertEqual(out['cold_pair_binding']['new_PBS'],0)
        self.assertTrue(out['cold_pair_binding']['previous_A_terminal_audit_and_settlement_required'])

    def test_unknown_recipe_support_or_template_mutation_is_rejected(self):
        for failure in ('common','support','hash','different-source'):
            with self.subTest(failure=failure):
                current,pair=fixture()
                if failure=='common':current['small']['shared-recipe.json']='changed'
                elif failure=='support':pair['B_COVERAGE']['metadata_windows']+=1
                elif failure=='hash':pair['B_COVERAGE']['small_inputs_base64']['shared-recipe.json']='YQ=='
                else:current['source']='d'*40
                with self.assertRaises(PermissionError):prepare_B(current,pair,source_sha='c'*40)

    def test_extra_arm_difference_with_valid_hash_still_rejected(self):
        current,pair=fixture();row=pair['B_COVERAGE'];value=b'unauthorized recipe'
        row['small_inputs_base64']['shared-recipe.json']=base64.b64encode(value).decode()
        row['small_input_sha256']['shared-recipe.json']=hashlib.sha256(value).hexdigest()
        with self.assertRaises(PermissionError):prepare_B(current,pair,source_sha='c'*40)
