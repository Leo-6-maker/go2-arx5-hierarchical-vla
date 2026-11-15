"""Ordinary RGB provider; distinct mobile recipe and its frozen new statistics."""
from .provider_v3 import Provider as OriginalProvider
from .mfree_snapshot_training_v1 import RECIPE
from .gripper_latent_v2 import VERSION, EPSILON


class Provider(OriginalProvider):
    def validate_statistics_metadata(self,stats):
        m=self.manifest;b=m['bindings']
        if (b['recipe']!=RECIPE or m['step']!=1000 or m['cursor']!=32000
                or b['codec_version']!=VERSION or stats['codec_version']!=VERSION or stats['epsilon']!=EPSILON
                or stats['schema']!='MFREE_NESTED_TRAIN_LATENT_STATISTICS_V1' or stats['split']!='TRAIN'
                or stats['state_gripper_transformed'] or b['boundary_version']!='MOBILE_ACCEPTED_BOUNDARY_050_V1'
                or any(b[k]for k in ('BASE_head','spatial_loss','reference_error_input','teacher_memory_input'))
                or m['statistics_sha256']!=b['derived_statistics_sha256']):
            raise ValueError('NEW_MFREE_FIXED1000_PACKAGE_AND_TRAIN_STATS_REQUIRED')
