"""Same RGB inference math; explicit PD4 quantile metadata identity only."""
from .provider_v3 import Provider as OriginalProvider
from .pd4_training_v71 import OPERATION,STATISTICS


class Provider(OriginalProvider):
    def validate_statistics_metadata(self,stats):
        value=self.manifest;b=value['bindings'];view=b.get('PD4_view')
        if (view not in STATISTICS or b.get('recipe')!=OPERATION
                or value['statistics_sha256']!=STATISTICS[view]
                or b['derived_statistics_sha256']!=STATISTICS[view]
                or b.get('statistics_codec_metadata_source')!='PACKAGE_CODEC_VERSION_AND_EPSILON_WITH_FROZEN_PD4_QUANTILES'
                or stats.get('fit_input')!='UNNORMALIZED_RELATIVE_XYZ_SO3_AND_LOGIT_GRIP'
                or stats.get('fit_unique_windows')!=244 or stats.get('split')!='TRAIN'):
            raise ValueError('EXACT_PD4_FROZEN_QUANTILES_AND_EXPLICIT_PACKAGE_CODEC_BINDING')
        # Base package contract already checks VERSION and EPSILON. The frozen
        # quantile file is copied byte-for-byte; no refit or added metadata.
