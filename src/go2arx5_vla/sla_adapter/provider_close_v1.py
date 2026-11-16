"""Explicit parent1000 + new500 identity; ordinary RGB, no optimizer/cache."""
from .provider_v3 import Provider as OriginalProvider
from .close_temporal_v1 import RECIPE
from .gripper_latent_v2 import VERSION,EPSILON


class Provider(OriginalProvider):
    def validate_statistics_metadata(self,stats):
        m=self.manifest;b=m['bindings']
        if (m['step']!=500 or m['cursor']!=16000 or b['recipe']!=RECIPE or b['parent_step']!=1000
                or b['parent_visits']!=32000 or b['new_planned_updates']!=500
                or b['codec_version']!=VERSION or stats['codec_version']!=VERSION
                or stats['epsilon']!=EPSILON or stats['split']!='TRAIN' or stats['state_gripper_transformed']
                or m['statistics_sha256']!=b['derived_statistics_sha256']):
            raise ValueError('DISTINCT_PARENT1000_NEW500_FROZEN_C20_RGB_PACKAGE')

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from scripts.sla.close_goal_g1_v1 import make_sampler
        from scripts.sla.mfree_c20_vectorfield_v3 import extracted_fields
        from . import batch_v2
        from openpi.models import pi0
        fields,_=extracted_fields(batch_v2,pi0);route=self.manifest['bindings']['inference_route']
        call=make_sampler(self.graph,fields,cached=route.startswith('cached'),precision='highest'if route.endswith('highest')else None)
        self.infer=lambda a,f,o,n:call(a,f,o,n)[0]
