"""Serve the matched V7-A projection through the original Go2 decoder and G7 report."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from v7_core_endpoint import ARMS, PLAN, STEPS, array_list_identity, byte_identity


DT = .002


def current_condition(previous, *, treatment):
    from go2arx5_vla.sla_adapter.dense_g7_v6 import previous_command_context
    from v7_previous_command import encode_previous_width

    width = previous_command_context(previous)
    times = np.asarray([previous["query_native_s"],previous["source_native_s"]],dtype=np.float64)
    ticks = np.rint(times/DT).astype(np.int64)
    require(ticks[1] >= 0 and np.max(np.abs(times-ticks*DT)) <= 1e-8,
            "Previous jaw command must come from the original physical clock")
    return encode_previous_width([width],ticks[:1],ticks[1:],[True],[True],treatment=treatment)


def provider_manifest(root, expected):
    root = root.resolve(strict=True)
    model = read(pin(root/"CHECKPOINT.json",expected))
    condition = model["v7_previous_command"]
    require(model["v7_plan"] == PLAN and model["role"] == "V7_A_CONTEXT_500" and model["phase"] == STEPS
            and model["v7_arm"] in ARMS and condition["arm"] == model["v7_arm"]
            and condition["file"] == "previous_command_projection.npy"
            and condition["input"] == "previous_actual_consumed_gripper_command"
            and condition["shape"] == [1,1024] and condition["dtype"] == "float32"
            and len(model["arrays"]) == 29 and len({r["key"] for r in model["arrays"]}) == 29
            and model["checkpoint_sha256"] == array_list_identity(model["arrays"],condition)
            and not any(model[k] for k in ("formal_inference_eligible","native_admission","hardware_allowed","physics")),
            "The Core inference identity differs from the independent V7 export")
    value = np.load(pin(root/condition["file"],condition["sha256"]),allow_pickle=False)
    require(value.shape == (1,1024) and value.dtype == np.float32 and np.isfinite(value).all()
            and byte_identity(value) == condition["array_sha256"]
            and bool(value.any()) == (model["v7_arm"] == ARMS[1]), "Invalid frozen V7 projection bytes")
    return model, value


class ContextProvider:
    def __init__(self, graph, frozen, parameters, tokenizer, statistics, projection, treatment):
        import jax
        from go2arx5_vla.sla_adapter import batch_v2
        from openpi.models import pi0
        from openpi_client.image_tools import resize_with_pad
        from scripts.sla.mfree_c20_vectorfield_v3 import extracted_fields
        from v7_full_model_paths import PROJECTION_KEY, auxiliary_h40, extend_target_io

        active = extend_target_io(parameters)
        target = active["target"]
        target.replace_by_pure_dict({**target.to_pure_dict(),PROJECTION_KEY:jax.device_put(projection)})
        self.active = dict(shared=active["shared"],private=target)
        self.frozen, self.tokenizer, self.statistics = frozen, tokenizer, statistics
        self.resize = resize_with_pad
        self.treatment = treatment
        fields, _ = extracted_fields(batch_v2,pi0)
        sampler = auxiliary_h40(graph,fields,rematerialize=False)

        def infer(active, frozen, observation, noise):
            return sampler(active["parameters"],frozen,observation,noise,active["condition"])

        self.infer = infer

    def predict(self, state, images, prompt, frame, *, noise_id, previous_reference):
        from go2arx5_vla.sla_adapter.scale20k_package import Provider

        condition = current_condition(previous_reference,treatment=self.treatment)
        # The request-local view leaves the original preprocessing, noise,
        # physical decoder and safety diagnostics in their normal entry point.
        active = dict(parameters=self.active,condition=condition)
        request = SimpleNamespace(active=active,frozen=self.frozen,statistics=self.statistics,
                                   tokenizer=self.tokenizer,resize=self.resize,infer=self.infer)
        return Provider.predict(request,state,images,prompt,frame,noise_id=noise_id,previous_reference=previous_reference)


def load_context_provider(path, expected, source, target, tokenizer):
    from go2arx5_vla.sla_adapter import scale20k_model, scale20k_package
    from go2arx5_vla.sla_adapter.model_v1 import Tokenizer
    from go2arx5_vla.sla_adapter.scale20k_contract import canonical_hash

    path = Path(path).resolve(strict=True)
    manifest, projection = provider_manifest(path,expected)
    require(source["parent_manifest"]["sha256"] == manifest["foundation_sha256"]
            and sha(Path(tokenizer)) == manifest["tokenizer_sha256"] and source["state_sha256"] == manifest["state_sha256"],
            "The original base, tokenizer and statistics are required")
    graphs, template, frozen, parent = scale20k_model.load(source["parent_manifest"],source["parent_params"])
    require(parent["frozen_fingerprint"] == manifest["frozen_model"]
            and canonical_hash(parent["inventory"]) == manifest["partition_sha256"], "The deployment base changed")
    active = scale20k_package.read_weights(path,manifest["arrays"],{k:template[k] for k in ("shared","target")})
    return ContextProvider(graphs["target"],frozen,active,Tokenizer(str(tokenizer)),target.statistics,
                           projection,manifest["v7_arm"] == ARMS[1]), manifest
