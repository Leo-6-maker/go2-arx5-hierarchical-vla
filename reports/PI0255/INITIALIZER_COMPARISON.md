# MT initializer metadata comparison

MT-Base: original public nvidia/GR00T-N1.7-3B, pinned revision 2fc962b973bccdd5d8ce4f67cc63b264d6886495. Index SHA256 407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59 matches the original Stage17/24 foundation identity. Original processor config SHA256 85c1b4690ae090559e79a45193e598b65d6146eedf14750884da65e6d31032be also matches that lineage. The included original NVIDIA licence is attribution, not a replacement licence for derived models.

MT-G: leoyuer/pi0246-gate-pass-campaign, revision efecea485ae77000599d19171b02491f8d514f42, prefix arms/PI0252/r1-speed-margin-20260928/G/checkpoint-2000; full Trainer tree cfbfeb54bea2b5ac266047dfd764cfa6b041a8a77b9688737ea70b56935d5226 comes from the original durable receipt. Current anonymous config/index reads match their original hashes.

Actual HTTP range reads obtained only four safetensor headers: 274960 transport bytes, zero model tensor data. All 1030 common keys have equal shapes. Base has one additional tied LM-head alias, paired with language_model.embed_tokens.weight. The existing Stage25 reconciliation rule is reused; the actual pinned Transformers source also declares that exact tied key. No arbitrary missing-key acceptance is proposed.

537 keys differ BF16→F32, exclusively within the already documented action-head scopes, including frozen vlln/self-attention storage. All other common stored dtypes agree. Both models are Gr00tN1d7, H40, max action/state132, 32 embodiment slots. The six config differences are recorded in initializer_metadata_receipt.json. A common proposed runtime uses the original G freeze flags and precision, with projector/diffusion trainable and backbone/vlln frozen.

The original foundation mapping has no new_embodiment entry; the verified frozen G/C_T mapping assigns it to existing slot10. No extra row or new architecture is required for the proposal. Base must retain its original slot10 tensors; G retains its adapted slot10 tensors. Adding the explicit mapping to Base is a named processor binding, not evidence of prior task adaptation. New C_MT/mask/normalization migration requires a common CPU-tested processor binding before readiness.

The common existing Cosmos-Reason2-2B tokenizer/processor metadata is pinned to9ce19a195e423419c349abfc86fd07178b230561. Nine small files were fully rehashed against the original receipt; no backbone weight was read. Both Base/G declare that same backbone model. Maximum tokenizer ID151668 fits151936 embedding rows. The gated tokenizer content is not redistributed; common_tokenizer_metadata_receipt.json publishes hashes and source identity only. Actual new C_MT processor/tokenization execution remains unproved.

These receipts prove metadata/header compatibility, not a successful model load, new-data support, training admission or generalization. New model loads/queries/backward/optimizer remain0. G has prior task-adaptation cost and cannot be presented as equal-total-compute to Base; proposed6000 NEW updates per arm are equal only for the next stage.

Current dependency: there are no admitted TRAIN motion episodes yet. TRAIN-derived C_MT values, actual every-window codec/gate support, a sealed6000 source/visit table and final executable paired training binding cannot be claimed complete. No validation/test values will be used to fill the missing TRAIN statistics.
