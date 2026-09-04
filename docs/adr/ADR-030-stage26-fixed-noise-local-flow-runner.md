# ADR-030: Stage26 fixed-noise teacher-forced local-flow runner

Status: Accepted for Stage26 preparation

## Context

Stage25 showed a large, distributed BASE-to-O1 parameter delta, so the next
scientific gate is the local flow field rather than a larger training run or
a sampler intervention. The existing Stage23 Phase-C protocol already freezes
the sample, universal 132D action ABI, full-width noise, active first-ten loss
mask, and nine paired probes.

## Decision

Add a project-owned runner that:

1. verifies the exact TINY1, source snapshot, processor, modality, Cosmos, and
   checkpoint index identities;
2. obtains one official processor-normalized H40x132 training sample;
3. generates each fixed noise tensor once on a CPU Torch generator;
4. loads BASE and O1 one at a time through the established local-only
   AutoModel.from_pretrained path and BF16 inference cast;
5. runs one shared backbone conditioning pass and nine direct
   teacher-forced action-head evaluations per checkpoint;
6. reports raw velocity hashes, active flow MSE, decoded profiles, and
   descriptive paired aggregates.

The direct helper is source-mapped to the official training forward and does
not call the complete model forward or free-running sampler. It performs no
training, backward, optimizer, PBS, WBC, or hardware operation.

## Consequences

The result answers whether O1 changed the local velocity field on the exact
fixed probes. It does not assign a scientific threshold or decide H2/C2.
Interpretation remains a GPT/PI gate. A source, path, hash, shape, finite,
device, or count mismatch stops before the corresponding work.

SCIENTIFIC_CONTRACT_CHANGED=false.
