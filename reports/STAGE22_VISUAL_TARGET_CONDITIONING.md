# Stage22 Visual Target Conditioning

The model-facing input is restricted to:

```text
native body RGB + native wrist RGB + language + StateV1[10]
```

The target actor position is not supplied to the model. It is retained only
for expert labels and post-hoc counterfactual analysis.

The profile evaluator uses the six `expert-0000` snapshots, which share the
fixed reset state and language while changing the target image/location. It
reports state equality, image hashes, endpoint displacement direction and
magnitude, and a descriptive correlation. It does not promote RGB sensitivity
to visual grounding automatically.

```text
VISUAL_TARGET_CONDITIONING: PENDING R0/R1 INFERENCE
allowed interim claim: RGB_SENSITIVITY_ONLY
stronger claim: VISUAL_TARGET_CONDITIONING_SUPPORTED_ON_SIM_TEST_SET only if
held-out direction and magnitude evidence supports it
```
