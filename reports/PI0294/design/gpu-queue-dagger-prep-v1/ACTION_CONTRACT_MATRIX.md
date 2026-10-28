# Q1 Action Contract Matrix

Actual receipt: CPU_ACTION_CONTRACT_TEST_RECEIPT.json. All outcomes here mean
CPU fixture behavior, not official-processor/model or physical qualification.

| Contract | Reused Or Added Coverage | Meaning |
|---|---|---|
| EEF10 source | codec/abi tests | exact H40x10, finite, row-rot6d, metres, original world/TCP |
| Relative once | nonzero rotated-anchor round trip | absolute input -> original SAP conversion once -> original decode; toy SAP only |
| H40 vs prefix8 | chunk transaction | full proposal; one exact8-point/200-tick advance; fresh next observation required |
| 20Hz/500Hz | integer timestamp tests | point50ms, tick2ms, prefix0.4s; H40 spans0..1.95s |
| Gripper | direction/range/timing tests |0 closed/1 open, reject LIBERO -1 or inversion, no clipping |
| Wrong mm/TCP/frame | abi negative tests | explicit semantics must match; numeric values cannot establish units alone |
| Short or EEF13 source | codec negative tests | no tail synthesis or BASE padding into M2048 |
| Future target EEF13 | prospective label-mask test | source/target domain separate; missingBASE masked, soft-zeroBASE explicitly supervised |
| BASE mapping | existing PI0294 BaseIntentTests | signedSE2, tilted mount origin,41endpoints, EEF world hold, zero vs absent, bounds |

The existing pure-CPU `scripts/pi0294/test_base_intent.py` is reused unchanged;
we did not duplicate its native adapter or qualification. These tests already
cover fixed world EEF plus BASE motion and nonfinite/speed/acceleration rejection.
Future BASE3 training still requires actual labels and a new compatible model.

Source masks accept10 genuinely labelled dimensions only. Prospective target
absentBASE must be masked false and structural-zero storage is explicitly not
an intent. A valid soft-zeroBASE has true labels and means a different command.
No target data or EEF13 model is created by these tests.

Input/codec fixtures use synthetic arrays, not actual saved E0 student states.
Official processor numerical round trip remains NOT_RUN. Existing native codec
and qualification receipts are not reclassified by this package.
