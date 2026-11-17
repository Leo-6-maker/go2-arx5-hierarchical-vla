# B1 actual fixed500 endpoint

B1 has actually completed **500 optimizer updates / 16,000 optimization visits** from the real SHIFT7 parent, with fresh Adam and lambda0. It performed4,000 completed micro4 forward/backward calls and one separately charged diagnostic backward over4 windows. No optimizer update was added by restoration. Parent cumulative1,500/48,000 -> this successor2,000/64,000; B2 is a separate model, not a continuation of B1.

| Evidence | Actual |
|---|---|
| Training source | `8ebdcbb7ab39c0f707c5b2ba94b0b097abc937e5` |
| Training spec | `48e33626b6db391997be167cd5a51e0b0f5bb008cfd259a0d51f40414039c5a4` |
| Real result | `6ab3b21ed91789aa883a88d8e875396c20b30910013f403b24377e33e6b4adf6` |
| step500 checkpoint | `ac675d77393b5d05c23c3a71efab09883b0045312efdafac4ffa064ab7b9d268` |
| Exported model manifest | `b9ba974c374b3c3a09d7742004fc799337d5e7f1b9c9c7d290af066766e4706c` |
| Frozen TRAIN statistics | `f28d50e6eeea09b332d4bd159eca4333bbe23b185f95f1d336d45311e364ab3e` |
| Frozen dense fingerprint | `507aa0a700c2e8cacfb398c76e2147da4a82cbf05bc2e7b4ec85d1562c091b06` |

All28 trainable arrays changed. All87 model/Adam/RNG arrays were actually reloaded on GPU and checked bit-exact; a separate CPU audit verified checkpoint and package bytes. This all87 reload occurred in the learner process. The independent ordinary-RGB16 package reload remains pending and is not claimed complete.

B1 generated the fixed96 TRAIN predictions **after** reloading step500: original full H40 **96/96**, including20 CLOSE cases, with20/20 crossing errors<=100ms (maximum50ms). These are seen TRAIN states, not native consumption or new task success. CN38, ROLL96 and independent RGB16 remain to be run by the finite followup.

Both models' saved pre-update states were independently byte-compared: all87 arrays are identical, as are the parent, visit map, LR, noise, flow time, state dropout and first indices. The whole checkpoint manifest SHAs differ because the declared loss bindings differ; array contents match. B1/B2 actual first diagnostic edges were equal; B2's objective increment agrees with .1 times that edge to FP32 rounding. This is an implementation check, not a policy-effect result.

Measured B1 service cost: **3,369.86 s wall / 2,829.06 s CPU**, host peak14,365,962,240 bytes, swap peak0. The learner function's narrower measured cost is3,363.72 s wall /2,823.71 s CPU. JAX live-buffer peak9,714,233,344 bytes. GPU busy-seconds were not separately measured. The unit exited0 with no remaining process.

B2 has automatically acquired the same GPU under its existing identity; its latest exact prefix is in ACTUAL.json. The followup remains live and waits for both endpoints before independent RGB/offline qualification and all24 E2 requests. **New native requests / integrations / newly demonstrated task successes: 0 / 0 / 0.** Historical failures and original V5.1 results remain unchanged.
