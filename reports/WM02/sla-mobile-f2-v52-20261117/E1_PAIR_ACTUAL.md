# E1 actual matched 500-step endpoints

Both B1 and B2 have completed **500 updates / 16,000 optimization visits each**. The independent audit matched all500 updates, all4,000 microbatch index/dropout/cursor entries and the LR schedule. All87 initial model/Adam/RNG arrays were identical. Noise and flow time use that identical initial RNG and the same sealed counter-based draw function on matched absolute cursors; the first bank was independently hash-matched. Every later draw was not separately hashed. Each endpoint changed all28 trainable arrays, then reloaded all87 saved arrays on GPU bit-exact. This full-state reload occurred in the learner process; the later separate inference process independently restored each ordinary-RGB package and reproduced16/16 outputs bit-exact without loading Adam.

| Arm | New updates / visits | Model manifest SHA256 | step500 SHA256 |
|---|---:|---|---|
| B1 | 500 / 16,000 | `b9ba974c374b3c3a09d7742004fc799337d5e7f1b9c9c7d290af066766e4706c` | `ac675d77393b5d05c23c3a71efab09883b0045312efdafac4ffa064ab7b9d268` |
| B2 | 500 / 16,000 | `f95a5bafc2c67cae722c74544fd2c6b6aa1496805f04179dd13c134338081e25` | `3d5b1128170ad3e26a9eb5511dcd6e96ccbf4e0ca09aaefbff4defa2d44b0e3c` |

Training source `8ebdcbb7ab39c0f707c5b2ba94b0b097abc937e5`; independent offline/native source `2797d0c2c458667e241aa0d24d6d0b005f3caf1a`. Common actual SHIFT7 parent `ef530bceebcc1691d9af9b5b1ea4ea89731f0c1d5d8f3e2287c3618dff7806eb`; frozen TRAIN statistics `f28d50e6eeea09b332d4bd159eca4333bbe23b185f95f1d336d45311e364ab3e`. Each new model is cumulatively2,000 updates/64,000 visits from the original SLA lineage; the campaign added1,000 updates/32,000 optimization visits across the two separate arms. Two diagnostic backward calls over8 windows are charged separately and are not optimizer updates.

Both TRAIN96 panels passed96/96 original full H40 checks. Each has20/20 CLOSE events within the original100ms standard, maximum50ms. The followup then performed exactly600 new offline generations: three independent RGB16 restorations(48), plus B1/B2 CN38 and ROLL96(552). Including192 TRAIN predictions, E1 used792 new generations. Scientific offline failures did not remove any E2 control.

| Cost | B1 | B2 |
|---|---:|---:|
| Learner wall / CPU seconds | 3363.72 / 2823.71 | 3359.53 / 2816.07 |
| Whole-service CPU seconds | 2,829.06 | 2,822.99 |
| Host peak bytes | 14,365,962,240 | 22,119,780,352 |
| JAX live-buffer peak bytes | 9,714,233,344 | 9,714,233,344 |

Both units exited0; swap peak0. B1 whole-service wall was3,369.86s. B2 whole-service wall was6,667.81s including its earlier serialized wait for B1; this is not GPU busy-time. GPU busy-seconds were not separately measured. The separate600-generation offline stage took218.38s wall.

**These results establish training/restoration and saved-input trajectory behavior, not mobile-grasp capability.** The fixed24-condition MuJoCo panel is running under its own actual counters. All previous failed conditions remain. See [offline distributions and gates](E1_OFFLINE_ACTUAL.md) and the timestamped [campaign state](ACTUAL.md).
