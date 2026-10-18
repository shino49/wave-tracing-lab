# Tight-anchor paired warmed renderer timing

Build `plt-resident-tight-interval-2026-10-13.5`. All four scenarios passed on native Dawn/Tint/Vulkan SwiftShader driver 5.0.0, a software fallback adapter. These are descriptive wall-clock measurements, not a browser/physical-GPU validation or a statistical speed claim.

## Result and recommendation

The independently useful result is the explicit `anchor-round` scheduling option. Its paired median renderer wall time is lower on all four tested fixtures while retaining exactly the old anchor audit, image, RNG and replay ownership. Keep tighter math experimental: it saves only five paths in the full 4,608-path transport corpus, still retains no complete paths in either large-world fixture, and its extra cost is not consistently recovered by scheduling.

In particular, 8× world scale replays all 128 timing paths in both guards. Tight-round has a 156.2 ms independent median versus 113.9 ms for anchor-batched; its paired median difference is +26.2 ms and six of eight paired runs are slower. It is not an overall or universal speed improvement. The 4× combined result is mixed: independent medians are 177.2 versus 178.8 ms, while the paired median difference is −13.4 ms with wide variability. Do not conflate a median of paired differences with a difference of independent medians.

### Per-mode renderer wall medians (ms)

| Existing fixture | Replay anchor → tight / 128 | Anchor batched | Anchor round | Tight batched | Tight round |
|---|---:|---:|---:|---:|---:|
| Unit two mirrors | 4 → 3 | 274.9 | 182.2 | 293.2 | 197.8 |
| 4× world scale | 128 → 128 | 177.2 | 144.5 | 259.1 | 178.8 |
| Half-scale three mirrors | 8 → 8 | 410.1 | 305.0 | 397.2 | 303.3 |
| 8× world scale | 128 → 128 | 113.9 | 74.5 | 182.3 | 156.2 |

### Paired differences, B minus A (ms)

Negative means B ran faster in that pair. Each comparison uses the same fixture, immutable RNG prefix and balanced quartet. All eight individual pairs remain in the JSON report; wide ranges are deliberately retained.

| Fixture | Comparison | Median difference | Observed range | B faster |
|---|---|---:|---:|---:|
| Unit two mirrors | anchor-scheduling-only | -64.6 | [-216.3, +7.7] | 7/8 |
| Unit two mirrors | tight-scheduling-only | -68.5 | [-170.6, +2.1] | 7/8 |
| Unit two mirrors | tightening-only-batched | +33.7 | [-105.9, +125.7] | 3/8 |
| Unit two mirrors | combined-versus-v20-mode | -47.1 | [-276.5, +59.0] | 5/8 |
| 4× world scale | anchor-scheduling-only | -39.1 | [-242.0, +176.8] | 6/8 |
| 4× world scale | tight-scheduling-only | -80.7 | [-276.9, -19.2] | 8/8 |
| 4× world scale | tightening-only-batched | +78.9 | [-109.2, +265.5] | 1/8 |
| 4× world scale | combined-versus-v20-mode | -13.4 | [-128.4, +89.3] | 5/8 |
| Half-scale three mirrors | anchor-scheduling-only | -101.3 | [-214.4, +8.3] | 6/8 |
| Half-scale three mirrors | tight-scheduling-only | -96.2 | [-214.5, +28.3] | 7/8 |
| Half-scale three mirrors | tightening-only-batched | +28.1 | [-158.8, +135.8] | 3/8 |
| Half-scale three mirrors | combined-versus-v20-mode | -64.2 | [-243.9, +18.0] | 6/8 |
| 8× world scale | anchor-scheduling-only | -33.4 | [-200.9, -15.5] | 8/8 |
| 8× world scale | tight-scheduling-only | -44.5 | [-164.1, +172.1] | 6/8 |
| 8× world scale | tightening-only-batched | +65.2 | [-104.3, +193.8] | 1/8 |
| 8× world scale | combined-versus-v20-mode | +26.2 | [-92.3, +236.1] | 2/8 |

## Method and correctness

Four existing fixture identities are used with the original camera and source geometry, each at 8×8 pixels and two samples per pixel (128 paths). Each mode receives two warmup runs, then eight balanced four-mode quartets. The balanced orders rotate every mode through every position. All other native validation workloads were idle before this benchmark started.

Totals: **128 measured runs / 16,384 measured path-runs**, plus **32 warmup runs / 4,096 warmup path-runs**. All 160 runs have complete independent CPU-shadow auditing and strict admission. Within each guard, all warmed runs and submission modes have exactly equal full audit/image/RNG/ownership/replay data. All analytic powers remain exact and there are zero numerical geometry replacements. The anchor first-batch retention counts match the preserved v20 report.

Each fixture uses a newly created native device and normal, uncached production runner setup. The four modes share that fixture's tight-compiled runner; anchor mode follows its old arithmetic/guard path. This is not a separately compiled historical v20 binary comparison. The transport harness's explicit shared-device cache is not used by this timing harness.

No extra 4×/8× path-load scenarios are measured in this bounded checkpoint. World-scale changes in fixture names are geometry transforms, not workload multipliers.

## Setup, full shadow and per-run costs

Compile/setup totals **280.636 s**, excluded from renderer run measurements. Independent complete CPU-shadow construction totals **33.376 ms**, also separate. One independent CPU reference is constructed per fixed-input fixture and reused to validate every repeated warmup/measured run. This benchmark does not charge repeated shadow construction to a renderer run; a workload with new inputs would need its own shadow construction. Every run still incurs both raw full-shadow comparison and strict admission, measured separately below.

| Fixture | Native compile/setup (s) | Independent full shadow construction (ms) |
|---|---:|---:|
| Unit two mirrors | 69.667 | 10.168 |
| 4× world scale | 72.114 | 7.121 |
| Half-scale three mirrors | 70.712 | 13.103 |
| 8× world scale | 68.143 | 2.984 |

Renderer wall time includes initialization, real GPU command scheduling/waits, status and replay readbacks, actual selective CPU replay, and final readback/decode. Raw shadow comparison and admission occur afterward. These counters describe overlapping phases within renderer time; do not add renderer time to its contained replay/readback components. They are host wall counters, not GPU timestamp queries.

### Per-run median cost components (ms)

| Fixture | Mode | Renderer | Selective CPU replay | Status readback | Replay readback | Final readback/decode | Raw full-shadow audit | Admission |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Unit two mirrors | anchor-batched | 274.927 | 0.780 | 0.962 | 0.255 | 0.906 | 1.459 | 9.232 |
| Unit two mirrors | anchor-round | 182.164 | 0.804 | 1.206 | 0.216 | 0.814 | 1.196 | 7.502 |
| Unit two mirrors | tight-batched | 293.166 | 0.617 | 1.080 | 0.152 | 0.903 | 1.607 | 9.330 |
| Unit two mirrors | tight-round | 197.816 | 0.559 | 1.338 | 0.175 | 0.970 | 1.225 | 8.988 |
| 4× world scale | anchor-batched | 177.189 | 5.604 | 0.650 | 0.285 | 0.732 | 1.556 | 10.629 |
| 4× world scale | anchor-round | 144.467 | 5.697 | 1.158 | 0.346 | 0.697 | 1.491 | 11.043 |
| 4× world scale | tight-batched | 259.139 | 5.826 | 0.809 | 0.336 | 0.739 | 1.638 | 9.998 |
| 4× world scale | tight-round | 178.847 | 6.199 | 1.339 | 0.371 | 0.700 | 1.458 | 9.907 |
| Half-scale three mirrors | anchor-batched | 410.111 | 0.812 | 1.075 | 0.166 | 0.966 | 1.898 | 10.317 |
| Half-scale three mirrors | anchor-round | 304.955 | 0.913 | 1.655 | 0.125 | 0.900 | 2.143 | 10.773 |
| Half-scale three mirrors | tight-batched | 397.176 | 0.762 | 1.184 | 0.131 | 0.966 | 2.015 | 11.035 |
| Half-scale three mirrors | tight-round | 303.299 | 0.760 | 1.661 | 0.127 | 0.978 | 1.945 | 9.579 |
| 8× world scale | anchor-batched | 113.912 | 5.329 | 0.434 | 0.121 | 0.720 | 1.455 | 10.129 |
| 8× world scale | anchor-round | 74.454 | 5.424 | 0.614 | 0.176 | 0.746 | 1.463 | 11.089 |
| 8× world scale | tight-batched | 182.333 | 5.925 | 0.722 | 0.299 | 0.709 | 1.514 | 9.994 |
| 8× world scale | tight-round | 156.243 | 5.561 | 1.000 | 0.284 | 0.734 | 1.470 | 9.464 |

## Reproduction and provenance

Run only after complete native transport validation and while other native validation workloads are idle:

`WEBGPU_EXECUTION_BACKEND=vulkan VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json node tests/plt-resident-tight-submit-timing.mjs --execute`

- Full data: `plt-resident-tight-submit-timing-validation.json`
- Transport prerequisite: `plt-resident-tight-validation.json`; SHA-256 `5accec3c0619d476fbc0c93e6d84a9adbd3be5b4a305b3ccfa64436b543eef6f`
- Timing harness SHA-256: `c9c69f4fcf27fcd5bf135c77b021f348d9621124e9d61a3a7f1b1506aeee99e3`
- Transport/helper harness SHA-256: `e7f80a17e065465991ac089b545e6450b3393100df95602d7000d6c150be083f`
- Frozen guard SHA-256: `794469d7223bcbad443fa088aaa7610f9e4c87443496fa8be1a49801bd06893f`
- Frozen walker SHA-256: `2fa2f1e966cdfc33b8462d8872f45261372ea9333ec56bd7581b3d74ab214c06`
- Frozen host SHA-256: `f7580b2cf3b8f7de8089d0a7103890330370b155719203324c484660f7170a72`
- Preserved v20 transport and warmed reports remain unchanged and SHA-pinned. Every current source/host hash equals the validated full transport snapshot.
