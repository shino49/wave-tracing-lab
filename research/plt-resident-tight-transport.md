# Tight-anchor paired transport checkpoint

Build `plt-resident-tight-interval-2026-10-13.5`. Native Dawn/Tint execution on Vulkan SwiftShader; not a browser or physical-GPU validation.

## Result

All 16 unchanged fixtures, 4,608 ordinary paths and four unsupported-wave paths pass all four modes: anchor-batched, anchor-round, tight-batched and tight-round. Each candidate requires its complete independent CPU shadow, strict existing geometry/power tolerance, exact dyadic analytic radiance where specified, complete contribution ownership and exact RNG words/final state. There are zero audit-discovered numerical replacements.

Within each guard, batched and whole-round submission produce exactly equal complete audits, images, pixel ownership, RNG buffers, replay keys/reasons/depths, rounds and dispatch counts. Four synthetic decoded-history corruptions are rejected independently at shadow/admission boundaries; this is not a claim about private proof-state coverage.

The preserved v20 anchor checkpoint is unchanged: 1,596 replay paths and 2,913 retained mirror paths. Every corresponding per-batch checkpoint and reason/depth histogram matches the pinned v20 report. Tight mode gives 1,591 replays and 2,918 retained mirror paths, an improvement of only five paths. No fixture regresses in replay count.

The main large-world retention goal is **not achieved**. Both 4× and 8× world-scale fixtures still replay all 192 paths. The three-mirror fixture is unchanged at 53 replays. Only the unit two-mirror goal improves (33 to 29 replays); one additional grazing escape path is retained. These are outcome measurements, not correctness gates.

| Existing fixture | Paths | Anchor replay | Tight replay | Fewer replays | Tight retained mirrors |
|---|---:|---:|---:|---:|---:|
| one-mirror-emission | 576 | 47 | 47 | 0 | 529 |
| eighth-scale-one-mirror-emission | 192 | 17 | 17 | 0 | 175 |
| two-mirrors-emission | 576 | 33 | 29 | 4 | 547 |
| quarter-scale-shifted-yzx | 576 | 29 | 29 | 0 | 547 |
| four-scale-shifted-zxy | 192 | 192 | 192 | 0 | 0 |
| one-mirror-escape | 192 | 33 | 33 | 0 | 159 |
| seam-after-mirror | 192 | 15 | 15 | 0 | 177 |
| grazing-mirror-escape | 192 | 15 | 14 | 1 | 79 |
| near-threshold-mirror | 192 | 192 | 192 | 0 | 0 |
| below-near-mirror | 192 | 192 | 192 | 0 | 0 |
| long-distance-mirror | 192 | 192 | 192 | 0 | 0 |
| unsupported-tilted-mirror | 192 | 192 | 192 | 0 | 0 |
| unsupported-decimal-world | 192 | 192 | 192 | 0 | 0 |
| half-scale-three-mirrors-negative-permutation | 576 | 53 | 53 | 0 | 523 |
| small-grid-two-mirrors | 192 | 10 | 10 | 0 | 182 |
| eight-scale-two-mirrors | 192 | 192 | 192 | 0 | 0 |

Unsupported tilted/decimal geometry, the bounded long-distance fixture and unsupported wave paths remain explicit CPU replay. The fixture declaration bytes are pinned to SHA-256 `e95b5afeb6615d15b813e271898e50bf2ca8bbd5a8eeac4fee3c774929af40f4`; their complete CPU summaries match v20.

## Native device reuse used for transport

The full correctness run explicitly uses the test-only `--reuse-native-device` option. The production runner is unchanged by this option. One real Dawn device is leased sequentially; shader modules and real pipelines are cached by the full exact source descriptor, entry point, specialization constants and automatic layout. All per-scene buffers, bind groups, queue submissions, dispatches, error scopes, readbacks and numerical execution remain native. Each runner releases its own buffers/listeners, with one final physical device destruction.

Observed cache accounting: 16 leases and releases, one physical adapter/device, 1 shader miss and 15 hits, 11 pipeline misses and 165 hits, one physical destroy. Pipeline creation wall time was 69.817 seconds; total reported shared setup was 70.507 seconds. This setup reuse is not a production renderer speed claim. The normal no-flag harness still creates genuine separate per-fixture devices.

The cache-control report covers infrastructure-only mocked lifecycle edges (late acquisition, unreleased/concurrent leases and exactly-once disposal); it supplies no numerical GPU evidence. Actual numerical evidence is exclusively the native transport report. Separate lifecycle suites retain ordinary physical-device ownership tests.

## Reproduction and evidence

- CPU fixture check: `node tests/plt-resident-tight-transport.mjs --cpu-only`
- Null-backend compile check: `node tests/plt-resident-tight-transport.mjs --reuse-native-device`
- Full native transport: `WEBGPU_EXECUTION_BACKEND=vulkan VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json node tests/plt-resident-tight-transport.mjs --execute --reuse-native-device`
- Omit `--reuse-native-device` to reproduce with separate devices and repeated compilation.
- `PLT_TIGHT_CASE=name` or `PLT_TIGHT_CASES=name1,name2` selects explicit existing fixtures and writes a case-labelled report, without overwriting full-suite evidence.
- Full results and source hashes: `plt-resident-tight-validation.json`
- Pinned baseline: `plt-resident-anchor-validation.json`, SHA-256 `e8410d3a98149bb53eed8d57e17d266fb9285abdabc52fa663cbb05c77a0adff`, durable commit `72fe8da5df129ab22aaa2940586d9811cb6face3`
- Initial large-world smoke reports are preserved as `plt-resident-tight-four-scale-shifted-zxy-initial-smoke-validation.json` and `plt-resident-tight-eight-scale-two-mirrors-initial-smoke-validation.json`; both showed zero complete retained paths.
- Final-source two-fixture native cache smoke: `plt-resident-tight-four-scale-shifted-zxy--eight-scale-two-mirrors-validation.json`
- Cache lifecycle controls: `plt-resident-tight-native-cache-control-validation.json`

The renderer wall times collected during correctness validation are not the warmed paired benchmark. The separate `plt-resident-tight-submit-timing-validation.json` measures four representative existing fixtures using uncached setup, per-mode warmups and balanced paired quartets. No physical-GPU, browser, larger-path-load or statistical speed claim is made.
