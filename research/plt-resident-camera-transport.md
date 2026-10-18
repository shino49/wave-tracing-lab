# Joint-camera transport checkpoint

Build `plt-resident-joint-camera-2026-10-13.1`. Actual Dawn/Tint Vulkan SwiftShader execution. This is not browser or physical-GPU validation.

## Result

All 24 fixtures pass both `anchor-round` and opt-in `camera-round`: the identical 16-fixture/4,608-path historical corpus, 4,563 additional adversarial paths, and four unsupported-wave controls. This is 9,175 distinct reference paths and 18,350 native mode-path executions.

Every native candidate is compared against its complete independent CPU shadow before admission. The existing strict geometry/power tolerances are unchanged. Dyadic analytic path powers and complete analytic images are exact where specified, RNG words and persistent final prefixes match exactly, replay ownership is complete and unique, and no audit-discovered numerical replacements occur. Both modes keep caller RNG input immutable.

The historical anchor baseline is preserved at every fixture/seed/batch checkpoint, including reason/depth histograms, rounds and dispatches: 1,596 replays and 2,913 retained mirror paths. Joint camera yields 1,349 replays and 3,160 retained mirror paths on that same 4,608-path corpus, reducing replay by 247 against anchor (242 against the preserved v21 tight-round result of 1,591). No historical fixture regresses.

The 4× world case improves from 0 to 180 retained mirror paths out of 192; the 8× case improves from 0 to 49 out of 192. The latter still replays 143 paths. The ordinary two-mirror case improves by four and grazing escape by fourteen. The three-mirror case remains unchanged at 53 replays, so the broader all-multibounce-improvement goal is not fully achieved.

Across the entire additional 4,563-path adversarial suite, camera replay increases from 801 to 1,802 (1,001 more), because the two explicitly unsupported-aspect controls add 1,442 replays. Separately, the six in-domain cameras account for 3,027 paths and improve from 707 to 266 replays (441 fewer). Both totals are reported, rather than selecting only supported-camera gains.

## Remaining replay diagnostics

All residual replays in these three representative fixtures carry the transport reason `axisIntervalUncertain`. The observed histograms are:

| Fixture | Residual replays | Interval reason-code histogram | Trigger-depth histogram |
|---|---:|---|---|
| 4× shifted two mirrors | 12 | 10: 8; 1152: 4 | depth 1: 8; depth 2: 4 |
| 8× two mirrors | 143 | 10: 10; 1152: 133 | depth 1: 10; depth 2: 133 |
| Half-scale three mirrors | 53 | 10: 53 | depth 0: 53 |

Code 10 is `RA_INTERVAL | RA_BARYCENTRIC`; code 1152 is `RA_NUMERIC | RA_NUMERIC_POINT`. These are separate recorded reason/depth histograms, not a newly measured joint distribution. In particular, the 8× residual is dominated by the numeric-point rejection code, and the three-mirror fixture fails the barycentric interval check at its initial query. No further root-cause or precision-improvement claim follows without a focused diagnostic.

## Exact fixture results

Positive reductions mean fewer complete CPU replays. Historical and additional cases remain separate in the machine-readable totals.

| Fixture | Paths | Anchor replay | Camera replay | Replay reduction | Camera retained mirrors |
|---|---:|---:|---:|---:|---:|
| one-mirror-emission | 576 | 47 | 47 | 0 | 529 |
| eighth-scale-one-mirror-emission | 192 | 17 | 17 | 0 | 175 |
| two-mirrors-emission | 576 | 33 | 29 | 4 | 547 |
| quarter-scale-shifted-yzx | 576 | 29 | 29 | 0 | 547 |
| four-scale-shifted-zxy | 192 | 192 | 12 | 180 | 180 |
| one-mirror-escape | 192 | 33 | 33 | 0 | 159 |
| seam-after-mirror | 192 | 15 | 15 | 0 | 177 |
| grazing-mirror-escape | 192 | 15 | 1 | 14 | 92 |
| near-threshold-mirror | 192 | 192 | 192 | 0 | 0 |
| below-near-mirror | 192 | 192 | 192 | 0 | 0 |
| long-distance-mirror | 192 | 192 | 192 | 0 | 0 |
| unsupported-tilted-mirror | 192 | 192 | 192 | 0 | 0 |
| unsupported-decimal-world | 192 | 192 | 192 | 0 | 0 |
| half-scale-three-mirrors-negative-permutation | 576 | 53 | 53 | 0 | 523 |
| small-grid-two-mirrors | 192 | 10 | 10 | 0 | 182 |
| eight-scale-two-mirrors | 192 | 192 | 143 | 49 | 49 |
| distinct-frame-square | 192 | 21 | 21 | 0 | 171 |
| distinct-frame-wide-17x9 | 459 | 31 | 31 | 0 | 428 |
| distinct-frame-tall-9x17 | 459 | 67 | 67 | 0 | 392 |
| minimum-fov-wide-31x8 | 744 | 87 | 87 | 0 | 657 |
| maximum-fov-tall-8x31 | 744 | 457 | 16 | 441 | 728 |
| extreme-jitter-13x11 | 429 | 44 | 44 | 0 | 385 |
| outside-camera-wide-aspect | 768 | 94 | 768 | -674 | 0 |
| outside-camera-tall-aspect | 768 | 0 | 768 | -768 | 0 |

## Camera coverage and fail-closed controls

Six additional in-domain fixtures use nine distinct, nonzero camera-frame components, so axis-aligned or repeated-component assumptions cannot satisfy this coverage. They cover 8×8, 17×9, 9×17, 31×8, 8×31 and 13×11 frames, FOV endpoints 5° and 120°, and both wide/tall non-power-of-two dimensions. Every in-domain fixture must retain at least one GPU-proven mirror path to pass; universal replay is not accepted as coverage.

The 13×11 fixture uses actual sequential Mulberry32 prefixes yielding raw camera jitter words 62 and 4,294,967,275 on each draw. The latter rounds to 1.0f before the GPU's [0,1) clamp, while CPU sampling uses the exact word/2³² value. Both low/high jitter endpoints are within 2⁻²⁵ of zero/one. These remain true GPU draws from uploaded RNG state, not CPU camera/path tapes. All 4,563 added paths have exact analytic radiance 4.

Aspect ratios 41/8 and 8/41 deliberately exceed the declared [0.25,4] camera domain. Both reject private camera provenance and replay all 768 paths, even though the old anchor guard retains many. These two intentional coverage restrictions are the only replay-count regressions. They must not be hidden by combining the adversarial and historical totals.

Four wave paths replay completely in both modes. Unsupported source geometry, bounded long-distance, and near-threshold historical cases remain fail-closed. Four synthetic decoded-history mutations (reflected origin, mirror direction, missing second mirror, stale sample owner) are rejected by both raw shadow comparison and admission. Those controls test the admission boundary, not private GPU proof-state integrity.

## Execution and timing scope

The explicit test-only native-device cache from the SHA-pinned v21 harness was used: one real adapter/device, 24 sequential runner leases, 24 releases, 11 real pipeline compilations, 253 exact pipeline-cache hits, and exactly one final device destruction. Every runner still creates/destroys its own scene/camera buffers and bind groups, and all dispatch, status, replay, readback and error-scope operations are genuine native operations. Production ownership is unchanged.

This correctness pass always executes anchor before camera and does not establish a speedup. Setup, CPU shadow construction, raw audit, admission, native renderer, readbacks and replay timings are recorded separately for diagnosis. Use `plt-resident-camera-submit-timing.mjs` for counterbalanced warmed pairs after other native work is idle. Both modes share the new camera-compiled runner, so setup is not a separately compiled historical-anchor comparison.

## Reproduction and immutable provenance

- CPU: `node tests/plt-resident-camera-transport.mjs --cpu-only`
- Native: `WEBGPU_EXECUTION_BACKEND=vulkan VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json node tests/plt-resident-camera-transport.mjs --execute --reuse-native-device`
- Validation: `research/plt-resident-camera-validation.json`, SHA-256 `32f0feae30e6a3ab6c0e7ebaecb843133cdbafbe7e4427b7519b8615e6e26979`
- Harness: `tests/plt-resident-camera-transport.mjs`, SHA-256 `c60544a4e19b83f0b603b948f75f9f5471ccbdfdb65daae1089bb2a95d395d7e`
- Guard SHA-256: `3eeb51b40f7a7288629e0d77b091bbec9596c2c9dd829a31cdbdef857353f54a`
- Walker SHA-256: `8cbee8c9d9a8c77e8d799e800385e9b64ecd055c2e72245e84b3eb91cecdb099`
- Host SHA-256: `c1fefb77d5e059aef42e1591ab79c4bf589eadeafaea171890b1d74735f1bf21`
- Immutable camera packing SHA-256: `2ebc9b1e1e57c212198fa5ad776e76c90df58fe466ebd3ff5a9e4e1570942d10`

The v20 baseline and every artifact in the pinned v21 validation manifest remain byte-for-byte unchanged. Source/host/harness hashes are checked again at the end of transport execution. The earlier two-fixture report is retained as `plt-resident-camera-four-scale-shifted-zxy--eight-scale-two-mirrors-initial-smoke-validation.json`; it predates final source-domain hardening and is not the full frozen-source evidence.
