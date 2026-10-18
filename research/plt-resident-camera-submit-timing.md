# Joint-camera paired warmed measurements

Build `plt-resident-joint-camera-2026-10-13.1`. Actual Dawn/Tint Vulkan SwiftShader execution after other planned native validation work had completed. All four scenarios pass; these are descriptive software-adapter wall times, not physical-GPU, browser or statistical performance evidence.

## Main result

Joint camera improves retained-path coverage, but this checkpoint does **not** demonstrate a speedup. The median paired camera-minus-anchor renderer deltas are −1.18 ms for ordinary two mirrors, +67.17 ms for 4× two mirrors, +1.59 ms for three mirrors, and +121.00 ms for 8× two mirrors. The two large-world cases are slower in this run despite fewer CPU replays.

Each scenario uses the same unchanged historical fixture, 8×8 frame, two samples, 128 paths, immutable initial RNG prefix, and whole-round submission schedule. The guard is the paired treatment. Each pair shares one camera-compiled runner; this is not a separately compiled historical-anchor setup comparison.

| Fixture | Replay / 128, anchor → camera | Retained mirrors, anchor → camera | Anchor renderer median ms | Camera renderer median ms | Paired delta median ms | Paired ratio median |
|---|---:|---:|---:|---:|---:|---:|
| two-mirrors-emission | 4 → 3 | 124 → 125 | 133.48 | 139.33 | -1.18 | 0.99 |
| four-scale-shifted-zxy | 128 → 5 | 0 → 123 | 131.76 | 230.70 | 67.17 | 1.51 |
| half-scale-three-mirrors-negative-permutation | 8 → 8 | 120 → 120 | 290.44 | 282.41 | 1.59 | 1.02 |
| eight-scale-two-mirrors | 128 → 97 | 0 → 31 | 90.42 | 225.36 | 121.00 | 2.42 |

The median of paired differences is calculated from the eight matched pairs, so it need not equal the difference of the two independent per-mode medians. Timing variation is substantial; the paired ranges below include both signs. No causal claim beyond the recorded workload/treatment comparison is made.

## Complete validation on every run

- Two warmups per mode and eight alternating AB/BA measured pairs per fixture.
- 32 measured pairs, 64 measured runs and 8,192 measured path-runs, plus 16 warmup runs and 2,048 warmup path-runs.
- Every warmup and measured run is checked against the complete independent CPU shadow with unchanged strict geometry/power tolerance, exact dyadic analytic power/image, exact RNG words/final states and complete replay ownership.
- Every anchor run preserves the exact historical v20/v21 first-batch replay/retention/reason/depth/round/dispatch checkpoint.
- Repeated runs of each guard produce identical complete audit, image, pixel ownership, final RNG, replay keys/reasons/depths, rounds and dispatch counts.
- Zero audit-discovered numerical geometry replacements. Caller RNG buffers remain unchanged.
- Timing starts only after the full frozen-source 24-fixture native transport report passes, including six in-domain and two explicit out-of-domain camera cases and the four wave controls.

## Setup, shadow and paired spread

No test-only device/pipeline reuse is enabled for this timing run. Each fixture creates and destroys its ordinary production runner/device and compiles its own pipelines. Total setup is 286.07 seconds and is excluded from renderer pairs. Independently constructed CPU shadows take 42.05 ms total; their construction is also excluded.

| Fixture | Setup seconds | CPU full-shadow construction ms | Paired camera − anchor range ms |
|---|---:|---:|---:|
| two-mirrors-emission | 75.62 | 16.00 | -111.88 … 68.08 |
| four-scale-shifted-zxy | 73.06 | 13.40 | -121.31 … 118.74 |
| half-scale-three-mirrors-negative-permutation | 69.47 | 7.79 | -129.05 … 116.50 |
| eight-scale-two-mirrors | 67.92 | 4.87 | -19.72 … 233.61 |

Renderer wall time includes native execution, dispatch submission/waits, status and output readbacks, decoding and selective CPU replay. Subsequent raw-shadow comparison and admission are excluded and measured separately. The following numbers are independent component medians; they are not an additive decomposition of the median renderer time. The complete report also records initialization submission/wait, observed outer run wall time and every run's raw component timings.

| Fixture / guard | Round submit/wait ms | Status readback ms | CPU selective replay ms | Replay readback ms | Final readback/decode ms | Raw shadow audit ms | Admission ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| two-mirrors-emission / anchor | 111.98 | 1.08 | 0.77 | 0.23 | 0.91 | 1.79 | 8.70 |
| two-mirrors-emission / camera | 113.99 | 1.18 | 0.53 | 0.18 | 0.84 | 1.43 | 8.97 |
| four-scale-shifted-zxy / anchor | 106.33 | 1.03 | 6.71 | 0.32 | 0.83 | 1.56 | 10.14 |
| four-scale-shifted-zxy / camera | 165.39 | 1.50 | 0.75 | 0.31 | 0.87 | 1.53 | 7.94 |
| half-scale-three-mirrors-negative-permutation / anchor | 258.83 | 1.50 | 0.72 | 0.13 | 0.89 | 1.81 | 10.63 |
| half-scale-three-mirrors-negative-permutation / camera | 239.41 | 1.53 | 0.73 | 0.15 | 0.89 | 2.05 | 9.87 |
| eight-scale-two-mirrors / anchor | 72.93 | 0.75 | 6.12 | 0.18 | 0.89 | 1.43 | 10.00 |
| eight-scale-two-mirrors / camera | 198.12 | 1.93 | 4.74 | 0.44 | 0.85 | 1.36 | 8.80 |

These host wall counters are not GPU timestamp queries. The workload is intentionally the same bounded 128-path historical workload; there are no larger-load results or hardware throughput claims. Lower replay counts do not imply lower end-to-end cost, and this run does not justify replacing the default backend or enabling unaudited display.

## Reproduction and provenance

Run other native jobs to completion first.

- CPU plan: `node tests/plt-resident-camera-submit-timing.mjs --cpu-only`
- Exact native command used: `WEBGPU_EXECUTION_BACKEND=vulkan VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json node tests/plt-resident-camera-submit-timing.mjs --execute`
- Native report: `research/plt-resident-camera-submit-timing-validation.json`, SHA-256 `2cac3f4773e1a207bc478330734a0299849ea8b0c3bcf83e3459a2ff4b319589`
- Full transport evidence: `research/plt-resident-camera-validation.json`, SHA-256 `32f0feae30e6a3ab6c0e7ebaecb843133cdbafbe7e4427b7519b8615e6e26979`
- Timing harness: `tests/plt-resident-camera-submit-timing.mjs`, SHA-256 `ca98df8b606fc09a373d639ea9189020e7c6b0b277c5d6ea23be89b7ff477777`
- Transport/helper harness: `tests/plt-resident-camera-transport.mjs`, SHA-256 `c60544a4e19b83f0b603b948f75f9f5471ccbdfdb65daae1089bb2a95d395d7e`

All shader and host hashes match the completed full transport snapshot, including the immutable camera-packing module. Source, host, both harnesses and the transport evidence report are checked again at the end of timing. The pinned v20 and v21 artifacts remain unchanged.
