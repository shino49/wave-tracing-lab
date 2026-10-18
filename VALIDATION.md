# Public source validation

Baseline: v38. This publication is a sanitized source snapshot with a fresh
public history. Browser source in `dist/` and test source in `tests/` are copied
byte-for-byte from the saved v38 baseline. Private deployment state, previous
history, screenshots, large standalone render dumps, logs and unrelated research
artifacts are excluded. Small numerical image fixtures used by tests remain. Required numerical fixtures and reference programs are
retained. SOURCE-MANIFEST.json records byte counts and SHA-256 hashes for all
published files except the manifest itself.

## Checks repeated on 2026-10-17

Node.js 24.19.0; pinned Happy DOM 20.14.6, wgsl_reflect 1.4.0, webgpu 0.6.2.
Tests execute in a separate copy so generated reports cannot alter the snapshot.

- `npm test`: passed; numerical fixtures and conservative-query checks.
- `npm run test:dom`: passed; simulated DOM and explicit CPU fallback.
- `npm run test:coherent-mesh-slab`: passed; core, camera, runtime, scene,
  spectral shader/mock checks and transport DOM suite.
- `npm run test:coherent-mesh-slab-mirror`: passed; bounded mirror/slab CPU
  routes and independent Fermat/ray-tube/reference checks.

These checks do not establish physical GPU execution, browser visual quality,
or hardware performance. Native WebGPU suites require a supported backend;
unavailable or skipped adapters must be reported as such. Historical numerical
baselines are inputs for comparison, not a claim of fresh hardware validation.

The renderer remains a bounded scalar model. See README.md and the method pages
for unsupported path classes and licensing limitations. Public source access
does not broaden those physical or legal claims.

## Known baseline limitations

The optional `npm run test:coherent` suite reaches a stale navigation assertion
in `tests/coherent-dom.mjs`: the page now includes `transport.html`, while the
old expected-link list omits it. This is a recorded baseline test failure; test
source is preserved unchanged and the suite is not reported as passed.

Resident timing/provenance tests checksum historical report, documentation and
source fixtures. These files are retained unchanged, including original timing
and generic software-adapter metadata. They are historical evidence, not new
hardware measurements. Running transport suites may overwrite their report
paths; use fresh disposable copies or restore fixtures between suites.

Optional spatial `--mono-only` and camera-mirror `final-fast` modes resume prior
runs. Generate their prior reports/render files with the normal mode first;
large historical render binaries are intentionally not included.
