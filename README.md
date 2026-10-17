# Wave Tracing Lab

Version 38 public source baseline of a static browser wave-optics laboratory.
The main experience is `dist/transport.html`: a progressive, coherent spectral
mesh renderer with native 512 × 512 output, bounded mirror transport, and a
finite dielectric slab model. The interface and method pages are in Chinese.

## Run locally

Use Node.js 20 or newer for tests (baseline checked with Node.js 24), and Python
3 for a local static server:

```sh
npm ci
python3 -m http.server 8000 --directory dist
```

Open http://localhost:8000/transport.html. No application build step is needed:
`dist/` contains the editable browser source and deployment assets. A static host
must serve JavaScript modules and WGSL files normally; HTTPS or localhost is
required for WebGPU. CPU fallback is explicitly identified. There is no server
renderer, remote GPU service, analytics endpoint, or runtime package install.

Other entries include `index.html` (double slit), `mesh.html` (UTD),
`coherent.html` (bounded 2-D Helmholtz reference), `tracer.html` (PLT),
`wave-demo.html`, `pinhole.html`, and the opt-in `resident-test.html`.

## Physics scope

The v38 source route builder handles one mirror followed by one slab passage,
or one slab passage followed by one mirror, using planar unfolding and a
stationary broad-face slab solution. Finite patches, physical-leg visibility,
ray-tube spreading, scalar s-Fresnel transmission, mirror phase, and spectral
Beer field loss enter the coherent sum. Examples are in `dist/examples/`.

This is a bounded scalar model, not a complete electromagnetic solver or a full
port of the native Wave Tracing system. It excludes coherent slab internal
reflection/Fabry–Pérot sums, two-mirror slab crossings, and slab-crossing UTD
diffraction. Unsupported paths report model-incomplete diagnostics. Native
Dawn/SwiftShader execution is software execution and does not establish browser
or physical-GPU correctness or performance. See the method pages in `dist/` for
mode-specific assumptions and attribution.

## Checks

```sh
npm test
npm run test:dom
npm run test:coherent-mesh
npm run test:coherent-mesh-slab
npm run test:coherent-mesh-slab-mirror
```

The complete script catalog is in `package.json`. Mathematical reference tests
may need Python packages from `requirements-test.txt`; install them into a
virtual environment. Native WebGPU suites use the pinned `webgpu` development
package and need a supported backend. An unavailable adapter is not a passed GPU
execution test. Browser-like DOM tests use Happy DOM; actual browser/hardware QA
is separate. Some historical timing suites consume retained numerical baselines;
regenerate their prerequisite tests before comparing a new environment.

`research/` contains independent reference scripts, fixtures and selected
numerical baselines required by the tests. Generated render arrays, screenshots,
logs and machine-specific deployment state are excluded. Test scripts may write
local reports to this directory; review deliberate fixture updates explicitly.

## Repository workflow

`main` is the stable, tested baseline; `dot/dev` is the assistant development
branch. Completed local development changes are committed and pushed to
`dot/dev`; meaningful tested milestones are reviewed through a pull request and
merged into `main`. See [CONTRIBUTING.md](CONTRIBUTING.md). This is a development
workflow, not a promise of a background filesystem watcher.

## Licensing and attribution

This is a mixed-license source distribution, not a blanket MIT release:

- Wave Tracing PLT adaptation: CC BY-NC 4.0. See
  [LICENSE-WAVE-TRACING.md](LICENSE-WAVE-TRACING.md) for attribution and scope.
- CIE datasets and their numeric JavaScript reformat: CC BY-SA 4.0. See
  [LICENSE-CIE-DATA.md](LICENSE-CIE-DATA.md).
- Cephes/SciPy Bessel coefficient adaptation: BSD-3-Clause notice retained in
  `dist/coherent-core.js`.
- Development dependencies retain their own licenses, recorded in the lockfile
  and distributed packages. They are not vendored here.

No additional license grant is asserted for components whose existing terms
are unspecified. Public availability does not grant commercial rights to the
CC BY-NC material. Preserve relevant notices when redistributing.
