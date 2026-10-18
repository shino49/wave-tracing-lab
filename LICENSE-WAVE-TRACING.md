# Wave Tracing adaptation attribution

New PLT browser modules (`dist/plt-*.js`, `dist/plt-compute.wgsl`), the new
`tracer.html` experience and related new PLT documentation/tests are an
independently implemented adaptation of the published Wave Tracing algorithm
and audited native control flow. They are provided under Creative Commons
Attribution-NonCommercial 4.0 International:
https://creativecommons.org/licenses/by-nc/4.0/
https://creativecommons.org/licenses/by-nc/4.0/legalcode

Native work: wave_tracer, copyright Shlomi Steinberg.
https://github.com/ssteinberg/wave_tracer
Audited snapshot: cf532564e18e1f932fdc95b387d1e9142d8efc4c.
Paper: Wave Tracing: Generalizing The Path Integral To Wave Optics.
https://ssteinberg.xyz/2025/08/28/generalizing_the_path_integral/paper.pdf

This implementation changes language/runtime, geometry storage/BVH, envelope
shape (conservative circular specialization), source/camera interface, numerical
UTD evaluation and parallel GPU execution. It explicitly corrects scalar
weighted addition, source-leg visibility, matched constant-radiance NEE, and
rotation-invariant numerical tangent decisions, and uses a normalized wrapped
angular proposal. It retains the native center-connection/local-plane and
Keller-angle-measure approximations. It is not a byte-for-byte port or a full
implementation of every feature in the publication. There is no endorsement by
the original authors. No unknown upstream executable was run.

Pre-existing independent numerical kernels and other Site modes retain their
existing attribution and terms; the new adaptation does not assert ownership
of upstream work or change unrelated components' terms.
