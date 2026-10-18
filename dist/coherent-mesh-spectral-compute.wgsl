// Prepend unchanged validated v16 utd-functions.wgsl. Geometry and sampled
// wavelengths are host inputs; every wavelength-dependent field is GPU-only.
// Distinct wavelengths NEVER interfere: the second pass reads intensities.
struct Parameters { sampleCount: u32, routeCount: u32, wavelengthCount: u32, sharedSpectrum: u32 };
struct SpectralSafety { nonFinite: atomic<u32>, outsideDomain: atomic<u32>, inspectedWavelengths: atomic<u32>, inspectedSamples: atomic<u32> };
struct SpectralRoute { field: vec4f, wedge: vec4f, medium: vec4f, deltas: vec4f };
struct SpectralNode { wave: vec4f, cie: vec4f };
@group(0) @binding(0) var<uniform> p: Parameters;
@group(0) @binding(1) var<storage, read> routes: array<SpectralRoute>;
@group(0) @binding(2) var<storage, read> wavelengths: array<SpectralNode>;
@group(0) @binding(3) var<storage, read_write> spectralFields: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> spectralOutput: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> spectralSafety: SpectralSafety;
fn spectral_finite(x: f32) -> bool { return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u; }
fn inspect_spectral(value: vec4f) {
  if (!(spectral_finite(value.x) && spectral_finite(value.y) && spectral_finite(value.z) && spectral_finite(value.w))) { atomicAdd(&spectralSafety.nonFinite, 1u); }
  if (any(value < vec4f(0.0)) || any(value > vec4f(1.0001e30))) { atomicAdd(&spectralSafety.outsideDomain, 1u); }
}
fn spectral_node(sample: u32, wavelength: u32) -> SpectralNode {
  return wavelengths[select(sample * p.wavelengthCount + wavelength, wavelength, p.sharedSpectrum == 1u)];
}
// Each factor is split into <=12-bit significands by clearing its low bits.
// Their four products are exact f32 values. Reduce EACH exact product to its
// fractional cycles before summing; do not subtract a huge rounded phase.
// This does not rely on fma being fused, nor on backend large-angle sin/cos.
fn centered_cycles(x: f32) -> f32 { return x - round(x); }
fn exact_product_cycles(a: f32, b: f32) -> f32 {
  let ah = bitcast<f32>(bitcast<u32>(a) & 0xfffff000u);
  let al = a - ah;
  let bh = bitcast<f32>(bitcast<u32>(b) & 0xfffff000u);
  let bl = b - bh;
  let t0 = centered_cycles(ah * bh);
  let t1 = centered_cycles(ah * bl);
  let t2 = centered_cycles(al * bh);
  let t3 = centered_cycles(al * bl);
  return centered_cycles(centered_cycles(t0 + t1) + centered_cycles(t2 + t3));
}
fn optical_cycles(excess: vec2f, inverseWavelength: vec2f) -> f32 {
  let a = exact_product_cycles(excess.x, inverseWavelength.x);
  let b = exact_product_cycles(excess.x, inverseWavelength.y);
  let c = exact_product_cycles(excess.y, inverseWavelength.x);
  let d = exact_product_cycles(excess.y, inverseWavelength.y);
  return centered_cycles(centered_cycles(a + b) + centered_cycles(c + d));
}
// Local Taylor cis, bounded |theta| <=2pi. Only quadrant reduction remains.
// Same controlled polynomial as the existing coherentValue-field native runner.
fn spectral_cis(theta: f32) -> vec2f {
  let quadrant = i32(round(theta * 0.63661977236758134));
  let q = f32(quadrant);
  let coarse = theta - q * 1.5703125;
  let r = coarse - q * 0.00048382679489661923;
  let z = r * r;
  let s = r * (1.0 + z * (-0.16666666666666667 + z * (0.008333333333333333 + z * (-0.0001984126984126984 + z * (0.0000027557319223985893 - z * 0.00000002505210838544172)))));
  let c = 1.0 + z * (-0.5 + z * (0.041666666666666664 + z * (-0.001388888888888889 + z * (0.0000248015873015873 + z * (-0.0000002755731922398589 + z * 0.00000000208767569878681)))));
  switch (quadrant & 3) {
    case 0: { return vec2f(c, s); }
    case 1: { return vec2f(-s, c); }
    case 2: { return vec2f(-c, -s); }
    default: { return vec2f(s, -c); }
  }
}
// Host supplied stable geometric deltas preserve the CPU's 1e-12 seam
// convention. Only exactly-zero delta is symmetric-zero; no 2e-6 widening.
// All wavelength-dependent UTD factors reuse the validated scalar v16 math.
fn spectral_cot_transition(delta: f32, n: f32, kL: f32, side: f32) -> vec2f {
  if (delta == 0.0) { return vec2f(0.0); }
  let t = delta * 0.5; let h = delta / (2.0 * n);
  let ratio = sign(delta) * n * utd_sinc(t) * utd_cos_small(h) / utd_sinc(h);
  let rootKL = sqrt(kL); let u = sqrt(2.0) * rootKL * abs(t * utd_sinc(t));
  let inv = 1.0 / (1.0 + u);
  return side * sqrt(2.0 * UTD_PI) * (rootKL * inv) * ratio *
    utd_cmul(vec2f(UTD_SQRT_HALF), utd_erfcx_scaled(u * inv));
}
fn spectral_utd(k: f32, record: SpectralRoute) -> vec2f {
  let n = record.wedge.y;
  if (n == 1.0) { return vec2f(0.0); }
  let kL = k * record.medium.x;
  let a = spectral_cot_transition(record.deltas.x, n, kL, 1.0) + spectral_cot_transition(record.deltas.y, n, kL, -1.0);
  let b = spectral_cot_transition(record.deltas.z, n, kL, 1.0) + spectral_cot_transition(record.deltas.w, n, kL, -1.0);
  return -utd_cmul(vec2f(UTD_SQRT_HALF, -UTD_SQRT_HALF), a + record.medium.z * b) /
    (2.0 * n * sqrt(2.0 * UTD_PI * k) * record.medium.y);
}
@compute @workgroup_size(64)
fn accumulateSpectralFields(@builtin(global_invocation_id) id: vec3u) {
  let sample = id.x; let wavelength = id.y;
  if (sample >= p.sampleCount || wavelength >= p.wavelengthCount) { return; }
  let node = spectral_node(sample, wavelength);
  var field = vec2f(0.0); var incoherent = 0.0; var geometric = 0.0;
  for (var r = 0u; r < p.routeCount; r++) {
    let record = routes[sample * p.routeCount + r];
    if (record.wedge.x == 0.0 || record.field.x == 0.0) { continue; }
    let phase = record.field.y - 6.283185307179586 * optical_cycles(record.field.zw, node.wave.xy);
    var routeField = record.field.x * spectral_cis(phase);
    if (record.wedge.x == 2.0) { routeField = utd_cmul(routeField, spectral_utd(node.wave.z, record)); }
    if (record.wedge.x == 3.0) { routeField *= exp(-0.5 * node.cie.w * record.wedge.y); }
    let routeIntensity = dot(routeField, routeField);
    field += routeField;
    incoherent += routeIntensity;
    if (record.wedge.x == 1.0 || record.wedge.x == 3.0) { geometric += routeIntensity; }
  }
  let coherentIntensity = dot(field, field);
  let result = vec4f(coherentIntensity, incoherent, geometric, 0.0);
  inspect_spectral(result);
  spectralFields[sample * p.wavelengthCount + wavelength] = result;
  atomicAdd(&spectralSafety.inspectedWavelengths, 1u);
}
@compute @workgroup_size(64)
fn integrateSpectralIntensities(@builtin(global_invocation_id) id: vec3u) {
  let sample = id.x;
  if (sample >= p.sampleCount) { return; }
  var coherentValue = vec4f(0.0); var incoherent = vec4f(0.0); var geometric = vec4f(0.0);
  for (var wavelength = 0u; wavelength < p.wavelengthCount; wavelength++) {
    let node = spectral_node(sample, wavelength);
    let weight = node.wave.w * vec4f(node.cie.xyz, 1.0);
    let intensity = spectralFields[sample * p.wavelengthCount + wavelength];
    coherentValue += intensity.x * weight;
    incoherent += intensity.y * weight;
    geometric += intensity.z * weight;
  }
  inspect_spectral(coherentValue); inspect_spectral(incoherent); inspect_spectral(geometric);
  spectralOutput[sample * 3u] = coherentValue;
  spectralOutput[sample * 3u + 1u] = incoherent;
  spectralOutput[sample * 3u + 2u] = geometric;
  atomicAdd(&spectralSafety.inspectedSamples, 1u);
}
