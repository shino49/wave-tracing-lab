// The CPU supplies deterministic route amplitude and already-reduced phase.
// No geometry, large-argument phase reduction, FFT, or CPU shadow runs here.
struct Parameters {
  sampleCount: u32,
  routeCount: u32,
  reserved0: u32,
  reserved1: u32,
};
struct Safety {
  nonFinite: atomic<u32>,
  outsideDomain: atomic<u32>,
  inspected: atomic<u32>,
  reserved: atomic<u32>,
};
@group(0) @binding(0) var<uniform> p: Parameters;
// Sample-major, route-major: amplitude, reduced radians, reserved zero, zero.
@group(0) @binding(1) var<storage, read> records: array<vec4f>;
// Coherent intensity, incoherent intensity, real field, imaginary field.
@group(0) @binding(2) var<storage, read_write> outputFields: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> safety: Safety;
fn finite(v: f32) -> bool {
  return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u;
}
// Short-interval Taylor cis uses the same controlled polynomial as the
// coherent boundary solver. Native WGSL sin/cos has adapter-dependent error
// (~2e-4 observed on SwiftShader). Input is already reduced to [-pi, pi];
// this only selects a local quadrant, never reduces a large optical phase.
fn reduced_cis(theta: f32) -> vec2f {
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
@compute @workgroup_size(64)
fn accumulateFields(@builtin(global_invocation_id) id: vec3u) {
  let sample = id.x;
  if (sample >= p.sampleCount) { return; }
  var field = vec2f(0.0);
  var incoherent = 0.0;
  // Route order is fixed by the CPU geometry producer and is not randomized.
  for (var route = 0u; route < p.routeCount; route++) {
    let record = records[sample * p.routeCount + route];
    let amplitude = record.x;
    let phase = record.y;
    field += amplitude * reduced_cis(phase);
    incoherent += amplitude * amplitude;
  }
  let coherentIntensity = dot(field, field);
  let result = vec4f(coherentIntensity, incoherent, field.x, field.y);
  if (!(finite(result.x) && finite(result.y) && finite(result.z) && finite(result.w))) {
    atomicAdd(&safety.nonFinite, 1u);
  }
  // A conservative host bound keeps normal results far from f32 overflow.
  if (result.x < 0.0 || result.y < 0.0 || any(abs(result) > vec4f(1.0001e30))) {
    atomicAdd(&safety.outsideDomain, 1u);
  }
  outputFields[sample] = result;
  atomicAdd(&safety.inspected, 1u);
}
