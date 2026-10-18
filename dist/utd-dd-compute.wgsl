// EXPERIMENTAL selected two-region conditional local-plane UTD operator.
// Prepend the unchanged utd-functions.wgsl include. CPU geometry supplies
// native region-centre anchored angles/ranges and the admission mask.
// GPU executes both coefficient stages and multiplies their conditional powers;
// it does not traverse cones, discover regions, or establish geometric admission.
// No cross-path phase, coherent sum across families, or calibrated irradiance.
struct DDRecord {
  angles: vec4f,     // phi_i1, phi_o1, phi_i2, phi_o2
  medium: vec4f,     // k, ro1 (also L1), ro2 (also L2), boundaryR
  admission: vec4f,  // x = exactly 1 admitted, 0 invalid; remaining lanes reserved
}
@group(0) @binding(0) var<storage, read> dd_records: array<DDRecord>;
// power, stage1 conditional power, stage2 conditional power, admitted flag.
@group(0) @binding(1) var<storage, read_write> dd_output: array<vec4f>;

@compute @workgroup_size(64)
fn dd_main(@builtin(global_invocation_id) invocation: vec3u) {
  let i = invocation.x;
  if (i >= arrayLength(&dd_records)) { return; }
  let record = dd_records[i];
  if (record.admission.x != 1.0) {
    dd_output[i] = vec4f(0.0);
    return;
  }
  let k = record.medium.x;
  let ro1 = record.medium.y;
  let ro2 = record.medium.z;
  let boundaryR = record.medium.w;
  let stage1 = utd_coefficient(k, 2.0, record.angles.x, record.angles.y, ro1, 1.0, boundaryR) / sqrt(ro1);
  let stage2 = utd_coefficient(k, 2.0, record.angles.z, record.angles.w, ro2, 1.0, boundaryR) / sqrt(ro2);
  let power1 = dot(stage1, stage1);
  let power2 = dot(stage2, stage2);
  dd_output[i] = vec4f(power1 * power2, power1, power2, 1.0);
}
