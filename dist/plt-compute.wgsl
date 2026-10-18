// Prepend the unchanged utd-functions.wgsl include.
// CPU owns general 3D BVH traversal, sampling, topology, and edge admission.
// These passes evaluate independent LOCAL coherent event operators, then the
// stochastic camera estimator. There is no cross-path coherent field sum.
struct PLTParameters { counts: vec4u, batch: vec4u };
struct PLTEvent { edgeRange: vec4u, field: vec4f };
struct PLTEdge { angles: vec4f, distances: vec4f };
struct PLTEventOutput { fields: vec4f, powers: vec4f };
struct PLTTerm { header: vec4u, ids0: vec4u, ids1: vec4u };
@group(0) @binding(0) var<uniform> plt: PLTParameters;
@group(0) @binding(1) var<storage, read> events: array<PLTEvent>;
@group(0) @binding(2) var<storage, read> edges: array<PLTEdge>;
@group(0) @binding(3) var<storage, read_write> eventResults: array<PLTEventOutput>;
@group(0) @binding(4) var<storage, read> terms: array<PLTTerm>;
@group(0) @binding(5) var<storage, read> pixelRanges: array<vec2u>;
@group(0) @binding(6) var<storage, read_write> camera: array<f32>;

@compute @workgroup_size(64)
fn plt_events(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= plt.batch.y) { return; }
  let index = plt.batch.x + id.x;
  if (index >= plt.counts.x) { return; }
  let event = events[index];
  var soft = vec2f(event.field.x, 0.0);
  var hard = soft;
  for (var j = 0u; j < event.edgeRange.y; j++) {
    let edge = edges[event.edgeRange.x+j];
    let a = edge.angles;
    let b = edge.distances;
    let phase = utd_cis(-a.x*b.z) / sqrt(b.x);
    soft += utd_cmul(utd_coefficient(a.x,a.y,a.z,a.w,b.w,b.y,-1.0),phase);
    hard += utd_cmul(utd_coefficient(a.x,a.y,a.z,a.w,b.w,b.y, 1.0),phase);
  }
  let sp = dot(soft,soft);
  let hp = dot(hard,hard);
  var selected = 0.5*(sp+hp);
  if (event.edgeRange.z == 0u) { selected = sp; }
  if (event.edgeRange.z == 1u) { selected = hp; }
  eventResults[index].fields = vec4f(soft,hard);
  eventResults[index].powers = vec4f(selected,sp,hp,f32(event.edgeRange.y));
}

@compute @workgroup_size(64)
fn plt_camera(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= plt.batch.y) { return; }
  let pixel = plt.batch.x + id.x;
  if (pixel >= plt.counts.y) { return; }
  let range = pixelRanges[pixel];
  var sum = 0.0;
  var correction = 0.0;
  for (var j = 0u; j < range.y; j++) {
    let term = terms[range.x+j];
    var value = bitcast<f32>(term.header.x);
    for (var depth = 0u; depth < term.header.y; depth++) {
      var eventId = 0u;
      if (depth < 4u) { eventId = term.ids0[depth]; }
      else { eventId = term.ids1[depth-4u]; }
      value *= eventResults[eventId].powers.x;
    }
    // Compensated sum bounds order-dependent error for long pixel term lists.
    let adjusted = value - correction;
    let next = sum + adjusted;
    correction = (next-sum)-adjusted;
    sum = next;
  }
  camera[pixel] = sum / f32(plt.counts.z);
}
