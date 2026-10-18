// Hybrid coherent finite-screen Helmholtz application, time exp(+i omega t).
// The float64 coupled boundary solve is CPU-owned. This kernel applies the
// incident angular spectrum and the solved single-layer density to EACH pixel.
// G(r)=-i H0^(2)(kr)/4=(-Y0(kr)-i J0(kr))/4.
// No conditional-power multiplication, ray mask, or GPU linear-system solve.

struct Parameters {
  counts: vec4u, // targets, incident quadrature nodes, boundary nodes, factors
  physics: vec4f, // k, unused, unused, unused
  batch: vec4u, // first target, target count in this batch, unused, unused
};
struct ComplexRow {
  position: vec2f,
  value0: vec2f,
  value1: vec2f,
  value2: vec2f,
};
struct FieldRow {
  value0: vec2f,
  value1: vec2f,
  value2: vec2f,
  intensity: f32,
  valid: f32,
};
@group(0) @binding(0) var<uniform> parameters: Parameters;
@group(0) @binding(1) var<storage, read> incident: array<ComplexRow>;
@group(0) @binding(2) var<storage, read> boundary: array<ComplexRow>;
@group(0) @binding(3) var<storage, read> factors: array<vec2f>;
@group(0) @binding(4) var<storage, read> targets: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> output: array<FieldRow>;

fn complex_product(a: vec2f, b: vec2f) -> vec2f {
  return vec2f(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);
}

// Bounded Cody-Waite reduction plus short-interval Taylor polynomials avoids
// adapter-dependent large-argument sin/cos accuracy. The exact binary high part
// keeps cancellation controlled; |r| <= pi/4 and |theta| <= 1024 on the host.
fn coherent_cis(theta: f32) -> vec2f {
  let quadrant=i32(round(theta*0.63661977236758134));
  let q=f32(quadrant);
  let coarse=theta-q*1.5703125;
  let r=coarse-q*0.00048382679489661923;
  let z=r*r;
  let s=r*(1.0+z*(-0.16666666666666667+z*(0.008333333333333333+z*(-0.0001984126984126984+z*(0.0000027557319223985893-z*0.00000002505210838544172)))));
  let c=1.0+z*(-0.5+z*(0.041666666666666664+z*(-0.001388888888888889+z*(0.0000248015873015873+z*(-0.0000002755731922398589+z*0.00000000208767569878681)))));
  switch (quadrant&3) {
    case 0: { return vec2f(c,s); }
    case 1: { return vec2f(-s,c); }
    case 2: { return vec2f(-c,-s); }
    default: { return vec2f(s,-c); }
  }
}

// Piecewise rational/asymptotic order-zero Bessel approximations. The bounded
// application domain is checked on the host (1e-4 <= kr <= 1024). The f32
// implementation is independently exercised through native WGSL readback.
fn bessel_jy0(x: f32) -> vec2f {
  if (x < 8.0) {
    let y=x*x;
    let jp=57568490574.0+y*(-13362590354.0+y*(651619640.7+y*(-11214424.18+y*(77392.33017+y*(-184.9052456)))));
    let jq=57568490411.0+y*(1029532985.0+y*(9494680.718+y*(59272.64853+y*(267.8532712+y))));
    let j=jp/jq;
    let yp=-2957821389.0+y*(7062834065.0+y*(-512359803.6+y*(10879881.29+y*(-86327.92757+y*228.4622733))));
    let yq=40076544269.0+y*(745249964.8+y*(7189466.438+y*(47447.26470+y*(226.1030244+y))));
    return vec2f(j,yp/yq+0.63661977236758134*j*log(x));
  }
  let z=8.0/x;
  let y=z*z;
  let phase=x-0.7853981633974483;
  let p=1.0+y*(-0.001098628627+y*(0.00002734510407+y*(-0.000002073370639+y*0.0000002093887211)));
  let q=-0.01562499995+y*(0.0001430488765+y*(-0.000006911147651+y*(0.0000007621095161-y*0.0000000934945152)));
  let scale=sqrt(0.63661977236758134/x);
  let cs=coherent_cis(phase);
  let c=cs.x;
  let s=cs.y;
  return scale*vec2f(c*p-z*s*q,s*p+z*c*q);
}

@compute @workgroup_size(64)
fn coherent_main(@builtin(global_invocation_id) invocation: vec3u) {
  if (invocation.x >= parameters.batch.y) { return; }
  let index=parameters.batch.x+invocation.x;
  if (index >= parameters.counts.x) { return; }
  let point=targets[index];
  if (point.z != 1.0) {
    output[index]=FieldRow(vec2f(0.0),vec2f(0.0),vec2f(0.0),0.0,0.0);
    return;
  }
  var u0=vec2f(0.0);
  var u1=vec2f(0.0);
  var u2=vec2f(0.0);
  for (var j=0u; j<parameters.counts.y; j++) {
    let row=incident[j];
    let phase=-dot(row.position,point.xy);
    let wave=coherent_cis(phase);
    u0+=complex_product(wave,row.value0);
    u1+=complex_product(wave,row.value1);
    u2+=complex_product(wave,row.value2);
  }
  for (var j=0u; j<parameters.counts.z; j++) {
    let row=boundary[j];
    let kr=parameters.physics.x*length(point.xy-row.position);
    let jy=bessel_jy0(kr);
    let green=-0.25*vec2f(jy.y,jy.x);
    u0+=complex_product(green,row.value0);
    u1+=complex_product(green,row.value1);
    u2+=complex_product(green,row.value2);
  }
  // rho = B B*, in a flux-orthonormal source basis. Host folds the raw-to-
  // orthonormal transform into these factors; each column is a COHERENT sum.
  var intensity=0.0;
  for (var c=0u; c<parameters.counts.w; c++) {
    let field=complex_product(u0,factors[c])
      +complex_product(u1,factors[parameters.counts.w+c])
      +complex_product(u2,factors[2u*parameters.counts.w+c]);
    intensity+=dot(field,field);
  }
  output[index]=FieldRow(u0,u1,u2,intensity,1.0);
}
