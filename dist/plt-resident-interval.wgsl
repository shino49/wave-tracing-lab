// Conservative scalar interval primitives; no bindings or entry points.
// Source: https://www.w3.org/TR/WGSL/#floating-point-evaluation (checked 2026-10-13).
// Contract: each valid RI encloses REAL arithmetic on the input f32 values.
// It does not enclose an independently rounded CPU/GPU expression, input upload
// error, geometry tolerances, or a separately evaluated normalize()/dot().
// Callers must propagate invalid != 0 as UNKNOWN, never as an empty interval.
//
// Endpoints are INTEGER bit patterns. Construct from raw source u32 storage,
// not bitcast<f32> then bitcast<u32>: WGSL even permits subnormal bitcasts to
// flush to zero. ri_point()/ri_bounds() reject unsupported raw patterns before
// any float reinterpretation. Accepted endpoints: zero or 2^-120 <= |x| <= 2^120.
// An interval spanning zero contains small real numbers; its endpoints alone
// must satisfy the supported range. Invalid results deliberately contain zeros.
//
// +,-,*: WGSL correctly-rounded means EITHER adjacent float, not nearest.
// One integer nextDown/nextUp therefore encloses the exact scalar result.
// /: WGSL permits 2.5 ULP, inverseSqrt: 2 ULP. Eight representable steps cover
// each: within these tiny error neighborhoods at most one binade boundary is
// crossed, whose spacing changes by two, so 2*ceil(2.5)+1 <= 8 steps suffice.
// sqrt uses an interval around inverseSqrt followed by interval reciprocal;
// no unjustified fixed-ULP assumption about native sqrt or normalize is used.
//
// Each arithmetic result is bitcast to u32 and adjusted in INTEGER space before
// reuse. This nonlinear bit operation is a semantic dependency, not a claim
// that a let, function boundary, or parenthesis prohibits reassociation. There
// is only one float arithmetic operator per scalar evaluation. Composite dot,
// norm, etc. must use these primitives, not unbounded reassociated expressions.
//
// Input bounds make add/sub overflow impossible. Integer exponent preflights
// conservatively exclude mul/div underflow/overflow BEFORE evaluation. Add/sub
// cancellation may underflow, but a non-proven zero or out-of-range output is
// invalidated. No test of NaN/Inf after potentially overflowing math is used.

struct RI { lo: u32, hi: u32, invalid: u32 }
struct RIVec3 { x: RI, y: RI, z: RI }
const RI_BAD_INPUT: u32 = 1u;
const RI_BAD_RANGE: u32 = 2u;
const RI_BAD_DOMAIN: u32 = 4u;
const RI_MIN_BITS: u32 = 0x03800000u; // 2^-120
const RI_MAX_BITS: u32 = 0x7b800000u; // 2^120
const RI_PAD_EXACT: u32 = 1u;
const RI_PAD_APPROX: u32 = 8u;

fn ri_unknown(reason: u32) -> RI { return RI(0u, 0u, reason | RI_BAD_INPUT); }
fn ri_supported(bits: u32) -> bool {
  let mag = bits & 0x7fffffffu;
  return mag == 0u || (mag >= RI_MIN_BITS && mag <= RI_MAX_BITS);
}
fn ri_canonical(bits: u32) -> u32 {
  if ((bits & 0x7fffffffu) == 0u) { return 0u; }
  return bits;
}
fn ri_lo(a: RI) -> f32 { return bitcast<f32>(a.lo); }
fn ri_hi(a: RI) -> f32 { return bitcast<f32>(a.hi); }
fn ri_bounds(lo: u32, hi: u32) -> RI {
  if (!ri_supported(lo) || !ri_supported(hi)) { return ri_unknown(RI_BAD_INPUT); }
  if (bitcast<f32>(lo) > bitcast<f32>(hi)) { return ri_unknown(RI_BAD_INPUT); }
  return RI(ri_canonical(lo), ri_canonical(hi), 0u);
}
fn ri_point(bits: u32) -> RI { return ri_bounds(bits, bits); }
fn ri_valid(a: RI) -> bool {
  if (a.invalid != 0u || !ri_supported(a.lo) || !ri_supported(a.hi)) { return false; }
  return ri_lo(a) <= ri_hi(a);
}
// Integer nextUp/down on NONZERO finite normal values, before conversion.
// Zero has no representable supported neighbor; known exact zeros are handled
// algebraically and any other zero is unknown. These are not generic IEEE APIs.
fn ri_next_up_bits(bits: u32, count: u32) -> u32 {
  if ((bits & 0x80000000u) == 0u) { return bits + count; }
  return bits - count;
}
fn ri_next_down_bits(bits: u32, count: u32) -> u32 {
  if ((bits & 0x80000000u) == 0u) { return bits - count; }
  return bits + count;
}
fn ri_pad(bits: u32, count: u32) -> RI {
  if (!ri_supported(bits) || (bits & 0x7fffffffu) == 0u) { return ri_unknown(RI_BAD_RANGE); }
  let lo = ri_next_down_bits(bits, count);
  let hi = ri_next_up_bits(bits, count);
  if (!ri_supported(lo) || !ri_supported(hi)) { return ri_unknown(RI_BAD_RANGE); }
  return RI(lo, hi, 0u);
}
fn ri_exponent(bits: u32) -> i32 { return i32((bits >> 23u) & 255u) - 127; }
fn ri_neg(a: RI) -> RI {
  if (!ri_valid(a)) { return ri_unknown(a.invalid); }
  return RI(ri_canonical(a.hi ^ 0x80000000u), ri_canonical(a.lo ^ 0x80000000u), 0u);
}
fn ri_scalar_add(a: u32, b: u32) -> RI {
  if (!ri_supported(a) || !ri_supported(b)) { return ri_unknown(RI_BAD_INPUT); }
  let aa = ri_canonical(a); let bb = ri_canonical(b);
  if (aa == 0u) { return ri_point(bb); }
  if (bb == 0u) { return ri_point(aa); }
  if (aa == (bb ^ 0x80000000u)) { return ri_point(0u); }
  return ri_pad(bitcast<u32>(bitcast<f32>(aa) + bitcast<f32>(bb)), RI_PAD_EXACT);
}
fn ri_scalar_mul(a: u32, b: u32) -> RI {
  if (!ri_supported(a) || !ri_supported(b)) { return ri_unknown(RI_BAD_INPUT); }
  if (ri_canonical(a) == 0u || ri_canonical(b) == 0u) { return ri_point(0u); }
  let e = ri_exponent(a) + ri_exponent(b);
  // |a*b| in [2^e, 2^(e+2)); leave output/padding range room.
  if (e < -120 || e > 118) { return ri_unknown(RI_BAD_RANGE); }
  return ri_pad(bitcast<u32>(bitcast<f32>(a) * bitcast<f32>(b)), RI_PAD_EXACT);
}
fn ri_scalar_div(a: u32, b: u32) -> RI {
  if (!ri_supported(a) || !ri_supported(b)) { return ri_unknown(RI_BAD_INPUT); }
  if (ri_canonical(b) == 0u) { return ri_unknown(RI_BAD_DOMAIN); }
  if (ri_canonical(a) == 0u) { return ri_point(0u); }
  let e = ri_exponent(a) - ri_exponent(b);
  // |a/b| in (2^(e-1), 2^(e+1)); |b| also meets WGSL division domain.
  if (e < -119 || e > 119) { return ri_unknown(RI_BAD_RANGE); }
  return ri_pad(bitcast<u32>(bitcast<f32>(a) / bitcast<f32>(b)), RI_PAD_APPROX);
}
fn ri_hull(a: RI, b: RI) -> RI {
  if (!ri_valid(a) || !ri_valid(b)) { return ri_unknown(a.invalid | b.invalid); }
  return RI(select(a.lo, b.lo, ri_lo(b) < ri_lo(a)),
            select(a.hi, b.hi, ri_hi(b) > ri_hi(a)), 0u);
}
fn ri_add(a: RI, b: RI) -> RI {
  if (!ri_valid(a) || !ri_valid(b)) { return ri_unknown(a.invalid | b.invalid); }
  let low = ri_scalar_add(a.lo, b.lo); let high = ri_scalar_add(a.hi, b.hi);
  if (low.invalid != 0u || high.invalid != 0u) { return ri_unknown(low.invalid | high.invalid); }
  return RI(low.lo, high.hi, 0u);
}
fn ri_sub(a: RI, b: RI) -> RI { return ri_add(a, ri_neg(b)); }
fn ri_mul(a: RI, b: RI) -> RI {
  if (!ri_valid(a) || !ri_valid(b)) { return ri_unknown(a.invalid | b.invalid); }
  return ri_hull(ri_hull(ri_scalar_mul(a.lo, b.lo), ri_scalar_mul(a.lo, b.hi)),
                 ri_hull(ri_scalar_mul(a.hi, b.lo), ri_scalar_mul(a.hi, b.hi)));
}
fn ri_div(a: RI, b: RI) -> RI {
  if (!ri_valid(a) || !ri_valid(b)) { return ri_unknown(a.invalid | b.invalid); }
  if (ri_lo(b) <= 0.0 && ri_hi(b) >= 0.0) { return ri_unknown(RI_BAD_DOMAIN); }
  return ri_hull(ri_hull(ri_scalar_div(a.lo, b.lo), ri_scalar_div(a.lo, b.hi)),
                 ri_hull(ri_scalar_div(a.hi, b.lo), ri_scalar_div(a.hi, b.hi)));
}
fn ri_square(a: RI) -> RI {
  if (!ri_valid(a)) { return ri_unknown(a.invalid); }
  let ends = ri_hull(ri_scalar_mul(a.lo, a.lo), ri_scalar_mul(a.hi, a.hi));
  if (ends.invalid != 0u) { return ends; }
  if (ri_lo(a) <= 0.0 && ri_hi(a) >= 0.0) { return RI(0u, ends.hi, 0u); }
  return ends;
}
fn ri_scalar_sqrt(a: u32) -> RI {
  if (!ri_supported(a)) { return ri_unknown(RI_BAD_INPUT); }
  if (ri_canonical(a) == 0u) { return ri_point(0u); }
  if ((a & 0x80000000u) != 0u) { return ri_unknown(RI_BAD_DOMAIN); }
  let inv = ri_pad(bitcast<u32>(inverseSqrt(bitcast<f32>(a))), RI_PAD_APPROX);
  return ri_div(ri_point(0x3f800000u), inv);
}
fn ri_sqrt(a: RI) -> RI {
  if (!ri_valid(a)) { return ri_unknown(a.invalid); }
  if (ri_lo(a) < 0.0) { return ri_unknown(RI_BAD_DOMAIN); }
  let low = ri_scalar_sqrt(a.lo); let high = ri_scalar_sqrt(a.hi);
  if (low.invalid != 0u || high.invalid != 0u) { return ri_unknown(low.invalid | high.invalid); }
  return RI(low.lo, high.hi, 0u);
}
fn ri_dot3(a: RIVec3, b: RIVec3) -> RI {
  return ri_add(ri_add(ri_mul(a.x, b.x), ri_mul(a.y, b.y)), ri_mul(a.z, b.z));
}
fn ri_length3(a: RIVec3) -> RI {
  return ri_sqrt(ri_add(ri_add(ri_square(a.x), ri_square(a.y)), ri_square(a.z)));
}
fn ri_normalize3(a: RIVec3) -> RIVec3 {
  let length = ri_length3(a);
  if (length.invalid != 0u || ri_lo(length) <= 0.0) {
    let bad = ri_unknown(length.invalid | RI_BAD_DOMAIN);
    return RIVec3(bad, bad, bad);
  }
  return RIVec3(ri_div(a.x, length), ri_div(a.y, length), ri_div(a.z, length));
}
