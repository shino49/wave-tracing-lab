// Mulberry32: raw words are exactly the existing sequential CPU stream.
// Float conversion is explicitly [0,1); discrete decisions use multiply-high.
var<private> rr_rng_count: u32;
var<private> rr_rng_hash: u32;
var<private> rr_rng_words: array<u32,64>;
fn rr_word(state: ptr<function,u32>) -> u32 {
  *state += 0x6D2B79F5u;
  var t = *state;
  t = (t ^ (t >> 15u)) * (t | 1u);
  t ^= t + (t ^ (t >> 7u)) * (t | 61u);
  let word = t ^ (t >> 14u);
  if (rr_rng_count < 64u) { rr_rng_words[rr_rng_count] = word; }
  else { rg_flags |= 16u; }
  rr_rng_count += 1u;
  rr_rng_hash = (rr_rng_hash ^ word) * 16777619u;
  return word;
}
fn rr_float(state: ptr<function,u32>) -> f32 {
  return min(f32(rr_word(state)) * 0x1p-32, bitcast<f32>(0x3f7fffffu));
}
fn rr_choice(state: ptr<function,u32>, count: u32) -> u32 {
  if (count == 0u || count > 65536u) { rg_flags |= 16u; return 0u; }
  let word = rr_word(state);
  return (((word >> 16u) * count) + (((word & 65535u) * count) >> 16u)) >> 16u;
}
