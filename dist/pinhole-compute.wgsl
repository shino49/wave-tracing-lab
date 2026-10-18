// Streaming complex-f32 radix-2 FFT. No spectral/PSF renormalization occurs here.
// Kernel layout is wrapped displacement: h[(dy mod M)*M+(dx mod M)].
struct Params {
  n: u32, m: u32, count: u32, span: u32,
  axis: u32, inverse: u32, channel: u32, mode: u32,
  weight: f32, tolerance: f32, pad0: f32, pad1: f32,
  pad2: vec4u,
};
struct Safety {
  nonFinite: atomic<u32>, negative: atomic<u32>, imaginary: atomic<u32>, roundoff: atomic<u32>,
  maxRoundoffBits: atomic<u32>, inspected: atomic<u32>, pad0: atomic<u32>, pad1: atomic<u32>,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> inputC: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> outputC: array<vec2f>;
@group(0) @binding(3) var<storage, read> realInput: array<f32>;
@group(0) @binding(4) var<storage, read> sourceFFT: array<vec2f>;
@group(0) @binding(5) var<storage, read_write> accumulator: array<f32>;
@group(0) @binding(6) var<storage, read_write> safety: Safety;
@group(0) @binding(7) var<storage, read> twiddles: array<vec2f>;
fn cmul(a: vec2f,b: vec2f) -> vec2f { return vec2f(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x); }
fn finiteValue(v:f32) -> bool { return (bitcast<u32>(v)&0x7f800000u)!=0x7f800000u; }
@compute @workgroup_size(64)
fn packReal(@builtin(global_invocation_id) id:vec3u) {
  let i=id.x; if(i>=p.count){return;} let x=i%p.m; let y=i/p.m;
  var v=0.0;
  if(p.mode==1u){v=realInput[i];} else if(x<p.n && y<p.n){v=realInput[y*p.n+x];}
  outputC[i]=vec2f(v,0.0);
}
@compute @workgroup_size(64)
fn bitReverse(@builtin(global_invocation_id) id:vec3u) {
  let i=id.x; if(i>=p.count){return;} let x=i%p.m;let y=i/p.m;
  var coord=select(x,y,p.axis==1u);var reversed=0u;var remaining=p.m;
  loop {if(remaining<=1u){break;}reversed=(reversed<<1u)|(coord&1u);coord>>=1u;remaining>>=1u;}
  let j=select(y*p.m+reversed,reversed*p.m+x,p.axis==1u);
  outputC[i]=inputC[j];
}
@compute @workgroup_size(64)
fn butterfly(@builtin(global_invocation_id) id:vec3u) {
  let i=id.x;if(i>=p.count){return;}let x=i%p.m;let y=i/p.m;
  let coord=select(x,y,p.axis==1u);let half=p.span/2u;let local=coord%p.span;let k=local%half;let base=coord-local;
  let ia=select(y*p.m+base+k,(base+k)*p.m+x,p.axis==1u);
  let ib=select(ia+half,ia+half*p.m,p.axis==1u);
  // Shared f32 twiddles avoid backend-dependent low-accuracy native trig.
  let w=twiddles[k*(p.m/p.span)];
  let t=cmul(inputC[ib],vec2f(w.x,select(w.y,-w.y,p.inverse==1u)));
  var v=inputC[ia]+select(-t,t,local<half);
  // Distribute inverse scaling across each radix-2 pass to bound intermediates.
  if(p.inverse==1u){v*=0.5;}outputC[i]=v;
}
@compute @workgroup_size(64)
fn multiply(@builtin(global_invocation_id) id:vec3u) {
  let i=id.x;if(i>=p.count){return;}outputC[i]=cmul(inputC[i],sourceFFT[i]);
}
@compute @workgroup_size(64)
fn accumulate(@builtin(global_invocation_id) id:vec3u) {
  let i=id.x;if(i>=p.n*p.n){return;}let x=i%p.n;let y=i/p.n;
  let z=inputC[y*p.m+x]*p.weight;
  atomicAdd(&safety.inspected,1u);
  if(!finiteValue(z.x)||!finiteValue(z.y)){atomicAdd(&safety.nonFinite,1u);return;}
  if(abs(z.y)>p.tolerance){atomicAdd(&safety.imaginary,1u);return;}
  if(z.x < -p.tolerance){atomicAdd(&safety.negative,1u);return;}
  if(z.x<0.0){atomicAdd(&safety.roundoff,1u);atomicMax(&safety.maxRoundoffBits,bitcast<u32>(-z.x));}
  let oi=p.channel*p.n*p.n+i;
  let result=accumulator[oi]+max(z.x,0.0);
  if(!finiteValue(result)){atomicAdd(&safety.nonFinite,1u);return;}
  accumulator[oi]=result;
}
