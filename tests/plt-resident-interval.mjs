/** Fail-closed WGSL interval arithmetic, independently checked on host doubles.
 * Compile: node tests/plt-resident-interval.mjs
 * Execute: VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json \
 * WEBGPU_EXECUTION_BACKEND=vulkan node tests/plt-resident-interval.mjs --execute
 * Null-backend compilation is never reported as execution. No files generated.
 * Host oracle does not reproduce the shader's padding algorithm. Exact dyadic
 * rational checks additionally cover add/sub/mul where binary64 can absorb a
 * tiny addend; double arithmetic alone cannot certify those boundary cases.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {create, globals} from 'webgpu';
Object.assign(globalThis, globals);
const source = fs.readFileSync(new URL('../dist/plt-resident-interval.wgsl', import.meta.url), 'utf8');
const execute = process.argv.includes('--execute');
const backend = execute ? process.env.WEBGPU_EXECUTION_BACKEND : 'null';
if (execute) assert(['vulkan','metal','d3d11','d3d12','opengl','opengles'].includes(backend), 'Explicit executable Dawn backend required; null cannot prove execution');
const fbuf = new ArrayBuffer(4), fu = new Uint32Array(fbuf), ff = new Float32Array(fbuf);
function bits(x) { ff[0] = x; return fu[0]; }
function value(x) { fu[0] = x; return ff[0]; }
const pair = (lo, hi=lo) => [bits(lo), bits(hi)];
const interval = p => p.map(value);
const cases = [];
function add(name, op, a, b=pair(0), c=pair(0), invalid=false) { cases.push({name,op,a,b,c,invalid}); }
// Binary64 is the independent numerical oracle; no Math.fround intermediates.
function oracle(q) {
 if(q.expected)return q.expected;
 const [al,ah]=interval(q.a),[bl,bh]=interval(q.b),[cl,ch]=interval(q.c);
 const products=[al*bl,al*bh,ah*bl,ah*bh];
 switch(q.op) {
  case 0:return [[al+bl,ah+bh]];
  case 1:return [[al-bh,ah-bl]];
  case 2:return [[Math.min(...products),Math.max(...products)]];
  case 3:{const p=[al/bl,al/bh,ah/bl,ah/bh];return [[Math.min(...p),Math.max(...p)]];}
  case 4:return [[al<=0&&ah>=0?0:Math.min(al*al,ah*ah),Math.max(al*al,ah*ah)]];
  case 5:return [[Math.sqrt(al),Math.sqrt(ah)]];
  case 6:{
   const choices=([l,h])=>l<=0&&h>=0?[l,0,h]:[l,(l+h)/2,h];
   const out=[[],[],[]];
   for(const x of choices([al,ah]))for(const y of choices([bl,bh]))for(const z of choices([cl,ch])){
    const n=Math.sqrt(x*x+y*y+z*z); if(n===0)continue;
    [x/n,y/n,z/n].forEach((v,k)=>out[k].push(v));
   }
   return out.map(x=>[Math.min(...x),Math.max(...x)]);
  }
  case 7:return [[al,ah]];
  case 8:return [[al,ah]]; // direct ri_pad probe: original exact f32 point
  case 9:return [[Math.min(...products)-ch,Math.max(...products)-cl]];
  default:throw Error('unknown op');
 }
}
// Exact dyadic arithmetic supplements the host-double oracle for finite f32s.
// Representation is signed integer * 2^exponent; no shader implementation reuse.
function dyadic(b) {
 const e=(b>>>23)&255,m=b&0x7fffff;
 const n=BigInt(e===0?m:m+0x800000)*(b>>>31?-1n:1n);
 return [n,e===0?-149:e-150];
}
function dadd(a,b) {const e=Math.min(a[1],b[1]);return [(a[0]<<BigInt(a[1]-e))+(b[0]<<BigInt(b[1]-e)),e];}
const dneg=a=>[-a[0],a[1]], dmul=(a,b)=>[a[0]*b[0],a[1]+b[1]];
function dcmp(a,b) {const v=dadd(a,dneg(b))[0];return v<0n?-1:v>0n?1:0;}
function exactBounds(q) {
 const [al,ah]=q.a.map(dyadic),[bl,bh]=q.b.map(dyadic),[cl,ch]=q.c.map(dyadic);
 if(q.op===0)return [dadd(al,bl),dadd(ah,bh)];
 if(q.op===1)return [dadd(al,dneg(bh)),dadd(ah,dneg(bl))];
 if(q.op===2||q.op===9){const p=[dmul(al,bl),dmul(al,bh),dmul(ah,bl),dmul(ah,bh)].sort(dcmp);return q.op===2?[p[0],p[3]]:[dadd(p[0],dneg(ch)),dadd(p[3],dneg(cl))];}
 return null;
}
const harness = `
struct RIProbeIn { dispatch:vec4u, ab:vec4u, c:vec4u }
struct RIProbeOut { x:vec4u, y:vec4u, z:vec4u }
@group(0) @binding(0) var<storage,read> ri_inputs:array<RIProbeIn>;
@group(0) @binding(1) var<storage,read_write> ri_outputs:array<RIProbeOut>;
fn ri_export(a:RI)->vec4u{return vec4u(a.lo,a.hi,a.invalid,0x51a7c0deu);}
@compute @workgroup_size(64) fn ri_probe(@builtin(global_invocation_id) gid:vec3u) {
 let i=gid.x;if(i>=arrayLength(&ri_inputs)){return;}
 let p=ri_inputs[i];let a=ri_bounds(p.ab.x,p.ab.y);let b=ri_bounds(p.ab.z,p.ab.w);let c=ri_bounds(p.c.x,p.c.y);
 var x=ri_unknown(1u);var y=ri_point(0u);var z=ri_point(0u);
 switch p.dispatch.x {
  case 0u:{x=ri_add(a,b);} case 1u:{x=ri_sub(a,b);}
  case 2u:{x=ri_mul(a,b);} case 3u:{x=ri_div(a,b);}
  case 4u:{x=ri_square(a);} case 5u:{x=ri_sqrt(a);}
  case 6u:{let n=ri_normalize3(RIVec3(a,b,c));x=n.x;y=n.y;z=n.z;}
  case 7u:{x=a;}
  case 8u:{x=ri_pad(p.ab.x,RI_PAD_APPROX);}
  case 9u:{x=ri_sub(ri_mul(a,b),c);}
  case 10u:{x=ri_pad(p.ab.x,RI_PAD_EXACT);}
  default:{}
 }
 ri_outputs[i]=RIProbeOut(ri_export(x),ri_export(y),ri_export(z));
}`;
const signs=[-1,1];
for(const s of signs)for(const t of signs)for(const x of [0.1,0.33333334,1,1.0000001192092896,Math.PI,2**-60,2**60])for(const y of [0.3,1.0000001192092896,1.9999998807907104,2**-50,2**50]){
 for(const op of [0,1,2,3])add(`sign-magnitude-${op}-${s}-${t}-${x}-${y}`,op,pair(s*x),pair(t*y));
}
for(const e of [-100,-60,-20,0,20,60,100]){
 const a=2**e;
 for(const offset of [-4,-1,0,1,4]){
  const b=value(bits(a)+offset);
  add(`cancellation-${e}-${offset}`,0,pair(a),pair(-b),pair(0),a!==b&&Math.abs(a-b)<2**-120);
  add(`subtraction-${e}-${offset}`,1,pair(a),pair(b),pair(0),a!==b&&Math.abs(a-b)<2**-120);
 }
 for(const sign of signs){const x=sign*a;add(`zero-${e}-${sign}`,0,pair(x),pair(0));add(`directed-endpoint-${e}-${sign}`,8,pair(x));}
 add(`absorption-plus-${e}`,0,pair(a),pair(2**-110));
 add(`absorption-minus-${e}`,1,pair(a),pair(2**-110));
 add(`sqrt-binade-${e}`,5,pair(a));
}
for(const a of [[-3,-0.2],[-3,0],[-3,7],[0,0],[0,7],[0.25,7],[0.99999994,1.00000012]]){
 add('square-crossing',4,pair(...a));
 for(const b of [[-5,-0.1],[-2,2],[0.1,5]])for(const op of [0,1,2,3])add(`interval-corner-${op}`,op,pair(...a),pair(...b),pair(0),op===3&&b[0]<=0&&b[1]>=0);
}
let state=0x913ab47;
function random(){state^=state<<13;state^=state>>>17;state^=state<<5;return (state>>>0)/4294967296;}
function fvalue(e=40){return Math.fround((random()<0.5?-1:1)*(1+random())*2**Math.floor((random()*2-1)*e));}
for(let i=0;i<1800;i++){
 const x=fvalue(),y=fvalue(),z=fvalue();
 const a=i%3?pair(x):pair(Math.min(x,x*1.01),Math.max(x,x*1.01));
 const b=i%5?pair(y):pair(Math.min(y,y*1.1),Math.max(y,y*1.1));
 const op=i%10;
 if(op===4)add(`random-square-${i}`,4,a);
 else if(op===5)add(`random-sqrt-${i}`,5,pair(Math.abs(x)));
 else if(op===6)add(`random-normalize-${i}`,6,a,b,pair(z));
 else if(op===7)add(`random-composition-${i}`,9,a,b,pair(z));
 else if(op===8)add(`random-directed-${i}`,8,pair(x));
 else add(`random-arithmetic-${i}`,op===9?3:op,a,b);
}
for(const components of [[[3,3],[4,4],[0,0]],[[-3,2],[4,5],[-2,2]],[[0,0],[0,0],[1,1]],[[-1,-1],[0,0],[0,0]],[[1e-8,1e-8],[1,1],[1e-5,1e-5]]])add('normalize-axis-or-box',6,...components.map(x=>pair(...x)));
// Raw bit-pattern inputs test fail-closed construction BEFORE f32 bitcasts.
for(const b of [1,0x007fffff,0x00800000,0x80000001,0x7f800000,0xff800000,0x7fc00000,0x7f7fffff])add(`unsupported-bits-${b.toString(16)}`,7,[b,b],pair(0),pair(0),true);
for(const x of [0,-0,2**-120,-(2**-120),2**120,-(2**120)])add('supported-endpoint-boundary',7,pair(x));
for(const x of [0,-0]){add('zero-sqrt',5,pair(x));add('zero-numerator',3,pair(x),pair(-3,-1));}
add('reversed',7,pair(2,1),pair(0),pair(0),true);
add('division-zero',3,pair(1),pair(0),pair(0),true);
add('division-span-zero',3,pair(1),pair(-1,1),pair(0),true);
add('sqrt-negative',5,pair(-1,1),pair(0),pair(0),true);
add('normalization-origin',6,pair(0),pair(0),pair(0),true);
add('normalization-containing-origin',6,pair(-1,1),pair(-1,1),pair(-1,1),true);
add('multiply-overflow-preflight',2,pair(2**120),pair(2**120),pair(0),true);
add('divide-overflow-preflight',3,pair(2**120),pair(2**-120),pair(0),true);
add('multiply-underflow-preflight',2,pair(2**-120),pair(2**-120),pair(0),true);
add('divide-underflow-preflight',3,pair(2**-120),pair(2**120),pair(0),true);
add('cancellation-subnormal',1,pair(value(0x03800001)),pair(value(0x03800000)),pair(0),true);
// Direct eight-step endpoint checks on normal interior values and both signs.
for(const sign of signs)for(const exponent of [-110,-10,0,10,110])for(const offset of [-8,-1,0,1,8])add('binade-directed-pad',8,pair(sign*value(bits(2**exponent)+offset)));
// This adapter may be more accurate than WGSL requires. Inject every permitted
// result near several binade boundaries, including round-up/down and worst-case
// 2.5-ULP division errors, into the real GPU padding code. All chosen magnitudes
// stay normal, so comparisons and bitcasts cannot invoke flush-to-zero.
let syntheticAccuracyCases=0;
function trueUlp(x){
 const v=Math.abs(x),r=Math.fround(v),b=bits(r);
 if(r===v)return Math.min(value(b+1)-r,r-value(b-1));
 return r<v?value(b+1)-r:r-value(b-1);
}
for(const sign of signs)for(const exponent of [-110,-10,0,10,110])for(const fraction of [-1.25,-0.25,0,0.25,1.25]){
 const exact=sign*(2**exponent+fraction*2**(exponent-23));
 const base=bits(Math.abs(exact)),ulp=trueUlp(exact);
 const surrounding=[];
 for(let offset=-8;offset<=8;offset++){
  const candidate=sign*value(base+offset);
  if(Math.abs(candidate-exact)<=2.5*ulp){
   add('allowed-2.5-ULP-result',8,pair(candidate));
   cases.at(-1).expected=[[exact,exact]];syntheticAccuracyCases++;
  }
  surrounding.push(candidate);
 }
 const low=Math.max(...surrounding.filter(x=>x<=exact));
 const high=Math.min(...surrounding.filter(x=>x>=exact));
 for(const candidate of new Set([low,high])){
  add('allowed-either-directed-rounding-result',10,pair(candidate));
  cases.at(-1).expected=[[exact,exact]];syntheticAccuracyCases++;
 }
}
// Host-only dyadic mutation control demonstrates why double-only tests are weak.
assert.equal(1+2**-100,1);
assert(dcmp(dyadic(bits(1)),dadd(dyadic(bits(1)),dyadic(bits(2**-100))))<0);

let gpu=create([`backend=${backend}`]);
const adapter=await gpu.requestAdapter();assert(adapter,'WebGPU adapter unavailable');
const device=await adapter.requestDevice();
const report={backend,actualGPUExecuted:execute,sourceSHA256:crypto.createHash('sha256').update(source).digest('hex'),cases:cases.length,syntheticAccuracyCases,adapter:{vendor:adapter.info.vendor,device:adapter.info.device,description:adapter.info.description},mutations:[]};
async function compile(src){
 device.pushErrorScope('validation');
 const module=device.createShaderModule({code:src+harness});
 const errors=(await module.getCompilationInfo()).messages.filter(x=>x.type==='error');
 assert.deepEqual(errors.map(x=>({line:x.lineNum,column:x.linePos,message:x.message})),[]);
 const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'ri_probe'}});
 assert.equal(await device.popErrorScope(),null);
 return pipeline;
}
async function run(pipeline){
 const words=new Uint32Array(cases.length*12);
 cases.forEach((q,i)=>{words.set([q.op,0,0,0,...q.a,...q.b,...q.c,0,0],i*12);});
 const buffers=[];
 const make=(size,usage)=>{const b=device.createBuffer({size,usage});buffers.push(b);return b;};
 try{
  const input=make(words.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
  const output=make(words.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
  const read=make(words.byteLength,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(input,0,words);
  const bind=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:input}},{binding:1,resource:{buffer:output}}]});
  const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
  pass.setPipeline(pipeline);pass.setBindGroup(0,bind);pass.dispatchWorkgroups(Math.ceil(cases.length/64));pass.end();
  encoder.copyBufferToBuffer(output,0,read,0,words.byteLength);device.queue.submit([encoder.finish()]);
  await read.mapAsync(GPUMapMode.READ);const result=new Uint32Array(read.getMappedRange().slice(0));read.unmap();return result;
 }finally{buffers.forEach(x=>x.destroy());}
}
function inspect(result){
 const failures=[];let valid=0,unknown=0,expectedUnknown=0,exactChecks=0;
 cases.forEach((q,i)=>{
  const at=i*12,flags=result[at+2];
  if(result[at+3]!==0x51a7c0de)failures.push({name:q.name,reason:'unwritten output / execution absent'});
  if(q.invalid){expectedUnknown++;if(!flags)failures.push({name:q.name,reason:'unsafe input was admitted'});return;}
  if(flags){unknown++;return;}valid++;
  const expected=oracle(q),exact=exactBounds(q);
  expected.forEach(([lo,hi],k)=>{
   const o=at+4*k,l=value(result[o]),h=value(result[o+1]);
   if(result[o+2]!==0||!Number.isFinite(l)||!Number.isFinite(h)||l>lo||h<hi||l>h)failures.push({name:q.name,k,reason:'double oracle under-enclosure',actual:[l,h],expected:[lo,hi]});
  });
  if(exact){exactChecks++;if(dcmp(dyadic(result[at]),exact[0])>0||dcmp(dyadic(result[at+1]),exact[1])<0)failures.push({name:q.name,reason:'exact dyadic oracle under-enclosure'});}
  if(q.op===8||q.op===10){
   const b=q.a[0],negative=!!(b>>>31),steps=q.op===10?1:8;
   const wantLo=negative?b+steps:b-steps,wantHi=negative?b-steps:b+steps;
   if(result[at]!==wantLo||result[at+1]!==wantHi)failures.push({name:q.name,reason:'wrong directed integer endpoints'});
  }
 });
 return {valid,unknown,expectedUnknown,exactChecks,failures};
}
try{
 const pipeline=await compile(source);console.log('PASS genuine Dawn/Tint interval compilation');
 if(execute){
  const baseline=inspect(await run(pipeline));
  assert.equal(baseline.failures.length,0,JSON.stringify(baseline.failures.slice(0,12),null,2));
  assert(baseline.valid>2000,'Most supported-domain cases must actually be admitted');
  assert.equal(baseline.unknown,0,'Supported cases must not be silently replaced by unknown');
  assert(baseline.exactChecks>800,'Exact oracle coverage missing');
  Object.assign(report,baseline);delete report.failures;
  console.log('PASS executable interval enclosure',JSON.stringify(baseline));
  const mutations=[
   ['remove-outward-rounding',s=>s.replace('RI_PAD_EXACT: u32 = 1u','RI_PAD_EXACT: u32 = 0u').replace('RI_PAD_APPROX: u32 = 8u','RI_PAD_APPROX: u32 = 0u')],
   ['one-step-approximate-padding',s=>s.replace('RI_PAD_APPROX: u32 = 8u','RI_PAD_APPROX: u32 = 1u')],
   ['omit-square-zero-interior',s=>s.replace('return RI(0u, ends.hi, 0u);','return ends;')],
   ['omit-denominator-zero-crossing-check',s=>s.replace('if (ri_lo(b) <= 0.0 && ri_hi(b) >= 0.0)', 'if (false)')]
  ];
  for(const [name,mutate]of mutations){const mutated=mutate(source);assert.notEqual(mutated,source);const check=inspect(await run(await compile(mutated)));assert(check.failures.length>0,`Mutation survived: ${name}`);if(name==='one-step-approximate-padding')assert(check.failures.some(f=>f.name==='allowed-2.5-ULP-result'&&f.reason==='double oracle under-enclosure'),'Too-small division padding must fail a legal WGSL error case, not merely endpoint bit equality');report.mutations.push({name,rejected:true,failures:check.failures.length,underEnclosures:check.failures.filter(f=>f.reason.includes('under-enclosure')).length,unsafeAdmissions:check.failures.filter(f=>f.reason==='unsafe input was admitted').length,first:check.failures.find(f=>f.reason.includes('under-enclosure'))??check.failures[0]});console.log('PASS mutation rejected',name,check.failures.length);}
 }
 console.log(JSON.stringify({...report,allPass:true},null,2));
}finally{device.destroy();gpu=null;}
