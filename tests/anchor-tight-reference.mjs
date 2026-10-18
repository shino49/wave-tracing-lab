/** Independent exact-Fraction check of ACTUAL tight helper native outputs.
 * The Python oracle never reproduces the shader's padding or recentering.
 * Compile: node tests/anchor-tight-reference.mjs
 * Execute: VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json WEBGPU_EXECUTION_BACKEND=vulkan node tests/anchor-tight-reference.mjs --execute
 */
import fs from 'node:fs';import assert from 'node:assert/strict';import crypto from 'node:crypto';import {spawnSync} from 'node:child_process';import {fileURLToPath} from 'node:url';
import {create,globals} from 'webgpu';Object.assign(globalThis,globals);
const execute=process.argv.includes('--execute'),backend=execute?process.env.WEBGPU_EXECUTION_BACKEND:'null';if(execute)assert.equal(backend,'vulkan');
const names=['plt-resident-interval.wgsl','plt-resident-geometry.wgsl','plt-resident-axis-guard.wgsl'],files=names.map(n=>fs.readFileSync(new URL('../dist/'+n,import.meta.url),'utf8')),source=files.join('\n');
const storage=new ArrayBuffer(4),v=new DataView(storage),bits=x=>{v.setFloat32(0,x,true);return v.getUint32(0,true)},value=b=>{v.setUint32(0,b,true);return v.getFloat32(0,true)};
const pair=(a,b=a)=>[bits(a),bits(b)],next=x=>value(bits(x)+(x<0?-1:1)),prev=x=>value(bits(x)+(x<0?1:-1));
const rows=[],put=(name,a,b,delta,base,extra={})=>rows.push({name,a,b,delta,base,op:0,...extra});
for(const a of [0,.5,1,1.5,2,3,-.5,-1,-1.5,-2,-3])for(const b of [.5,1,1.5,2,-.5,-1,-1.5,-2])for(const delta of [-16,-1,0,1,16]){
 const c=Math.fround(-(a/b)*delta);put('binade-and-cancellation',pair(a),pair(b),pair(delta),pair(c),{mustValid:true});
}
let state=0xf284397b;const rand=()=>{state^=state<<13;state^=state>>>17;state^=state<<5;return(state>>>0)/2**32};
for(let i=0;i<1500;i++){
 const a=Math.fround((rand()<.5?-1:1)*(.5+rand())*2**(i%11-5)),b=Math.fround((rand()<.5?-1:1)*(.5+rand())*2**(i%7-3));
 const d=Math.fround((rand()<.5?-1:1)*(.5+rand())*2**(i%13-6)),c=Math.fround(-(a/b)*d);
 put('non-singleton-cancel-'+i,pair(prev(a),next(a)),pair(prev(b),next(b)),i%2?pair(d):pair(prev(d),next(d)),i%3?pair(c):pair(prev(c),next(c)),{mustValid:true});
}
for(const ea of [-100,-99,-90,-50,-1,0,1,47,48])for(const eb of [-100,-99,-90,-50,-1,0,1,47,48])for(const sign of [-1,1]){
 const a=sign*2**ea,b=2**eb;put('compact-boundary',pair(a),pair(b),pair(1),pair(0));
}
put('denominator-crosses-zero',pair(1),pair(-1,1),pair(1),pair(0),{mustInvalid:true});
put('denominator-touches-zero',pair(1),pair(0,1),pair(1),pair(0),{mustInvalid:true});
put('denominator-exact-zero',pair(1),pair(0),pair(1),pair(0),{mustInvalid:true});
put('invalid-input-endpoint',[1,1],pair(1),pair(1),pair(0),{mustInvalid:true});
put('intersection-empty',pair(0,1),pair(2,3),pair(1),pair(0),{op:1,mustInvalid:true});
put('intersection-touching',pair(0,1),pair(1,3),pair(1),pair(0),{op:1,mustValid:true});
put('intersection-overlap',pair(-4,2),pair(1,3),pair(1),pair(0),{op:1,mustValid:true});
put('sticky-prior-failure',pair(1),pair(2),pair(1),pair(0),{op:2,mustInvalid:true});
put('near-zero-ratio-residual',pair(2**-99),pair(1),pair(1),pair(-(2**-99)));
put('ratio-interval-crosses-zero',pair(-1,2),pair(1,2),pair(-4,5),pair(-3,2),{mustValid:true});
const harness=`
struct RefIn {header:vec4u,a:vec2u,b:vec2u,d:vec2u,c:vec2u}
@group(0) @binding(2) var<storage,read> probe:array<RefIn>;
@group(0) @binding(3) var<storage,read_write> result:array<vec4u>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){
 let p=probe[id.x];ra_bad=false;
 let a=ra_ibounds(p.a.x,p.a.y);let b=ra_ibounds(p.b.x,p.b.y);let d=ra_ibounds(p.d.x,p.d.y);let c=ra_ibounds(p.c.x,p.c.y);
 if(p.header.x==1u){let r=ra_anchor_tight_intersect(a,b);result[id.x*4u]=bitcast<vec4u>(vec4f(r.lo,r.hi,0,0));result[id.x*4u+3u]=vec4u(select(0u,1u,ra_bad),0u,0u,0x1a51c021u);return;}
 if(p.header.x==2u){ra_bad=true;}
 let coarse=ra_idiv(a,b);let old=ra_iadd(c,ra_imul(coarse,d));let tight=ra_anchor_tight_ratio(a,b);let point=ra_anchor_tight_product_add(c,tight,d);
 result[id.x*4u]=bitcast<vec4u>(vec4f(coarse.lo,coarse.hi,tight.full.lo,tight.full.hi));
 result[id.x*4u+1u]=bitcast<vec4u>(vec4f(old.lo,old.hi,point.lo,point.hi));
 result[id.x*4u+2u]=bitcast<vec4u>(vec4f(tight.center,tight.residual.lo,tight.residual.hi,0));
 result[id.x*4u+3u]=vec4u(select(0u,1u,ra_bad),0u,0u,0x1a51c021u);
}`;
const sha=x=>crypto.createHash('sha256').update(x).digest('hex'),report={createdAt:new Date().toISOString(),actualGPUExecuted:false,executionRequested:execute,backend,allPass:false,fullCPUShadowMandatory:true,physicalGPUValidated:false,browserValidated:false,cases:rows.length,sourceSHA256:Object.fromEntries(names.map((n,i)=>[n,sha(files[i])])),testSHA256:sha(fs.readFileSync(new URL(import.meta.url))),oracleSHA256:sha(fs.readFileSync(new URL('../research/anchor-tight-reference.py',import.meta.url))),mutations:[]};
const path=new URL('../research/anchor-tight-reference-'+(execute?'validation':'compile-validation')+'.json',import.meta.url),save=()=>fs.writeFileSync(path,JSON.stringify(report,null,2)+'\n');
let gpu=create([`backend=${backend}`]);const adapter=await gpu.requestAdapter(),device=await adapter.requestDevice();report.adapter={description:adapter.info.description,vendor:adapter.info.vendor,device:adapter.info.device};
async function compile(src){const started=performance.now(),m=device.createShaderModule({code:src+harness}),info=await m.getCompilationInfo();assert.deepEqual(info.messages.filter(x=>x.type==='error'),[]);const p=await device.createComputePipelineAsync({layout:'auto',compute:{module:m,entryPoint:'main'}});return {p,compileMs:performance.now()-started};}
async function dispatch(p){const data=new Uint32Array(rows.length*12);rows.forEach((r,i)=>data.set([r.op,0,0,0,...r.a,...r.b,...r.delta,...r.base],i*12));const input=device.createBuffer({size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),output=device.createBuffer({size:rows.length*64,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:rows.length*64,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});try{device.queue.writeBuffer(input,0,data);const group=device.createBindGroup({layout:p.getBindGroupLayout(0),entries:[{binding:2,resource:{buffer:input}},{binding:3,resource:{buffer:output}}]}),e=device.createCommandEncoder(),pass=e.beginComputePass();pass.setPipeline(p);pass.setBindGroup(0,group);pass.dispatchWorkgroups(rows.length);pass.end();e.copyBufferToBuffer(output,0,read,0,rows.length*64);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);const raw=new Uint32Array(read.getMappedRange().slice(0));read.unmap();const payload=rows.map((r,i)=>({...r,output:Array.from(raw.slice(i*16,i*16+16))}));const checked=spawnSync('python3',[fileURLToPath(new URL('../research/anchor-tight-reference.py',import.meta.url))],{input:JSON.stringify(payload),encoding:'utf8',maxBuffer:8*1024*1024});assert.equal(checked.status,0,checked.stderr);return JSON.parse(checked.stdout);}finally{input.destroy();output.destroy();read.destroy();}}
try{const baseline=await compile(source);report.compileMs=baseline.compileMs;save();if(execute){report.baseline=await dispatch(baseline.p);report.actualGPUExecuted=true;save();assert.equal(report.baseline.failures.length,0,JSON.stringify(report.baseline.failures.slice(0,10)));
 const mutations=[['accept-empty-intersection',s=>s.replace('if(lo>hi){return ra_iunknown(RA_INTERVAL);}return RAI(lo,hi);','if(lo>hi){return RAI(min(a.lo,b.lo),max(a.hi,b.hi));}return RAI(lo,hi);')],['drop-centered-ratio-residual',s=>s.replace('let residual=ra_idiv(ra_isub(a,ra_imul(ra_point(q),b)),b);','let residual=RAI(0.0,0.0);')]];
 for(const [name,mutate] of mutations){const altered=mutate(source);assert.notEqual(altered,source);const result=await dispatch((await compile(altered)).p);assert(result.failures.length>0,name+' survived');report.mutations.push({name,rejected:true,failures:result.failures.length,firstFailure:result.failures[0]});save();}
}report.allPass=true;console.log(JSON.stringify(report,null,2));}finally{device.destroy();gpu=null;save();}
