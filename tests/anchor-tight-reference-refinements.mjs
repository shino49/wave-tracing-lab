/** Independent exact-rational native oracle for tight norm and centered camera refinements. */
import fs from 'node:fs';import crypto from 'node:crypto';import assert from 'node:assert/strict';import {spawnSync} from 'node:child_process';import {fileURLToPath} from 'node:url';
import {create,globals} from 'webgpu';Object.assign(globalThis,globals);
const execute=process.argv.includes('--execute'),backend=execute?process.env.WEBGPU_EXECUTION_BACKEND:'null';if(execute)assert.equal(backend,'vulkan');
const files=['plt-resident-interval.wgsl','plt-resident-geometry.wgsl','plt-resident-axis-guard.wgsl'],texts=files.map(f=>fs.readFileSync(new URL('../dist/'+f,import.meta.url),'utf8')),source=texts.join('\n');
const ab=new ArrayBuffer(4),dv=new DataView(ab),bits=x=>{dv.setFloat32(0,x,true);return dv.getUint32(0,true)},value=x=>{dv.setUint32(0,x,true);return dv.getFloat32(0,true)},pair=(a,b=a)=>[bits(a),bits(b),0,0],prev=x=>value(bits(x)+(x<0?1:-1)),next=x=>value(bits(x)+(x<0?-1:1));
const rows=[],norm=(name,a,b,c,extra={})=>rows.push({op:0,name,a,b,c,offset:0,sign:0,...extra}),base=(name,p,C,E,offset,sign,extra={})=>rows.push({op:1,name,a:[bits(p),bits(C),bits(E),0],b:[0,0,0,0],c:[0,0,0,0],offset,sign,...extra});
for(const x of [0,1,-1,.5,-.5,1.5,-1.5])for(const y of [0,.25,-.25,1,-1])for(const z of [0,.75,-.75,1,-1])norm('norm-axis-and-sign',pair(x),pair(y),pair(z),{[x||y||z?'mustValid':'mustInvalid']:true});
let state=0x123981bc;const rand=()=>{state^=state<<13;state^=state>>>17;state^=state<<5;return(state>>>0)/2**32};
for(let i=0;i<600;i++){
 const vals=Array.from({length:3},()=>Math.fround((rand()<.5?-1:1)*(.1+rand())*2**(i%17-8)));
 norm('raw-box-'+i,...vals.map(x=>pair(prev(x),next(x))),{mustValid:true});
}
for(const e of [-100,-99,-51,-50,-49,-24,-1,0,1,23,24,25,47,48])for(const sign of [-1,1])norm('norm-compact-boundary',pair(sign*2**e),pair(0),pair(0));
norm('norm-prior-sticky-failure',pair(1),pair(1),pair(1),{mustInvalid:true,sticky:true});
norm('norm-raw-crosses-zero',pair(-.1,.1),pair(1),pair(1),{mustInvalid:true});norm('norm-raw-touches-zero',pair(0,.1),pair(1),pair(1),{mustInvalid:true});
const offsets=[-201326592,-16777216,-16777215,-65537,-1,0,1,65537,16777215,16777216,201326592];
for(const offset of offsets)for(const p of [-4096,-256,-1,0,1,256,4096])for(const sign of [0,1]){
 const fixed=p-offset/65536,C=Math.fround((sign?-1:1)*fixed);
 for(const E of [0,2**-42,2**-16])base('exact-grid-centered',p,C,E,offset,sign,{mustValid:true});
}
for(const offset of [-16777215,0,16777215])for(const p of [2**-20,next(1),prev(1),next(256),prev(256)])for(const sign of [0,1])for(const C of [-256,-1,0,1,256])base('nongrid-or-unrepresentable-fixed',p,C,2**-42,offset,sign,{mustValid:true});
for(const p of [-2,-1,1,2])for(const C of [-4,prev(-2),-2,next(-2),-1,-.5,.5,1,prev(2),2,next(2),4])for(const sign of [0,1])base('sterbenz-boundary',p,C,2**-42,0,sign,{mustValid:true});
for(const p of [-16,-2,-1,1,2,16])for(const sign of [0,1])for(const cameraSign of [-1,1])base('nonsterbenz-zero-error-rounding',p,cameraSign*2**-24,0,0,sign,{mustValid:true});
const harness=`
struct Probe {header:vec4u,a:vec4u,b:vec4u,c:vec4u}
@group(0) @binding(2) var<storage,read> input:array<Probe>;
@group(0) @binding(3) var<storage,read_write> output:array<vec4u>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){
 let p=input[id.x];ra_bad=false;var values:vec4f;var valid=1u;
 if(p.header.x==0u){let raw=RAVec3(ra_ibounds(p.a.x,p.a.y),ra_ibounds(p.b.x,p.b.y),ra_ibounds(p.c.x,p.c.y));var coarse:RAAnchor;if(!ra_bad){coarse=ra_anchor_initialize(raw);}var refined=coarse;if(p.header.w!=0u){ra_bad=true;}if(coarse.valid==1u){refined=ra_anchor_tight_refine_norm(coarse);}values=vec4f(refined.rawNorm.lo,refined.rawNorm.hi,coarse.rawNorm.lo,coarse.rawNorm.hi);valid=refined.valid;}
 else{var a:RAAnchor;a.offsetGrid.x=bitcast<i32>(p.header.y);a.signMask=p.header.z;let plane=bitcast<f32>(p.a.x);let camera=vec3f(bitcast<f32>(p.a.y),0,0);let error=vec3f(bitcast<f32>(p.a.z),0,0);let d=ra_anchor_tight_delta(a,0u,plane,camera,error);let c=ra_anchor_tight_base(a,0u,plane,camera,error);values=vec4f(d.lo,d.hi,c.lo,c.hi);}
 output[id.x*2u]=bitcast<vec4u>(values);output[id.x*2u+1u]=vec4u(select(0u,1u,ra_bad||valid!=1u),valid,0u,0xac712122u);
}`;
const sha=s=>crypto.createHash('sha256').update(s).digest('hex'),report={createdAt:new Date().toISOString(),backend,actualGPUExecuted:false,executionRequested:execute,allPass:false,fullCPUShadowMandatory:true,physicalGPUValidated:false,browserValidated:false,cases:rows.length,sourceSHA256:Object.fromEntries(files.map((f,i)=>[f,sha(texts[i])])),testSHA256:sha(fs.readFileSync(new URL(import.meta.url))),oracleSHA256:sha(fs.readFileSync(new URL('../research/anchor-tight-reference-refinements.py',import.meta.url))),mutations:[]};
const path=new URL('../research/anchor-tight-reference-refinements-'+(execute?'validation':'compile-validation')+'.json',import.meta.url),save=()=>fs.writeFileSync(path,JSON.stringify(report,null,2)+'\n');
let gpu=create([`backend=${backend}`]);const adapter=await gpu.requestAdapter(),device=await adapter.requestDevice();report.adapter={description:adapter.info.description,vendor:adapter.info.vendor,device:adapter.info.device};
async function compile(src){const start=performance.now(),module=device.createShaderModule({code:src+harness});assert.deepEqual((await module.getCompilationInfo()).messages.filter(x=>x.type==='error'),[]);const p=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});return {p,compileMs:performance.now()-start};}
async function run(p){const words=new Uint32Array(rows.length*16);rows.forEach((r,i)=>words.set([r.op,r.offset>>>0,r.sign,r.sticky?1:0,...r.a,...r.b,...r.c],i*16));const input=device.createBuffer({size:words.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),out=device.createBuffer({size:rows.length*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:rows.length*32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});try{device.queue.writeBuffer(input,0,words);const group=device.createBindGroup({layout:p.getBindGroupLayout(0),entries:[{binding:2,resource:{buffer:input}},{binding:3,resource:{buffer:out}}]}),enc=device.createCommandEncoder(),pass=enc.beginComputePass();pass.setPipeline(p);pass.setBindGroup(0,group);pass.dispatchWorkgroups(rows.length);pass.end();enc.copyBufferToBuffer(out,0,read,0,rows.length*32);device.queue.submit([enc.finish()]);await read.mapAsync(GPUMapMode.READ);const result=new Uint32Array(read.getMappedRange().slice(0));read.unmap();const checked=spawnSync('python3',[fileURLToPath(new URL('../research/anchor-tight-reference-refinements.py',import.meta.url))],{input:JSON.stringify(rows.map((r,i)=>({...r,output:Array.from(result.slice(i*8,i*8+8))}))),encoding:'utf8',maxBuffer:8*1024*1024});assert.equal(checked.status,0,checked.stderr);return JSON.parse(checked.stdout);}finally{input.destroy();out.destroy();read.destroy();}}
try{const first=await compile(source);report.compileMs=first.compileMs;save();if(execute){report.baseline=await run(first.p);report.actualGPUExecuted=true;save();assert.equal(report.baseline.failureCount,0,JSON.stringify(report.baseline.failures.slice(0,10)));
 const mutations=[
  ['drop-camera-packing-radius',s=>s.replace('return ra_iadd(centered,ra_symmetric(cameraError[k]));','return centered;')],
  ['skip-grid-difference-roundtrip',s=>s.replace('if(i32(f32(difference))==difference){return ra_point(f32(difference)*0x1p-16);}','if(true){return ra_point(f32(difference)*0x1p-16);}')],
  ['drop-true-norm-residual',s=>s.replace('next.rawNorm=ra_anchor_tight_intersect(a.rawNorm,ra_iadd(q,residual));','next.rawNorm=q;')],
  ['assume-all-centered-additions-exact',s=>s.replace('return ra_add_scalar(a,b);','return ra_point(a+b);')],
 ];
 for(const[name,mutate]of mutations){const altered=mutate(source);assert.notEqual(altered,source,name);const result=await run((await compile(altered)).p);assert(result.failureCount>0,name+' survived');report.mutations.push({name,rejected:true,failureCount:result.failureCount,firstFailure:result.failures[0]});save();}
 }report.allPass=true;console.log(JSON.stringify(report,null,2));}finally{save();device.destroy();gpu=null;}
