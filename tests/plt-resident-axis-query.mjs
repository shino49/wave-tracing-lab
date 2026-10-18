/** Reproduce bounded cursor-based axis guard tests.
 * Compile: node tests/plt-resident-axis-query.mjs
 * Execute: VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json \
 * WEBGPU_EXECUTION_BACKEND=vulkan node tests/plt-resident-axis-query.mjs --execute
 * No null-backend compilation is called execution. Synthetic query evidence
 * does not replace full transport shadow or establish unchecked admission.
 */
import fs from 'node:fs';import assert from 'node:assert/strict';import crypto from 'node:crypto';import {rayNearestBrute} from '../dist/plt-geometry.js';import {create,globals} from 'webgpu';Object.assign(globalThis,globals);const execute=process.argv.includes('--execute');const backend=execute?process.env.WEBGPU_EXECUTION_BACKEND:'null';if(execute)assert(['vulkan','metal','d3d11','d3d12','opengl','opengles'].includes(backend),'Explicit executable backend required; null is compile-only');const compileTimes={};let gpu=create([`backend=${backend}`]);let a=await gpu.requestAdapter();let d=await a.requestDevice();console.log('adapter ready');let source=['plt-resident-interval.wgsl','plt-resident-geometry.wgsl','plt-resident-axis-guard.wgsl'].map(x=>fs.readFileSync(new URL('../dist/'+x,import.meta.url),'utf8')).join('\n');source+=`
struct ProbeInput {origin:vec3f,error:vec3f,direction:vec3f,hit:RGHit}
@group(0) @binding(2) var<storage,read_write> prepared:array<RAQueryState>;
@group(0) @binding(3) var<storage,read_write> progress:array<RAScanProgress>;
@group(0) @binding(4) var<storage,read_write> planes:array<RAPlaneState>;
@group(0) @binding(5) var<storage,read_write> outputs:array<vec4u>;
fn probe_input(i:u32)->ProbeInput {
 var o=vec3f(0.0);var e=vec3f(0.0);var dir=vec3f(0,0,1);var h=RGHit(1u,0u,1.0,vec3f(0,0,1),vec3f(0,0,1),0u);
 if(i==1u){o=vec3f(0,0,1);e=vec3f(0,0,1e-12);dir=vec3f(0,0,-1);h=RGHit(1u,1u,2.0,vec3f(0,0,-1),vec3f(0,0,1),0u);}
 if(i==2u){h=RGHit();}if(i==3u){h.distance=1.001;}if(i==4u){o.y=-2.0;h.point.y=-2.0;}
 if(i==5u){dir=vec3f(1,0,0);h=RGHit(1u,2u,1.0,vec3f(1,0,0),vec3f(1,0,0),0u);}
 if(i==6u){o=vec3f(1,0,0);e=vec3f(1e-12,0,0);dir=vec3f(-1,0,0);h=RGHit(1u,3u,2.0,vec3f(-1,0,0),vec3f(1,0,0),0u);}
 if(i==7u){dir=vec3f(0,1,0);h=RGHit(1u,4u,1.0,vec3f(0,1,0),vec3f(0,1,0),0u);}
 if(i==8u){o=vec3f(0,1,0);e=vec3f(0,1e-12,0);dir=vec3f(0,-1,0);h=RGHit(1u,5u,2.0,vec3f(0,-1,0),vec3f(0,1,0),0u);}
 if(i==9u){o=vec3f(3,0,0);dir=vec3f(1,0,0);h=RGHit();}
 if(i==10u){o=vec3f(0,0,1);e=vec3f(0,0,2e-7);dir=vec3f(0,0,-1);h=RGHit(1u,1u,2.0,vec3f(0,0,-1),vec3f(0,0,1),0u);}
 if(i==11u){h.triangle=1u;h.point.z=-1.0;}
 if(i==12u){h.point.x=0.01;}
 if(i==13u){dir=vec3f(1,0,1e-5);h=RGHit(1u,2u,1.0,vec3f(1,0,1e-5),vec3f(1,0,0),0u);}
 if(i>=14u&&i<=18u){dir=vec3f(0.125,0.0625,1);h=RGHit(1u,0u,1.0097184,vec3f(0.125,0.0625,1),vec3f(0,0,1),0u);}
 if(i==15u){o=vec3f(0.125,0.0625,1);e=vec3f(1e-7,1e-7,1e-12);dir.z=-1;h=RGHit(1u,1u,2.0194368,vec3f(0.375,0.1875,-1),vec3f(0,0,1),0u);}
 if(i==16u){dir*=2.0;}
 if(i==17u){dir.x=-dir.x;dir.y=-dir.y;h.point.x=-h.point.x;h.point.y=-h.point.y;}
 return ProbeInput(o,e,dir,h);
}
@compute @workgroup_size(1) fn prepare_stage(@builtin(global_invocation_id) id:vec3u){let i=id.x;let p=probe_input(i);prepared[i]=ra_prepare(p.origin,p.error,p.direction,select(vec3f(0.0),vec3f(1e-8),i==18u));progress[i]=ra_scan_begin(p.hit,prepared[i]);}
@compute @workgroup_size(1) fn plane_stage(@builtin(global_invocation_id) id:vec3u){let i=id.x;let p=probe_input(i);planes[i]=ra_scan_plane(p.origin,p.error,prepared[i],progress[i]);}
@compute @workgroup_size(1) fn step_stage(@builtin(global_invocation_id) id:vec3u){let i=id.x;let p=probe_input(i);progress[i]=ra_scan_step(p.hit,planes[i],progress[i]);}
@compute @workgroup_size(1) fn finish_stage(@builtin(global_invocation_id) id:vec3u){let i=id.x;let p=probe_input(i);let r=ra_finish(p.origin,p.error,p.direction,p.hit,prepared[i],ra_scan_end(p.hit,progress[i]));outputs[i*2u]=vec4u(r.accepted,r.reason,r.distance.lo,r.distance.hi);outputs[i*2u+1u]=bitcast<vec4u>(vec4f(r.originError,r.planeResidual));}
`;let m=d.createShaderModule({code:source});let info=await m.getCompilationInfo();if(info.messages.some(x=>x.type==='error'))throw new Error(JSON.stringify(info.messages));let pipelines=[];for(let entryPoint of ['prepare_stage','plane_stage','step_stage','finish_stage']){console.log('compiling',entryPoint);let start=Date.now();pipelines.push(await d.createComputePipelineAsync({layout:'auto',compute:{module:m,entryPoint}}));compileTimes[entryPoint]=Date.now()-start;console.log('ready',entryPoint,compileTimes[entryPoint]);}
if(!execute){console.log(JSON.stringify({mode:'compile-only',pipelines:4,compileTimes}));d.destroy();gpu=null;process.exit(0);}
const count=19,nt=6,expected=[1,1,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1];
const names=['initial-z','mirror-z-with-residual','omitted-hit','wrong-distance','edge-contact','initial-x','mirror-x-with-residual','initial-y','mirror-y-with-residual','definite-escape','self-plane-near-uncertainty','wrong-identity','wrong-point','non-grazing-domain-rejection','oblique-initial','oblique-mirror-with-residual','oblique-nonunit-stored-direction','oblique-negative-tangents','oblique-direction-deviation'];
let buf=new ArrayBuffer((8+nt*5)*16),u=new Uint32Array(buf),f=new Float32Array(buf),triangles=[];u[2]=nt;u[8]=8;u[30]=1;f[31]=1;
for(let n=0;n<nt;n++){let k=[2,0,1][Math.floor(n/2)],i=(k+1)%3,j=(k+2)%3,plane=n%2?-1:1,pa=[0,0,0],pb=[0,0,0],pc=[0,0,0],normal=[0,0,0];pa[k]=pb[k]=pc[k]=plane;pa[i]=-2;pa[j]=-2;pb[i]=2;pb[j]=-2;pc[i]=0;pc[j]=2;normal[k]=1;let o=(8+n*5)*4;f.set(pa,o);f.set(pb,o+4);f.set(pc,o+8);f.set(normal,o+12);u[o+15]=0;u[o+19]=n;triangles.push({id:n,a:pa,b:pb,c:pc,normal,materialIndex:0});}
let scene=d.createBuffer({size:buf.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});d.queue.writeBuffer(scene,0,buf);let prep=d.createBuffer({size:count*64,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});let scan=d.createBuffer({size:count*52,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});let planes=d.createBuffer({size:count*88,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});let out=d.createBuffer({size:count*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});let read=d.createBuffer({size:count*32,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});let sets=[[[1,scene],[2,prep],[3,scan]],[[1,scene],[2,prep],[3,scan],[4,planes]],[[1,scene],[3,scan],[4,planes]],[[1,scene],[2,prep],[3,scan],[5,out]]];let groups=sets.map((entries,j)=>d.createBindGroup({layout:pipelines[j].getBindGroupLayout(0),entries:entries.map(([binding,buffer])=>({binding,resource:{buffer}}))}));for(let j of [0,...Array.from({length:nt},()=>[1,2]).flat(),3]){let enc=d.createCommandEncoder(),pass=enc.beginComputePass();pass.setPipeline(pipelines[j]);pass.setBindGroup(0,groups[j]);pass.dispatchWorkgroups(count);pass.end();d.queue.submit([enc.finish()]);await d.queue.onSubmittedWorkDone();console.log('executed stage',j);}

let enc=d.createCommandEncoder();enc.copyBufferToBuffer(out,0,read,0,count*32);d.queue.submit([enc.finish()]);await read.mapAsync(GPUMapMode.READ);let raw=read.getMappedRange(0,count*32).slice(0),ur=new Uint32Array(raw),fr=new Float32Array(raw);const rows=[];let verified=0;
for(let i=0;i<count;i++){let row={name:names[i],accepted:ur[i*8],reason:ur[i*8+1],distance:[fr[i*8+2],fr[i*8+3]],originError:[...fr.slice(i*8+4,i*8+7)],planeResidual:fr[i*8+7]};assert.equal(row.accepted,expected[i],names[i]);rows.push(row);}
// Independent existing CPU triangle implementation, including its normalization.
for(const [index,axis,mirror] of [[0,2,false],[1,2,true],[5,0,false],[6,0,true],[7,1,false],[8,1,true]]){const origin=[0,0,0],direction=[0,0,0];direction[axis]=mirror?-1:1;if(mirror)origin[axis]=1+5e-13;const hit=rayNearestBrute({triangles},origin,direction,{minDistance:1e-7,maxDistance:Infinity});assert(hit);assert.equal(hit.triangleId,[2,4,0][axis]+(mirror?1:0));const r=rows[index];assert(hit.distance>=r.distance[0]&&hit.distance<=r.distance[1]);for(let k=0;k<3;k++){let gpuPoint=k===axis?(mirror?-1:1):0;assert(Math.abs(hit.point[k]-gpuPoint)<=r.originError[k]);}verified++;}
assert.equal(rayNearestBrute({triangles},[3,0,0],[1,0,0],{minDistance:1e-7,maxDistance:Infinity}),null);verified++;
// Oblique directions exercise cancellation of the common query-normalization
// factor. The non-unit case verifies this is a ratio identity, not a hidden
// unit-vector assumption. Deviated CPU inputs remain within the GPU envelopes.
for(let index=14;index<=18;index++){
 const origin=index===15?[0.125+5e-8,0.0625-5e-8,1+5e-13]:[0,0,0];
 let direction=[0.125,0.0625,1];let gpuPoint=[0.125,0.0625,1];
 if(index===15){direction[2]=-1;gpuPoint=[0.375,0.1875,-1];}
 if(index===16)direction=direction.map(x=>2*x);
 if(index===17){direction[0]*=-1;direction[1]*=-1;gpuPoint=[-0.125,-0.0625,1];}
 if(index===18)direction=[0.125+5e-9,0.0625-5e-9,1+5e-9];
 const hit=rayNearestBrute({triangles},origin,direction,{minDistance:1e-7,maxDistance:Infinity});assert(hit);assert.equal(hit.triangleId,index===15?1:0);
 const r=rows[index];assert(hit.distance>=r.distance[0]&&hit.distance<=r.distance[1]);
 for(let k=0;k<3;k++)assert(Math.abs(hit.point[k]-gpuPoint[k])<=r.originError[k],names[index]+' component '+k);verified++;
}
const report={backend,execution:true,actualGPUExecuted:true,allPass:true,physicalGPUValidated:false,browserValidated:false,mode:'focused bounded query guard; full CPU shadow remains mandatory',shaderSHA256:crypto.createHash('sha256').update(source).digest('hex'),createdAt:new Date().toISOString(),pipelines:4,compileTimes,cases:count,accepted:rows.filter(r=>r.accepted).length,rejected:rows.filter(r=>!r.accepted).length,cpuComparisons:verified,rows};fs.writeFileSync(new URL('../research/plt-resident-axis-query-validation.json',import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));read.unmap();d.destroy();gpu=null;
