/** Native resident FSD math: Dawn/Tint compile, optional SwiftShader dispatch.
 * Uses an explicit test geometry adapter that returns known clipped wedges and
 * unobstructed endpoint-only legs. Integrated scene geometry is tested separately.
 * All saddles, proposal PDFs, draws, local UTD records and coherent power are
 * generated in WGSL; no CPU operator/event replay is uploaded.
 * WEBGPU_EXECUTION_BACKEND=vulkan VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json node tests/plt-resident-fsd.mjs --execute
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {create,globals} from 'webgpu';
import {makeAperture,sampleFSD,pdfFSD,evaluateFSD,wrappedNormalPDF,diffractionPointDirection,stableBrokenPathExcess} from '../dist/plt-fsd.js';
import {preparePLTScene,collectRegionGeometry} from '../dist/plt-geometry.js';
import {packResidentScene} from '../dist/plt-resident-geometry.js';
import {normalize,sub} from '../dist/mesh-geometry.js';
Object.assign(globalThis,globals);
const read=name=>fs.readFileSync(new URL(`../dist/${name}`,import.meta.url),'utf8');
const execute=process.argv.includes('--execute'),failures=[];let checks=0;
const test=async(name,fn)=>{try{await fn();checks++;console.log(`PASS ${name}`);}catch(e){failures.push({name,error:e.stack});console.error(`FAIL ${name}: ${e.stack}`);}};
const adapter=`
struct RGBeam {origin:vec3f,direction:vec3f,radius0:f32,tanAngle:f32,near:f32,far:f32}
struct RGRegion {center:vec3f,direction:vec3f,radius:f32,depth:f32}
struct RGEdge {valid:u32,index:u32,edgeId:u32,a:vec3f,b:vec3f,v:vec3f,e:vec3f,n1:vec3f,n2:vec3f,t1:vec3f,t2:vec3f,l:f32,alpha:f32,originalA:vec3f,originalB:vec3f,clippedStart:u32,clippedEnd:u32}
@group(0) @binding(0) var<storage,read> test_jobs:array<vec4f>;
@group(0) @binding(1) var<storage,read> test_edges:array<vec4f>;
@group(0) @binding(2) var<storage,read_write> test_out:array<vec4f>;
var<private> rg_flags:u32;
var<private> test_zero:bool;
fn rg_wedge_count()->u32 {return u32(test_edges[0].x);}
fn rg_clip(index:u32,region:RGRegion)->RGEdge {
 var e:RGEdge;if(index>=rg_wedge_count()){return e;}let p=1u+index*10u;
 e.valid=1u;e.index=index;e.edgeId=u32(test_edges[p].w);e.a=test_edges[p].xyz;e.b=test_edges[p+1u].xyz;
 e.v=(e.a+e.b)*0.5;e.l=distance(e.a,e.b);e.e=normalize(e.b-e.a);e.alpha=test_edges[p+1u].w;
 e.n1=test_edges[p+2u].xyz;e.n2=test_edges[p+3u].xyz;e.t1=test_edges[p+4u].xyz;e.t2=test_edges[p+5u].xyz;
 e.originalA=test_edges[p+6u].xyz;e.originalB=test_edges[p+7u].xyz;
 e.clippedStart=u32(test_edges[p+8u].x);e.clippedEnd=u32(test_edges[p+8u].y);return e;
}
fn rg_clip_active(index:u32,region:RGRegion)->RGEdge {return rg_clip(index,region);}
fn rg_world(a:vec3f)->vec3f {return a;}
fn rg_endpoint_epsilon(a:vec3f,b:vec3f)->f32 {return max(1e-10,2.842170943040401e-14*max(1.0,max(max(abs(a.x),max(abs(a.y),abs(a.z))),max(abs(b.x),max(abs(b.y),abs(b.z))))));}
fn rg_visible(a:vec3f,b:vec3f,epsilon:f32)->bool {return test_edges[0].y==0.0;}
fn rg_cone_contains(beam:RGBeam,point:vec3f,tolerance:f32)->bool {let q=point-beam.origin;let t=dot(q,beam.direction);return t>=beam.near-tolerance&&t<=beam.far+tolerance&&length(q-beam.direction*t)<=beam.radius0+beam.tanAngle*t+tolerance;}
fn rr_word(state:ptr<function,u32>)->u32 {
 *state+=0x6d2b79f5u;var t=*state;t=(t^(t>>15u))*(t|1u);t^=t+((t^(t>>7u))*(t|61u));return t^(t>>14u);
}
fn rr_float(state:ptr<function,u32>)->f32 {let word=rr_word(state);if(test_zero){return 0.0;}return min(f32(word)*2.3283064365386963e-10,bitcast<f32>(0x3f7fffffu));}
fn rr_choice(state:ptr<function,u32>,count:u32)->u32 {let word=rr_word(state);if(test_zero){return 0u;}return rf_discrete(word,count);}
`;
const entry=`
@compute @workgroup_size(64) fn rf_test(@builtin(global_invocation_id) invocation:vec3u){
 let i=invocation.x;if(i>=arrayLength(&test_jobs)/8u){return;}let b=i*8u;let o=i*16u;
 let source=test_jobs[b].xyz;let k=test_jobs[b].w;let destination=test_jobs[b+1u].xyz;let boundary=u32(test_jobs[b+1u].w);
 let incoming=test_jobs[b+3u].xyz;
 let region=RGRegion(test_jobs[b+2u].xyz,incoming,test_jobs[b+2u].w,test_jobs[b+3u].w);
 let beam=RGBeam(test_jobs[b+4u].xyz,test_jobs[b+5u].xyz,test_jobs[b+4u].w,test_jobs[b+5u].w,test_jobs[b+6u].x,test_jobs[b+6u].y);
 rg_flags=0u;test_zero=test_jobs[b+6u].w==8.0;
 var rng=bitcast<u32>(test_jobs[b+6u].z);
 let event=rf_evaluate(region,incoming,k,boundary,source,destination,beam);let evaluationFlags=rg_flags;
 let record=rf_record(region,incoming,k,source,destination,0u);
 let sample=rf_sample(region,incoming,k,source,&rng);
 test_out[o]=vec4f(event.F,f32(event.edgeCount),event.directAmplitude,f32(event.directBoundary));
 test_out[o+1u]=vec4f(event.soft,event.hard);
 test_out[o+2u]=vec4f(f32(rg_flags),f32(event.diagnostics),f32(event.endpointExclusions),f32(event.invalidEdges));
 test_out[o+3u]=vec4f(record.point,record.ri);
 test_out[o+4u]=vec4f(record.ro,record.sinBeta,record.phiI,record.phiO);
 test_out[o+5u]=vec4f(record.n,record.L,record.phaseExcess,record.fermatResidual);
 test_out[o+6u]=vec4f(f32(record.valid),f32(record.edgeId),f32(record.excluded),f32(record.diagnostics));
 test_out[o+7u]=vec4f(sample.direction,sample.weight);
 test_out[o+8u]=vec4f(sample.pdf,f32(sample.valid),bitcast<f32>(sample.edgeId),f32(sample.count));
 test_out[o+9u]=vec4f(sample.point,f32(sample.isDirect));
 test_out[o+10u]=vec4f(bitcast<f32>(rng),f32(sample.selected),bitcast<f32>(event.edgeHash),f32(evaluationFlags));
 let edge=rf_orient(0u,region,normalize(incoming));let saddle=rf_direction_saddle(edge.edge,source,sample.direction);
 test_out[o+11u]=vec4f(saddle.point,f32(saddle.valid));
 test_out[o+13u]=vec4f(rf_sincos(test_jobs[b+7u].x),f32(rg_flags),0.0);
 test_out[o+12u]=vec4f(rf_wrapped_normal(test_jobs[b+7u].x,test_jobs[b+7u].y),rf_excess(source,region.center,destination),f32(rf_discrete(bitcast<u32>(test_jobs[b+7u].z),bitcast<u32>(test_jobs[b+7u].w))),0.0);
}
`;
const source=[read('utd-functions.wgsl'),adapter,read('plt-resident-fsd.wgsl'),entry].join('\n');
const edge=(id=0,y=0,l=40)=>({id,edgeId:id,a:[0,y-l/2,0],b:[0,y+l/2,0],v:[0,y,0],e:[0,1,0],l,alpha:0,n1:[0,0,1],n2:[0,0,-1],t1:[1,0,0],t2:[1,0,0],originalA:[0,y-l/2,0],originalB:[0,y+l/2,0],clippedStart:false,clippedEnd:false});
const emptyScene=preparePLTScene({vertices:[[10000,10000,10000],[10001,10000,10000],[10000,10001,10000]],faces:[[0,1,2]]});
const defaultJob={source:[-3,0,-4],destination:[5,2,4],k:100,boundary:'unpolarized',center:[0,0,0],radius:20,depth:40,incoming:normalize([3,0,4]),seed:171,angle:.123,sigma:.3,word:0xffffffff,count:65537};
function job(j={}){j={...defaultJob,...j};j.beam=j.beam??{origin:j.source,direction:normalize(sub(j.destination,j.source)),radius0:50,tanAngle:.1,near:0,far:1000};return j;}
function cpu(j,edges){const aperture=makeAperture({center:j.center,wedges:edges},j.incoming,j.k,j.boundary);return{aperture,event:evaluateFSD(emptyScene,aperture,j.source,j.destination,j.beam)};}
function random(seed){let state=seed>>>0;const fn=()=>{state=(state+0x6d2b79f5)>>>0;let t=state;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};fn.state=()=>state;return fn;}
function packJobs(jobs){const a=new Float32Array(jobs.length*32),u=new Uint32Array(a.buffer);for(const [i,j] of jobs.entries()){const b=i*32;a.set([...j.source,j.k,...j.destination,['soft','hard','unpolarized'].indexOf(j.boundary),...j.center,j.radius,...j.incoming,j.depth,...j.beam.origin,j.beam.radius0,...j.beam.direction,j.beam.tanAngle,j.beam.near,j.beam.far,0,j.mode??0,j.angle,j.sigma,0,0],b);u[b+26]=j.seed;u[b+30]=j.word;u[b+31]=j.count;}return a;}
function packEdges(edges,blocked=false){const data=new Float32Array(4+edges.length*40);data[0]=edges.length;data[1]=+blocked;edges.forEach((e,i)=>data.set([...e.a,e.edgeId,...e.b,e.alpha,...e.n1,0,...e.n2,0,...e.t1,0,...e.t2,0,...e.originalA,0,...e.originalB,0,+e.clippedStart,+e.clippedEnd,0,0,0,0,0,0],4+i*40));return data;}
const near=(actual,expected,label,relative=.003,absolute=2e-5)=>assert(Number.isFinite(actual)&&Math.abs(actual-expected)<=absolute+relative*Math.abs(expected),`${label}: ${actual} != ${expected}`);
let gpu,device,pipeline;
async function dispatch(jobs,edges,blocked=false){const buffers=[];const buffer=(data,usage)=>{const b=device.createBuffer({size:Math.max(16,data.byteLength??data),usage});buffers.push(b);if(data.byteLength)device.queue.writeBuffer(b,0,data);return b;};try{const input=buffer(packJobs(jobs),GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),edgeb=buffer(edges.data??packEdges(edges,blocked),GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),out=buffer(jobs.length*256,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC),readback=buffer(jobs.length*256,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:input}},{binding:1,resource:{buffer:edgeb}},{binding:2,resource:{buffer:out}}]});device.pushErrorScope('validation');const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(jobs.length/64));pass.end();encoder.copyBufferToBuffer(out,0,readback,0,jobs.length*256);device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();assert.equal(await device.popErrorScope(),null);await readback.mapAsync(GPUMapMode.READ);const result=new Float32Array(readback.getMappedRange().slice(0));readback.unmap();return jobs.map((_,i)=>result.slice(i*64,(i+1)*64));}finally{for(const b of buffers)b.destroy();}}
await test('Dawn/Tint compiles native resident FSD include and executable entry',async()=>{gpu=create(['backend=null']);const adapter=await gpu.requestAdapter();device=await adapter.requestDevice();device.pushErrorScope('validation');const module=device.createShaderModule({code:source});const messages=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');assert.deepEqual(messages,[]);assert.equal(await device.popErrorScope(),null);pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'rf_test'}});device.destroy();device=null;gpu=null;});
if(execute&&failures.length===0){
 const backend=process.env.WEBGPU_EXECUTION_BACKEND;assert(backend&&backend!=='null','Select a real execution backend explicitly');gpu=create([`backend=${backend}`]);const adapter=await gpu.requestAdapter();device=await adapter.requestDevice();pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module:device.createShaderModule({code:source}),entryPoint:'rf_test'}});
 try{
  await test('production geometry and FSD generate real endpoint-leg records on GPU',async()=>{
    // One scene wedge per invocation mirrors the staged resident renderer and
    // avoids compiling a duplicated BVH/coefficient call tree in one kernel.
    const productionEntry=`
      @group(0) @binding(0) var<storage,read> test_jobs:array<vec4f>;
      @group(0) @binding(2) var<storage,read_write> test_out:array<vec4f>;
      @compute @workgroup_size(64) fn rf_production_test(@builtin(global_invocation_id) id:vec3u){
        let i=id.x;if(i>=arrayLength(&test_jobs)/8u){return;}let b=i*8u;let o=i*16u;
        let region=RGRegion(test_jobs[b+2u].xyz,test_jobs[b+3u].xyz,test_jobs[b+2u].w,test_jobs[b+3u].w);
        rg_flags=0u;let record=rf_record(region,test_jobs[b+3u].xyz,test_jobs[b].w,test_jobs[b].xyz,test_jobs[b+1u].xyz,bitcast<u32>(test_jobs[b+7u].z));
        test_out[o+3u]=vec4f(record.point,record.ri);test_out[o+4u]=vec4f(record.ro,record.sinBeta,record.phiI,record.phiO);
        test_out[o+5u]=vec4f(record.n,record.L,record.phaseExcess,record.fermatResidual);
        test_out[o+6u]=vec4f(f32(record.valid),f32(record.edgeId),f32(record.excluded),f32(record.diagnostics));
        test_out[o+10u]=vec4f(0,0,0,f32(rg_flags));
      }`;
    const fullSource=[read('utd-functions.wgsl'),read('plt-resident-geometry.wgsl'),read('plt-resident-rng.wgsl'),read('plt-resident-fsd.wgsl'),productionEntry].join('\n');
    const module=device.createShaderModule({code:fullSource});assert.deepEqual((await module.getCompilationInfo()).messages.filter(x=>x.type==='error'),[]);
    const previous=pipeline;pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'rf_production_test'}});
    try{const scene=preparePLTScene({vertices:[[0,-100,0],[100,-100,0],[100,100,0],[0,100,0]],faces:[[0,1,2],[0,2,3]]}),packed=packResidentScene(scene),world=job(),local=job({...world,source:packed.toLocal(world.source),destination:packed.toLocal(world.destination),center:packed.toLocal(world.center),beam:{...world.beam,origin:packed.toLocal(world.beam.origin)}});
      const region=collectRegionGeometry(scene,{center:world.center,direction:world.incoming,radius:world.radius,depth:world.depth}),aperture=makeAperture(region,world.incoming,world.k,world.boundary),expected=evaluateFSD(scene,aperture,world.source,world.destination,world.beam),results=await dispatch(scene.wedges.map((_,word)=>job({...local,word})),packed),accepted=results.filter(r=>r[24]);
      assert.equal(accepted.length,expected.edgeCount);for(const r of results)assert.equal(r[43],0,'ordinary endpoint-only FSD legs must be admitted');
      for(const r of accepted){const rec=expected.records.find(e=>e.edgeId===r[25]);assert(rec);const point=packed.toLocal(rec.point);[...point,rec.ri,rec.ro,rec.sinBeta,rec.phiI,rec.phiO,rec.n,rec.L,rec.phaseExcess,rec.fermatResidual].forEach((x,c)=>near(r[12+c],x,'production record',3e-6,2e-5));}
    }finally{pipeline=previous;}
  });
  await test('native soft/hard/unpolarized direct-plus-edge coherent fields agree with CPU',async()=>{const edges=[edge()],jobs=['soft','hard','unpolarized'].flatMap(boundary=>[[-3,0,4],[5,2,4],[4,0,-3],[3,0,4]].map(destination=>job({boundary,destination})));const got=await dispatch(jobs,edges);jobs.forEach((j,i)=>{const e=cpu(j,edges).event,r=got[i];assert.equal(r[43],0,'evaluation must be admitted');near(r[0],e.F,'F');assert.equal(r[1],e.edgeCount);assert.equal(r[2],e.directWeight);assert.equal(r[3],+e.diagnostics.directBoundary);[...e.soft,...e.hard].forEach((x,c)=>near(r[4+c],x,'field'));if(e.records.length){const rec=e.records[0];[...rec.point,rec.ri,rec.ro,rec.sinBeta,rec.phiI,rec.phiO,rec.n,rec.L,rec.phaseExcess,rec.fermatResidual].forEach((x,c)=>near(r[12+c],x,'record'));}});});
  await test('convex wedge orientation and rigid 3-D transforms retain native fields',async()=>{
    const base={...edge(),alpha:Math.PI/2,n2:[-1,0,0],t2:[0,0,-1]};
    for(const angle of [0,.371,1.139]){const c=Math.cos(angle),s=Math.sin(angle),rotate=([x,y,z])=>[c*x-s*y,s*x+c*y,z],e={...base};for(const key of ['a','b','v','e','n1','n2','t1','t2','originalA','originalB'])e[key]=rotate(base[key]);
      const j=job({source:rotate(defaultJob.source),destination:rotate(defaultJob.destination),incoming:rotate(defaultJob.incoming)}),[r]=await dispatch([j],[e]),expected=cpu(j,[e]).event;assert.equal(r[43],0);assert.equal(r[1],expected.edgeCount);near(r[0],expected.F,'rotated convex FSD');[...expected.soft,...expected.hard].forEach((x,i)=>near(r[4+i],x,'rotated convex field'));
    }
  });
  await test('native wrapped-normal image/Fourier branches and high-word choices',async()=>{const jobs=[.025,.1,.4,.99,1,2,5,30].flatMap(sigma=>[-12.4,-3,-.123,0,.123,3,13.4].map(angle=>job({sigma,angle})));const got=await dispatch(jobs,[edge()]);jobs.forEach((j,i)=>{near(got[i][48],wrappedNormalPDF(j.angle,j.sigma),'normal',.003,2e-5);assert.equal(got[i][50],65536);});const choices=[0,1,0x7fffffff,0x80000000,0xffffff00,0xffffffff].flatMap(word=>[1,2,3,257,65535,65536,65537].map(count=>job({word,count})));const results=await dispatch(choices,[edge()]);choices.forEach((j,i)=>assert.equal(results[i][50],Number((BigInt(j.word)*BigInt(j.count))>>32n)));});
  await test('deterministic polynomial trig meets fixed strict geometry bounds over f32 angles',async()=>{
    const angles=[0,-0,1e-8,-1e-8,Math.PI/4,-Math.PI/4,Math.PI/2,-Math.PI/2,Math.PI,-Math.PI,2*Math.PI,-2*Math.PI,99999.9,-99999.9,100000,-100000];
    for(let i=0;i<1024;i++){angles.push((i-511.5)*100000/512);angles.push((i-511.5)*Math.PI/256);}
    for(let i=-40;i<=40;i++)for(const offset of [-1e-6,0,1e-6])angles.push(i*Math.PI/2+offset);
    const jobs=angles.map(angle=>job({angle:Math.fround(angle)})),got=await dispatch(jobs,[edge()]);let maxError=0;
    jobs.forEach((j,i)=>{const values=[Math.cos(j.angle),Math.sin(j.angle)];values.forEach((x,c)=>{const actual=got[i][52+c];maxError=Math.max(maxError,Math.abs(actual-x));near(actual,x,'polynomial trig',3e-6,2e-5);assert(Math.abs(actual-x)<1e-6,`f32 trig regression at ${j.angle}: ${actual} vs ${x}`);});assert.equal(got[i][54]&4,0);});
    console.log(`Native bounded f32 trig maximum absolute error: ${maxError}`);
    const rejected=await dispatch([job({angle:100001}),job({angle:-100001})],[edge()]);for(const r of rejected)assert(r[54]&4,'unbounded proposal trig must fail closed');
  });
  await test('native Mulberry samples preserve branch/seed and angular PDF',async()=>{const jobs=Array.from({length:256},(_,i)=>job({seed:i+1})),edges=[edge()],results=await dispatch(jobs,edges);let direct=0,edgeDraws=0;for(let i=0;i<jobs.length;i++){const j=jobs[i],r=results[i],a=cpu(j,edges).aperture,rng=random(j.seed),s=sampleFSD(a,j.source,rng);assert.equal(r[33],+s.valid);assert.equal(r[39],+s.isDirect);assert.equal(new Uint32Array(r.buffer)[40],rng.state());if(!s.valid)continue;assert.equal(r[8],0,'sample must be admitted');near(r[31],s.weight,'weight');near(r[32],s.pdf.value,'pdf');s.wo.forEach((x,c)=>near(r[28+c],x,'sample direction',3e-6,2e-5));s.point.forEach((x,c)=>near(r[36+c],x,'sample point',3e-6,2e-5));if(s.isDirect)direct++;else{edgeDraws++;near(r[32],pdfFSD(a,j.source,[...r.slice(28,31)]),'rescanned pdf');}}assert(direct>80&&edgeDraws>80);});
  await test('all 257 native wedges participate without a 128-edge cap',async()=>{const edges=Array.from({length:257},(_,i)=>edge(i,0,40)),j=job(),[r]=await dispatch([j],edges),e=cpu(j,edges).event;assert.equal(r[1],257);assert.equal(r[35],257);assert.equal(r[43],0);near(r[0],e.F,'257 edge coherent power');[...e.soft,...e.hard].forEach((x,c)=>near(r[4+c],x,'257 edge field'));});
  await test('finite directional saddle is reconstructed on GPU from sampled direction',async()=>{const edges=[edge()],jobs=Array.from({length:64},(_,i)=>job({seed:600+i})),results=await dispatch(jobs,edges);for(let i=0;i<jobs.length;i++){const r=results[i];if(!r[33]||r[39])continue;const a=cpu(jobs[i],edges).aperture,p=diffractionPointDirection(a.edges[0],jobs[i].source,[...r.slice(28,31)]);assert(p);assert.equal(r[47],1);p.point.forEach((x,c)=>near(r[44+c],x,'directional point'));}});
  await test('finite endpoint exclusions and low-frequency warnings survive native evaluation',async()=>{const edges=[edge(0,0,2)],jobs=[job({source:[-3,5,-4],destination:[5,5,4]}),job({source:[-3,1,-4],destination:[5,1,4]}),job({k:.1})],results=await dispatch(jobs,edges);assert.equal(results[0][1],0);assert.equal(results[0][10],1);assert.equal(results[1][1],0);assert.equal(results[1][10],1);assert(results[1][8]&2,'endpoint predicate must fail closed');assert(results[2][9]&8);assert(results[2][9]&32);});
  await test('endpoint-only blocked leg adapter suppresses all edge and direct transport',async()=>{const [r]=await dispatch([job()],[edge()],true);assert.equal(r[0],0);assert.equal(r[1],0);assert.equal(r[2],0);});
  await test('empty aperture remains direct discrete sampling with no fabricated UTD',async()=>{const j=job(),[r]=await dispatch([j],[]);assert.equal(r[0],1);assert.equal(r[1],0);assert.equal(r[31],1);assert.equal(r[32],1);assert.equal(r[35],0);assert.equal(r[39],1);});
  await test('zero Box-Muller is explicitly rejected rather than silently floored',async()=>{const [r]=await dispatch([job({mode:8})],[edge()]);assert(r[8]&8);assert.equal(r[33],0);assert.equal(r[31],0);});
  await test('phase above 100000 fails closed before coefficient use',async()=>{const [r]=await dispatch([job({k:1e8})],[edge()]);assert(r[8]&4);assert(r[9]&256);assert.equal(r[1],0);});
  await test('seam threshold remains 2e-6 and nearby ambiguous classification fails closed',async()=>{const phi=Math.atan2(4,3),source=[-3,0,-4],destination=[5*Math.cos(phi+2.1e-6),0,5*Math.sin(phi+2.1e-6)],j=job({source,destination}),[r]=await dispatch([j],[edge()]);assert(r[43]&2);const e=cpu(j,[edge()]).event;assert.equal(e.directWeight,1,'CPU semantic epsilon is unchanged');assert.match(read('utd-functions.wgsl'),/UTD_BOUNDARY_EPS:f32 = 0\.000002/);/* Ambiguous GPU classification is deliberately untrusted, regardless of its provisional weight. */});
  await test('stable broken-path excess avoids long nearly-equal subtraction',async()=>{const j=job({source:[-1e9,0,0],destination:[1e9,0,0],center:[0,1,0],k:1}),[r]=await dispatch([j],[edge()]);near(r[49],stableBrokenPathExcess(j.source,j.center,j.destination),'excess',.003,1e-14);assert(r[49]>0);});
 }finally{device.destroy();device=null;gpu=null;}
}
console.log(JSON.stringify({suite:'plt-resident-fsd',checks,failures,executionRequested:execute,nativeExecution:execute&&failures.length===0,browserValidated:false,physicalGPUValidated:false,fullWaveValidated:false,scope:'GPU-generated FSD saddles, sampling, PDF, UTD records, and local coherent intensity with explicit clipped-wedge/visibility test adapter'}));
if(failures.length)process.exitCode=1;
