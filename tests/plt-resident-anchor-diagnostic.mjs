/** Read-only native diagnostic of baseline v19 camera-box uncertainty.
 * No candidate admission. Staged kernels inspect true GPU camera / geometry
 * and baseline compact interval operators. A forced diagnostic continuation
 * after rejection is labeled and cannot update any production output.
 * VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json WEBGPU_EXECUTION_BACKEND=vulkan node tests/plt-resident-anchor-diagnostic.mjs
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {create,globals} from 'webgpu';
import {preparePLTScene,rayNearestBrute} from '../dist/plt-geometry.js';
import {randomGenerator,sampleCamera,cameraFrame} from '../dist/plt-core.js';
import {normalize} from '../dist/mesh-geometry.js';
import {packResidentScene} from '../dist/plt-resident-geometry.js';
import {packResidentParameters} from '../dist/plt-resident-gpu.js';
Object.assign(globalThis,globals);
const baseline='e4d5d8a6090606ac028ce5cb06e2a17f7ecadd89';
const files=['utd-functions.wgsl','plt-resident-geometry.wgsl','plt-resident-interval.wgsl','plt-resident-axis-guard.wgsl','plt-resident-rng.wgsl','plt-resident-fsd.wgsl','plt-resident-walker.wgsl'];
let source=files.map(f=>execFileSync('git',['show',`${baseline}:dist/${f}`],{encoding:'utf8',cwd:new URL('..',import.meta.url)})).join('\n');
source=source.replace('let unit=ra_cpu_normalize(raw);','ad_raw=raw;let unit=ra_cpu_normalize(raw);');
source=source.replace('let error=vec3f(rri_error(normalized.x','ad_normalized=normalized;let error=vec3f(rri_error(normalized.x');
const shader=source+`
var<private> ad_raw:RAVec3;
var<private> ad_normalized:RAVec3;
struct ADState { raw:RAVec3, normalized:RAVec3, prepared:RAQueryState, plane:RAPlaneState, hit:RGHit, pointErrors:vec3f, rawPointErrors:vec3f, depth:u32 }
@group(0) @binding(9) var<storage,read_write> ads:array<ADState>;
@group(0) @binding(10) var<storage,read_write> ado:array<vec4f>;
fn ad_put(i:u32,k:u32,v:vec3f){ado[i*64u+k]=vec4f(v,0);}
fn ad_interval(i:u32,k:u32,v:RAI){ado[i*64u+k]=vec4f(v.lo,v.hi,0,0);}
fn ad_vector(i:u32,k:u32,v:RAVec3){ado[i*64u+k]=vec4f(v.x.lo,v.y.lo,v.z.lo,0);ado[i*64u+k+1u]=vec4f(v.x.hi,v.y.hi,v.z.hi,0);}
@compute @workgroup_size(32) fn ad_camera(@builtin(global_invocation_id) id:vec3u){
 let i=id.x;if(i>=arrayLength(&rr_initial)){return;}rr_lane_index=i;var s:RRLane;s.header=vec4u(rr_initial[i].xy,0u,0u);rr_start(&s);rr_lanes[i]=s;
 var a:ADState;a.raw=ad_raw;a.normalized=ad_normalized;ads[i]=a;
 ad_vector(i,0u,a.raw);ad_vector(i,2u,a.normalized);ad_put(i,4u,s.direction.xyz);ad_put(i,5u,s.intervalDirectionError.xyz);ad_put(i,6u,s.origin.xyz);ad_put(i,7u,s.intervalOriginError.xyz);
}
@compute @workgroup_size(32) fn ad_geometry(@builtin(global_invocation_id) id:vec3u){let i=id.x;if(i>=arrayLength(&rr_lanes)){return;}var a=ads[i];var s=rr_lanes[i];rg_flags=0u;a.hit=rg_ray(rr_beam(&s));ads[i]=a;
 let b=8u+a.depth*28u;ad_put(i,b,s.origin.xyz);ad_put(i,b+1u,s.direction.xyz);ad_put(i,b+2u,s.intervalOriginError.xyz);ad_put(i,b+3u,a.hit.point);ado[i*64u+b+4u]=vec4f(a.hit.distance,f32(a.hit.triangle),f32(a.hit.valid),f32(rg_flags));}
@compute @workgroup_size(32) fn ad_prepare(@builtin(global_invocation_id) id:vec3u){let i=id.x;if(i>=arrayLength(&rr_lanes)){return;}var a=ads[i];let s=rr_lanes[i];a.prepared=ra_prepare_unit(s.origin.xyz,s.intervalOriginError.xyz,s.direction.xyz,s.intervalDirectionError.xyz,1u);ads[i]=a;let b=8u+a.depth*28u;ad_vector(i,b+5u,a.prepared.stored);ad_vector(i,b+7u,a.prepared.query);}
@compute @workgroup_size(32) fn ad_plane(@builtin(global_invocation_id) id:vec3u){let i=id.x;if(i>=arrayLength(&rr_lanes)){return;}var a=ads[i];let s=rr_lanes[i];ra_bad=false;a.plane=ra_plane_query(rg_triangle(a.hit.triangle),a.hit.triangle,s.origin.xyz,s.intervalOriginError.xyz,a.prepared.query,a.prepared.kappa);ads[i]=a;let b=8u+a.depth*28u;ad_interval(i,b+9u,a.plane.distance);ad_interval(i,b+10u,a.plane.planeDelta);}
@compute @workgroup_size(32) fn ad_normalized_point(@builtin(global_invocation_id) id:vec3u){let i=id.x/3u;let k=id.x%3u;if(i>=arrayLength(&rr_lanes)){return;}let a=ads[i];let s=rr_lanes[i];ra_bad=false;let axis=a.plane.axis;let b=8u+a.depth*28u;var error=0.0;
 if(k!=axis){let den=ra_component(a.prepared.stored,axis);let ratio=ra_idiv(ra_component(a.prepared.stored,k),den);ad_interval(i,b+11u+k,ratio);let product=ra_imul(ratio,a.plane.planeDelta);ad_interval(i,b+14u+k,product);error=ra_point_error(s.origin[k],s.intervalOriginError[k],ra_component(a.prepared.stored,k),den,a.plane.planeDelta,a.prepared.kappa,a.hit.point[k]);}
 else{let residual=ra_iadd(ra_imul(ra_imul(ra_point(ra_absmax(a.plane.planeDelta)),a.prepared.kappa),ra_ipoint(0x28000000u)),ra_ipoint(0x2a800000u));error=residual.hi;}
 ads[i].pointErrors[k]=error;ado[i*64u+b+17u][k]=error;
}
@compute @workgroup_size(32) fn ad_raw_point(@builtin(global_invocation_id) id:vec3u){let i=id.x/3u;let k=id.x%3u;if(i>=arrayLength(&rr_lanes)){return;}let a=ads[i];let s=rr_lanes[i];ra_bad=false;let axis=a.plane.axis;let b=8u+a.depth*28u;var error=a.pointErrors[k];
 if(k!=axis){let den=ra_component(a.raw,axis);let ratio=ra_idiv(ra_component(a.raw,k),den);ad_interval(i,b+18u+k,ratio);let product=ra_imul(ratio,a.plane.planeDelta);ad_interval(i,b+21u+k,product);error=ra_point_error(s.origin[k],s.intervalOriginError[k],ra_component(a.raw,k),den,a.plane.planeDelta,a.prepared.kappa,a.hit.point[k]);}
 ads[i].rawPointErrors[k]=error;ado[i*64u+b+24u][k]=error;
}
@compute @workgroup_size(32) fn ad_finish(@builtin(global_invocation_id) id:vec3u){let i=id.x;if(i>=arrayLength(&rr_lanes)){return;}let a=ads[i];let s=rr_lanes[i];ra_bad=false;let tri=ra_plane_test(a.plane);let scanned=RAScanState(select(0u,1u,tri.classification==1u),tri.reason,1u,tri);let finished=ra_finish(s.origin.xyz,s.intervalOriginError.xyz,s.direction.xyz,a.hit,a.prepared,scanned);let b=8u+a.depth*28u;ado[i*64u+b+25u]=vec4f(f32(tri.classification),f32(tri.reason),f32(finished.accepted),f32(finished.reason));}
@compute @workgroup_size(32) fn ad_advance(@builtin(global_invocation_id) id:vec3u){let i=id.x;if(i>=arrayLength(&rr_lanes)){return;}var a=ads[i];var s=rr_lanes[i];let axis=a.plane.axis;s.origin=vec4f(a.hit.point,0);s.direction[axis]=-s.direction[axis];s.intervalOriginError=vec4f(a.pointErrors,0);if(axis==0u){a.raw.x=ra_ineg(a.raw.x);}if(axis==1u){a.raw.y=ra_ineg(a.raw.y);}if(axis==2u){a.raw.z=ra_ineg(a.raw.z);}a.depth+=1u;ads[i]=a;rr_lanes[i]=s;}
`;
const options={width:8,height:8,samplesPerPixel:3,maxDepth:6,wave:false,nee:false,seed:7391};
const materials=[{type:'mirror',albedo:.5,emission:0},{type:'mirror',albedo:.75,emission:0},{type:'diffuse',albedo:0,emission:8,twoSidedEmission:true}];
function quad(axis,position,lo,hi,materialIndex){const a=(axis+1)%3,b=(axis+2)%3,p=(u,v)=>{const p=[0,0,0];p[axis]=position;p[a]=u;p[b]=v;return p;};return {mesh:{vertices:[p(lo,lo),p(hi,lo),p(hi,hi),p(lo,hi)],faces:[[0,1,2],[0,2,3]]},materialIndex};}
function fixture(name,scale=1,shift=[0,0,0],perm=[0,1,2]){const entries=[quad(0,2,-8,8,0),quad(2,3,-8,8,1),quad(2,-2,-16,16,2)];const point=p=>perm.map((a,k)=>scale*p[a]+shift[k]);for(const e of entries)e.mesh.vertices=e.mesh.vertices.map(point);const camera={position:point([0,0,0]),target:point([4,1,4]),up:perm.map(a=>[0,1,0][a]),fov:10};const scene=preparePLTScene(entries,{materials});scene.camera=camera;return {name,scene,camera,scale,shift,perm};}
const fixtures=[fixture('two-mirrors-emission'),fixture('four-scale-shifted-zxy',4,[-32,16,-8],[2,0,1])];
// Independent exact integer proof of the raw-ratio normal-only CPU64 budget.
// A(kappa)=((1+u)^4/(1-u)^2)*(1+32*kappa*u)-1.
// A(kappa)/kappa decreases for kappa>=1; proving kappa=1 suffices.
const N=1n<<53n,den=N**3n*(N-1n)**2n,num=(N+1n)**4n*(N+32n);
assert((num-den)*N<39n*den,'Raw camera cancellation needs <39*kappa*u64');
const lowerDen=N**3n*(N+1n)**2n,lowerNum=(N-1n)**4n*(N-32n);
assert((lowerDen-lowerNum)*N<39n*lowerDen,'Raw camera lower budget');
const backend=process.env.WEBGPU_EXECUTION_BACKEND??'vulkan';let gpu=create([`backend=${backend}`]);const adapter=await gpu.requestAdapter();assert(adapter);const device=await adapter.requestDevice();
const names=['ad_camera','ad_geometry','ad_prepare','ad_plane','ad_normalized_point','ad_raw_point','ad_finish','ad_advance'];
const module=device.createShaderModule({code:shader});const info=await module.getCompilationInfo();assert.deepEqual(info.messages.filter(x=>x.type==='error'),[]);
const pipelines={};for(const name of names){let t=performance.now();pipelines[name]=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:name,constants:{RR_AXIS_INTERVAL_ENABLED:1}}});console.log('compiled',name,performance.now()-t);}
const report={baseline,createdAt:new Date().toISOString(),actualGPUExecuted:true,allPass:false,fullCPUShadowMandatory:true,physicalGPUValidated:false,browserValidated:false,hardwarePerformanceClaim:false,backend,adapter:adapter.info.description,sourceSHA256:crypto.createHash('sha256').update(shader).digest('hex'),roundingProof:{u64:'2^-53',upperFactor:'((1+u)^4/(1-u)^2)*(1+32*kappa*u)-1',relativeBoundU64:39,existingBudgetU64:64,absoluteWorldAdditionBudget:'4*u64*512',normalOnly:true,tinyComponentPolicy:'Additional gradual-underflow handling or rejection required; no claim of a complete raw-ratio admission proof'},scope:'Diagnostic only; candidate-triangle audit is not complete scene admission; baseline normalized origin errors are propagated beyond rejection solely to inspect bottlenecks',cases:[]};
const save=()=>fs.writeFileSync(new URL('../research/plt-resident-anchor-diagnostic.json',import.meta.url),JSON.stringify(report,null,2)+'\n');
try{for(const fixture of fixtures){
 const {scene,camera}=fixture,packed=packResidentScene(scene),parameters=packResidentParameters(camera,options,packed,{tile:{x:0,y:0,width:8,height:8},sampleStart:0,samplesPerPixel:1});new Uint32Array(parameters)[38]=4;
 const count=64*3,initial=new Uint32Array(count*4);const cpu=[];
 for(let sample=0;sample<3;sample++)for(let pixel=0;pixel<64;pixel++){const i=sample*64+pixel,rng=randomGenerator((7391^Math.imul(pixel+1,0x9E3779B1))>>>0);for(let s=0;s<sample;s++){rng();rng();}const state=rng.getState();initial.set([pixel,state,sample,0],i*4);const words=[rng()*4294967296,rng()*4294967296],frame=cameraFrame(camera),half=Math.tan((camera.fov*Math.PI/180)/2),u=(2*((pixel%8)+words[0]/4294967296)/8-1)*half,v=(1-2*(Math.floor(pixel/8)+words[1]/4294967296)/8)*half;const raw=frame.forward.map((x,k)=>x+(frame.right[k]*u+frame.up[k]*v));const beam=sampleCamera(camera,pixel%8,Math.floor(pixel/8),8,8,randomGenerator(state),.025,false);const hits=[];for(let depth=0;depth<2;depth++){const h=rayNearestBrute(scene,beam.origin,beam.direction,beam);hits.push({origin:beam.origin,direction:beam.direction,query:normalize(beam.direction),hit:h});beam.origin=h.point;beam.direction=beam.direction.map((x,k)=>h.normal[k]?-x:x);}cpu.push({pixel,sample,raw,hits});}
 const buffers=[],make=(size,usage)=>{const b=device.createBuffer({size,usage});buffers.push(b);return b;};
 const uniform=make(160,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST),geo=make(packed.buffer.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),inputs=make(initial.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),lanes=make(count*448,GPUBufferUsage.STORAGE),weights=make(count*48*16,GPUBufferUsage.STORAGE),state=make(count*384,GPUBufferUsage.STORAGE),output=make(count*64*16,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC),read=make(count*64*16,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
 device.queue.writeBuffer(uniform,0,parameters);device.queue.writeBuffer(geo,0,packed.buffer);device.queue.writeBuffer(inputs,0,initial);
 const resource={0:uniform,1:geo,2:inputs,3:lanes,7:weights,9:state,10:output};
 const bindings={ad_camera:[0,1,2,3,7,9,10],ad_geometry:[1,3,9,10],ad_prepare:[1,3,9,10],ad_plane:[1,3,9,10],ad_normalized_point:[3,9,10],ad_raw_point:[3,9,10],ad_finish:[1,3,9,10],ad_advance:[3,9]};
 const groups=Object.fromEntries(names.map(name=>[name,device.createBindGroup({layout:pipelines[name].getBindGroupLayout(0),entries:bindings[name].map(binding=>({binding,resource:{buffer:resource[binding]}}))})]));
 const encoder=device.createCommandEncoder();const dispatch=name=>{const pass=encoder.beginComputePass();pass.setPipeline(pipelines[name]);pass.setBindGroup(0,groups[name]);pass.dispatchWorkgroups(Math.ceil(count*(name.includes("_point")?3:1)/32));pass.end();};
 dispatch('ad_camera');for(let depth=0;depth<2;depth++){for(const name of ['ad_geometry','ad_prepare','ad_plane','ad_normalized_point','ad_raw_point','ad_finish'])dispatch(name);if(depth===0)dispatch('ad_advance');}encoder.copyBufferToBuffer(output,0,read,0,count*64*16);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);const result=new Float32Array(read.getMappedRange().slice(0));read.unmap();
 const rows=[];for(let i=0;i<count;i++){const vec=(k,n=3)=>Array.from(result.subarray(i*256+k*4,i*256+k*4+n)),interval=k=>vec(k,2),vinterval=k=>({lo:vec(k),hi:vec(k+1)});const depths=[];for(let depth=0;depth<2;depth++){const b=8+depth*28,c=cpu[i].hits[depth],axis=c.hit.normal.findIndex(x=>x!==0);depths.push({depth,origin:vec(b),direction:vec(b+1),originError:vec(b+2),point:vec(b+3),distance:vec(b+4,1)[0],candidate:vec(b+4,4),stored:vinterval(b+5),query:vinterval(b+7),t:interval(b+9),planeDelta:interval(b+10),normalizedRatios:[0,1,2].map(k=>k===axis?null:interval(b+11+k)),normalizedProducts:[0,1,2].map(k=>k===axis?null:interval(b+14+k)),normalizedPointError:vec(b+17),rawRatios:[0,1,2].map(k=>k===axis?null:interval(b+18+k)),rawProducts:[0,1,2].map(k=>k===axis?null:interval(b+21+k)),rawPointError:vec(b+24),triangleFinish:vec(b+25,4),cpu:{origin:c.origin.map((x,k)=>x-packed.origin[k]),stored:c.direction,query:c.query,point:c.hit.point.map((x,k)=>x-packed.origin[k]),distance:c.hit.distance,axis}});}
 rows.push({pixel:cpu[i].pixel,sample:cpu[i].sample,rawCPU:cpu[i].raw,raw:vinterval(0),normalized:vinterval(2),direction:vec(4),directionError:vec(5),cameraOrigin:vec(6),cameraOriginError:vec(7),depths});}
 const quant=a=>{const s=[...a].sort((a,b)=>a-b);return {min:s[0],median:s[Math.floor(s.length/2)],max:s.at(-1),mean:s.reduce((a,b)=>a+b,0)/s.length};};
 const summary=[0,1].map(depth=>{const ds=rows.map(r=>r.depths[depth]),hist={};for(const d of ds)hist[d.triangleFinish[3]]=(hist[d.triangleFinish[3]]??0)+1;return {depth,finishReasons:hist,normalizedPointError:quant(ds.map(d=>Math.max(...d.normalizedPointError))),rawPointError:quant(ds.map(d=>Math.max(...d.rawPointError))),normalizedRawRatioWidth:quant(ds.flatMap(d=>d.normalizedRatios.map((q,k)=>q?(q[1]-q[0])/(d.rawRatios[k][1]-d.rawRatios[k][0]):null).filter(x=>x!==null))),originError:quant(ds.map(d=>Math.max(...d.originError))),tWidth:quant(ds.map(d=>d.t[1]-d.t[0])),rawBelowPointThreshold:ds.filter(d=>Math.max(...d.rawPointError)<=2e-5).length,normalizedBelowPointThreshold:ds.filter(d=>Math.max(...d.normalizedPointError)<=2e-5).length};});
 const diagnostics={enclosureChecks:0,maxActualPointError:0,rawPointDiagnosticUnderEnclosures:0};
 for(const row of rows){for(let k=0;k<3;k++){assert(row.raw.lo[k]<=row.rawCPU[k]&&row.rawCPU[k]<=row.raw.hi[k],'Native raw camera enclosure');diagnostics.enclosureChecks++;}
 for(const d of row.depths){assert(d.candidate[2]===1,'Missing native candidate');assert(d.t[0]<=d.cpu.distance&&d.cpu.distance<=d.t[1],'CPU64 true distance enclosure');diagnostics.enclosureChecks++;
 for(let k=0;k<3;k++){assert(d.stored.lo[k]<=d.cpu.stored[k]&&d.cpu.stored[k]<=d.stored.hi[k],'CPU64 stored direction enclosure');assert(d.query.lo[k]<=d.cpu.query[k]&&d.cpu.query[k]<=d.query.hi[k],'CPU64 query direction enclosure');const actual=Math.abs(d.point[k]-d.cpu.point[k]);assert(actual<=d.normalizedPointError[k],'Baseline point enclosure');diagnostics.maxActualPointError=Math.max(diagnostics.maxActualPointError,actual);diagnostics.rawPointDiagnosticUnderEnclosures+=actual>d.rawPointError[k]?1:0;diagnostics.enclosureChecks+=3;}
 const k=d.cpu.axis,mid=(d.query.lo[k]+d.query.hi[k])/2,delta=(d.planeDelta[0]+d.planeDelta[1])/2;
 d.uncertaintyBreakdown={distanceHalfWidth:(d.t[1]-d.t[0])/2,originContributionLinearized:d.originError[k]/Math.abs(mid),directionContributionLinearized:Math.abs(delta)*(d.query.hi[k]-d.query.lo[k])/2/(mid*mid),planeDeltaHalfWidth:(d.planeDelta[1]-d.planeDelta[0])/2};
 d.actualPointError=d.point.map((x,k)=>Math.abs(x-d.cpu.point[k]));}}
 for(const sum of summary){const ds=rows.map(r=>r.depths[sum.depth]);sum.originShareOfDistanceHalfWidthLinearized=quant(ds.map(d=>d.uncertaintyBreakdown.originContributionLinearized/d.uncertaintyBreakdown.distanceHalfWidth));sum.directionShareOfDistanceHalfWidthLinearized=quant(ds.map(d=>d.uncertaintyBreakdown.directionContributionLinearized/d.uncertaintyBreakdown.distanceHalfWidth));}
 const representativePixel=rows.find(r=>r.sample===0&&r.depths[1].triangleFinish[3]===640)?.pixel??0;
 const retained=new Set(rows.filter(r=>r.sample===0&&[representativePixel,27].includes(r.pixel)));
 for(const depth of [0,1]){for(const key of ['normalizedPointError','rawPointError'])retained.add(rows.reduce((a,b)=>Math.max(...a.depths[depth][key])>=Math.max(...b.depths[depth][key])?a:b));for(const reason of [10,640,1152]){const row=rows.find(r=>r.depths[depth].triangleFinish[3]===reason);if(row)retained.add(row);}}
 for(const row of rows)if(Math.max(...row.depths[0].rawPointError)>2e-5)retained.add(row);
 const record={name:fixture.name,scale:fixture.scale,rebase:packed.origin,count,summary,diagnostics,representativePixel,rowsSelection:'Representative pixels, maxima, each rejection reason and all first-hit raw bounds exceeding threshold; statistics and enclosure checks use every path',rows:[...retained]};report.cases.push(record);console.log(JSON.stringify({name:fixture.name,summary},null,2));save();buffers.forEach(b=>b.destroy());
}report.allPass=true;}finally{device.destroy();gpu=null;save();}
