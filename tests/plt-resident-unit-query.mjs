/** Restricted CPU camera/axis-mirror query enclosure.
 * Compile: node tests/plt-resident-unit-query.mjs
 * Execute: VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json \
 * WEBGPU_EXECUTION_BACKEND=vulkan node tests/plt-resident-unit-query.mjs --execute
 * Uses the real WGSL API and native Dawn/Tint. Exact BigInt rational checks do
 * not reuse f32 interval arithmetic. This is a focused proof-contract probe;
 * complete independent CPU shadow remains mandatory for all transport.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {sampleCamera} from '../dist/plt-core.js';
import {normalize,sub,scale,dot} from '../dist/mesh-geometry.js';
import {rayNearestBrute} from '../dist/plt-geometry.js';
import {create,globals} from 'webgpu';
Object.assign(globalThis,globals);
const execute=process.argv.includes('--execute');
const backend=execute?process.env.WEBGPU_EXECUTION_BACKEND:'null';
if(execute)assert(['vulkan','metal','d3d11','d3d12','opengl','opengles'].includes(backend),'An executable native backend is required; null is compile-only');
const files=['plt-resident-interval.wgsl','plt-resident-geometry.wgsl','plt-resident-axis-guard.wgsl'];
const sources=files.map(name=>fs.readFileSync(new URL('../dist/'+name,import.meta.url),'utf8'));
const source=sources.join('\n');
const f32buf=new ArrayBuffer(4),u32=new Uint32Array(f32buf),f32=new Float32Array(f32buf);
const bits=x=>{f32[0]=x;return u32[0];},value=b=>{u32[0]=b;return f32[0];};
const ceilF32=x=>{const b=bits(x);return value(b+(value(b)<x?1:0));};
// Exact rationals. tau cannot be stored as a JS Number: it is exact 2^-1075.
const R=(n,d=1n)=>({n:BigInt(n),d:BigInt(d)}),add=(a,b)=>R(a.n*b.d+b.n*a.d,a.d*b.d),neg=a=>R(-a.n,a.d),minus=(a,b)=>add(a,neg(b)),mul=(a,b)=>R(a.n*b.n,a.d*b.d),div=(a,b)=>R(a.n*b.d,a.d*b.n),sq=a=>mul(a,a),cmp=(a,b)=>{const n=a.n*b.d-b.n*a.d;return n<0n?-1:n>0n?1:0;};
const pow2=e=>e>=0?R(1n<<BigInt(e)):R(1n,1n<<BigInt(-e));
const one=R(1),u=pow2(-53),tau=pow2(-1075),nu=n=>mul(R(n),u);
function rational(x){assert(Number.isFinite(x));const b=new ArrayBuffer(8),v=new DataView(b);v.setFloat64(0,x);const w=v.getBigUint64(0),e=Number(w>>52n&2047n),m=w&0xfffffffffffffn,s=w>>63n?-1n:1n;return mul(R(s*(e?m+(1n<<52n):m)),pow2(e?e-1075:-1074));}
const norm2=d=>d.map(rational).map(sq).reduce(add,R(0));
const L=div(sq(minus(one,u)),add(one,nu(16))),U=div(sq(add(one,u)),minus(one,nu(16)));
const relativeUpper=minus(div(U,L),one),relativeLower=minus(one,div(L,U));
assert(cmp(relativeUpper,nu(37))<0&&cmp(relativeLower,nu(37))<0,'normal-only 37u bound');
const relaxedLower=minus(one,nu(19)),relaxedUpper=add(one,nu(19));
assert(cmp(minus(L,mul(R(2),tau)),relaxedLower)>0&&cmp(add(U,mul(R(2),tau)),relaxedUpper)<0,'gradual-underflow norm bound');
const queryUpper=div(sq(add(one,u)),mul(relaxedLower,minus(one,nu(16))));
const queryLower=div(sq(minus(one,u)),mul(relaxedUpper,add(one,nu(16))));
assert(cmp(minus(queryUpper,one),nu(38))<0&&cmp(minus(one,queryLower),nu(38))<0,'gradual-underflow 38u bound');
assert(cmp(mul(nu(26),pow2(-100)),tau)>0,'64u padding absorbs the additive underflow term');
const rows=[],cameraNames=[],cpuStats={cameraCalls:0,mirrorSteps:0,exactNormChecks:0,exactHypotChecks:0,exactMirrorChecks:0,cpuTriangleChecks:0,subnormalCameraComponents:0,signedZeros:0,edgeMirrorChecks:0,normalNormalizationReciprocals:0};
const addRow=row=>{row.id=rows.length;rows.push(row);};
let state=0x417936b;const random=()=>{state^=state<<13;state^=state>>>17;state^=state<<5;return(state>>>0)/4294967296;};
const cameras=[
 {name:'axis',position:[0,0,0],target:[0,0,2],up:[0,1,0],fov:20},
 {name:'off-axis-narrow',position:[3,-7,2],target:[0,1,0],up:[0,1,0],fov:5},
 {name:'off-axis-wide',position:[3,-7,2],target:[0,1,0],up:[0,1,0],fov:120},
 {name:'diagonal',position:[0,0,0],target:[4,1,4],up:[0,1,0],fov:10},
 {name:'permuted',position:[16,-8,32],target:[17,-4,36],up:[1,0,0],fov:42},
 {name:'tiny-component',position:[0,0,0],target:[2**-90,.3,1],up:[1,1,0],fov:42},
 {name:'subnormal-component',position:[0,0,0],target:[Number.MIN_VALUE,.3,1],up:[1,1,0],fov:42},
 {name:'negative-signed-zero',position:[0,0,0],target:[-0,0,-1],up:[0,1,0],fov:120},
];
// Actual source sampleCamera and mirror operation; independently establish the
// reference norm and hypot assumptions by comparing squared exact dyadics.
for(const camera of cameras){cameraNames.push(camera.name);for(let i=0;i<48;i++){
 const center=i===0,x=center?15:Math.floor(random()*31),y=center?11:Math.floor(random()*23),rs=center?[.5,.5]:[random(),random()];let at=0;
 const oldHypot=Math.hypot,calls=[];Math.hypot=(...args)=>{const h=oldHypot(...args);calls.push({args,h});return h;};
 let beam;try{beam=sampleCamera(camera,x,y,31,23,()=>rs[at++],.025,false);}finally{Math.hypot=oldHypot;}
 cpuStats.cameraCalls++;
 for(const {args,h}of calls){assert(Number.isFinite(h)&&h>=2**-1022&&Number.isFinite(1/h)&&1/h>=2**-1022,'normal finite normalization norm/reciprocal required');cpuStats.normalNormalizationReciprocals++;const trueSquared=norm2(args),actualSquared=sq(rational(h));assert(cmp(actualSquared,mul(trueSquared,sq(minus(one,nu(16)))))>=0&&cmp(actualSquared,mul(trueSquared,sq(add(one,nu(16)))))<=0,'actual CPU hypot violates declared contract');cpuStats.exactHypotChecks++;}
 let direction=beam.direction;
 for(let depth=0;depth<=6;depth++){
  const n2=norm2(direction);assert(cmp(n2,sq(relaxedLower))>=0&&cmp(n2,sq(relaxedUpper))<=0,'camera/mirror norm invariant');cpuStats.exactNormChecks++;
  cpuStats.subnormalCameraComponents+=depth===0?direction.filter(v=>v!==0&&Math.abs(v)<2**-1022).length:0;
  cpuStats.signedZeros+=direction.filter(v=>Object.is(v,-0)).length;
  const query=normalize(direction),center=direction.map(Math.fround),error=direction.map((v,k)=>ceilF32(Math.max(Math.abs(v-center[k]),1e-13)));
  const name=`${camera.name}/${i}/mirror${depth}`;addRow({name,mode:1,provenance:1,direction:center,error,reference:query,stored:direction,valid:true,camera:true});
  if(depth===0)addRow({name:name+'/generic',mode:0,provenance:0,direction:center,error,reference:query,valid:true,genericCamera:true,pair:rows.length-1});
  // A real source triangle query also normalizes independently through checkedRay.
  if(i<4){const axis=query.map(Math.abs).indexOf(Math.max(...query.map(Math.abs))),a=[-8,-8,-8],b=[-8,-8,-8],c=[-8,-8,-8],n=[0,0,0],j=(axis+1)%3,k=(axis+2)%3,plane=Math.sign(query[axis]);a[axis]=b[axis]=c[axis]=plane;a[j]=-8;a[k]=-8;b[j]=8;b[k]=-8;c[j]=0;c[k]=8;n[axis]=1;const hit=rayNearestBrute({triangles:[{id:0,a,b,c,normal:n,materialIndex:0}]},[0,0,0],direction,{minDistance:1e-7,maxDistance:Infinity});assert(hit);assert(Math.abs(hit.distance-plane/query[axis])<2e-14);cpuStats.cpuTriangleChecks++;}
  if(depth===6)break;
  const axis=[0,1,2,2,1,0][depth],normal=[0,0,0];normal[axis]=depth%2?-1:1;
  const next=sub(direction,scale(normal,2*dot(direction,normal)));
  for(let k=0;k<3;k++)assert.equal(next[k]||0,(k===axis?-direction[k]:direction[k])||0,'actual CPU mirror differs from exact sign flip');
  assert.equal(cmp(norm2(next),norm2(direction)),0,'mirror changed exact squared norm');cpuStats.exactMirrorChecks++;cpuStats.mirrorSteps++;direction=next;
 }
}}
assert(cpuStats.subnormalCameraComponents>0,'subnormal camera output fixture did not execute');
// Axis reflection has the same exact numerical identity even for signed zero
// and binary64 minimum subnormals. It does not require bitwise zero-sign identity.
for(const direction of [[-0,0,1],[0,-1,-0],[Number.MIN_VALUE,.3,1],[-Number.MIN_VALUE,-0,-1]]){
 cpuStats.signedZeros+=direction.filter(x=>Object.is(x,-0)).length;
 for(let axis=0;axis<3;axis++)for(const sign of [-1,1]){const normal=[0,0,0];normal[axis]=sign;const next=sub(direction,scale(normal,2*dot(direction,normal)));for(let k=0;k<3;k++)assert.equal(next[k]||0,(k===axis?-direction[k]:direction[k])||0);assert.equal(cmp(norm2(next),norm2(direction)),0);cpuStats.edgeMirrorChecks++;}
} 
// Arbitrary standalone nonunit directions MUST retain full normalization.
for(const direction of [[.125,.0625,1],[.25,.125,2],[1,1,0],[.125,-.25,.5],[2,0,0],[1+2**-23,0,0],[-1,-1,-1]]){
 const name='nonunit/'+direction.join(',');addRow({name,mode:0,provenance:0,direction,error:[0,0,0],reference:normalize(direction),valid:true,nonunit:true});
 for(const provenance of [0,2,0xffffffff])addRow({name:name+'/missing-provenance'+provenance,mode:1,provenance,direction,error:[0,0,0],valid:false});
}
for(const direction of [[1,0,-0],[0,-1,0],[-0,0,1]]){
 addRow({name:'exact-axis/'+direction,mode:1,provenance:1,direction,error:[0,0,0],reference:normalize(direction),valid:true});
 addRow({name:'unit-missing-provenance/'+direction,mode:1,provenance:0,direction,error:[0,0,0],valid:false});
}
// Deliberately minimal intervals require a genuine positive CPU-rounding budget.
// Witness Q=(1+u)^2/(1-16u) for an exact D=1 is legal under the assumed error
// model; comparisons below are exact, not rounded-to-double at that endpoint.
const axisWitness=div(sq(add(one,u)),minus(one,nu(16)));
for(const direction of [[1,0,0],[-1,0,0]])addRow({name:'permitted-hypot-error/'+direction[0],mode:1,provenance:1,direction,error:[0,0,0],reference:direction,valid:true,witness:direction[0]>0?axisWitness:neg(axisWitness)});
for(const m of [2**-50,2**-30,.1,.5,1,2])for(const sign of [-1,1])addRow({name:'padding-budget/'+sign*m,mode:2,provenance:1,direction:[Math.fround(sign*m),0,0],error:[0,0,0],valid:true,padding:true});
addRow({name:'compact-padding-underflow-fails-closed',mode:2,provenance:1,direction:[2**-100,0,0],error:[0,0,0],valid:false});
const harness=`
struct UnitInput {dispatch:vec4u,direction:vec4f,error:vec4f}
struct UnitOutput {status:vec4u,lo:vec4u,hi:vec4u,padding:vec4u}
@group(0) @binding(2) var<storage,read> unitInputs:array<UnitInput>;
@group(0) @binding(3) var<storage,read_write> unitOutputs:array<UnitOutput>;
@compute @workgroup_size(64) fn unit_probe(@builtin(global_invocation_id) id:vec3u){
 let i=id.x;if(i>=arrayLength(&unitInputs)){return;}let input=unitInputs[i];var q:RAQueryState;
 if(input.dispatch.x==0u){q=ra_prepare(vec3f(0),vec3f(0),input.direction.xyz,input.error.xyz);}
 else if(input.dispatch.x==1u){q=ra_prepare_unit(vec3f(0),vec3f(0),input.direction.xyz,input.error.xyz,input.dispatch.y);}
 else {ra_bad=false;let a=ra_point(input.direction.x);let e=ra_unit_query_error(a);q=RAQueryState(select(1u,0u,ra_bad),0u,RAVec3(a,a,a),RAVec3(e,e,e),RAI(1,1));}
 let qvalid=q.valid;let lo=vec4u(bitcast<u32>(q.query.x.lo),bitcast<u32>(q.query.y.lo),bitcast<u32>(q.query.z.lo),0u);
 let hi=vec4u(bitcast<u32>(q.query.x.hi),bitcast<u32>(q.query.y.hi),bitcast<u32>(q.query.z.hi),0u);
 unitOutputs[i]=UnitOutput(vec4u(qvalid,q.reason,0x71a7c0deu,input.dispatch.x),lo,hi,vec4u(0));
}`;
let gpu=create([`backend=${backend}`]);const adapter=await gpu.requestAdapter();assert(adapter);const device=await adapter.requestDevice();
const report={createdAt:new Date().toISOString(),backend,executionRequested:execute,actualGPUExecuted:false,allPass:false,status:'compiling',scope:'Restricted unit-query API, exact CPU camera/mirror invariant and independent normalized-query enclosures; not a full transport test or universal hypot guarantee',fullCPUShadowMandatory:true,physicalGPUValidated:false,browserValidated:false,hardwarePerformanceClaim:false,cases:rows.length,cameraFixtures:cameraNames,cpu:cpuStats,exactProof:{normalRelativeBoundU64:37,underflowRelativeBoundU64:38,paddingU64:64,zeroHandledSeparately:true,proofRationalComparisons:7},sourceSHA256:crypto.createHash('sha256').update(source+harness).digest('hex'),filesSHA256:Object.fromEntries(files.map((name,i)=>[name,crypto.createHash('sha256').update(sources[i]).digest('hex')])),hostSHA256:Object.fromEntries(['plt-core.js','mesh-geometry.js','plt-geometry.js'].map(name=>[name,crypto.createHash('sha256').update(fs.readFileSync(new URL('../dist/'+name,import.meta.url))).digest('hex')])),adapter:{vendor:adapter.info.vendor,device:adapter.info.device,description:adapter.info.description},mutations:[]};
const reportPath=new URL('../research/plt-resident-unit-query-'+(execute?'validation':'compile-validation')+'.json',import.meta.url),save=()=>fs.writeFileSync(reportPath,JSON.stringify(report,null,2)+'\n');
async function compile(src){const started=performance.now();const module=device.createShaderModule({code:src+harness});const info=await module.getCompilationInfo();assert.deepEqual(info.messages.filter(m=>m.type==='error'),[]);const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'unit_probe'}});return{pipeline,compileMs:performance.now()-started};}
async function run(pipeline){const words=new Uint32Array(rows.length*12),floats=new Float32Array(words.buffer);for(const q of rows){words.set([q.mode,q.provenance,0,0],q.id*12);floats.set(q.direction,q.id*12+4);floats.set(q.error,q.id*12+8);}const buffers=[];const make=(size,usage)=>{const b=device.createBuffer({size,usage});buffers.push(b);return b;};try{
 const data=new ArrayBuffer(128),s=new Uint32Array(data),f=new Float32Array(data);s[2]=1;s[30]=1;f[31]=1;
 const scene=make(128,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),input=make(words.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),output=make(rows.length*64,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC),read=make(rows.length*64,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);device.queue.writeBuffer(scene,0,data);device.queue.writeBuffer(input,0,words);
 const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:1,resource:{buffer:scene}},{binding:2,resource:{buffer:input}},{binding:3,resource:{buffer:output}}]});const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(rows.length/64));pass.end();encoder.copyBufferToBuffer(output,0,read,0,rows.length*64);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);const out=new Uint32Array(read.getMappedRange().slice(0));read.unmap();return out;
 }finally{buffers.forEach(b=>b.destroy());}}
function inspect(result){const failures=[],widthRatios=[];let accepted=0,rejected=0,exactChecks=0;const fail=(q,reason,data={})=>failures.push({name:q.name,reason,...data});for(const q of rows){const at=q.id*16,valid=result[at]===1;if(result[at+2]!==0x71a7c0de)fail(q,'missing dispatch sentinel');if(valid!==q.valid){fail(q,valid?'missing provenance or unsupported arithmetic admitted':'valid query rejected',{flag:result[at+1]});continue;}if(!valid){rejected++;continue;}accepted++;const lo=[0,1,2].map(k=>value(result[at+4+k])),hi=[0,1,2].map(k=>value(result[at+8+k]));
 if(q.padding){const M=rational(Math.abs(q.direction[0]));const required=add(mul(nu(38),M),tau);if(cmp(rational(hi[0]),required)<0)fail(q,'analytic CPU-rounding budget under-enclosure',{actual:hi[0]});exactChecks++;continue;}
 for(let k=0;k<3;k++){if(!(Number.isFinite(lo[k])&&Number.isFinite(hi[k])&&lo[k]<=q.reference[k]&&q.reference[k]<=hi[k]))fail(q,'actual CPU normalized query under-enclosure',{component:k,lo:lo[k],hi:hi[k],reference:q.reference[k]});if(q.stored){const exactDelta=minus(rational(q.reference[k]),rational(q.stored[k])),limit=add(mul(nu(38),rational(Math.abs(q.stored[k]))),tau);assert(cmp(exactDelta,limit)<=0&&cmp(exactDelta,neg(limit))>=0);exactChecks++;}}
 if(q.witness){if(cmp(rational(lo[0]),q.witness)>0||cmp(rational(hi[0]),q.witness)<0)fail(q,'permitted hypot-rounding witness under-enclosure',{lo:lo[0],hi:hi[0]});exactChecks++;}
 if(q.genericCamera){const pa=q.pair*16;for(let k=0;k<3;k++){const unitWidth=value(result[pa+8+k])-value(result[pa+4+k]);if(unitWidth>0)widthRatios.push((hi[k]-lo[k])/unitWidth);}}
 }
 return{accepted,rejected,exactChecks,failures,meanGenericToUnitWidthRatio:widthRatios.reduce((a,b)=>a+b,0)/widthRatios.length,minGenericToUnitWidthRatio:Math.min(...widthRatios)};}
try{save();const baseline=await compile(source);report.compileMs=baseline.compileMs;console.log('PASS genuine Dawn/Tint unit-query compilation',baseline.compileMs);if(execute){const out=await run(baseline.pipeline);report.actualGPUExecuted=true;const result=inspect(out);assert.equal(result.failures.length,0,JSON.stringify(result.failures.slice(0,12),null,2));assert(result.meanGenericToUnitWidthRatio>2,'restricted enclosure must actually tighten the query');Object.assign(report,result);delete report.failures;console.log('PASS native unit-query enclosures',JSON.stringify(result));
 const mutations=[
  ['omit-provenance-check',s=>s.replace('if(provenance!=1u){return ra_query_failure(RA_UNSUPPORTED);}','if(false){return ra_query_failure(RA_UNSUPPORTED);}')],
  ['remove-unit-rounding-padding',s=>s.replace('ra_ipoint(0x28000000u)); // 64*u64','ra_ipoint(0u)); // 64*u64')],
  ['reduce-unit-rounding-padding-to-16u',s=>s.replace('ra_ipoint(0x28000000u)); // 64*u64','ra_ipoint(0x27000000u)); // 64*u64')],
  ['apply-unit-shortcut-to-generic-rays',s=>s.replace('direction,directionError,false);','direction,directionError,true);')],
 ];
 for(const[name,mutate]of mutations){const altered=mutate(source);assert.notEqual(altered,source);const check=inspect(await run((await compile(altered)).pipeline));assert(check.failures.length>0,`Mutation survived: ${name}`);const under=check.failures.filter(f=>f.reason.includes('under-enclosure'));if(name.includes('padding'))assert(under.length>0,'Padding mutation must fail actual or exact analytical enclosure, not a source-string check');report.mutations.push({name,rejected:true,failures:check.failures.length,underEnclosures:under.length,first:under[0]??check.failures[0]});console.log('PASS rejected mutation',name,check.failures.length);}
 }report.status='passed';report.allPass=true;console.log(JSON.stringify(report,null,2));
}catch(error){report.status='failed';report.failure={name:error.name,message:error.message};throw error;}finally{device.destroy();gpu=null;save();}
