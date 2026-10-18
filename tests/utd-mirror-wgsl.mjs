// Execute the shared production UTD shader, without test-side modifications,
// for one-mirror, one-local-region families.
// This is native Dawn/Tint compile + dispatch + buffer readback, not a WGSL
// interpreter, a replacement formula, or evidence from a physical user GPU.
// Run from the repo root (or supply it as argv[2]). Example:
// WEBGPU_EXECUTION_BACKEND=vulkan VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json node tests/utd-mirror-wgsl.mjs
// WEBGPU_EXECUTION_REPORT optionally overrides the saved report path.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';

const root=path.resolve(process.argv[2]||process.cwd());
const backend=process.env.WEBGPU_EXECUTION_BACKEND;
assert.ok(['vulkan','metal','d3d11','d3d12','opengl','opengles'].includes(backend),
  'Set WEBGPU_EXECUTION_BACKEND to an executable native backend; null is refused.');
const reportPath=path.resolve(process.env.WEBGPU_EXECUTION_REPORT||path.join(root,'research/utd-mirror-native-validation.json'));
const load=name=>import(pathToFileURL(path.join(root,'dist',name)).href);
const [{builtInSceneOBJ,evaluatePoint,computeReceiverPoint},{prepareMesh,transformPoint},
  {prepareMirrorExperiment,prepareReflectedUTDScene},{packUTDSceneGPU},{parityGate},{makeGPUEngine}]=await Promise.all([
  load('utd-scene.js'),load('mesh-geometry.js'),load('utd-mirror.js'),load('utd-gpu-pack.js'),load('utd-engine.js'),load('utd-gpu-engine.js'),
]);
const {create,globals}=process.env.WEBGPU_MODULE
  ? await import(pathToFileURL(path.resolve(process.env.WEBGPU_MODULE)).href)
  : await import('webgpu');
const packagePath=process.env.WEBGPU_MODULE
  ? path.join(path.dirname(path.resolve(process.env.WEBGPU_MODULE)),'package.json')
  : createRequire(import.meta.url).resolve('webgpu/package.json');
const webgpuVersion=JSON.parse(fs.readFileSync(packagePath,'utf8')).version;
Object.assign(globalThis,globals);
let gpu=create([`backend=${backend}`]);
const adapter=await gpu.requestAdapter();
assert.ok(adapter,`No executable ${backend} adapter available`);
const adapterInfo=Object.fromEntries(['vendor','architecture','device','description','isFallbackAdapter'].map(k=>[k,adapter.info[k]]));
const device=await adapter.requestDevice();
const sourceFiles=['utd-functions.wgsl','utd-compute.wgsl'];
const source=sourceFiles.map(name=>fs.readFileSync(path.join(root,'dist',name),'utf8')).join('\n');
const hashFile=name=>createHash('sha256').update(fs.readFileSync(path.join(root,'dist',name))).digest('hex');
const sourceHashes=Object.fromEntries(sourceFiles.map(name=>[name,hashFile(name)]));
const referenceFiles=['utd-scene.js','utd-mirror.js','utd-gpu-pack.js','utd-engine.js','utd-gpu-engine.js'];
const referenceHashes=Object.fromEntries(referenceFiles.map(name=>[name,hashFile(name)]));
const report={execution:'Native Dawn/Tint WGSL compile, independent dispatch and readback; software-adapter execution is not physical GPU validation',
  scope:'Backend agreement for one selected ideal-mirror plus one coherent local UTD region; not independent physical validation or general multi-region transport',
  sourcePrivacy:'The application source and uploaded meshes remain browser-local; this regression uses built-in synthetic geometry only',
  requestedBackend:backend,adapter:adapterInfo,nodeVersion:process.version,webgpuVersion,
  sourceHashes,referenceHashes,tolerance:{absolute:2e-5,relative:.005,boundaryEpsilon:2e-6},
  gateContract:{basePoints:25,maximumTotalPoints:81,unchanged:true},
  regressionEvidence:{
    observedBeforeFix:{date:'2026-10-12',backend:'Native Dawn/Tint Vulkan SwiftShader',
      computeShaderSHA256:'ce709ccb7d66f127e83ad0abb947f549382b5c466a0cdf61d4bf9294772e45ef',
      fixture:'80-degree tilted screen, positive-edge local GO reflection seam, both mirror orders, 33x33',
      failingPixels:[[16,1],[16,13],[16,18],[16,25],[16,30]],failuresPerRoute:5,
      expected:{geometricPower:.25,power:.13152742100530138},actual:{geometricPower:0,power:.06714341044425964},
      maxToleranceRatio:196.85039370078738,sampledGatePassed:true,actualEngineMatchedIndependentDispatch:true},
    diagnosis:{method:'Temporary native diagnostic entrypoint using the original shader functions; authoritative regression remained unmodified',
      pixel:[16,1],triangle0:{u:.4772739112377167,v:.5227271914482117,sum:1.0000011026859283,t:20.000001907348633},
      facetRange:20.000001907348633,tmax:20.0000057220459,
      inCone:true,regionValue:.02273664064705372,incomingOccluded:false,outgoingOccluded:false,physicalBorderCount:1,
      cause:'f32 ray barycentric u+v exceeded the fixed membership threshold before the existing symmetric physical-edge half-weight band was considered; tmax was not the cause'},
    correction:'Shared CPU reflectionFields and WGSL utd_reflections classify the same facet-restricted physical border before rejection and accept onFacet OR borderCount>0. Half weight, corner invalidation, cone/region/visibility, ray thresholds, tmax, numerical tolerance and parity gate are retained.',
    retainedCoverage:'Both full 33x33 seam grids, center GO quarter-power and two-edge assertions, exact masks and edge counts, and actual engine output versus independent native dispatch',
  },scenes:[],passed:false};
const previousNavigator=Object.getOwnPropertyDescriptor(globalThis,'navigator'),previousFetch=globalThis.fetch;
Object.defineProperty(globalThis,'navigator',{configurable:true,value:{gpu,userAgent:`Node ${process.version}; native Dawn ${webgpuVersion}`}});
let fetchedSources=[];
globalThis.fetch=async input=>{
  const name=String(input).replace(/^\.\//,'');
  assert.ok(sourceFiles.includes(name),`Unexpected engine fetch: ${input}`);
  fetchedSources.push(name);
  return {ok:true,status:200,text:async()=>fs.readFileSync(path.join(root,'dist',name),'utf8')};
};

function uiState(mesh,preset,boundary='soft'){
  assert.deepEqual(mesh.bounds.center,[0,0,0],'Built-in default fixtures must be centered at the actual UI origin');
  return {wavelengthMm:.05,sigma0Mm:1,incidentDirection:[0,0,1],
    sourceOrigin:[0,0,-Math.max(5,Math.hypot(...mesh.bounds.size))],detectorCenter:[0,0,20],
    detectorU:[1,0,0],detectorV:[0,1,0],detectorWidthMm:6,detectorHeightMm:6,
    surface:['box','wedge'].includes(preset)?'closed-solid':'open-plate',boundary};
}
function prepareCase(item){
  const mesh=prepareMesh(builtInSceneOBJ(item.preset),item.meshTransform);
  const state={...uiState(mesh,item.preset,item.boundary),...item.state};
  const base=prepareMirrorExperiment(mesh,state,{order:item.order,distanceMm:10});
  if(!item.transform)return {scene:base,requestedState:state};
  // Rigidly transform the PHYSICAL scene, including the mirror, and invoke the
  // general physical API. This is not merely a tilted receiver or a new shader.
  const physical=base.mirrorTransport.physical,mirror=base.mirrorTransport.mirror;
  const point=p=>transformPoint(p,item.transform);
  const direction=p=>transformPoint(p,{rotationDeg:item.transform.rotationDeg});
  const transformed={...physical,sourceOrigin:point(physical.sourceOrigin),incidentDirection:direction(physical.incidentDirection),
    detectorCenter:point(physical.detectorCenter),detectorU:direction(physical.detectorU),detectorV:direction(physical.detectorV),
    mirror:{point:point(mirror.point),normal:direction(mirror.normal),order:item.order}};
  return {scene:prepareReflectedUTDScene(prepareMesh(builtInSceneOBJ(item.preset),item.transform),transformed),requestedState:transformed};
}
function checkGate(gate,w,h){
  assert.equal(gate.passed,true,'Unmodified application parity gate must pass');
  assert.deepEqual(gate.tolerance,report.tolerance,'The parity gate tolerance must not be relaxed');
  assert.equal(gate.coverage.basePoints,25,'All original baseline checks remain');
  assert.ok(gate.points===25||gate.points===81,'The gate remains the 25-point baseline or 81 total points');
  const xs=[0,Math.floor((w-1)/4),Math.floor((w-1)/2),Math.floor(3*(w-1)/4),w-1];
  const ys=[0,Math.floor((h-1)/4),Math.floor((h-1)/2),Math.floor(3*(h-1)/4),h-1];
  assert.deepEqual(gate.samples.slice(0,25).map(p=>[p.x,p.y,p.stage]),ys.flatMap(y=>xs.map(x=>[x,y,'base'])));
}
async function dispatch(scene,w,h,pipeline){
  const packed=packUTDSceneGPU(scene,w,h),resources=[];
  try{
    const make=(size,usage,data)=>{
      const b=device.createBuffer({size:Math.max(16,Math.ceil(size/16)*16),usage});resources.push(b);
      if(data)device.queue.writeBuffer(b,0,data);return b;
    };
    const uniformData=packed.createUniforms(0,8);
    const uniform=make(uniformData.byteLength,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,uniformData);
    const storage=a=>make(a.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,a);
    const triangles=storage(packed.triangles),edges=storage(packed.edges),adjacency=storage(packed.adjacency),facets=storage(packed.facets);
    const bytes=w*h*16,out=make(bytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC),read=make(bytes,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[uniform,triangles,edges,adjacency,out,facets].map((buffer,binding)=>({binding,resource:{buffer}}))});
    device.pushErrorScope('validation');
    try{
      for(let base=0;base<h;base+=8){
        device.queue.writeBuffer(uniform,0,packed.createUniforms(base,Math.min(8,h-base)));
        const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
        pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(w/8),1);pass.end();
        device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
      }
      const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(out,0,read,0,bytes);device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      try{return new Float32Array(read.getMappedRange(0,bytes).slice(0));}finally{read.unmap();}
    }finally{const error=await device.popErrorScope();assert.equal(error,null,error?.message);}
  }finally{for(const buffer of resources)buffer.destroy();}
}

const cases=[];
for(const order of ['before','after']){
  for(const size of [33,65,97])cases.push({label:`Exact screen UI defaults; mirror ${order}; ${size}x${size}`,preset:'screen',order,size,boundary:'soft',engine:true});
  cases.push({label:`Hard screen; mirror ${order}; 33x33`,preset:'screen',order,size:33,boundary:'hard'});
  for(const preset of ['box','wedge'])for(const boundary of ['soft','hard'])
    cases.push({label:`${boundary} ${preset}; mirror ${order}; 33x33`,preset,order,size:33,boundary,engine:boundary==='hard'});
  cases.push({label:`Oblique screen incidence; mirror ${order}; 33x33`,preset:'screen',order,size:33,boundary:'soft',state:{incidentDirection:[.1,0,Math.sqrt(.99)]}});
  cases.push({label:`Tilted transformed physical screen and mirror ${order}; 33x33`,preset:'screen',order,size:33,boundary:'soft',
    transform:{rotationDeg:[17,-23,31],translation:[.25,-.4,.7]},engine:true});
}
cases.push({label:'Opaque-slit refined 81-point gate; mirror after; 33x33',preset:'opaque-double-slit',order:'after',size:33,boundary:'soft',engine:true,expectedGatePoints:81});
// Regression for facet-membership rounding preempting the symmetric physical
// reflection seam. Before the shared border-admission correction, x=16 at
// y=1,13,18,25,30 lost GO in BOTH routes; the sampled gate missed all five.
// Exhaustive comparison below must retain this fixture even if its gate passes.
const theta=80*Math.PI/180,reflectedDirection=[-Math.sin(2*theta),0,-Math.cos(2*theta)],edgeTangent=[Math.cos(theta),0,-Math.sin(theta)];
for(const order of ['before','after'])cases.push({
  label:`Local specular GO plus diffraction at half-weight seam; mirror ${order}; 33x33`,
  preset:'screen',order,size:33,boundary:'soft',meshTransform:{rotationDeg:[0,80,0]},engine:true,
  state:{sourceOrigin:[0,0,-20],detectorCenter:edgeTangent.map((v,i)=>v+20*reflectedDirection[i]),
    detectorU:[reflectedDirection[2],0,-reflectedDirection[0]],detectorV:[0,1,0],detectorWidthMm:1,detectorHeightMm:1},
  expectedCenter:{reflectionCount:1,edgeCount:2,geometricPower:.25},
});
const failures=[];
try{
  const module=device.createShaderModule({label:'Unmodified UTD shader: reflected-family regression',code:source});
  const info=await module.getCompilationInfo();
  report.compilationMessages=info.messages.map(m=>({type:m.type,lineNum:m.lineNum,linePos:m.linePos,message:m.message}));
  assert.equal(info.messages.filter(m=>m.type==='error').length,0,'Native WGSL compilation errors');
  const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});
  for(const item of cases){
    const r={label:item.label,preset:item.preset,order:item.order,boundary:item.boundary,width:item.size,height:item.size,
      transform:item.transform??null,meshTransform:item.meshTransform??null,localSpecularPixels:0,valid:0,total:item.size**2,cpuPeak:0,gpuPeak:0,maskMismatches:0,edgeMismatches:0,
      malformedOutputs:0,numericalFailures:0,maxToleranceRatio:0,maxAbsolutePowerError:0,gate:null,failures:[]};
    report.scenes.push(r);
    try{
      const {scene,requestedState}=prepareCase(item),w=item.size,h=item.size;
      r.requestedState=requestedState;r.parameters=scene.params;r.mirrorTransport=scene.mirrorTransport;
      assert.equal(scene.supported,true,`Reflected scene must be admitted: ${JSON.stringify(scene.unsupported)}`);
      const data=await dispatch(scene,w,h,pipeline);
      try{r.gate=parityGate(scene,{width:w,height:h,data});checkGate(r.gate,w,h);
        if(item.expectedGatePoints)assert.equal(r.gate.points,item.expectedGatePoints,'Fixture must exercise the intended gate coverage');}
      catch(error){r.gate=error.parity??r.gate??{passed:false};r.gateError=error.message;}
      for(let y=0;y<h;y++)for(let x=0;x<w;x++){
        const position=computeReceiverPoint(scene,x,y,w,h),expected=evaluatePoint(scene,position,{boundaryEpsilon:report.tolerance.boundaryEpsilon});
        const i=(y*w+x)*4,actual=[...data.slice(i,i+4)],gpuValid=actual[2]===1;
        if(expected.valid&&expected.reflectionCount>0)r.localSpecularPixels++;
        if(item.expectedCenter&&x===(w-1)/2&&y===(h-1)/2){
          assert.equal(expected.valid,true,'The local specular seam must be a valid first-order path');
          assert.equal(expected.reflectionCount,item.expectedCenter.reflectionCount,'The fixture must retain local GO reflection');
          assert.equal(expected.edgeCount,item.expectedCenter.edgeCount,'The fixture must retain both coherent UTD edges');
          assert.ok(Math.abs(expected.geometricPower-item.expectedCenter.geometricPower)<1e-8,'The seam must use symmetric half-amplitude GO, quarter power');
          r.localSpecularSeam={position,expected:{power:expected.f,geometricPower:expected.geometricPower,reflectionCount:expected.reflectionCount,edgeCount:expected.edgeCount,valid:expected.valid},
            actual:{power:actual[0],geometricPower:actual[1],valid:gpuValid,edgeCount:actual[3]}};
        }
        const ratio=Math.abs(actual[0]-expected.f)/(2e-5+.005*Math.abs(expected.f));
        const geometricRatio=Math.abs(actual[1]-expected.geometricPower)/(2e-5+.005*Math.abs(expected.geometricPower));
        const maskMismatch=gpuValid!==expected.valid,edgeMismatch=actual[3]!==expected.edgeCount;
        const malformed=!actual.every(Number.isFinite)||actual[0]<0||actual[1]<0||![0,1].includes(actual[2])||!Number.isInteger(actual[3])||actual[3]<0||actual[3]>scene.wedges.length;
        const numericFailure=expected.valid&&(ratio>1||geometricRatio>1||malformed);
        if(maskMismatch)r.maskMismatches++;if(edgeMismatch)r.edgeMismatches++;if(malformed)r.malformedOutputs++;if(numericFailure)r.numericalFailures++;
        if(expected.valid){r.valid++;r.cpuPeak=Math.max(r.cpuPeak,Math.fround(expected.f));r.gpuPeak=Math.max(r.gpuPeak,actual[0]);
          r.maxToleranceRatio=Math.max(r.maxToleranceRatio,ratio,geometricRatio);r.maxAbsolutePowerError=Math.max(r.maxAbsolutePowerError,Math.abs(actual[0]-expected.f));}
        if(maskMismatch||edgeMismatch||malformed||numericFailure)r.failures.push({x,y,position,actual:{power:actual[0],geometricPower:actual[1],valid:gpuValid,validityFlag:actual[2],edgeCount:actual[3]},
          expected:{power:expected.f,geometricPower:expected.geometricPower,valid:expected.valid,edgeCount:expected.edgeCount,invalidReasons:expected.invalidReasons},
          toleranceRatio:ratio,geometricToleranceRatio:geometricRatio});
      }
      if(item.engine){
        fetchedSources=[];const engine=await makeGPUEngine(scene,w,h),progress=[];
        try{
          const cancelled=new AbortController();cancelled.abort();
          await assert.rejects(engine.run({signal:cancelled.signal}),{name:'AbortError'});
          const output=await engine.run({onProgress:value=>progress.push(value)});
          checkGate(output.parity,w,h);
          assert.equal(output.valid,r.valid,'Application engine valid-pixel count');
          assert.deepEqual(progress,Array.from({length:Math.ceil(h/8)},(_,i)=>Math.min((i+1)*8,h)/h),'Application engine executes every row batch');
          assert.deepEqual(output.data,data,'Application engine output must bitwise match independent native dispatch');
          assert.deepEqual(fetchedSources.sort(),[...sourceFiles].sort(),'Mirror mode must fetch exactly the existing shader sources');
          r.actualEngine={passed:true,backend:output.backend,valid:output.valid,batches:progress.length,
            preAbortedRunRejected:true,normalRunAfterAbortPassed:true,bitwiseEqualToIndependentDispatch:true,parity:output.parity};
        }finally{engine.destroy();engine.destroy();}
        await assert.rejects(engine.run(),{name:'AbortError'});
        r.actualEngine.destroyedRunRejected=true;r.actualEngine.repeatedDestroySafe=true;
      }
      r.passed=Boolean(r.gate?.passed&&!r.gateError&&!r.maskMismatches&&!r.edgeMismatches&&!r.malformedOutputs&&!r.numericalFailures);
    }catch(error){r.error=error.message;r.passed=false;}
    if(!r.passed)failures.push(item.label);
    console.log(`${r.passed?'PASS':'FAIL'} ${item.label}: ${r.valid}/${r.total} valid; gate ${r.gate?.points??0}; max tolerance ratio ${r.maxToleranceRatio}; masks ${r.maskMismatches}; edges ${r.edgeMismatches}${r.error?`; ${r.error}`:''}`);
  }
  for(const name of sourceFiles)assert.equal(hashFile(name),sourceHashes[name],`Shader changed during regression: ${name}`);
  for(const name of referenceFiles)assert.equal(hashFile(name),referenceHashes[name],`Application source changed during regression: ${name}`);
  report.summary={scenes:report.scenes.length,pixels:report.scenes.reduce((s,r)=>s+r.total,0),validPixels:report.scenes.reduce((s,r)=>s+r.valid,0),
    engineLifecycleCases:report.scenes.filter(r=>r.actualEngine?.passed).length,
    localSpecularSeamCases:report.scenes.filter(r=>r.localSpecularSeam).length,localSpecularPixels:report.scenes.reduce((s,r)=>s+r.localSpecularPixels,0),
    maxToleranceRatio:Math.max(...report.scenes.map(r=>r.maxToleranceRatio)),maskMismatches:report.scenes.reduce((s,r)=>s+r.maskMismatches,0),
    edgeMismatches:report.scenes.reduce((s,r)=>s+r.edgeMismatches,0),numericalFailures:report.scenes.reduce((s,r)=>s+r.numericalFailures,0),failedScenes:failures};
  report.passed=failures.length===0;
  assert.equal(failures.length,0,`Native reflected-family WGSL failures: ${failures.join(', ')}`);
}catch(error){report.error=error.message;throw error;}
finally{
  fs.writeFileSync(reportPath,JSON.stringify(report,null,2)+'\n');
  console.log(`Native reflected-family report: ${reportPath}`);
  console.log('Adapter provenance:',JSON.stringify(adapterInfo));
  globalThis.fetch=previousFetch;
  if(previousNavigator)Object.defineProperty(globalThis,'navigator',previousNavigator);else delete globalThis.navigator;
  device.destroy();gpu=null;
}
