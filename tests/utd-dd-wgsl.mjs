// Native Dawn/Tint execution of the bounded experimental DD conditional operator.
// No browser visual claim, physical-device claim, or independent physics claim.
// From repo root:
// VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json WEBGPU_EXECUTION_BACKEND=vulkan node tests/utd-dd-wgsl.mjs
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
const reportPath=path.resolve(process.env.WEBGPU_EXECUTION_REPORT||path.join(root,'research/utd-dd-native-validation.json'));
const load=file=>import(pathToFileURL(path.join(root,'dist',file)).href);
const {evaluateDDRecord,packDDRecords,ddParityGate,makeDDGPUEngine,DD_PARITY_TOLERANCE,DD_GPU_SCOPE,DD_MAX_RECORDS}=await load('utd-dd-gpu.js');
const {create,globals}=process.env.WEBGPU_MODULE
  ?await import(pathToFileURL(path.resolve(process.env.WEBGPU_MODULE)).href):await import('webgpu');
const packagePath=process.env.WEBGPU_MODULE
  ?path.join(path.dirname(path.resolve(process.env.WEBGPU_MODULE)),'package.json'):createRequire(import.meta.url).resolve('webgpu/package.json');
const webgpuVersion=JSON.parse(fs.readFileSync(packagePath,'utf8')).version;
Object.assign(globalThis,globals);
let gpu=create([`backend=${backend}`]);
const adapter=await gpu.requestAdapter();assert.ok(adapter,`No executable ${backend} adapter`);
const adapterInfo=Object.fromEntries(['vendor','architecture','device','description','isFallbackAdapter'].map(key=>[key,adapter.info[key]]));
const device=await adapter.requestDevice();
const hash=text=>createHash('sha256').update(text).digest('hex');
const read=relative=>fs.readFileSync(path.join(root,relative),'utf8');
const sourceFiles=['utd-functions.wgsl','utd-dd-compute.wgsl'];
const sourceByFile=Object.fromEntries(sourceFiles.map(name=>[name,read(`dist/${name}`)]));
const source=sourceFiles.map(name=>sourceByFile[name]).join('\n');
const preservedFiles=['utd-functions.wgsl','utd-compute.wgsl','utd-engine.js','utd-gpu-engine.js'];
const preservedHashes=Object.fromEntries(preservedFiles.map(name=>[name,hash(read(`dist/${name}`))]));
const geometryFiles=['utd-dd.js','wave-packet.js','mesh-geometry.js','utd-scene.js'];
const geometryHashes=Object.fromEntries(geometryFiles.map(name=>[name,hash(read(`dist/${name}`))]));
const {traceDoubleDiffraction}=await load('utd-dd.js');
const fixtureNames=['dd-frustum-check.json','dd-frustum-k128.json','dd-frustum-k256.json'];
const fixtures=fixtureNames.map(name=>({name,data:JSON.parse(read(`research/${name}`))}));
const report={schemaVersion:1,execution:'Actual native Dawn/Tint WGSL compilation, dispatch and mapped buffer readback',
  scope:DD_GPU_SCOPE,limitations:[
    'Numerical CPU/WGSL agreement only; independent double-diffraction physics validation is separate and incomplete.',
    'No physical user GPU or browser visual validation.',
    'The GPU evaluates pre-vetted coefficient records, not geometric cone traversal or region discovery.',
    'No UI integration, publishing, coherent cross-family phase, or calibrated irradiance is implemented by this experiment.',
  ],requestedBackend:backend,adapter:adapterInfo,nodeVersion:process.version,webgpuVersion,
  driverSelection:{VK_ICD_FILENAMES:process.env.VK_ICD_FILENAMES??null},
  testSHA256:hash(read('tests/utd-dd-wgsl.mjs')),
  sourceHashes:Object.fromEntries(sourceFiles.map(name=>[name,hash(sourceByFile[name])])),
  referenceHashes:Object.fromEntries(['utd-dd-gpu.js','utd-math.js'].map(name=>[name,hash(read(`dist/${name}`))])),
  geometryReferenceHashes:geometryHashes,
  geometryCoverageScope:'Exhaustive backend agreement for records produced by CPU geometric admission; not GPU traversal or finite-Gaussian physical validation. Hard-boundary coverage is numerical only; independent PDE reference is soft only.',
  historicalFixtureScope:'dd-frustum JSONs are historical coefficient-only stress inputs. Their obsolete saved geometric proof is not accepted traversal evidence.',
  fixtureHashes:Object.fromEntries(fixtureNames.map(name=>[name,hash(read(`research/${name}`))])),
  existingSources:preservedHashes,existing25And81PointGate:{changed:false,replaced:false},
  tolerance:DD_PARITY_TOLERANCE,outputLayout:['conditionalPower','stage1Power','stage2Power','valid'],
  cases:[],negativeControls:[],lifecycle:{},passed:false};
let mutation=null,fetched=[];
const previousFetch=globalThis.fetch;
globalThis.fetch=async input=>{
  const name=String(input).replace(/^\.\//,'');assert.ok(sourceFiles.includes(name),`Unexpected DD shader fetch ${input}`);
  fetched.push(name);
  const text=mutation&&name==='utd-dd-compute.wgsl'?mutation(sourceByFile[name]):sourceByFile[name];
  return {ok:true,status:200,text:async()=>text};
};
function recordFor(fixture,boundary) {
  const {nativeCentreEvaluation:value,parameters}=fixture;
  return {k:parameters.k,boundary,angles:[...value.angles],ro1:value.ro1,ro2:value.ro2,valid:true};
}
function angle(v) { const x=Math.atan2(v[0],-v[1]);return x<0?x+2*Math.PI:x; }
async function dispatch(records,pipeline) {
  const buffers=[];let mapped=false,scope=false;
  const make=(size,usage,data)=>{const buffer=device.createBuffer({size,usage});buffers.push(buffer);if(data)device.queue.writeBuffer(buffer,0,data);return buffer;};
  try {
    const packed=packDDRecords(records),bytes=records.length*16;
    const input=make(packed.data.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,packed.data);
    const output=make(bytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
    const readback=make(bytes,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[input,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
    device.pushErrorScope('validation');scope=true;
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
    pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(records.length/64));pass.end();
    encoder.copyBufferToBuffer(output,0,readback,0,bytes);device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();await readback.mapAsync(GPUMapMode.READ);mapped=true;
    const data=new Float32Array(readback.getMappedRange(0,bytes).slice(0));readback.unmap();mapped=false;
    const error=await device.popErrorScope();scope=false;assert.equal(error,null,error?.message);
    return data;
  }finally{
    if(mapped)buffers.at(-1).unmap();
    if(scope)await device.popErrorScope();
    for(const buffer of buffers)buffer.destroy();
  }
}
const cases=[];
for(const {name,data:fixture} of fixtures) {
  // Historical records remain useful numerical inputs only. Their obsolete
  // geometric proof is deliberately not used as admitted traversal evidence.
  for(const boundary of ['soft','hard']) {
    const centre=recordFor(fixture,boundary),expected=evaluateDDRecord(centre).power;
    assert.ok(Math.abs(expected-fixture.nativeCentreEvaluation.powers[boundary])/expected<1e-12,'CPU reference must agree with historical coefficient-only fixture');
    cases.push({label:`Historical coefficient-only anchor; k=${centre.k}; ${boundary}`,records:[centre],fixture:name,expectedCentrePower:expected});
    const records=[],size=33;
    for(let y=0;y<size;y++)for(let x=0;x<size;x++) {
      // A two-parameter in-plane receiver-position sweep. This is coefficient
      // coverage, not an admitted 3D detector scene or a new geometry oracle.
      const receiver=[fixture.parameters.receiver[0]+(x/(size-1)-.5),fixture.parameters.receiver[1]+(y/(size-1)-.5),0];
      const leg=receiver.map((value,i)=>value-fixture.parameters.E2[i]);
      records.push({...centre,angles:[...centre.angles.slice(0,3),angle(leg)],ro2:Math.hypot(...leg),valid:(y*size+x)%17!==0});
    }
    cases.push({label:`Historical in-plane receiver coefficient sweep; k=${centre.k}; ${boundary}; 33x33`,records,fixture:name,
      sampling:'33x33 in-plane receiver-position parameter sweep; deterministic synthetic invalid masks every 17th record; no geometric admission claim',engine:true});
  }
}
const tinyRecords=[.1,.02,.01,.002].flatMap(a=>[256,1024].map(k=>({k,boundary:'soft',angles:[a,2,a,2],ro1:8,ro2:8,valid:true})));
cases.push({label:'Tiny selected powers with cancellation-sensitive soft coefficients',records:tinyRecords,engine:true});
cases.push({label:'Exact zero soft-boundary coefficients and explicit invalid masks',records:[
  {valid:false},{...recordFor(fixtures[0].data,'soft'),angles:[0,2,0,2]},
  {valid:false},{...recordFor(fixtures[0].data,'hard'),angles:[0,2,0,2]},
],engine:true});

try {
  const grid={sigma:[.0675,.07,.0725],aim1:[.09,.10,.11],aim2:[.15,.175,.20],receiverY:[-2.25,-2,-1.75]};
  const transforms=[null,{rotationDeg:[17,-23,31],translation:[.25,-.4,.7]}];
  report.geometryGrid={k:256,...grid,casesPerBoundaryPerTransform:81,transforms,
    selection:'Fixed Cartesian product declared before evaluation; every case retained',
    rigidTransformChecks:0};
  const traceRecord=config=>{
    const traced=traceDoubleDiffraction(config);
    assert.equal(traced.coefficientRecord.valid,traced.valid,'CPU mask and coefficient record must agree');
    if(traced.valid) {
      const reference=evaluateDDRecord(traced.coefficientRecord);
      assert.ok(Math.abs(reference.power-traced.conditionalPower)<=1e-24+1e-12*Math.abs(traced.conditionalPower),
        'Geometry-owned conditional power must match the GPU gate reference');
      for(let stage=0;stage<2;stage++)assert.ok(Math.abs(reference[`stage${stage+1}Power`]-traced.localStages[stage].localPower)
        <=1e-24+1e-12*Math.abs(traced.localStages[stage].localPower),'Both geometry-owned stage powers must agree');
    } else {
      assert.equal(traced.conditionalPower,null,'A rejected geometry is not a valid zero-power scene');
      assert.ok(traced.admissionReasons.length>0,'Rejected geometry retains its reason');
    }
    return {record:traced.coefficientRecord,proof:{config,admitted:traced.valid,
      admissionReasons:traced.admissionReasons,conditionalPower:traced.conditionalPower,
      regionCount:traced.regions?.length??0,geometryChecks:traced.checks,
      traversalDecisions:traced.geometryTrace.map(stage=>({stage:stage.stage,valid:stage.valid,
        decisions:stage.steps.map(step=>step.decision)}))}};
  };
  for(const boundary of ['soft','hard']) {
    let canonical=null;
    for(const transform of transforms) {
      const records=[],proofs=[];
      for(const sigma of grid.sigma)for(const aim1 of grid.aim1)for(const aim2 of grid.aim2)for(const y of grid.receiverY) {
        const config={k:256,sigma,aim1,aim2,receiver:[16,y,0],boundary,...transform};
        const {record,proof}=traceRecord(config);
        assert.equal(record.valid,true,`Fixed geometry must be admitted: ${JSON.stringify(proof)}`);
        records.push(record);proofs.push(proof);
      }
      assert.equal(records.length,81,'All predeclared geometric cases must be retained');
      if(canonical) {
        for(let i=0;i<records.length;i++) {
          for(let angle=0;angle<4;angle++)assert.ok(Math.abs(records[i].angles[angle]-canonical[i].angles[angle])<1e-11,
            'A common rigid transform must preserve every coefficient angle');
          for(const range of ['ro1','ro2','ri2'])assert.ok(Math.abs(records[i][range]-canonical[i][range])<1e-10,
            'A common rigid transform must preserve every physical range');
          report.geometryGrid.rigidTransformChecks++;
        }
      } else canonical=records;
      cases.push({label:`CPU-geometry-derived fixed81 grid; k=256; ${boundary}; ${transform?'rigid transform':'canonical'}`,
        records,geometryProofs:proofs,engine:true,
        sampling:'All 81 fixed CPU-admitted configurations; coefficient execution only, no GPU traversal or finite-Gaussian physics claim'});
    }
    const configs=[{k:128},{k:64},{k:256},{k:256,aim1:0,aim2:0},{k:512}];
    const mixed=configs.map(config=>traceRecord({...config,boundary}));
    assert.deepEqual(mixed.map(item=>item.record.valid),[true,false,true,false,true],'Actual CPU rejection masks are fixed test expectations');
    cases.push({label:`CPU-admission defaults and rejected geometries; ${boundary}`,
      records:mixed.map(item=>item.record),geometryProofs:mixed.map(item=>item.proof),engine:true,
      sampling:'Admitted k128/k256/k512 defaults mixed with actual k64 and zero-aim traversal rejections'});
  }
  const module=device.createShaderModule({label:'Unmodified experimental DD numerical operator',code:source});
  const info=await module.getCompilationInfo();
  report.compilationMessages=info.messages.map(message=>({type:message.type,lineNum:message.lineNum,linePos:message.linePos,message:message.message}));
  assert.equal(info.messages.filter(message=>message.type==='error').length,0,'Native WGSL must compile');
  const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'dd_main'}});
  for(const item of cases) {
    const data=await dispatch(item.records,pipeline),gate=ddParityGate(item.records,data);
    assert.equal(gate.points,item.records.length,'Every output must be checked');
    const result={label:item.label,fixture:item.fixture??null,sampling:item.sampling??null,count:item.records.length,
      ...(item.geometryProofs?{cpuGeometryDerived:true,geometryProofs:item.geometryProofs}:{}),
      ...(item.expectedCentrePower===undefined?{}:{expectedCentrePower:item.expectedCentrePower,actualCentrePower:data[0]}),gate};
    if(item.engine) {
      fetched=[];
      const engine=await makeDDGPUEngine(item.records,{gpu});
      try {
        const aborted=new AbortController();aborted.abort();
        await assert.rejects(engine.run({signal:aborted.signal}),{name:'AbortError'});
        const output=await engine.run();
        assert.deepEqual(output.data,data,'Production DD engine must bitwise match independent native dispatch');
        assert.equal(output.parity.passed,true);
        assert.deepEqual(fetched.sort(),[...sourceFiles].sort(),'DD engine must fetch exactly its two shader sources');
        assert.deepEqual((await engine.run()).data,data,'Repeated run must be stable');
        result.actualEngine={passed:true,preAbortedRunRejected:true,repeatedRunBitwiseEqual:true,
          bitwiseEqualToIndependentDispatch:true,returnsOnlyGatedData:true};
      }finally{engine.destroy();engine.destroy();}
      await assert.rejects(engine.run(),{name:'AbortError'});
      result.actualEngine.destroyedRunRejected=true;
    }
    report.cases.push(result);
    console.log(`${item.label}: PASS ${gate.valid}/${gate.points}; maximum tolerance ratio ${gate.maxToleranceRatio}`);
  }

  // Numerical gate controls are exhaustive and fail closed at tiny magnitudes.
  const record=recordFor(fixtures[2].data,'soft'),single=await dispatch([record],pipeline);
  const gateControls=[
    ['zero selected power',0,0],['negative selected power',0,-single[0]],
    ['NaN selected power',0,NaN],['infinite selected power',0,Infinity],
    ['corrupt first-stage conditional power',1,single[1]*1.02],
    ['corrupt second-stage conditional power',2,single[2]*1.02],
    ['nonbinary mask',3,.5],
  ];
  for(const [label,lane,value] of gateControls) {
    const corrupted=single.slice();corrupted[lane]=value;
    assert.throws(()=>ddParityGate([record],corrupted),{name:'DDParityError'});
    report.negativeControls.push({label,method:'mutated readback unit control',rejected:true});
  }
  const tiny=await dispatch(tinyRecords,pipeline),last=tinyRecords.length-1;
  assert.ok(tiny[4*last]>0&&tiny[4*last]<1e-18,'Fixture must meaningfully exercise sub-1e-18 selected powers');
  const tinyCorrupted=tiny.slice();tinyCorrupted[4*last]*=2;
  assert.throws(()=>ddParityGate(tinyRecords,tinyCorrupted),{name:'DDParityError'});
  report.negativeControls.push({label:'Double a selected power below 1e-18',method:'mutated readback unit control',
    originalPower:tiny[4*last],absoluteError:Math.abs(tinyCorrupted[4*last]-tiny[4*last]),
    oldSingleRegionAbsoluteTolerance:2e-5,rejected:true});
  const invalidNonzero=new Float32Array([1e-30,0,0,0]);
  assert.throws(()=>ddParityGate([{valid:false}],invalidNonzero),{name:'DDParityError'});
  report.negativeControls.push({label:'Invalid-mask output must be exactly zero',method:'mutated readback unit control',rejected:true});
  const distributed=Array.from({length:193},()=>record),distributedData=await dispatch(distributed,pipeline);
  distributedData[4*127]*=1.1;
  let failure;
  try{ddParityGate(distributed,distributedData);}catch(error){failure=error;}
  assert.equal(failure?.name,'DDParityError');assert.equal(failure.parity.points,193);
  assert.equal(failure.parity.failures[0].index,127);
  report.negativeControls.push({label:'Unsampled interior record corruption',method:'mutated readback unit control',index:127,fullArrayPoints:193,rejected:true});

  // Native fault control compiles, executes and reads a deliberately faulty
  // IN-MEMORY shader; no production shader file is modified. The actual engine
  // must reject its readback, never resolving a publishable result.
  const replacements=[
    ['native zero selected power','vec4f(power1 * power2, power1, power2, 1.0)','vec4f(0.0, power1, power2, 1.0)'],
    ['native negative selected power','vec4f(power1 * power2, power1, power2, 1.0)','vec4f(-power1 * power2, power1, power2, 1.0)'],
    ['native nonfinite selected power','vec4f(power1 * power2, power1, power2, 1.0)',
      'vec4f(bitcast<f32>(0x7fc00000u | bitcast<u32>(record.admission.y)), power1, power2, 1.0)'],
  ];
  for(const [label,find,replace] of replacements) {
    assert.equal(sourceByFile['utd-dd-compute.wgsl'].split(find).length,2,'Fault replacement must target one output expression');
    mutation=text=>text.replace(find,replace);
    const engine=await makeDDGPUEngine([record,...tinyRecords],{gpu});
    let resolved=false,error;
    try {await engine.run().then(()=>{resolved=true;}).catch(value=>{error=value;});}
    finally{engine.destroy();mutation=null;}
    assert.equal(resolved,false,'No unverified native output may resolve from the engine');
    assert.equal(error?.name,'DDParityError','Native fault must be rejected by the actual full-array gate');
    assert.equal(error.parity.passed,false);
    if(label.includes('nonfinite'))assert.ok(error.parity.nonfinite>0);
    if(label.includes('negative'))assert.ok(error.parity.negative>0);
    report.negativeControls.push({label,method:'native faulty WGSL compile, dispatch, readback through actual engine',
      faultSourceSHA256:hash(sourceByFile['utd-functions.wgsl']+'\n'+sourceByFile['utd-dd-compute.wgsl'].replace(find,replace)),
      rejected:true,unverifiedResultReturned:false,gate:error.parity});
  }

  // Host input bounds and an async mutation race cannot bypass CPU admission.
  for(const bad of [[],[{...record,valid:1}],[{...record,boundary:'mixed'}],
    [{...record,k:NaN}],[{...record,ro1:0}],[{...record,k:1e6,ro1:1e6}],
    [{...record,angles:[0,1,2,Infinity]}],Array(DD_MAX_RECORDS+1).fill({valid:false})])
    assert.throws(()=>packDDRecords(bad),RangeError);
  assert.throws(()=>ddParityGate([record],new Float32Array(3)),RangeError);
  const mutable=[{...record,angles:[...record.angles]}];
  const pending=makeDDGPUEngine(mutable,{gpu});mutable[0].angles[0]=NaN;mutable[0].valid=false;
  const snapshotEngine=await pending;
  try{assert.deepEqual((await snapshotEngine.run()).data,single,'Inputs must be snapshotted before asynchronous setup');}
  finally{snapshotEngine.destroy();}
  const lifecycleEngine=await makeDDGPUEngine(Array.from({length:193},()=>record),{gpu});
  try {
    const abort=new AbortController();
    const cancelledRun=lifecycleEngine.run({signal:abort.signal});abort.abort();
    await assert.rejects(cancelledRun,{name:'AbortError'});
    const active=lifecycleEngine.run();
    await assert.rejects(lifecycleEngine.run(),/already in progress/);
    assert.equal((await active).parity.passed,true);
    const destroyedRun=lifecycleEngine.run();lifecycleEngine.destroy();
    await assert.rejects(destroyedRun,{name:'AbortError'});
  }finally{lifecycleEngine.destroy();}
  report.lifecycle={preAbortedRunsRejected:true,normalRunsAfterAbortPassed:true,repeatedRunsStable:true,
    cancellationAfterSubmitWithholdsOutput:true,normalRunAfterMidflightAbortPassed:true,
    concurrentRunRejected:true,destroyDuringRunWithholdsOutput:true,
    repeatedDestroySafe:true,destroyedRunRejected:true,inputSnapshotRaceChecked:true,inputDomainRejections:8};
  report.passed=true;
} catch(error) {
  report.error={name:error.name,message:error.message,parity:error.parity??null};
  throw error;
} finally {
  report.geometrySourceHashesStableDuringExecution=Object.entries(geometryHashes).every(([name,expected])=>hash(read(`dist/${name}`))===expected);
  report.unchangedExistingSourceHashesVerified=Object.entries(preservedHashes).every(([name,expected])=>hash(read(`dist/${name}`))===expected);
  const changedDuringRun=!report.geometrySourceHashesStableDuringExecution||!report.unchangedExistingSourceHashesVerified;
  if(changedDuringRun) {
    report.passed=false;
    report.error??={name:'SourceMutationError',message:'Reference or preserved sources changed during execution; rerun against frozen sources'};
  }
  report.cpuGeometryDerivedRecords=report.cases.filter(item=>item.cpuGeometryDerived).reduce((sum,item)=>sum+item.count,0);
  report.cpuGeometryDerivedValidRecords=report.cases.filter(item=>item.cpuGeometryDerived).reduce((sum,item)=>sum+item.gate.valid,0);
  report.totalNativeReferenceRecords=report.cases.reduce((sum,item)=>sum+item.count,0);
  report.totalValidReferenceRecords=report.cases.reduce((sum,item)=>sum+item.gate.valid,0);
  fs.writeFileSync(reportPath,JSON.stringify(report,null,2)+'\n');
  globalThis.fetch=previousFetch;
  device.destroy();gpu=null;
  console.log('Adapter provenance:',JSON.stringify(adapterInfo));
  console.log(`Saved ${reportPath}`);
  assert.equal(changedDuringRun,false,'Reference sources must remain stable for a hash-bound execution report');
}
