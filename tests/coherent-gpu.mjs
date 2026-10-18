/** Default: unit/lifecycle checks and native Dawn/Tint compile (null backend).
 * --execute: ALSO actual native WGSL dispatch/readback. Requires an explicit
 * WEBGPU_EXECUTION_BACKEND (null refused). Software adapters do NOT establish
 * browser behavior or physical GPU behavior. No browser/server is launched.
 * COHERENT_GPU_REPORT optionally saves this exact run's JSON report.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {solveCoherent,evaluateGrid,evaluatePoints,createState,besselJ0,besselY0} from '../dist/coherent-core.js';
import {packCoherentGPU,coherentParityGate,makeCoherentGPUEngine,computeCoherentGPU,COHERENT_GPU_TOLERANCE,COHERENT_GPU_LIMITS} from '../dist/coherent-gpu.js';

const source=fs.readFileSync(new URL('../dist/coherent-field.wgsl',import.meta.url),'utf8');
const execute=process.argv.includes('--execute'),tests=[],failures=[];
const report={suite:'coherent-gpu',browserValidated:false,physicalGPUValidated:false,
  executionRequested:execute,nodeVersion:process.version,shaderSHA256:createHash('sha256').update(source).digest('hex'),
  sourceHashes:Object.fromEntries(['coherent-core.js','coherent-gpu.js','coherent-field.wgsl'].map(name=>[name,
    createHash('sha256').update(fs.readFileSync(new URL(`../dist/${name}`,import.meta.url))).digest('hex')])),
  tolerance:COHERENT_GPU_TOLERANCE,limits:COHERENT_GPU_LIMITS,tests,failures,nativeCases:[]};
const test=async(name,fn)=>{try{await fn();tests.push(name);console.log(`PASS ${name}`);}catch(error){failures.push({name,error:error.stack,parity:error.parity});console.error(`FAIL ${name}: ${error.message}`);}};
const delay=()=>new Promise(resolve=>setTimeout(resolve,0));
const deferred=()=>{let resolve;const promise=new Promise(r=>{resolve=r;});return {promise,resolve};};
const rawReadback=reference=>{const data=new Float32Array(reference.mask.length*8);for(let i=0;i<reference.mask.length;i++){
  data.set(reference.fieldsRaw.slice(6*i,6*i+6),8*i);data[8*i+6]=reference.intensity[i];data[8*i+7]=reference.mask[i];
}return data;};
const smallSolution=solveCoherent({screenCount:2}),smallGrid={width:17,height:13},smallReference=evaluateGrid(smallSolution,smallGrid);

await test('full-array gate rejects interior complex, intensity, mask and nonfinite corruption',()=>{
  const valid=smallReference.mask.findIndex(x=>x===1),masked=smallReference.mask.findIndex(x=>x===0);
  assert.equal(coherentParityGate(smallReference,rawReadback(smallReference)).points,17*13);
  for(const [index,lane,value] of [[valid,0,2],[valid,6,2],[valid,7,0],[valid,1,NaN],[valid,6,-1],[masked,2,.01]]){
    const data=rawReadback(smallReference);data[8*index+lane]=value;
    assert.throws(()=>coherentParityGate(smallReference,data),error=>error.name==='CoherentGPUParityError'&&!error.parity.passed);
  }
  const data=rawReadback(smallReference);for(let i=0;i<smallReference.mask.length;i++)if(smallReference.mask[i])data[8*i]+=1;
  assert.throws(()=>coherentParityGate(smallReference,data),error=>error.parity.fieldFailures===smallReference.mask.filter(x=>x===1).length&&error.parity.failures.length===16);
});
await test('gate thresholds are fixed absolute-plus-relative complex norm and intensity bounds',()=>{
  assert.deepEqual(COHERENT_GPU_TOLERANCE,{fieldAbsolute:2e-5,fieldRelative:.003,intensityAbsolute:2e-5,intensityRelative:.002});
  const reference={xy:new Float64Array([1,1]),fieldsRaw:new Float64Array(6),intensity:new Float64Array(1),mask:new Uint8Array([1])};
  const near=new Float32Array([1e-5,1e-5,0,0,0,0,1e-5,1]);assert(coherentParityGate(reference,near).passed);
  near[0]=1.5e-5;near[1]=1.5e-5;assert.throws(()=>coherentParityGate(reference,near),/parity/);
});
await test('packing preserves all complex columns/factors and rejects unbounded or malformed input',()=>{
  const packed=packCoherentGPU(smallSolution,smallReference);assert.equal(packed.sourceNodes,320);assert.equal(packed.boundaryNodes,128);assert.equal(packed.factorCount,4);
  assert.deepEqual([...packed.waves.slice(2,8)],[...Float32Array.from(smallSolution.sources.weightedSpectrumRaw.slice(0,6))]);
  assert.deepEqual([...packed.factors],[...Float32Array.from(smallSolution.state.factorRaw)]);
  assert.throws(()=>packCoherentGPU(smallSolution,{...smallReference,mask:new Float32Array(smallReference.mask).fill(.5)}),/mask/);
  assert.throws(()=>packCoherentGPU({...smallSolution,k:Infinity},smallReference),/wavenumber/);
  assert.throws(()=>packCoherentGPU(smallSolution,{...smallReference,width:99999}),/width/);
  const singular={xy:smallSolution.boundary.xy.slice(0,2),fieldsRaw:new Float64Array(6),intensity:new Float64Array(1),mask:new Uint8Array([1])};
  assert.throws(()=>packCoherentGPU(smallSolution,singular),/Bessel/);
  assert.throws(()=>packCoherentGPU(smallSolution,{...singular,xy:new Float64Array([1e5,0])}),/phase/);
  assert.throws(()=>packCoherentGPU(smallSolution,smallReference,{state:{factorRaw:new Float64Array(30)}}),/factors/);
});

const {create,globals}=process.env.WEBGPU_MODULE?await import(pathToFileURL(path.resolve(process.env.WEBGPU_MODULE)).href):await import('webgpu');
report.webgpuVersion=JSON.parse(fs.readFileSync(process.env.WEBGPU_MODULE?
  path.join(path.dirname(path.resolve(process.env.WEBGPU_MODULE)),'package.json'):new URL('../node_modules/webgpu/package.json',import.meta.url),'utf8')).version;
Object.assign(globalThis,globals);

// This inline mock verifies host lifecycle only. It does not execute the shader.
function mockGPU({fixture=rawReadback(smallReference),compileError,mapError,rangeError,queueError,submitError,
  validationError,pipelineError,workWait,onWork,mapWait,onMap,limit=1e9}={}){
  const lost=deferred(),listeners={},buffers=[];let scopes=0,unmaps=0,reads=0,destroyed=false,submitted=0;
  const device={limits:{maxBufferSize:limit,maxStorageBufferBindingSize:limit},lost:lost.promise,
    addEventListener:(key,fn)=>{listeners[key]=fn;},removeEventListener:key=>{delete listeners[key];},
    pushErrorScope:()=>{scopes++;},popErrorScope:async()=>{scopes--;return validationError?{message:validationError}:null;},
    createShaderModule:()=>({getCompilationInfo:async()=>({messages:compileError?[{type:'error',lineNum:1,linePos:2,message:compileError}]:[]})}),
    createComputePipelineAsync:async()=>{if(pipelineError)throw new Error(pipelineError);return {getBindGroupLayout:()=>({})};},
    createBuffer:options=>{const b={destroyed:false,size:options.size,buffer:new ArrayBuffer(options.size),
      destroy(){this.destroyed=true;},async mapAsync(){onMap?.();if(mapWait)await mapWait;if(mapError)throw new Error(mapError);},
      getMappedRange(){reads++;if(rangeError)throw new Error(rangeError);return fixture.buffer.slice(0);},unmap(){unmaps++;}};buffers.push(b);return b;},
    createBindGroup:()=>({}),createCommandEncoder:()=>({beginComputePass:()=>({setPipeline(){},setBindGroup(){},dispatchWorkgroups(){},end(){}}),copyBufferToBuffer(){},finish:()=>({})}),
    queue:{writeBuffer(){},submit(){submitted++;if(submitError)throw new Error(submitError);},async onSubmittedWorkDone(){onWork?.();if(workWait)await workWait;if(queueError)throw new Error(queueError);}},
    destroy(){destroyed=true;}};
  return {gpu:{requestAdapter:async()=>({info:{description:'CONTROL-FLOW MOCK; NO WGSL EXECUTION'},requestDevice:async()=>device})},
    lose:reason=>lost.resolve({message:reason}),uncaptured:reason=>listeners.uncapturederror?.({error:{message:reason}}),
    get reads(){return reads;},get unmaps(){return unmaps;},get submitted(){return submitted;},
    check(){assert.equal(scopes,0);assert(destroyed);assert(buffers.every(x=>x.destroyed));}};
}
async function withMock(options,body){const mock=mockGPU(options);let engine;try{
  engine=await makeCoherentGPUEngine(smallSolution,smallGrid,{gpu:mock.gpu,reference:smallReference,shaderSource:source});await body(engine,mock);
}finally{engine?.destroy();}mock.check();}
await test('mock success is hybrid-labelled, CPU time separate, GPU phase preview uses GPU fields',async()=>{
  const data=rawReadback(smallReference),first=smallReference.mask.findIndex(x=>x===1);data[8*first]+=1e-6;
  await withMock({fixture:data},async(engine,mock)=>{const progress=[],result=await engine.run({onProgress:p=>progress.push(p)});
    assert.equal(result.backend,'WebGPU hybrid');assert.equal(result.timings.cpuReferenceMs,null);assert.equal(result.timings.cpuReferenceProvided,true);
    assert(result.parity.passed);assert.equal(progress.at(-1),1);assert.equal(mock.reads,1);assert.equal(mock.unmaps,1);
    let re=0,im=0;for(let m=0;m<3;m++){const a=6*first+2*m,b=2*m;re+=result.fieldsRaw[a]*smallSolution.state.coherentRaw[b]-result.fieldsRaw[a+1]*smallSolution.state.coherentRaw[b+1];im+=result.fieldsRaw[a]*smallSolution.state.coherentRaw[b+1]+result.fieldsRaw[a+1]*smallSolution.state.coherentRaw[b];}
    assert.equal(result.coherentField[2*first],re);assert.equal(result.coherentField[2*first+1],im);
  });
});
await test('missing adapter, compile/pipeline/buffer errors fail closed',async()=>{
  await assert.rejects(makeCoherentGPUEngine(smallSolution,smallGrid,{reference:smallReference,gpu:null}),/unavailable/);
  await assert.rejects(makeCoherentGPUEngine(smallSolution,smallGrid,{reference:smallReference,gpu:{requestAdapter:async()=>null}}),/adapter/);
  for(const options of [{compileError:'broken shader'},{pipelineError:'pipeline failed'},{limit:32},{validationError:'validation failed'}]){
    const mock=mockGPU(options);await assert.rejects(makeCoherentGPUEngine(smallSolution,smallGrid,{reference:smallReference,gpu:mock.gpu,shaderSource:source}));mock.check();
  }
});
await test('map, read range, queue and parity failures destroy resources without publishing',async()=>{
  for(const options of [{mapError:'map failed'},{rangeError:'range failed'},{queueError:'queue failed'},{submitError:'submit failed'}])
    await withMock(options,async(engine)=>{await assert.rejects(engine.run(),/failed/);});
  const corrupt=rawReadback(smallReference);corrupt[0]=NaN;
  await withMock({fixture:corrupt},async(engine,mock)=>{await assert.rejects(engine.run(),error=>error.name==='CoherentGPUParityError'&&error.diagnostic.parity.nonfinite===1);assert.equal(mock.unmaps,1);});
});
await test('cancellation and destruction never return partial or stale pixels',async()=>{
  const pre=new AbortController();pre.abort();await withMock({},async(engine,mock)=>{await assert.rejects(engine.run({signal:pre.signal}),error=>error.name==='AbortError');assert.equal(mock.submitted,0);});
  for(const during of ['work','map']){
    const entered=deferred(),resume=deferred(),controller=new AbortController();
    const options=during==='work'?{workWait:resume.promise,onWork:()=>entered.resolve()}:{mapWait:resume.promise,onMap:()=>entered.resolve()};
    await withMock(options,async(engine,mock)=>{const run=engine.run({signal:controller.signal});await entered.promise;controller.abort();resume.resolve();await assert.rejects(run,error=>error.name==='AbortError');assert.equal(mock.reads,0);});
  }
  await withMock({},async(engine)=>{await assert.rejects(engine.run({onProgress:()=>engine.destroy()}),error=>error.name==='AbortError');engine.destroy();await assert.rejects(engine.run(),error=>error.name==='AbortError');});
});
await test('device loss and uncaptured errors are explicit and suppress readback',async()=>{
  for(const kind of ['lose','uncaptured'])await withMock({},async(engine,mock)=>{mock[kind]('device failed');await delay();await assert.rejects(engine.run(),/device failed/);assert.equal(mock.reads,0);});
  const entered=deferred(),resume=deferred();await withMock({workWait:resume.promise,onWork:()=>entered.resolve()},async(engine,mock)=>{
    const run=engine.run();await entered.promise;mock.lose('device lost while executing');await delay();resume.resolve();await assert.rejects(run,/device lost/);assert.equal(mock.reads,0);
  });
});
await test('concurrent runs rejected; numerical reference snapshots are stable across await',async()=>{
  const entered=deferred(),resume=deferred();await withMock({workWait:resume.promise,onWork:()=>entered.resolve()},async(engine)=>{
    const run=engine.run();await entered.promise;await assert.rejects(engine.run(),/already in progress/);resume.resolve();assert((await run).parity.passed);
  });
  const reference={...smallReference,fieldsRaw:smallReference.fieldsRaw.slice(),intensity:smallReference.intensity.slice(),
    xy:smallReference.xy.slice(),mask:smallReference.mask.slice()},mock=mockGPU();
  const engine=await makeCoherentGPUEngine(smallSolution,smallGrid,{reference,gpu:mock.gpu,shaderSource:source});
  try{reference.fieldsRaw.fill(999);reference.intensity.fill(999);reference.xy.fill(999);reference.mask.fill(0);
    assert((await engine.run()).parity.passed);
  }finally{engine.destroy();}mock.check();
});

await test('native Dawn/Tint WGSL module and pipeline compile; negative control rejected',async()=>{
  let gpu=create(['backend=null']);const adapter=await gpu.requestAdapter();assert(adapter);const device=await adapter.requestDevice();
  try{
    device.pushErrorScope('validation');const module=device.createShaderModule({code:source}),info=await module.getCompilationInfo();
    assert.equal(info.messages.filter(x=>x.type==='error').length,0);assert.equal(await device.popErrorScope(),null);
    await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'coherent_main'}});
    device.pushErrorScope('validation');const bad=device.createShaderModule({code:'struct Bad { meta: vec4f };'}),badInfo=await bad.getCompilationInfo();
    assert(badInfo.messages.some(x=>x.type==='error'));assert(await device.popErrorScope());report.nativeCompilePassed=true;
  }finally{device.destroy();gpu=null;}
});

if(execute){
  const backend=process.env.WEBGPU_EXECUTION_BACKEND;
  assert(['vulkan','metal','d3d11','d3d12','opengl','opengles'].includes(backend),'Actual execution requires WEBGPU_EXECUTION_BACKEND; null is refused');
  report.requestedBackend=backend;report.execution='Native Dawn WGSL dispatch and readback; software adapter is not browser or physical GPU evidence';
  let gpu=create([`backend=${backend}`]);
  try{
    const cases=[
      {name:'one-screen UI default 144x108',options:{screenCount:1},grid:{width:144,height:108}},
      {name:'two-screen UI default 144x108',options:{screenCount:2},grid:{width:144,height:108}},
      {name:'two-screen n96 partial-coherence shifted',options:{n:96,gamma:.43,phase:1.7,separation:1.8,offset:-.2},grid:{width:144,height:108}},
      {name:'two-screen n144 incoherent shifted',options:{n:144,gamma:0,phase:2.4,separation:3,offset:1},grid:{width:97,height:73}},
      {name:'two-screen nq480 coherent destructive phase',options:{nq:480,phase:Math.PI},grid:{width:97,height:73}},
      {name:'zero source power',options:{inputPower:0},grid:{width:33,height:25}},
      {name:'source power scale seven',options:{inputPower:7,gamma:.55,phase:-.8},grid:{width:65,height:49}},
    ];
    for(const item of cases)await test(`actual WGSL: ${item.name}`,async()=>{
      const solution=solveCoherent(item.options),progress=[],result=await computeCoherentGPU(solution,item.grid,{gpu,shaderSource:source,onProgress:p=>progress.push(p)});
      assert(result.parity.passed);assert.equal(result.parity.points,item.grid.width*item.grid.height);assert.equal(progress.at(-1),1);
      assert.equal(progress.length,Math.ceil(result.mask.length/COHERENT_GPU_LIMITS.batchPoints));assert(result.fieldsRaw.every(Number.isFinite));assert(result.coherentField.every(Number.isFinite));
      report.adapter=result.adapter;report.nativeCases.push({name:item.name,parameters:item.options,grid:item.grid,parity:result.parity,timings:result.timings});
    });
    // Frozen independent SciPy 1.17.0 j0/y0 fixtures; dense sweep additionally
    // uses the independently implemented CPU Cephes evaluator.
    const golden=[
      [.001,.9999997500000155,-4.471416611375923],[.01,.9999750001562495,-3.005455637083646],
      [.1,.99750156206604,-1.5342386513503667],[.5,.938469807240813,-.4445187335067066],
      [1,.7651976865579665,.08825696421567697],[2,.22389077914123562,.5103756726497451],
      [4,-.3971498098638473,-.016940739325064846],[5,-.1775967713143383,-.30851762524903303],
      [7.9,.1943618448412782,.20652094814437574],[7.999,.17188537228232045,.22336330730718532],
      [8,.1716508071375539,.22352148938756622],[8.001,.1714160996715328,.22367942818892078],
      [10,-.24593576445134832,.05567116728359961],[16,-.1748990739836291,.09581099708071256],
      [32,.1380790097465559,-.028742484654433412],[64,.09259001221604803,.03706710323208859],
      [128,.0014722223281849597,.07050806047147914],[256,-.03665349806171347,-.03381290171792464],
      [512,-.022865672770482643,.02684321904623385],[1024,.014610399860870881,-.02020482957725712],
    ];
    await test('actual WGSL: Bessel Green function dense range and independent SciPy fixtures',async()=>{
      const values=[...golden.map(x=>x[0]),...Array.from({length:4096},(_,i)=>Math.exp(Math.log(.0001001)+i/4095*Math.log(1024/.0001001))),
        ...Array.from({length:2048},(_,i)=>6+i/2047*4)];
      const n=values.length,xy=new Float64Array(n*2),fieldsRaw=new Float64Array(n*6),intensity=new Float64Array(n),mask=new Uint8Array(n).fill(1);
      for(let i=0;i<n;i++){const x=values[i],j=i<golden.length?golden[i][1]:besselJ0(x),y=i<golden.length?golden[i][2]:besselY0(x);xy[2*i]=x;fieldsRaw[6*i]=-.25*y;fieldsRaw[6*i+1]=-.25*j;intensity[i]=(j*j+y*y)/16;}
      const synthetic={k:1,sources:{q:[0],kx:[0],weightedSpectrumRaw:new Float64Array(6)},boundary:{xy:[0,0],weightedPhiRaw:[1,0,0,0,0,0]},state:{factorRaw:[1,0,0,0,0,0]}};
      const reference={xy,fieldsRaw,intensity,mask},result=await computeCoherentGPU(synthetic,{xy},{reference,gpu,shaderSource:source});
      let maxError=0;for(let i=0;i<n;i++)maxError=Math.max(maxError,4*Math.hypot(result.fieldsRaw[6*i]-fieldsRaw[6*i],result.fieldsRaw[6*i+1]-fieldsRaw[6*i+1]));
      // Component-amplitude error in (J0,Y0), stricter than display gate.
      assert(maxError<8e-6,`Bessel vector absolute error ${maxError}`);
      report.nativeCases.push({name:'Bessel Green function sweep',points:n,range:[Math.min(...values),Math.max(...values)],independentSciPyFixtures:golden.length,maxBesselVectorError:maxError,parity:result.parity});
    });
    await test('actual WGSL: source factors retain constructive/destructive interference',async()=>{
      const xy=new Float64Array([1,0,2,0]),fieldsRaw=new Float64Array([1,0,1,0,0,0,1,0,1,0,0,0]),mask=new Uint8Array([1,1]);
      const synthetic={k:1,sources:{q:[0],kx:[0],weightedSpectrumRaw:[1,0,1,0,0,0]},boundary:{xy:[0,0],weightedPhiRaw:new Float64Array(6)}};
      const signals=[];for(const sign of [1,-1]){
        synthetic.state={factorRaw:[Math.SQRT1_2,0,sign*Math.SQRT1_2,0,0,0]};
        const intensity=new Float64Array(2).fill(sign===1?2:0),reference={xy,fieldsRaw,intensity,mask};
        const result=await computeCoherentGPU(synthetic,{xy},{reference,gpu,shaderSource:source});signals.push(result.intensity[0]);
      }
      assert(Math.abs(signals[0]-2)<1e-6);assert.equal(signals[1],0);report.nativeCases.push({name:'Coherent interference negative control',constructive:signals[0],destructive:signals[1],incoherentWouldBe:1});
    });
    await test('actual WGSL: shader perturbation is withheld by the production all-pixel gate',async()=>{
      const wrong=source.replace('output[index]=FieldRow(u0,u1,u2,intensity,1.0);','output[index]=FieldRow(u0+vec2f(0.01,0.0),u1,u2,intensity,1.0);');
      assert.notEqual(wrong,source);await assert.rejects(computeCoherentGPU(smallSolution,smallGrid,{reference:smallReference,gpu,shaderSource:wrong}),error=>error.name==='CoherentGPUParityError'&&error.parity.fieldFailures>0);
    });
  }finally{gpu=null;}
}
report.passed=failures.length===0;report.passedTests=tests.length;report.failedTests=failures.length;
if(process.env.COHERENT_GPU_REPORT)fs.writeFileSync(process.env.COHERENT_GPU_REPORT,JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({passed:report.passed,tests:tests.length,failed:failures.length,nativeCompilePassed:report.nativeCompilePassed,
  nativeExecutionCases:report.nativeCases.length,adapter:report.adapter,browserValidated:false,physicalGPUValidated:false},null,2));
if(failures.length)process.exitCode=1;
