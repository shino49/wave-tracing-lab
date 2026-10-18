/** Native Tint compile is mandatory. --execute requires a real executable backend.
 * VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json WEBGPU_EXECUTION_BACKEND=vulkan node tests/plt-spectral-gpu.mjs --execute
 * SwiftShader verifies native WGSL execution; not browser, physical GPU, or physical accuracy.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {create,globals} from 'webgpu';
import {coefficient} from '../dist/utd-math.js';
import {packPLTGPU,pltEventParityGate,pltPixelParityGate,makePLTGPUEngine,renderPLTGPU,createPLTGPUBatchRunner,PLT_GPU_LIMITS,PLT_GPU_TOLERANCE} from '../dist/plt-gpu.js';
Object.assign(globalThis,globals);
const read=name=>fs.readFileSync(new URL(`../dist/${name}`,import.meta.url),'utf8'),hash=value=>createHash('sha256').update(value).digest('hex');
const shaderNames=['utd-functions.wgsl','plt-compute.wgsl'],source=shaderNames.map(read).join('\n'),execute=process.argv.includes('--execute');
const tests=[],failures=[],report={suite:'plt-spectral-gpu',schemaVersion:1,browserValidated:false,physicalGPUValidated:false,executionRequested:execute,
  scope:'Every local event and every XYZ camera component computed by native WGSL, checked against CPU references at unchanged v13 tolerances; no independent physical-accuracy claim',
  sourceHashes:Object.fromEntries([...shaderNames,'plt-gpu.js','utd-math.js'].map(name=>[name,hash(read(name))])),
  testSHA256:hash(fs.readFileSync(new URL('./plt-spectral-gpu.mjs',import.meta.url))),limits:PLT_GPU_LIMITS,tolerance:PLT_GPU_TOLERANCE,tests,failures,nativeCases:[],negativeControls:[]};
const test=async(name,body)=>{try{await body();tests.push(name);console.log(`PASS ${name}`);}catch(error){failures.push({name,error:error.stack,validation:error.validation,diagnostic:error.diagnostic});console.error(`FAIL ${name}: ${error.stack}`);}};
function evaluate(op){
  let sr=op.directAmplitude,si=0,hr=sr,hi=0;
  for(const edge of op.edges){const k=edge.k??op.k,L=edge.ro*edge.sinBeta**2,p=-k*edge.phaseExcess,c=Math.cos(p)/Math.sqrt(edge.ro),s=Math.sin(p)/Math.sqrt(edge.ro);
    for(const boundary of ['soft','hard']){const d=coefficient(k,edge.n,edge.phiI,edge.phiO,L,edge.sinBeta,boundary,2e-6),re=d[0]*c-d[1]*s,im=d[0]*s+d[1]*c;if(boundary==='soft'){sr+=re;si+=im;}else{hr+=re;hi+=im;}}
  }
  const sp=sr*sr+si*si,hp=hr*hr+hi*hi;return[sr,si,hr,hi,op.boundary==='soft'?sp:op.boundary==='hard'?hp:(sp+hp)/2,sp,hp,op.edges.length];
}
function reference(job){
  job.referenceXYZ=new Float64Array(job.width*job.height*3);
  for(const term of job.terms){let value=term.weight;for(const id of term.eventIds)value*=job.events[id].referenceF;
    for(let c=0;c<3;c++)job.referenceXYZ[3*term.pixel+c]+=value*term.xyzWeight[c]/job.samplesPerPixel;}
  job.reference=Float64Array.from({length:job.width*job.height},(_,i)=>job.referenceXYZ[3*i+1]);return job;
}
function synthetic({width=7,height=5,samplesPerPixel=7,eventCount=61,maxEdges=7,phaseLimit=128,zero=false}={}){
  let state=39101;const random=()=>((state=(Math.imul(state,1664525)+1013904223)>>>0)/2**32);
  const events=Array.from({length:eventCount},(_,i)=>{const wavelengthNm=380+400*random(),k=2*Math.PI/(wavelengthNm*1e-6),op={k,boundary:['soft','hard','unpolarized'][i%3],directAmplitude:zero?0:i%4===0?.5:i%4===1?0:1,
    edges:Array.from({length:zero?0:i%(maxEdges+1)},()=>({k,n:1.1+random()*.9,phiI:.2+random()*2.4,phiO:.2+random()*2.4,ro:.3+random()*3,sinBeta:.3+random()*.7,phaseExcess:random()*Math.min(2,phaseLimit/k)}))};
    return {...op,referenceF:evaluate(op)[4],wavelengthNm};});
  // Independent wavelength-dependent source/material scalars in each term and
  // non-proportional XYZ weights exercise actual three-channel accumulation.
  const terms=[];for(let pixel=0;pixel<width*height;pixel++)for(let s=0;s<samplesPerPixel;s++){
    const sourceRadiance=.1+3*random(),materialReflectance=random(),xyzWeight=s%5===0?[0,random(),0]:[.1+2*random(),.2+random(),.01+3*random()];
    terms.push({pixel,weight:zero?0:sourceRadiance*materialReflectance,eventIds:Array.from({length:eventCount?s%7:0},()=>Math.floor(random()*eventCount)),xyzWeight});}
  terms.reverse();return reference({width,height,samplesPerPixel,spectralMode:'visible',events,terms});
}
const rawEvents=job=>Float32Array.from(job.events.flatMap(event=>evaluate(event.operator??event)));
function mockGPU(job,{camera=Float32Array.from(job.referenceXYZ),events=rawEvents(job),limit=1e9,pendingWork=null,onWork=()=>{}}={}){
  const buffers=[],listeners={};let scopes=0,destroyed=false,reads=0,resolveLost;
  const lost=new Promise(resolve=>resolveLost=resolve),device={limits:{maxBufferSize:limit,maxStorageBufferBindingSize:limit},lost,
    addEventListener:(name,fn)=>listeners[name]=fn,removeEventListener:name=>delete listeners[name],
    pushErrorScope:()=>scopes++,popErrorScope:async()=>{scopes--;return null;},
    createShaderModule:()=>({getCompilationInfo:async()=>({messages:[]})}),createComputePipelineAsync:async()=>({getBindGroupLayout:()=>({})}),
    createBuffer(options){const data=options.label==='PLT event readback'?events:camera,b={...options,destroyed:false,destroy(){this.destroyed=true;},async mapAsync(){},getMappedRange(){reads++;return data.buffer.slice(0);},unmap(){}};buffers.push(b);return b;},
    createBindGroup:()=>({}),createCommandEncoder:()=>({beginComputePass:()=>({setPipeline(){},setBindGroup(){},dispatchWorkgroups(){},end(){}}),copyBufferToBuffer(){},finish:()=>({})}),
    queue:{writeBuffer(){},submit(){},async onSubmittedWorkDone(){onWork();if(pendingWork)await pendingWork;}},destroy:()=>destroyed=true};
  return {gpu:{requestAdapter:async()=>({info:{description:'CONTROL-FLOW MOCK ONLY'},requestDevice:async()=>device})},lose:()=>resolveLost({message:'spectral device lost'}),
    get reads(){return reads;},get buffers(){return buffers;},check(){assert.equal(scopes,0);assert(destroyed);assert(buffers.every(x=>x.destroyed));}};
}
await test('interleaved virtual pixels preserve every term, scalar event tape and immutable snapshots',()=>{
  const job=synthetic(),packed=packPLTGPU(job),u=new Uint32Array(packed.termData),f=new Float32Array(packed.termData);
  assert.equal(packed.physicalPixels,35);assert.equal(packed.channels,3);assert.equal(packed.pixels,105);assert.equal(packed.termCount,job.terms.length*3);
  assert.equal(packed.physicalTermCount,job.terms.length);assert.equal(packed.createUniforms()[1],105);
  for(let pixel=0;pixel<35;pixel++)for(let c=0;c<3;c++){
    const virtual=3*pixel+c,expected=job.terms.filter(t=>t.pixel===pixel);assert.equal(packed.ranges[virtual*2+1],expected.length);
    for(let j=0;j<expected.length;j++){const base=12*(packed.ranges[virtual*2]+j),term=expected[j];assert.equal(f[base],Math.fround(term.weight*term.xyzWeight[c]));assert.equal(u[base+1],term.eventIds.length);assert.deepEqual([...u.slice(base+4,base+4+term.eventIds.length)],term.eventIds);}
  }
  const saved={reference:packed.reference.slice(),data:packed.termData.slice(0),uniforms:packed.createUniforms()};
  job.referenceXYZ.fill(99);job.reference.fill(99);job.terms[0].xyzWeight.fill(99);job.events.length=0;job.samplesPerPixel=99;
  assert.deepEqual(packed.reference,saved.reference);assert.deepEqual(packed.termData,saved.data);assert.deepEqual(packed.createUniforms(),saved.uniforms);
});
await test('strict XYZ shape, finite nonnegative values, matching Y, physical indices and multiplied weights',()=>{
  const changes=[j=>delete j.referenceXYZ,j=>j.referenceXYZ=new Float32Array(j.referenceXYZ),j=>j.referenceXYZ=new Float64Array(j.referenceXYZ.length-1),
    j=>j.reference=new Float64Array(j.reference.length*3),j=>j.reference[0]+=1,j=>j.referenceXYZ[0]=-1,j=>j.referenceXYZ[2]=NaN,
    j=>delete j.terms[0].xyzWeight,j=>j.terms[0].xyzWeight=[1,2],j=>j.terms[0].xyzWeight=[1,2,3,4],j=>j.terms[0].xyzWeight=new Float32Array([1,2,3]),
    j=>j.terms[0].xyzWeight=[1,,3],j=>j.terms[0].xyzWeight=[1,NaN,3],j=>j.terms[0].xyzWeight=[Infinity,1,3],j=>j.terms[0].xyzWeight=[1,2,-1],
    j=>{j.terms[0].weight=1e30;j.terms[0].xyzWeight=[2,2,2];},j=>j.terms[0].pixel=j.width*j.height];
  for(const mutate of changes){const job=synthetic();mutate(job);assert.throws(()=>packPLTGPU(job),RangeError);}
});
await test('tripled pixel and term counts stay within original allocation budgets',()=>{
  const job=reference({width:21845,height:1,samplesPerPixel:1,spectralMode:'visible',events:[],terms:[]});assert.equal(packPLTGPU(job).pixels,65535);
  job.width++;job.reference=new Float64Array(job.width);job.referenceXYZ=new Float64Array(job.width*3);assert.throws(()=>packPLTGPU(job),/expanded camera component count/);
  const over=synthetic();over.terms=Array(Math.floor(PLT_GPU_LIMITS.terms/3)+1);assert.throws(()=>packPLTGPU(over),/expanded term count/);
  const perPixel=reference({width:1,height:1,samplesPerPixel:1,spectralMode:'visible',events:[],terms:Array(PLT_GPU_LIMITS.termsPerPixel+1).fill({pixel:0,weight:0,xyzWeight:[0,0,0],eventIds:[]})});
  assert.throws(()=>packPLTGPU(perPixel),/pixel 0 term count/);
});
await test('every XYZ gate rejects isolated X/Y/Z corruption with physical pixel and channel identity',()=>{
  const job=synthetic(),packed=packPLTGPU(job),good=Float32Array.from(job.referenceXYZ),gate=pltPixelParityGate(packed,good);
  assert.equal(gate.coverage,'every-color-component');assert.equal(gate.count,3*job.width*job.height);assert.equal(gate.absoluteErrors.length,good.length);assert.equal(gate.channelOrder,'XYZ');
  for(const pixel of [0,job.width*job.height-1])for(let c=0;c<3;c++)for(const value of [-1,Infinity,NaN,1e10]){
    const corrupted=good.slice(),index=3*pixel+c;corrupted[index]=value;
    assert.throws(()=>pltPixelParityGate(packed,corrupted),error=>error.name==='PLTGPUParityError'&&error.validation.failures.some(f=>f.index===index&&f.pixel===pixel&&f.channel==='XYZ'[c]));
  }
  assert.throws(()=>pltPixelParityGate(packed,new Float32Array(packed.physicalPixels)),/exact Float32Array/);
  const eventValues=rawEvents(job);eventValues[eventValues.length-4]+=2;assert.throws(()=>pltEventParityGate(packed,eventValues),error=>error.name==='PLTGPUParityError');
});
await test('mono packing retains scalar ABI bytes, one-channel gate, and ignores spectral-only fields',()=>{
  const spectral=synthetic(),mono={...spectral,spectralMode:'monochrome',reference:new Float64Array(spectral.width*spectral.height)};
  for(const t of mono.terms){let v=t.weight;for(const id of t.eventIds)v*=mono.events[id].referenceF;mono.reference[t.pixel]+=v/mono.samplesPerPixel;}
  const packed=packPLTGPU(mono),again=packPLTGPU({...mono,spectralMode:undefined,referenceXYZ:null,terms:mono.terms.map(t=>({...t,xyzWeight:null}))});
  assert.equal(packed.channels,1);assert.equal(packed.pixels,35);assert.equal(packed.termCount,mono.terms.length);
  for(const key of ['eventData','edgeData','termData','ranges','reference'])assert.deepEqual(packed[key],again[key]);
  assert.equal(pltPixelParityGate(packed,Float32Array.from(mono.reference)).coverage,'every-pixel');
  const spectralPacked=packPLTGPU(spectral);assert.deepEqual(packed.eventData,spectralPacked.eventData);
  // Spectral reduction changes only the periodic phase distance lane.
  for(let i=0;i<packed.edgeData.length;i++)if(i%8!==6)assert.equal(packed.edgeData[i],spectralPacked.edgeData[i]);
});
await test('visible phase reduction is periodic through ±1e8, with unchanged monochrome phase admission',()=>{
  const job=synthetic(),edge=job.events.find(e=>e.edges.length).edges[0];edge.k=2*Math.PI/.00036;
  assert.equal(PLT_GPU_LIMITS.maxPhase,100000);assert.equal(PLT_GPU_LIMITS.maxSpectralRawPhase,1e8);
  assert.equal(PLT_GPU_TOLERANCE.eventRelative,.003);assert.equal(PLT_GPU_TOLERANCE.pixelRelative,.003);
  for(const phase of [-1e8,-99999999,-1e6,-100001,-Math.PI,-1,0,1,Math.PI,100001,1e6,99999999,1e8]){
    edge.phaseExcess=phase/edge.k;const packed=packPLTGPU(job),gpuPhase=Math.fround(packed.edgeData[0]*packed.edgeData[6]);
    assert(Math.abs(gpuPhase)<=Math.PI+1e-6);assert(Math.abs(Math.cos(gpuPhase)-Math.cos(phase))<1e-6);assert(Math.abs(Math.sin(gpuPhase)-Math.sin(phase))<1e-6);
    assert(packed.phaseReduction.maxAbsolutePackedPhase<=Math.PI+1e-6);assert.equal(packed.phaseReduction.rawPhaseLimit,1e8);
  }
  for(const phase of [1e8+1,-1e8-1]){edge.phaseExcess=phase/edge.k;assert.throws(()=>packPLTGPU(job),/visible raw local phase/);}
  edge.phaseExcess=100001/edge.k;assert.throws(()=>packPLTGPU({...job,spectralMode:'monochrome'}),/local phase/);
  edge.k=100000001;edge.phaseExcess=0;assert.throws(()=>packPLTGPU(job),/ k /);

});
await test('dense periodic-reduction sweep bounds the actual f32 product phase error',()=>{
  const op={boundary:'unpolarized',directAmplitude:1,edges:[{k:1,n:1.5,phiI:.3,phiO:2.3,ro:1,sinBeta:1,phaseExcess:0}],referenceF:1};
  const job=reference({width:1,height:1,samplesPerPixel:1,spectralMode:'visible',events:[op],terms:[]});let maxError=0,samples=0;
  for(const nm of [360,380,450,550,650,780,830])for(let i=0;i<=1024;i++){
    const edge=op.edges[0],phase=-99999999+199999998*i/1024;edge.k=2*Math.PI/(nm*1e-6);edge.phaseExcess=phase/edge.k;
    const packed=packPLTGPU(job),actual=Math.fround(packed.edgeData[0]*packed.edgeData[6]),raw=edge.k*edge.phaseExcess;
    maxError=Math.max(maxError,Math.abs(Math.sin(actual)-Math.sin(raw)),Math.abs(Math.cos(actual)-Math.cos(raw)));samples++;
  }
  assert(maxError<5e-7);report.phaseReductionSweep={samples,maxAbsolutePhasorComponentError:maxError,absoluteLimit:5e-7,rawPhaseRange:[-99999999,99999999],wavelengthRangeNm:[360,830]};
});
await test('successful engine returns verified GPU XYZ and Y extraction with complete component metadata',async()=>{
  const job=synthetic(),mock=mockGPU(job);try{const result=await renderPLTGPU(job,{gpu:mock.gpu,shaderSource:source});assert.equal(result.spectralMode,'visible');assert.equal(result.xyz.length,job.referenceXYZ.length);
    assert.deepEqual(result.image,Float32Array.from({length:job.reference.length},(_,i)=>result.xyz[3*i+1]));assert.equal(result.validation.coverage,'every-event-and-every-color-component');
    assert.equal(result.validation.pixels.count,result.xyz.length);assert.equal(result.validation.events.count,job.events.length);assert.equal(mock.reads,2);
  }finally{mock.check();}
});
await test('adapter allocation checks include every expanded component and release buffers on rejection',async()=>{
  const job=synthetic(),mock=mockGPU(job,{limit:job.terms.length*48+16});await assert.rejects(renderPLTGPU(job,{gpu:mock.gpu,shaderSource:source}),/adapter buffer limits/);mock.check();
});
await test('X/Z-only failures with correct luminance withhold all output and poison a batch runner',async()=>{
  for(const c of [0,2]){const job=synthetic(),camera=Float32Array.from(job.referenceXYZ);camera[c]+=10;const mock=mockGPU(job,{camera}),runner=createPLTGPUBatchRunner({gpu:mock.gpu,shaderSource:source});
    await assert.rejects(runner.run(job),error=>error.name==='PLTGPUParityError'&&error.validation.failures[0].channel==='XYZ'[c]);
    await assert.rejects(runner.run(job),error=>error.name==='AbortError');runner.destroy();mock.check();}
});
await test('spectral cancellation and device loss withhold output and clean expanded resources',async()=>{
  for(const kind of ['cancel','loss']){const job=synthetic(),abort=new AbortController();let resume,entered;const ready=new Promise(resolve=>entered=resolve),pendingWork=new Promise(resolve=>resume=resolve),mock=mockGPU(job,{pendingWork,onWork:entered});
    const engine=await makePLTGPUEngine(job,{gpu:mock.gpu,shaderSource:source});const pending=engine.run({signal:abort.signal});await ready;
    if(kind==='cancel')abort.abort();else mock.lose();await assert.rejects(pending,kind==='cancel'?e=>e.name==='AbortError':/spectral device lost/);assert.equal(mock.reads,0);resume();engine.destroy();mock.check();}
});
await test('native Tint compiles unchanged event/camera shaders with expanded XYZ host ABI',async()=>{
  let gpu=create(['backend=null']);const adapter=await gpu.requestAdapter(),device=await adapter.requestDevice();try{
    device.pushErrorScope('validation');const module=device.createShaderModule({code:source});assert.deepEqual((await module.getCompilationInfo()).messages.filter(x=>x.type==='error'),[]);assert.equal(await device.popErrorScope(),null);
    for(const entryPoint of ['plt_events','plt_camera'])await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint}});report.nativeCompilePassed=true;
  }finally{device.destroy();gpu=null;}
});
function retain(name,job,result){
  assert.equal(result.validation.coverage,'every-event-and-every-color-component');assert(result.validation.passed);assert.equal(result.xyz.length,job.width*job.height*3);
  assert.equal(result.validation.events.count,job.events.length);assert.equal(result.validation.pixels.count,job.referenceXYZ.length);
  assert(result.xyz.every(x=>Number.isFinite(x)&&x>=0));assert.deepEqual(result.image,Float32Array.from({length:job.width*job.height},(_,i)=>result.xyz[i*3+1]));
  report.adapter=result.adapter;report.nativeCases.push({name,width:job.width,height:job.height,samplesPerPixel:job.samplesPerPixel,eventCount:job.events.length,termCount:job.terms.length,
    pathStats:job.stats??null,phaseReduction:result.phaseReduction,xyzTotals:[0,1,2].map(c=>result.xyz.reduce((sum,v,i)=>sum+(i%3===c?v:0),0)),validation:result.validation,timings:result.timings});
}
if(execute){
  const backend=process.env.WEBGPU_EXECUTION_BACKEND;assert(['vulkan','metal','d3d11','d3d12','opengl','opengles'].includes(backend),'Explicit executable backend required; null is refused');
  let gpu=create([`backend=${backend}`]);report.requestedBackend=backend;report.execution='Actual native Dawn/Tint WGSL compilation, dispatch and mapped readback';report.driverSelection={VK_ICD_FILENAMES:process.env.VK_ICD_FILENAMES??null};
  try{
    const cases=[['visible-k mixed boundaries and six local event factors',synthetic()],['multi-dispatch interleaved XYZ and event batches',synthetic({width:43,height:29,eventCount:4103})],
      ['128-edge coherent local sums',synthetic({width:9,height:5,eventCount:129,maxEdges:128})],['zero power and zero spectral channels',synthetic({zero:true})],
      ['source/material terms without diffraction events',synthetic({eventCount:0})],['empty event and term tapes',reference({width:5,height:3,samplesPerPixel:1,spectralMode:'visible',events:[],terms:[]})]];
    for(const [name,job] of cases)await test(`actual WGSL: ${name}`,async()=>retain(name,job,await renderPLTGPU(job,{gpu,shaderSource:source})));
    await test('actual WGSL: one reused device executes differently sized XYZ tapes',async()=>{
      const runner=createPLTGPUBatchRunner({gpu,shaderSource:source});try{for(const [name,job] of cases)retain(`reused ${name}`,job,await runner.run(job));}finally{runner.destroy();}
    });
    await test('actual visible k and high bounded phase: every event and XYZ component at fixed tolerances',async()=>{
      const events=[];for(const wavelengthNm of [360,380,450,550,650,780,830])for(const phase of [-1e8,-100001,-99900,0,1,1024,65536,99900,100001,1e8])for(const boundary of ['soft','hard','unpolarized']){
        const k=2*Math.PI/(wavelengthNm*1e-6),op={k,boundary,directAmplitude:.5,edges:[{k,n:1.5,phiI:.31,phiO:2.21,ro:.7,sinBeta:.707,phaseExcess:phase/k}]};events.push({...op,referenceF:evaluate(op)[4]});}
      const job=reference({width:events.length,height:1,samplesPerPixel:1,spectralMode:'visible',events,terms:events.map((_,pixel)=>({pixel,weight:1,xyzWeight:[.15+(pixel%3),.9,.03+(pixel%5)],eventIds:[pixel]}))});
      retain('visible k 7569..17453 and raw local phases -1e8..1e8',job,await renderPLTGPU(job,{gpu,shaderSource:source}));
    });
    await test('actual broad visible-phase cancellation stress now passes unchanged every-event/XYZ gates',async()=>{
      // These same tapes failed before visible-only f64 reduction (ratios
      // 1.210804 and 1.131419 on SwiftShader). They must now pass, retaining
      // every original CPU reference and the exact inherited tolerances.
      for(const [name,job] of [['4103 optical-k events at phases up to 33000',synthetic({width:43,height:29,eventCount:4103,phaseLimit:100000})],
        ['128-edge optical-k cancellation',synthetic({width:9,height:5,eventCount:129,maxEdges:128,phaseLimit:100000})]])retain(name,job,await renderPLTGPU(job,{gpu,shaderSource:source}));
    });
    await test('actual shader mutations isolated to X, Y or last Z cannot pass camera admission',async()=>{
      const job=synthetic();for(const target of [0,1,job.referenceXYZ.length-1]){
        const modified=source.replace('camera[pixel] = sum / f32(plt.counts.z);',`camera[pixel] = sum / f32(plt.counts.z); if (pixel == ${target}u) { camera[pixel] += 1.0; }`);assert.notEqual(source,modified);
        await assert.rejects(renderPLTGPU(job,{gpu,shaderSource:modified}),error=>{assert.equal(error.name,'PLTGPUParityError');assert(error.validation.failures.some(f=>f.index===target));report.negativeControls.push({name:`isolated ${'XYZ'[target%3]} shader mutation`,validation:error.validation});return true;});
      }
    });
    await test('actual unused local event corruption still rejects the complete spectral result',async()=>{
      const job=synthetic();job.terms.forEach(t=>t.eventIds=[]);reference(job);
      const modified=source.replace('eventResults[index].powers = vec4f(selected,sp,hp,f32(event.edgeRange.y));','eventResults[index].powers = vec4f(selected,sp,hp,f32(event.edgeRange.y)); if (index + 1u == plt.counts.x) { eventResults[index].powers.x += 1.0; }');assert.notEqual(source,modified);
      await assert.rejects(renderPLTGPU(job,{gpu,shaderSource:modified}),error=>{assert.equal(error.name,'PLTGPUParityError');assert.equal(error.validation.failures.at(-1).index,job.events.length-1);report.negativeControls.push({name:'unused local event shader mutation',validation:error.validation});return true;});
    });
    // Real core integration is deliberately mandatory for the execution suite.
    // A missing visible implementation is a failed test, never a silent skip.
    await test('actual scene spectral buildPLTBatch replays full GPU XYZ from source/material path samples',async()=>{
      const {createPLTScene,createPLTRandomStates,buildPLTBatch}=await import('../dist/plt-core.js');
      report.sceneReferenceHashes=Object.fromEntries(['plt-core.js','plt-spectrum.js','plt-cie.js','plt-fsd.js','plt-geometry.js','mesh-geometry.js'].map(name=>[name,hash(read(name))]));
      const totals=[];
      for(const [preset,wave,spectralSource,spectralMaterial] of [['micro-gallery',true,'D65','neutral'],['micro-gallery',true,'E','neutral'],
        ['micro-gallery',true,'D65','red'],['micro-gallery',true,'D65','green'],['micro-gallery',true,'D65','blue'],['micro-gallery',true,'blackbody3000','neutral'],['baffles',false,'D65','neutral']]){
        const options={width:12,height:10,samplesPerPixel:4,maxDepth:4,spectralMode:'visible',spectralSource,spectralMaterial,wave,seed:7391},scene=createPLTScene({preset,...options});
        const job=await buildPLTBatch(scene,scene.camera,options,{tile:{x:0,y:0,width:12,height:10},samplesPerPixel:4,rngStates:createPLTRandomStates(options),yieldEvery:1024});
        assert(job.referenceXYZ instanceof Float64Array,'core must supply sampled spectral XYZ');assert(job.terms.length>0);assert(job.terms.every(t=>Array.isArray(t.xyzWeight)&&t.xyzWeight.length===3));
        assert(job.stats.surfaceInteractions>0);if(wave)assert(job.events.length>0);
        const result=await renderPLTGPU(job,{gpu,shaderSource:source});retain(`real spectral ${preset} ${spectralSource} ${spectralMaterial} wave=${wave}`,job,result);
        totals.push([0,1,2].map(c=>result.xyz.reduce((sum,v,i)=>sum+(i%3===c?v:0),0)));
      }
      for(let i=1;i<6;i++)assert.notDeepEqual(totals[i],totals[0],'source and material spectra must change actual GPU XYZ');
    });
    await test('actual progressive irregular tiles and sample batches retain exhaustive XYZ parity',async()=>{
      const {executePLTTask}=await import('../dist/plt-runtime.js');
      const request={backend:'auto',scene:{preset:'micro-gallery'},options:{width:13,height:11,samplesPerPixel:3,maxDepth:4,spectralMode:'visible',spectralSource:'blackbody3000',spectralMaterial:'red',seed:7391},progressive:{tileSize:5,batchSamples:2}};
      const runner=createPLTGPUBatchRunner({gpu,shaderSource:source});let previews=0;
      try{const result=await executePLTTask(request,{gpuRunner:(job,opts)=>runner.run(job,opts),preview:value=>{previews++;for(const r of value.results){assert(r.xyz instanceof Float64Array);assert.deepEqual(r.image,Float64Array.from({length:r.image.length},(_,i)=>r.xyz[3*i+1]));}}});
        const cpu=await executePLTTask({...request,backend:'cpu'});assert.equal(result.warnings.length,0);
        for(let i=0;i<result.results.length;i++){const r=result.results[i];assert(r.validation.passed);assert.equal(r.validation.acceptedBatches,18);assert.equal(r.validation.cpuFallbackBatches,0);assert(r.sampleCounts.every(n=>n===3));
          const gate=pltPixelParityGate({reference:cpu.results[i].xyz,pixels:r.xyz.length,physicalPixels:r.image.length,channels:3},Float32Array.from(r.xyz));assert.equal(gate.count,429);
          report.nativeCases.push({name:`progressive irregular XYZ ${r.wave?'wave':'geometric'}`,width:r.width,height:r.height,samplesPerPixel:3,validation:r.validation,finalComponentGate:gate});
        }assert(previews>0);
      }finally{runner.destroy();}
    });
    await test('actual X-only shader rejection discards prior GPU XYZ and returns whole-image CPU fallback',async()=>{
      const {executePLTTask}=await import('../dist/plt-runtime.js');
      const request={backend:'auto',scene:{preset:'micro-gallery'},options:{width:8,height:8,samplesPerPixel:2,maxDepth:3,spectralMode:'visible',spectralSource:'D65',spectralMaterial:'blue',seed:7391},progressive:{tileSize:5,batchSamples:1}};
      const modified=source.replace('camera[pixel] = sum / f32(plt.counts.z);','camera[pixel] = sum / f32(plt.counts.z); if (pixel == 0u) { camera[pixel] += 1.0; }');let calls=0;
      const result=await executePLTTask(request,{gpuRunner:async(job,options)=>{calls++;return renderPLTGPU(job,{...options,gpu,shaderSource:calls===2?modified:source});}}),cpu=await executePLTTask({...request,backend:'cpu'});
      assert.equal(calls,2);assert(result.warnings.length>0);for(let i=0;i<result.results.length;i++){
        const r=result.results[i];assert.equal(r.validation.passed,false);assert.match(r.backend,/CPU/);assert.deepEqual(r.xyz,cpu.results[i].xyz);assert.deepEqual(r.image,cpu.results[i].image);
      }
      report.negativeControls.push({name:'progressive isolated X shader mutation, complete CPU XYZ fallback',nativeCalls:calls,validation:result.results.map(r=>r.validation)});
    });
    await test('ordinary-scale real visible scenes preserve full XYZ after phase-only host reduction',async()=>{
      const {createPLTScene,createPLTRandomStates,buildPLTBatch}=await import('../dist/plt-core.js');
      for(const preset of ['gallery','prism','baffles']){
        const options={width:12,height:10,samplesPerPixel:4,maxDepth:4,spectralMode:'visible',spectralSource:'D65',spectralMaterial:'neutral',wave:true,seed:7391},scene=createPLTScene({preset,...options});
        const job=await buildPLTBatch(scene,scene.camera,options,{tile:{x:0,y:0,width:12,height:10},samplesPerPixel:4,rngStates:createPLTRandomStates(options),yieldEvery:1024});
        const phases=job.events.flatMap(o=>o.edges.map(e=>Math.abs((e.k??o.k)*e.phaseExcess))),maxPhase=Math.max(...phases);assert(maxPhase>PLT_GPU_LIMITS.maxPhase);
        const result=await renderPLTGPU(job,{gpu,shaderSource:source});assert(result.phaseReduction.maxAbsoluteRawPhase>PLT_GPU_LIMITS.maxPhase);
        assert(result.phaseReduction.maxAbsolutePackedPhase<=Math.PI+1e-6);retain(`real visible ${preset} f64 phase-reduced`,job,result);
      }
    });
  }finally{gpu=null;}
}
report.passed=failures.length===0;report.completedAt=new Date().toISOString();
if(process.env.PLT_SPECTRAL_GPU_REPORT)fs.writeFileSync(process.env.PLT_SPECTRAL_GPU_REPORT,JSON.stringify(report,(_,value)=>ArrayBuffer.isView(value)?Array.from(value):value,2)+'\n');
console.log(`${tests.length} passed, ${failures.length} failed`);if(failures.length)process.exitCode=1;
