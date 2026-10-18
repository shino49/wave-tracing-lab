/** Actual CPU task orchestration and injected GPU-contract QA.
 * No real browser, graphics driver or physical GPU is exercised here. */
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {executeCoherentTask,validateRequest,DISPLAY_GRID} from '../dist/coherent-runtime.js';
import {DEFAULT_OPTIONS} from '../dist/coherent-core.js';

const tests=[],failures=[],originalNavigator=Object.getOwnPropertyDescriptor(globalThis,'navigator');
Object.defineProperty(globalThis,'navigator',{value:{},configurable:true});
const test=async(name,fn)=>{try{await fn();tests.push(name);}catch(error){failures.push({name,error:error.stack});}};
const cpu=options=>executeCoherentTask({backend:'cpu',options});
const finiteTimings=result=>{for(const key of ['solveMs','cpuFieldMs','totalMs'])assert(Number.isFinite(result.timings[key])&&result.timings[key]>=0,key);assert(result.timings.totalMs>=result.timings.solveMs+result.timings.cpuFieldMs);};
const sameGrid=(actual,expected)=>{for(const key of ['fieldsRaw','intensity','mask','coherentField'])assert.deepEqual(actual[key],expected[key],key);};
let reference;

await test('Actual CPU default run produces bounded finite fields, scalar flux, timings and progress',async()=>{
 const progress=[];reference=await executeCoherentTask({backend:'cpu',options:{}},{progress:text=>progress.push(text),gpuRunner:()=>{throw new Error('CPU selection must not invoke GPU');}});
 assert.equal(reference.action,'run');assert.equal(reference.backend,'CPU 双精度');assert.equal(reference.parity,null);assert.equal(reference.timings.gpuMs,null);assert.deepEqual(reference.warnings,[]);finiteTimings(reference);
 assert.equal(reference.grid.width,DISPLAY_GRID.width);assert.equal(reference.grid.height,DISPLAY_GRID.height);assert.equal(reference.grid.mask.length,144*108);assert(reference.grid.fieldsRaw instanceof Float64Array);
 assert(reference.grid.diagnostics.validPixels>0);assert(reference.grid.diagnostics.maskedPixels>0);assert(reference.grid.intensity.every(v=>Number.isFinite(v)&&v>=0));
 assert(reference.measurement.detectorFlux>=0);assert(reference.measurement.diagnostics.energyPass);assert(reference.diagnostics.relativeResidual<1e-10);assert.equal(reference.diagnostics.reference.pass,true);
 assert.equal(reference.diagnostics.executionContext,'non-browser test execution');assert.equal(reference.diagnostics.serverCompute,false);assert.match(reference.diagnostics.scope,/scalar.*no electromagnetic watts/);assert(progress.length>=2);assert(progress.every(t=>typeof t==='string'));
 for(let i=0;i<reference.grid.mask.length;i++)if(!reference.grid.mask[i]){assert.equal(reference.grid.intensity[i],0);assert(reference.grid.fieldsRaw.slice(i*6,i*6+6).every(v=>v===0));}
});
await test('Missing Worker WebGPU uses an explicit CPU fallback with identical fields',async()=>{
 const r=await executeCoherentTask({options:{}});assert.equal(r.backend,'CPU 双精度');assert.equal(r.parity,null);assert.equal(r.timings.gpuMs,null);assert.match(r.warnings.join(' '),/未提供 WebGPU.*CPU/);sameGrid(r.grid,reference.grid);finiteTimings(r);
});
await test('Request validator accepts documented endpoints and rejects all invalid state values',async()=>{
 const ranges={gamma:[0,1],phase:[-Math.PI,Math.PI],inputPower:[.25,2],detectorCenter:[-Math.PI/4,Math.PI/4],detectorHalfAngle:[.1,.6]};
 for(const [key,[lo,hi]]of Object.entries(ranges)){
  for(const value of [lo,hi])assert.equal(validateRequest({options:{[key]:value}})[key],value);
  for(const value of [lo-1e-6,hi+1e-6,NaN,Infinity,-Infinity,null,'1'])assert.throws(()=>validateRequest({options:{[key]:value}}),new RegExp(key));
 }
 for(const backend of ['gpu','server','',0])if(backend)assert.throws(()=>validateRequest({backend}),/backend/);
 assert.throws(()=>validateRequest({action:'upload'}),/action/);assert.deepEqual(validateRequest({}),DEFAULT_OPTIONS);
});
await test('Actual task rejects geometry, resolution and quadrature limits before invoking GPU',async()=>{
 const bad={screenCount:[0,3,1.5,NaN],separation:[1.799,3.001,NaN],offset:[-.201,1.001,Infinity],n:[32,65,145,NaN],nq:[159,641,160.5],nang:[79,321,80.5]};
 for(const [key,values]of Object.entries(bad))for(const value of values)await assert.rejects(executeCoherentTask({options:{[key]:value}},{gpuRunner:()=>assert.fail('Invalid request reached GPU')}),new RegExp(key));
});
await test('GPU success receives actual CPU references and reports only passing hybrid output with engine timing names',async()=>{
 let calls=0;const progress=[];const r=await executeCoherentTask({options:{}},{progress:text=>progress.push(text),gpuRunner:async(solution,grid,options)=>{
  calls++;assert.equal(grid,DISPLAY_GRID);assert.equal(options.state,solution.state);sameGrid(options.reference,reference.grid);assert.equal(typeof options.onProgress,'function');
  options.onProgress(.5);return {...options.reference,parity:{passed:true,points:144*108,coverage:'every-pixel-every-complex-mode-and-intensity'},diagnostics:{...options.reference.diagnostics,gpuParityPassed:true},adapter:{vendor:'mock'},buildId:'mock-build',shaderFingerprintFNV1a:'mock-fingerprint',timings:{cpuReferenceProvided:true,gpuSetupWallMs:12,gpuDispatchReadbackWallMs:23.5,parityGateMs:4}};
 }});
 assert.equal(calls,1);assert.equal(r.backend,'CPU + WebGPU');assert.equal(r.parity.passed,true);assert.equal(r.timings.gpuMs,23.5);assert.equal(r.timings.gpuDetails.gpuSetupWallMs,12);assert.deepEqual(r.warnings,[]);sameGrid(r.grid,reference.grid);finiteTimings(r);assert(progress.some(s=>s.includes('50%')));assert(progress.every(s=>typeof s==='string'));assert.equal(r.diagnostics.gpu.adapter.vendor,'mock');assert.equal(r.diagnostics.gpu.buildId,'mock-build');assert.equal(r.diagnostics.gpu.shaderFingerprintFNV1a,'mock-fingerprint');
});
await test('GPU throw preserves useful rejection diagnostics and publishes only CPU fields',async()=>{
 const rejected={passed:false,points:144*108,failures:[{index:17,actual:[99],expected:[0]}]};
 const r=await executeCoherentTask({options:{}},{gpuRunner:async()=>{const e=new Error('mock device/parity rejection');e.diagnostic={stage:'coherent-gpu-field',parity:rejected};throw e;}});
 assert.equal(r.backend,'CPU 双精度');assert.equal(r.parity.passed,false);assert.deepEqual(r.parity.failures,rejected.failures);assert.match(r.warnings.join(' '),/mock device\/parity rejection/);sameGrid(r.grid,reference.grid);assert.equal(r.timings.gpuMs,null);
});
await test('GPU returned failing gate cannot expose corrupted GPU fields or claim admission',async()=>{
 const r=await executeCoherentTask({options:{}},{gpuRunner:async(_s,_g,{reference:r})=>({...r,intensity:new Float32Array(r.intensity.length).fill(99),parity:{passed:false,failures:[{index:5}]},timings:{gpuDispatchReadbackWallMs:7}})});
 assert.equal(r.backend,'CPU 双精度');assert.equal(r.parity.passed,false);assert.match(r.warnings.join(' '),/GPU.*未提供通过/);sameGrid(r.grid,reference.grid);
});
await test('GPU result missing gate fails closed without publishing an unverified field',async()=>{
 const r=await executeCoherentTask({options:{}},{gpuRunner:async(_s,_g,{reference:r})=>({...r,parity:null})});assert.equal(r.backend,'CPU 双精度');assert.equal(r.parity.passed,false);sameGrid(r.grid,reference.grid);
});
await test('GPU failure after a passed intermediate gate cannot become a passing result badge',async()=>{
 const r=await executeCoherentTask({options:{}},{gpuRunner:async()=>{const e=new Error('mock device lost after parity');e.diagnostic={parity:{passed:true,points:144*108}};throw e;}});
 assert.equal(r.backend,'CPU 双精度');assert.equal(r.parity.passed,false);sameGrid(r.grid,reference.grid);
});
await test('Explicit CPU bypasses an injected GPU path even when navigator.gpu exists',async()=>{
 navigator.gpu={};const r=await executeCoherentTask({backend:'cpu',options:{}},{gpuRunner:async()=>assert.fail('GPU must be bypassed')});delete navigator.gpu;assert.equal(r.backend,'CPU 双精度');assert.equal(r.parity,null);sameGrid(r.grid,reference.grid);
});
await test('Single-screen task uses one screen and a changed geometry excludes the fixed reference',async()=>{
 const one=await cpu({screenCount:1});assert.equal(one.screens.length,1);assert.equal(one.options.screenCount,1);assert.equal(one.diagnostics.reference.applicable,true);
 const changed=await cpu({separation:2.4,offset:.5});assert.equal(changed.screens.length,2);assert.equal(changed.diagnostics.reference.applicable,false);assert.equal(changed.diagnostics.reference.pass,null);assert(changed.measurement.diagnostics.energyPass);
});
await test('Noncoherent task supplies its own matching baseline independent of source phase',async()=>{
 const r=await cpu({gamma:0,phase:Math.PI}),zero=await cpu({gamma:0,phase:0});assert.equal(r.measurement.detectorFlux,r.baseline.detectorFlux);assert.equal(r.measurement.detectorFlux,zero.measurement.detectorFlux);assert.deepEqual(r.grid.intensity,zero.grid.intensity);
});
await test('64-to-96 actual refinement compares accepted pixels and reports only convergence evidence',async()=>{
 const progress=[];const r=await executeCoherentTask({action:'refine',backend:'auto',options:{}},{progress:s=>progress.push(s),gpuRunner:()=>assert.fail('Refinement must not run GPU')});
 assert.equal(r.action,'refine');assert.equal(r.refinement.coarseN,64);assert.equal(r.refinement.fineN,96);assert.equal(r.refinement.threshold,2e-4);assert.equal(r.refinement.validPixels,reference.grid.diagnostics.validPixels);
 for(const k of ['fieldError','detectorOperatorError','detectorSignalChange','elapsedMs'])assert(Number.isFinite(r.refinement[k])&&r.refinement[k]>=0,k);assert.equal(r.refinement.pass,r.refinement.fieldError<=2e-4&&r.refinement.detectorOperatorError<=2e-4);assert.equal(r.grid,undefined);assert.equal(r.backend,undefined);assert(progress.some(s=>s.includes('64')&&s.includes('96')));
});
await test('96-to-144 refinement works and 144 cannot be refined further',async()=>{
 const r=await executeCoherentTask({action:'refine',backend:'cpu',options:{n:96}});assert.equal(r.refinement.coarseN,96);assert.equal(r.refinement.fineN,144);await assert.rejects(executeCoherentTask({action:'refine',options:{n:144}}),/144/);
});

if(originalNavigator)Object.defineProperty(globalThis,'navigator',originalNavigator);else delete globalThis.navigator;
const reportPath=new URL('../research/coherent-browser-flow-validation.json',import.meta.url);
let report={};try{report=JSON.parse(fs.readFileSync(reportPath,'utf8'));}catch{}
report.generatedAt=new Date().toISOString();report.limitations=['Happy-dom and Canvas spy only; no browser visual or accessibility-tree verification.','GPU contract paths are injected mocks; these tests do not establish GPU shader execution or physical-device support.'];
report.runtime={passed:tests.length,failed:failures.length,tests,failures,actualCpuSolverExecuted:true,displayGrid:DISPLAY_GRID,defaultCpuSample:reference?{detectorFlux:reference.measurement.detectorFlux,incoherentDetectorFlux:reference.baseline.detectorFlux,totalFlux:reference.measurement.totalFlux,validPixels:reference.grid.diagnostics.validPixels,maskedPixels:reference.grid.diagnostics.maskedPixels,relativeResidual:reference.diagnostics.relativeResidual,executionContext:reference.diagnostics.executionContext}:null};fs.writeFileSync(reportPath,JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report.runtime,null,2));if(failures.length)process.exitCode=1;
