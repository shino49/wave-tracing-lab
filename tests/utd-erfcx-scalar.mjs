/** Equivalent scalar-unrolled erfcx regression.
 * Always runs native Dawn/Tint compilation, coefficient identity, and independent
 * forward-Chebyshev checks. SciPy is used if already installed; never installed.
 * Execute all WGSL probes and exhaustive production parity gates with:
 * VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json \
 * WEBGPU_EXECUTION_BACKEND=vulkan node tests/utd-erfcx-scalar.mjs --execute
 * Set ERFCX_SCALAR_REPORT to save a report. This constructs public default scenes;
 * it never opens a user diagnostic report, uploaded geometry, or browser state.
 * SwiftShader is executable WGSL evidence, not physical NVIDIA verification.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {create,globals} from 'webgpu';
import {ERFCX_CHEBYSHEV,erfcxScaled} from '../dist/utd-math.js';
import {createPLTScene,createPLTRandomStates,buildPLTBatch,validatePLTOptions} from '../dist/plt-core.js';
import {renderPLTGPU,PLT_GPU_TOLERANCE} from '../dist/plt-gpu.js';
Object.assign(globalThis,globals);
const read=name=>fs.readFileSync(new URL(`../dist/${name}`,import.meta.url),'utf8');
const hash=value=>createHash('sha256').update(value).digest('hex');
const functions=read('utd-functions.wgsl'),compute=read('plt-compute.wgsl'),execute=process.argv.includes('--execute');
// Fixed before observing execution results. The existing f32 transition suite
// uses an absolute 4e-7 error ceiling; production event/pixel gates are unchanged.
const NUMERIC_TOLERANCE=Object.freeze({f64Absolute:2e-11,f32Absolute:4e-7});
const EXPECTED_PARITY=Object.freeze({eventAbsolute:2e-5,eventRelative:.003,pixelAbsolute:2e-5,pixelRelative:.003,boundaryEpsilon:2e-6});
const tests=[],failures=[],report={suite:'utd-erfcx-scalar',schemaVersion:1,executionRequested:execute,
  nativeCompilePassed:false,browserValidated:false,physicalGPUValidated:false,physicalNVIDIAValidated:false,
  scope:'Equivalent degree-40 polynomial, independent mathematical oracle, and exhaustive first-tile event/camera parity for reconstructed default scenes; not independent physical accuracy',
  inputPrivacy:'Only source code, generated public default scenes, and numeric test inputs; no user reports, uploads, account data, or browser state',
  tolerance:NUMERIC_TOLERANCE,productionTolerance:PLT_GPU_TOLERANCE,
  sourceHashes:Object.fromEntries(['utd-functions.wgsl','utd-math.js','plt-compute.wgsl','plt-gpu.js','plt-core.js','plt-spectrum.js','plt-fsd.js','plt-geometry.js','mesh-geometry.js'].map(name=>[name,hash(read(name))])),
  testSHA256:hash(fs.readFileSync(new URL('./utd-erfcx-scalar.mjs',import.meta.url))),tests,failures,nativeCases:[],negativeControls:[]};
const test=async(name,body)=>{try{await body();tests.push(name);console.log(`PASS ${name}`);}catch(error){failures.push({name,error:error.stack,validation:error.validation?compactGate(error.validation):undefined});console.error(`FAIL ${name}: ${error.stack}`);}};
function functionBody(source,name){
  const start=source.indexOf(`fn ${name}(`);assert(start>=0,`${name} must exist`);
  const open=source.indexOf('{',start);let depth=1,end=open+1;
  for(;depth&&end<source.length;end++){if(source[end]==='{')depth++;else if(source[end]==='}')depth--;}
  assert.equal(depth,0,'balanced function body');return {start,end,text:source.slice(open+1,end-1)};
}
const strip=text=>text.replace(/\/\*[\s\S]*?\*\//g,'').replace(/\/\/[^\n]*/g,'').replace(/\s+/g,'');
const body=functionBody(functions,'utd_erfcx_scaled');
// Validate the algorithm and numeric literals, not a historical whole-file hash.
// The source recognizer permits whitespace, comments and alternate decimal
// spellings while requiring two independent scalar streams and all 40 steps.
function scalarCoefficients(sourceBody){
  let text=strip(sourceBody);
  const prefix='lets=2.0*t-1.0;letx=2.0*s;varbr=0.0;varbi=0.0;varcr=0.0;varci=0.0;';
  assert(text.startsWith(prefix),'separate f32 scalar streams and unchanged t-to-s mapping');text=text.slice(prefix.length);
  const values=Array(41),number='([+-]?(?:\\d*\\.\\d+|\\d+\\.?\\d*)(?:e[+-]?\\d+)?)';
  const step=new RegExp('^\\{letnr=x\\*br-cr\\+\\('+number+'\\);letni=x\\*bi-ci\\+\\('+number+'\\);cr=br;ci=bi;br=nr;bi=ni;\\}','i');
  for(let j=40;j>=1;j--){const match=text.match(step);assert(match,`literal scalar Clenshaw coefficient ${j}`);values[j]=[Number(match[1]),Number(match[2])];text=text.slice(match[0].length);}
  const end=new RegExp('^returnvec2f\\(s\\*br-cr\\+\\('+number+'\\),s\\*bi-ci\\+\\('+number+'\\)\\);$','i');
  const match=text.match(end);assert(match,'only the final scalar values are assembled into vec2f');values[0]=[Number(match[1]),Number(match[2])];return values;
}
let literals;
await test('same 41 exact-f32 coefficient pairs and scalar-only unrolled degree-40 algorithm',()=>{
  assert.equal(ERFCX_CHEBYSHEV.length,41);literals=scalarCoefficients(body.text);
  const table=functions.match(/const\s+UTD_ERFCX_CHEBYSHEV\s*=\s*array<vec2f,\s*41>\(([\s\S]*?)\);/);assert(table,'preserved 41-pair generated coefficient table');
  const pairs=[...table[1].matchAll(/vec2f\(([^,]+),([^\)]+)\)/g)].map(m=>[Number(m[1]),Number(m[2])]);assert.equal(pairs.length,41);
  for(let j=0;j<41;j++)for(let c=0;c<2;c++){
    assert.equal(Math.fround(literals[j][c]),Math.fround(ERFCX_CHEBYSHEV[j][c]),`literal ${j}/${c}`);
    assert.equal(Math.fround(pairs[j][c]),Math.fround(ERFCX_CHEBYSHEV[j][c]),`table ${j}/${c}`);
  }
  assert.deepEqual(PLT_GPU_TOLERANCE,EXPECTED_PARITY);
  report.algorithm={degree:40,coefficientPairs:41,scalarStreams:2,unrolledSteps:40,dynamicCoefficientAccess:false,vectorRecurrence:false,
    exactF32CoefficientSHA256:hash(Buffer.from(Float32Array.from(literals.flat()).buffer)),coefficientIdentityPassed:true};
});
// Forward basis generation and explicit summation are independent of Clenshaw.
function forwardChebyshev(t,coefficients){
  const s=2*t-1;let prev=1,current=s,re=coefficients[0][0],im=coefficients[0][1];
  for(let j=1;j<coefficients.length;j++){re+=coefficients[j][0]*current;im+=coefficients[j][1]*current;const next=2*s*current-prev;prev=current;current=next;}
  return [re,im];
}
const bits=new DataView(new ArrayBuffer(4));
function nextF32(value,direction){value=Math.fround(value);if(value===0)return direction>0?2**-149:-(2**-149);bits.setFloat32(0,value,true);let u=bits.getUint32(0,true);u+=(value>0)===(direction>0)?1:-1;bits.setUint32(0,u,true);return bits.getFloat32(0,true);}
const targeted=[0,nextF32(0,1),2**-126,1e-30,1e-20,1e-12,1e-8,1e-6,.0001,.001,.01,.1,.5,.9860957183805704,.99,.999,.9999,1-1e-6,1-1e-8,nextF32(1,-1),1];
for(const t of [.5,.9860957183805704])targeted.push(nextF32(t,-1),nextF32(t,1));
const requested=[...Array.from({length:10001},(_,i)=>i/10000),...targeted],inputs=Float32Array.from(requested);
const coefficientsF32=ERFCX_CHEBYSHEV.map(pair=>pair.map(Math.fround));
const cpuReference=Array.from(inputs,t=>forwardChebyshev(t,coefficientsF32));
report.coverage={denseSamples:10001,denseRange:[0,1],totalSamples:inputs.length,includesEndpoints:true,includesF32Nextafter:true,
  targeted:targeted.map(t=>({requested:t,packedF32:Math.fround(t)})),comparison:'Every real and imaginary component; explicit absolute-error gates'};
function numericGate(actual,expected,limit){
  assert.equal(actual.length,expected.length*2);let maxAbsoluteError=0,worst=null,numericalFailures=0;
  const componentFailures=[0,0];
  for(let i=0;i<expected.length;i++)for(let c=0;c<2;c++){
    const got=actual[2*i+c],error=Math.abs(got-expected[i][c]);
    if(!Number.isFinite(got)||!Number.isFinite(error)||error>limit){numericalFailures++;componentFailures[c]++;}
    if(error>maxAbsoluteError){maxAbsoluteError=error;worst={index:i,requested:requested[i],packedF32:inputs[i],component:c?'imaginary':'real',actual:got,expected:expected[i][c],absoluteError:error};}
  }
  return {passed:numericalFailures===0,count:expected.length,componentCount:actual.length,absoluteTolerance:limit,numericalFailures,componentFailures,maxAbsoluteError,worst};
}
await test('independent forward Chebyshev agrees with CPU f64 Clenshaw at every dense and targeted input',()=>{
  const expected=requested.map(t=>forwardChebyshev(t,ERFCX_CHEBYSHEV)),actual=requested.flatMap(t=>erfcxScaled(t));
  report.cpuForwardGate=numericGate(actual,expected,NUMERIC_TOLERANCE.f64Absolute);assert(report.cpuForwardGate.passed);
});
let scipyRequested=null,scipyPacked=null;
await test('independent SciPy complex erfcx oracle when already installed',()=>{
  const program=`import json,sys,math\ntry:\n import scipy\n from scipy.special import erfcx\nexcept ImportError:\n sys.exit(77)\nvalues=json.load(sys.stdin)\ndef scaled(t):\n if t == 1: return [1/math.sqrt(2*math.pi),-1/math.sqrt(2*math.pi)]\n u=t/(1-t)\n z=erfcx(complex(u/math.sqrt(2),u/math.sqrt(2)))/(1-t)\n return [float(z.real),float(z.imag)]\njson.dump({'version':scipy.__version__,'values':[scaled(t) for t in values]},sys.stdout,allow_nan=False)\n`;
  const result=spawnSync(process.env.PYTHON??'python',['-c',program],{input:JSON.stringify([...requested,...inputs]),encoding:'utf8',maxBuffer:16*1024*1024});
  if(result.status===77||result.error?.code==='ENOENT'){
    report.scipy={available:false,skipped:true,reason:result.status===77?'SciPy is not installed':'Python executable is not available'};
    assert(!process.argv.includes('--require-scipy'),'--require-scipy prohibits an oracle skip');return;
  }
  assert.equal(result.status,0,result.stderr||String(result.error));const oracle=JSON.parse(result.stdout);
  scipyRequested=oracle.values.slice(0,requested.length);scipyPacked=oracle.values.slice(requested.length);
  report.scipy={available:true,version:oracle.version,formula:'erfcx(exp(i*pi/4)*t/(1-t))/(1-t); analytic endpoint exp(-i*pi/4)/sqrt(pi) at t=1',
    f64Polynomial:numericGate(requested.flatMap(t=>forwardChebyshev(t,ERFCX_CHEBYSHEV)),scipyRequested,NUMERIC_TOLERANCE.f64Absolute),
    f32Coefficients:numericGate(cpuReference.flat(),scipyPacked,NUMERIC_TOLERANCE.f32Absolute)};
  assert(report.scipy.f64Polynomial.passed);assert(report.scipy.f32Coefficients.passed);
});
const probe=`
@group(0) @binding(0) var<storage,read> erfcxInput:array<f32>;
@group(0) @binding(1) var<storage,read_write> erfcxOutput:array<vec2f>;
@compute @workgroup_size(64) fn erfcxProbe(@builtin(global_invocation_id) id:vec3u) {
 // Logical count excludes allocation padding in the storage buffers.
 if(id.x>=${inputs.length}u){return;}
 erfcxOutput[id.x]=utd_erfcx_scaled(erfcxInput[id.x]);
}`;
async function pipeline(device,code,entryPoint){
  device.pushErrorScope('validation');
  try{const module=device.createShaderModule({code});assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==='error'),[]);
    return await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint}});
  }finally{assert.equal(await device.popErrorScope(),null);}
}
await test('mandatory native Dawn/Tint compiles production scalar probe plus complete event/camera shaders',async()=>{
  let gpu=create(['backend=null']);const adapter=await gpu.requestAdapter();assert(adapter);const device=await adapter.requestDevice();
  try{await pipeline(device,functions+'\n'+probe,'erfcxProbe');for(const entryPoint of ['plt_events','plt_camera'])await pipeline(device,functions+'\n'+compute,entryPoint);report.nativeCompilePassed=true;}
  finally{device.destroy();gpu=null;}
});
async function dispatchProbe(device,source){
  const p=await pipeline(device,source+'\n'+probe,'erfcxProbe'),buffers=[];
  const buffer=(size,usage,data)=>{const b=device.createBuffer({size:Math.max(16,Math.ceil(size/16)*16),usage});buffers.push(b);if(data)device.queue.writeBuffer(b,0,data);return b;};
  device.pushErrorScope('validation');
  try{
    const input=buffer(inputs.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,inputs),bytes=inputs.length*8;
    const out=buffer(bytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC),read=buffer(bytes,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
    const group=device.createBindGroup({layout:p.getBindGroupLayout(0),entries:[input,out].map((b,binding)=>({binding,resource:{buffer:b}}))});
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(p);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(inputs.length/64));pass.end();encoder.copyBufferToBuffer(out,0,read,0,bytes);device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);const result=new Float32Array(read.getMappedRange(0,bytes).slice(0));read.unmap();return result;
  }finally{for(const b of buffers)b.destroy();assert.equal(await device.popErrorScope(),null);}
}
function compactGate(gate){const {absoluteErrors,toleranceRatios,...summary}=gate;return summary;}
function mutateImaginaryCoefficient(){
  const original=body.text,pattern=/(return\s+vec2f\(s\*br-cr\+\([^)]+\),s\*bi-ci\+\()([^)]+)(\)\);)/;
  let replacements=0;const mutated=original.replace(pattern,(_,before,value,after)=>{replacements++;return before+String(Number(value)+.03125)+after;});
  assert.equal(replacements,1,'negative control changes only imaginary c0');
  return functions.slice(0,body.start)+functions.slice(body.start,body.end).replace(original,mutated)+functions.slice(body.end);
}
const fixtureSpecs=[
  {name:'monochrome gallery first 32x32 tile',preset:'gallery',options:{width:48,height:36,samplesPerPixel:16,maxDepth:4,seed:7391,spectralMode:'monochrome'}},
  {name:'visible micro-gallery D65 red first 32x32 tile',preset:'micro-gallery',options:{width:48,height:36,samplesPerPixel:64,maxDepth:4,seed:7391,spectralMode:'visible',spectralSource:'D65',spectralMaterial:'red'}},
];
if(execute){
  const backend=process.env.WEBGPU_EXECUTION_BACKEND;
  await test('explicit executable native backend is required',()=>assert(['vulkan','metal','d3d11','d3d12','opengl','opengles'].includes(backend),'Set WEBGPU_EXECUTION_BACKEND; null is never execution evidence'));
  if(['vulkan','metal','d3d11','d3d12','opengl','opengles'].includes(backend)){
    let gpu=create([`backend=${backend}`]),device;
    try{
      const adapter=await gpu.requestAdapter();assert(adapter,'Executable adapter required');device=await adapter.requestDevice();
      report.adapter=Object.fromEntries(['vendor','architecture','device','description','isFallbackAdapter'].map(key=>[key,adapter.info[key]]));
      report.requestedBackend=backend;report.driverSelection={VK_ICD_FILENAMES:process.env.VK_ICD_FILENAMES??null};
      report.execution='Actual native Dawn/Tint WGSL dispatch and mapped buffer readback';
      await test('executed WGSL all dense/targeted real and imaginary components agree with forward f64 sum and SciPy',async()=>{
        const actual=await dispatchProbe(device,functions);report.nativeForwardGate=numericGate(actual,cpuReference,NUMERIC_TOLERANCE.f32Absolute);assert(report.nativeForwardGate.passed);
        if(scipyPacked){report.nativeScipyGate=numericGate(actual,scipyPacked,NUMERIC_TOLERANCE.f32Absolute);assert(report.nativeScipyGate.passed);}
        report.targetedResults=targeted.map((t,j)=>({requested:t,packedF32:inputs[10001+j],actual:[actual[2*(10001+j)],actual[2*(10001+j)+1]],forwardReference:cpuReference[10001+j],scipyReference:scipyPacked?.[10001+j]}));
      });
      const mutated=mutateImaginaryCoefficient();
      await test('wrong imaginary coefficient is rejected at exactly the same numerical tolerances',async()=>{
        const actual=await dispatchProbe(device,mutated),gate=numericGate(actual,cpuReference,NUMERIC_TOLERANCE.f32Absolute);
        assert(!gate.passed);assert.equal(gate.componentFailures[0],0,'real stream must remain correct');assert.equal(gate.componentFailures[1],inputs.length,'every imaginary value must expose the wrong constant');
        const oracleGate=scipyPacked?numericGate(actual,scipyPacked,NUMERIC_TOLERANCE.f32Absolute):null;if(oracleGate)assert(!oracleGate.passed);
        report.negativeControls.push({name:'imaginary c0 + 0.03125',detected:true,toleranceUnchanged:true,gate,scipyGate:oracleGate});
      });
      for(const spec of fixtureSpecs)await test(`full production event and pixel gates: ${spec.name}`,async()=>{
        const options=validatePLTOptions(spec.options,{progressive:true}),scene=createPLTScene({preset:spec.preset,...options}),tile={x:0,y:0,width:32,height:32};
        const job=await buildPLTBatch(scene,scene.camera,options,{tile,sampleStart:0,samplesPerPixel:1,rngStates:createPLTRandomStates(options),yieldEvery:1024});
        assert(job.events.length>0);assert(job.terms.length>0);
        const result=await renderPLTGPU(job,{gpu,shaderSource:functions+'\n'+compute});
        assert(result.validation.passed);assert.equal(result.validation.events.count,job.events.length);assert.equal(result.validation.pixels.count,1024*(options.spectralMode==='visible'?3:1));
        assert.equal(result.validation.coverage,options.spectralMode==='visible'?'every-event-and-every-color-component':'every-event-and-every-pixel');
        assert.deepEqual(result.validation.events.tolerance,{absolute:EXPECTED_PARITY.eventAbsolute,relative:EXPECTED_PARITY.eventRelative});
        assert.deepEqual(result.validation.pixels.tolerance,{absolute:EXPECTED_PARITY.pixelAbsolute,relative:EXPECTED_PARITY.pixelRelative});
        const summary={name:spec.name,preset:spec.preset,frameOptions:options,tile,sampleStart:0,testedBatchSamplesPerPixel:1,
          scope:'First progressive tile and first sample batch only; frame spp describes requested configuration, not complete-frame execution',
          events:job.events.length,terms:job.terms.length,edgeRecords:job.edgeRecords,pathStats:job.stats,
          validation:{...result.validation,events:compactGate(result.validation.events),pixels:compactGate(result.validation.pixels)},phaseReduction:result.phaseReduction,timings:result.timings};
        report.nativeCases.push(summary);
        await assert.rejects(renderPLTGPU(job,{gpu,shaderSource:mutated+'\n'+compute}),error=>{
          assert.equal(error.name,'PLTGPUParityError');assert.equal(error.validation.coverage,'every-event');assert(error.validation.numericalFailures>0);
          report.negativeControls.push({name:`imaginary coefficient mutation rejected by ${spec.name}`,detected:true,toleranceUnchanged:true,gate:compactGate(error.validation)});return true;
        });
        // Admission must still reject a camera-only corruption after correct
        // events, in both mono and XYZ modes, without relaxing either gate.
        const cameraFault=compute.replace('camera[pixel] = sum / f32(plt.counts.z);','camera[pixel] = sum / f32(plt.counts.z); if(pixel==0u){camera[pixel] += 1.0;}');assert.notEqual(cameraFault,compute);
        await assert.rejects(renderPLTGPU(job,{gpu,shaderSource:functions+'\n'+cameraFault}),error=>{
          assert.equal(error.name,'PLTGPUParityError');assert.equal(error.validation.coverage,options.spectralMode==='visible'?'every-color-component':'every-pixel');assert(error.validation.failures.some(f=>f.index===0));
          report.negativeControls.push({name:`camera-only mutation rejected by ${spec.name}`,detected:true,toleranceUnchanged:true,gate:compactGate(error.validation)});return true;
        });
      });
    }catch(error){failures.push({name:'executable backend setup',error:error.stack});console.error(error.stack);}
    finally{device?.destroy();gpu=null;}
  }
}
report.passed=failures.length===0;report.completedAt=new Date().toISOString();
if(process.env.ERFCX_SCALAR_REPORT)fs.writeFileSync(process.env.ERFCX_SCALAR_REPORT,JSON.stringify(report,null,2)+'\n');
console.log(`${tests.length} passed, ${failures.length} failed; native execution ${execute?'requested':'not requested'}; physical NVIDIA not verified`);
if(failures.length)process.exitCode=1;
