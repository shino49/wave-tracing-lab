// CPU float64 coupled BEM solve; GPU applies the complex linear field operator.
// A GPU buffer is never returned to the view before an exhaustive CPU gate.
import {evaluateGrid,evaluatePoints} from './coherent-core.js';

export const COHERENT_GPU_BUILD_ID='coherent-hybrid-2026-10-12.1';
export const COHERENT_GPU_SCOPE='Float64 CPU coupled BEM solve + WebGPU complex field application; full-array CPU verification before display';
export const COHERENT_GPU_LIMITS=Object.freeze({points:65536,sourceNodes:512,boundaryNodes:512,factors:4,
  minKr:1e-4,maxKr:1024,maxPhase:1024,maxComplexTerms:200000000,batchPoints:1024});
// Declared before execution. Backend parity is not a physical-error bound.
export const COHERENT_GPU_TOLERANCE=Object.freeze({fieldAbsolute:2e-5,fieldRelative:.003,
  intensityAbsolute:2e-5,intensityRelative:.002});
const now=()=>globalThis.performance?.now?.()??Date.now();
function fail(message){throw new RangeError(`Coherent GPU: ${message}`);}
function count(value,max,label,min=1){if(!Number.isInteger(value)||value<min||value>max)fail(`${label} must be ${min}..${max}`);}
function finite(array,length,label){
  if(!array||array.length!==length)fail(`${label} has the wrong length`);
  for(let i=0;i<array.length;i++)if(!Number.isFinite(array[i])||Math.abs(array[i])>1e12)fail(`${label}[${i}] is outside the finite bounded domain`);
}
function aborted(){const error=new Error('Coherent GPU calculation cancelled or destroyed');error.name='AbortError';return error;}
function checkSignal(signal){if(signal?.aborted)throw aborted();}
function timeout(promise,ms,label){let timer;return Promise.race([promise,new Promise((_,reject)=>{
  timer=setTimeout(()=>reject(new Error(label)),ms);
})]).finally(()=>clearTimeout(timer));}
function adapterSummary(adapter){const info={};for(const key of ['vendor','architecture','device','description','isFallbackAdapter']){
  try{const value=adapter.info?.[key];if(typeof value==='string'||typeof value==='boolean')info[key]=value;}catch{}
}return info;}
function fingerprint(source){let hash=2166136261;for(let i=0;i<source.length;i++){hash^=source.charCodeAt(i);hash=Math.imul(hash,16777619);}return (hash>>>0).toString(16).padStart(8,'0');}

function checkReference(reference){
  const n=reference?.intensity?.length;count(n,COHERENT_GPU_LIMITS.points,'target count');
  finite(reference.xy,2*n,'target coordinates');finite(reference.fieldsRaw,6*n,'CPU complex reference');
  finite(reference.intensity,n,'CPU intensity reference');finite(reference.mask,n,'CPU validity mask');
  for(let i=0;i<n;i++){
    if(reference.mask[i]!==0&&reference.mask[i]!==1)fail('CPU mask must be exactly 0 or 1');
    if(reference.intensity[i]<0)fail('CPU intensity reference must be nonnegative');
    if(!reference.mask[i]&&(reference.intensity[i]!==0||reference.fieldsRaw.slice(6*i,6*i+6).some(value=>value!==0)))
      fail('masked CPU pixels must carry canonical zero fields and intensity');
  }
  if(reference.width!==undefined||reference.height!==undefined){
    count(reference.width,COHERENT_GPU_LIMITS.points,'grid width');count(reference.height,COHERENT_GPU_LIMITS.points,'grid height');
    if(reference.width*reference.height!==n)fail('grid dimensions differ from reference length');
  }
  return n;
}
function snapshotReference(reference){checkReference(reference);return {...reference,
  xy:Float64Array.from(reference.xy),fieldsRaw:Float64Array.from(reference.fieldsRaw),
  intensity:Float64Array.from(reference.intensity),mask:Uint8Array.from(reference.mask)};}

/** Snapshot all GPU inputs. Incident and density values retain three complex
 * raw-mode columns; factors supply the coherent interference and mixture. */
export function packCoherentGPU(solution,reference,{state=solution.state}={}){
  const n=checkReference(reference),sources=solution?.sources,boundary=solution?.boundary;
  if(!Number.isFinite(solution?.k)||solution.k<=0||solution.k>1024)fail('wavenumber is outside (0,1024]');
  const nq=sources?.q?.length,nb=boundary?.xy?.length/2,nf=state?.factorRaw?.length/6;
  count(nq,COHERENT_GPU_LIMITS.sourceNodes,'source nodes');count(nb,COHERENT_GPU_LIMITS.boundaryNodes,'boundary nodes');
  count(nf,COHERENT_GPU_LIMITS.factors,'coherent factors');
  if(n*(nq+nb)*3>COHERENT_GPU_LIMITS.maxComplexTerms)fail('field application exceeds the bounded work budget');
  finite(sources.q,nq,'source q');finite(sources.kx,nq,'source kx');
  finite(sources.weightedSpectrumRaw,nq*6,'weighted source spectrum');
  finite(boundary.xy,nb*2,'boundary coordinates');finite(boundary.weightedPhiRaw,nb*6,'weighted boundary density');
  finite(state.factorRaw,nf*6,'coherent source factors');
  const waves=new Float32Array(nq*8),density=new Float32Array(nb*8),targets=new Float32Array(n*4);
  const factors=Float32Array.from(state.factorRaw);
  for(let j=0;j<nq;j++){
    waves[8*j]=sources.kx[j];waves[8*j+1]=sources.q[j];
    waves.set(sources.weightedSpectrumRaw.slice(6*j,6*j+6),8*j+2);
  }
  for(let j=0;j<nb;j++){
    density[8*j]=boundary.xy[2*j];density[8*j+1]=boundary.xy[2*j+1];
    density.set(boundary.weightedPhiRaw.slice(6*j,6*j+6),8*j+2);
  }
  for(let i=0;i<n;i++){
    const x=reference.xy[2*i],y=reference.xy[2*i+1];targets.set([x,y,reference.mask[i],0],4*i);
    if(!reference.mask[i])continue;
    // Check the ACTUAL packed f32 geometry as well as the original coordinates.
    const fx=targets[4*i],fy=targets[4*i+1];
    for(let j=0;j<nq;j++){
      const phase=Math.abs(waves[8*j]*fx+waves[8*j+1]*fy);
      if(phase>COHERENT_GPU_LIMITS.maxPhase)fail('target phase exceeds the verified GPU trigonometric domain');
    }
    for(let j=0;j<nb;j++){
      const kr=Math.fround(solution.k)*Math.hypot(fx-density[8*j],fy-density[8*j+1]);
      if(kr<COHERENT_GPU_LIMITS.minKr||kr>COHERENT_GPU_LIMITS.maxKr)
        fail('target distance is outside the bounded Bessel domain');
    }
  }
  const createUniforms=(first=0,batch=Math.min(n,COHERENT_GPU_LIMITS.batchPoints))=>{
    const data=new ArrayBuffer(48),u=new Uint32Array(data),f=new Float32Array(data);
    u.set([n,nq,nb,nf]);f[4]=solution.k;u.set([first,batch,0,0],8);return data;
  };
  return {waves,density,factors,targets,count:n,sourceNodes:nq,boundaryNodes:nb,factorCount:nf,createUniforms};
}

/** Compares every raw complex mode AND mixture intensity of every valid pixel.
 * All invalid records must be exactly zero. Throws rather than returning any
 * failed GPU data to the view. This establishes backend parity only. */
export function coherentParityGate(reference,data){
  const n=checkReference(reference);
  if(!(data instanceof Float32Array)||data.length!==n*8)fail('GPU readback must be an exact stride-8 Float32Array');
  const gate={passed:true,coverage:'every-pixel-every-complex-mode-and-intensity',points:n,valid:0,invalid:0,
    tolerance:{...COHERENT_GPU_TOLERANCE},nonfinite:0,negativeIntensity:0,maskMismatches:0,invalidNonzero:0,
    fieldFailures:0,intensityFailures:0,maxFieldToleranceRatio:0,maxIntensityToleranceRatio:0,
    maxAbsoluteFieldError:0,maxAbsoluteIntensityError:0,failures:[]};
  const t=COHERENT_GPU_TOLERANCE;
  for(let i=0;i<n;i++){
    const actual=data.subarray(8*i,8*i+8),valid=reference.mask[i]===1;let bad=false;
    if(!actual.every(Number.isFinite)){gate.nonfinite++;bad=true;}
    if(actual[6]<0){gate.negativeIntensity++;bad=true;}
    if(actual[7]!==reference.mask[i]){gate.maskMismatches++;bad=true;}
    if(!valid){gate.invalid++;if(actual.some(value=>value!==0)){gate.invalidNonzero++;bad=true;}}
    else{
      gate.valid++;
      for(let m=0;m<3;m++){
        const re=reference.fieldsRaw[6*i+2*m],im=reference.fieldsRaw[6*i+2*m+1];
        const error=Math.hypot(actual[2*m]-re,actual[2*m+1]-im);
        const ratio=error/(t.fieldAbsolute+t.fieldRelative*Math.hypot(re,im));
        if(!Number.isFinite(ratio)||ratio>1){gate.fieldFailures++;bad=true;}
        if(Number.isFinite(error))gate.maxAbsoluteFieldError=Math.max(gate.maxAbsoluteFieldError,error);
        if(Number.isFinite(ratio))gate.maxFieldToleranceRatio=Math.max(gate.maxFieldToleranceRatio,ratio);
      }
      const error=Math.abs(actual[6]-reference.intensity[i]),ratio=error/(t.intensityAbsolute+t.intensityRelative*reference.intensity[i]);
      if(!Number.isFinite(ratio)||ratio>1){gate.intensityFailures++;bad=true;}
      if(Number.isFinite(error))gate.maxAbsoluteIntensityError=Math.max(gate.maxAbsoluteIntensityError,error);
      if(Number.isFinite(ratio))gate.maxIntensityToleranceRatio=Math.max(gate.maxIntensityToleranceRatio,ratio);
    }
    if(bad){gate.passed=false;if(gate.failures.length<16)gate.failures.push({index:i,
      expected:{fields:[...reference.fieldsRaw.slice(6*i,6*i+6)],intensity:reference.intensity[i],mask:reference.mask[i]},actual:[...actual]});}
  }
  if(!gate.passed){const error=new Error('Coherent GPU full-grid CPU parity failed; unverified GPU field withheld');
    error.name='CoherentGPUParityError';error.parity=gate;throw error;}
  return gate;
}

/** Each engine owns one device. Failure is explicit; caller may choose a clearly
 * labelled CPU fallback. A supplied reference must correspond to this solution,
 * grid and state. CPU reference and GPU wall times are reported separately. */
export async function makeCoherentGPUEngine(solution,grid,{state=solution.state,reference:providedReference,
  gpu=globalThis.navigator?.gpu,shaderSource,onLost=()=>{},signal:setupSignal}={}){
  checkSignal(setupSignal);
  const cpuStart=now();
  const rawReference=providedReference??(grid?.xy?{...evaluatePoints(solution,grid.xy,state),xy:grid.xy,
    ...(grid.width!==undefined?{width:grid.width,height:grid.height}:{})}:evaluateGrid(solution,grid,state));
  const cpuReferenceMs=providedReference?null:now()-cpuStart;
  const setupStart=now(),reference=snapshotReference(rawReference),packed=packCoherentGPU(solution,reference,{state});
  const coherentRaw=state.coherentRaw?Float64Array.from(state.coherentRaw):null;
  if(coherentRaw)finite(coherentRaw,6,'coherent preview source');
  if(!gpu)throw new Error('WebGPU unavailable; coherent GPU field was not computed');
  const adapter=await timeout(gpu.requestAdapter({powerPreference:'low-power'}),20000,'Coherent WebGPU adapter request timed out');
  checkSignal(setupSignal);if(!adapter)throw new Error('No WebGPU adapter for coherent field application');
  const pendingDevice=adapter.requestDevice();let device;
  try{device=await timeout(pendingDevice,20000,'Coherent WebGPU device request timed out');}
  catch(error){pendingDevice.then(lateDevice=>{try{lateDevice.destroy();}catch{}},()=>{});throw error;}
  let destroyed=false,running=false,lostReason=null;const resources=[];
  const reportLost=reason=>{if(!destroyed){lostReason=reason;try{onLost(reason);}catch{}}};
  const onUncaptured=event=>reportLost(event.error?.message||'Coherent WebGPU uncaptured error');
  device.addEventListener('uncapturederror',onUncaptured);
  device.lost.then(info=>reportLost(info.message||'Coherent WebGPU device lost'));
  const destroy=()=>{if(destroyed)return;destroyed=true;for(const resource of resources){try{resource.destroy();}catch{}}
    device.removeEventListener?.('uncapturederror',onUncaptured);device.destroy();};
  const check=signal=>{if(destroyed||signal?.aborted||setupSignal?.aborted)throw aborted();if(lostReason)throw new Error(lostReason);};
  const scoped=async action=>{
    const filters=['out-of-memory','internal','validation'];for(const filter of filters)device.pushErrorScope(filter);
    let result,failure;try{result=await action();}catch(error){failure=error;}
    for(let i=filters.length-1;i>=0;i--){try{const error=await device.popErrorScope();if(error)failure??=new Error(error.message);}catch(error){failure??=error;}}
    if(failure)throw failure;return result;
  };
  try{
    check(setupSignal);
    const source=shaderSource??await timeout((async()=>{const response=await fetch(new URL('./coherent-field.wgsl',import.meta.url),{signal:setupSignal});
      if(!response.ok)throw new Error(`Cannot load coherent WGSL (${response.status})`);return response.text();})(),30000,'Coherent WGSL loading timed out');
    check(setupSignal);
    const pipeline=await scoped(async()=>{
      const module=device.createShaderModule({label:'Coherent full complex field',code:source});
      const info=await module.getCompilationInfo(),errors=info.messages.filter(item=>item.type==='error');
      if(errors.length)throw new Error(errors.map(item=>`WGSL ${item.lineNum}:${item.linePos} ${item.message}`).join('; '));
      return timeout(device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'coherent_main'}}),30000,'Coherent WGSL pipeline compilation timed out');
    });
    check(setupSignal);
    let uniform,output,readback,group;const bytes=packed.count*32;
    await scoped(async()=>{
      const make=(size,usage,data)=>{
        size=Math.max(16,Math.ceil(size/16)*16);
        if(size>device.limits.maxBufferSize||((usage&GPUBufferUsage.STORAGE)!==0&&size>device.limits.maxStorageBufferBindingSize))
          throw new Error('Coherent field exceeds adapter buffer limits');
        const buffer=device.createBuffer({size,usage});resources.push(buffer);if(data)device.queue.writeBuffer(buffer,0,data);return buffer;
      };
      uniform=make(48,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,packed.createUniforms());
      const storage=data=>make(data.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,data);
      const input=[storage(packed.waves),storage(packed.density),storage(packed.factors),storage(packed.targets)];
      output=make(bytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
      readback=make(bytes,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
      group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[uniform,...input,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
    });
    check(setupSignal);
    const setupMs=now()-setupStart,adapterInfo=adapterSummary(adapter),shaderFingerprintFNV1a=fingerprint(source);
    return {type:'WebGPU hybrid',width:reference.width,height:reference.height,count:packed.count,destroy,
      async run({signal,onProgress=()=>{}}={}){
        check(signal);if(running)throw new Error('A coherent GPU field run is already in progress');running=true;
        let mapped=false;
        try{
          const dispatchStart=now();
          const data=await scoped(async()=>{
            for(let first=0;first<packed.count;first+=COHERENT_GPU_LIMITS.batchPoints){
              check(signal);const batch=Math.min(COHERENT_GPU_LIMITS.batchPoints,packed.count-first);
              device.queue.writeBuffer(uniform,0,packed.createUniforms(first,batch));
              const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
              pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(batch/64));pass.end();
              device.queue.submit([encoder.finish()]);
              await timeout(device.queue.onSubmittedWorkDone(),30000,'Coherent GPU batch execution timed out');
              check(signal);onProgress((first+batch)/packed.count);await new Promise(resolve=>setTimeout(resolve,0));
            }
            check(signal);const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(output,0,readback,0,bytes);device.queue.submit([encoder.finish()]);
            await timeout(readback.mapAsync(GPUMapMode.READ),30000,'Coherent GPU readback timed out');mapped=true;check(signal);
            const values=new Float32Array(readback.getMappedRange(0,bytes).slice(0));readback.unmap();mapped=false;return values;
          });
          check(signal);const gpuDispatchReadbackWallMs=now()-dispatchStart,gateStart=now(),parity=coherentParityGate(reference,data),parityGateMs=now()-gateStart;
          check(signal);
          const fieldsRaw=new Float32Array(packed.count*6),intensity=new Float32Array(packed.count),mask=new Uint8Array(packed.count);
          const coherentField=coherentRaw?new Float64Array(packed.count*2):undefined;
          let maxIntensity=0;
          for(let i=0;i<packed.count;i++){
            fieldsRaw.set(data.subarray(8*i,8*i+6),6*i);intensity[i]=data[8*i+6];mask[i]=data[8*i+7];maxIntensity=Math.max(maxIntensity,intensity[i]);
            if(coherentField)for(let m=0;m<3;m++){
              const re=fieldsRaw[6*i+2*m],im=fieldsRaw[6*i+2*m+1],cr=coherentRaw[2*m],ci=coherentRaw[2*m+1];
              coherentField[2*i]+=re*cr-im*ci;coherentField[2*i+1]+=re*ci+im*cr;
            }
          }
          check(signal);
          return {...reference,fieldsRaw,intensity,mask,coherentField,backend:'WebGPU hybrid',scope:COHERENT_GPU_SCOPE,
            diagnostics:{...reference.diagnostics,validPixels:parity.valid,maskedPixels:parity.invalid,maxIntensity,
              finite:true,nonnegative:true,gpuParityPassed:true},
            parity,adapter:adapterInfo,buildId:COHERENT_GPU_BUILD_ID,shaderFingerprintFNV1a,
            timings:{cpuReferenceMs,cpuReferenceProvided:!!providedReference,gpuSetupWallMs:setupMs,gpuDispatchReadbackWallMs,parityGateMs},
            valid:parity.valid,invalid:parity.invalid};
        }catch(error){
          error.diagnostic={buildId:COHERENT_GPU_BUILD_ID,stage:'coherent-gpu-field',adapter:adapterInfo,
            shaderFingerprintFNV1a,points:packed.count,error:error.message,parity:error.parity??null};
          destroy();throw error;
        }finally{if(mapped){try{readback.unmap();}catch{}}running=false;}
      }};
  }catch(error){destroy();throw error;}
}

export async function computeCoherentGPU(solution,grid,options={}){
  const engine=await makeCoherentGPUEngine(solution,grid,options);
  try{return await engine.run(options);}finally{engine.destroy();}
}
