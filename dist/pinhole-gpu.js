/** Native-size 2D complex-f32 FFT convolution for incoherent pinhole images.
 * CPU: scene geometry, spectral PSF construction, input validation.
 * WebGPU: source/kernel FFT, complex product, inverse FFT, planar accumulation.
 * There is no CPU shadow render or physical-model oracle in this ordinary path.
 */
export const PINHOLE_GPU_BUILD_ID='pinhole-streaming-fft-2026-10-13.1';
export const PINHOLE_GPU_LIMITS=Object.freeze({maxWidth:512,maxFFTSize:1024,maxOutputChannels:6,maxOutputMagnitude:1e30});
export const PINHOLE_GPU_TOLERANCE=Object.freeze({roundoffRelative:5e-5,roundoffAbsolute:1e-30});
const now=()=>globalThis.performance?.now?.()??Date.now();
const abortError=()=>Object.assign(new Error('Pinhole GPU calculation cancelled or destroyed'),{name:'AbortError'});
const fail=message=>{throw new RangeError(`Pinhole GPU: ${message}`);};
const power2=x=>Number.isInteger(x)&&x>=2&&(x&(x-1))===0;
function arrayInfo(values,length,label,{signed=false}={}){
  if(!(values instanceof Float32Array)||values.length!==length)fail(`${label} must be Float32Array(${length})`);
  let sum=0,max=0;for(let i=0;i<length;i++){const v=values[i];if(!Number.isFinite(v)||(!signed&&v<0)||Math.abs(v)>1e30)fail(`${label}[${i}] must be finite${signed?'':' and nonnegative'} within f32 bounds`);sum+=Math.abs(v);max=Math.max(max,Math.abs(v));}
  return{sum,max};
}
function adapterInfo(adapter){const info={};for(const k of ['vendor','architecture','device','description','isFallbackAdapter'])try{const v=adapter.info?.[k];if(typeof v==='string'||typeof v==='boolean')info[k]=v;}catch{}return info;}
export function pinholeGPUCapacity({width=512,fftSize=2*width,outputChannels=6}={}){
  if(!power2(width)||width>PINHOLE_GPU_LIMITS.maxWidth)fail('width must be a power of two in 2..512');
  if(!power2(fftSize)||fftSize<2*width||fftSize>PINHOLE_GPU_LIMITS.maxFFTSize)fail('fftSize must be a power of two, at least 2*width, and at most 1024');
  if(!Number.isInteger(outputChannels)||outputChannels<1||outputChannels>6)fail('outputChannels must be an integer in 1..6');
  const complexBytes=fftSize*fftSize*8,scalarBytes=fftSize*fftSize*4,outputBytes=width*width*outputChannels*4,uniformBytes=256*96,safetyBytes=32,twiddleBytes=fftSize*4;
  return{width,fftSize,outputChannels,complexBytes,scalarBytes,outputBytes,uniformBytes,safetyBytes,twiddleBytes,
    residentBytes:3*complexBytes+scalarBytes+2*outputBytes+uniformBytes+2*safetyBytes+twiddleBytes,
    debugFFTPeakBytes:4*complexBytes+scalarBytes+2*outputBytes+uniformBytes+2*safetyBytes+twiddleBytes,
    maxSingleBufferBytes:Math.max(complexBytes,scalarBytes,outputBytes,uniformBytes)};
}
/** Creates one owned device and bounded streaming buffer set. All methods are
 * sequential; cancellation/failure permanently poisons the runner. Partial
 * accumulated output is withheld after any failure. Timings are host elapsed
 * submission+queue completion timings, NOT timestamp-query GPU kernel times.
 */
export async function createPinholeGPURunner(options={}){
  const capacity=pinholeGPUCapacity(options),{width:n,fftSize:m,outputChannels}=capacity;
  const gpu=options.gpu??globalThis.navigator?.gpu;
  if(!gpu)throw new Error('WebGPU unavailable; no CPU replacement was run');
  const timeoutMs=options.timeoutMs??120000;if(!Number.isFinite(timeoutMs)||timeoutMs<=0)fail('timeoutMs must be positive');
  let device,dead=false,busy=false,reason=null,lossReject;const buffers=[];
  const fatal=new Promise((_,reject)=>{lossReject=reject;});fatal.catch(()=>{});
  const stats={...capacity,sourceTransforms:0,kernelTransforms:0,inverseTransforms:0,accumulations:0,fftPasses:0,submissions:0,
    initializationMs:0,sourceFFTMs:0,kernelConvolutionMs:0,readbackMs:0,maxRoundoffTolerance:0,maxConvolutionBound:0,
    timingScope:'Host elapsed upload/encoding/submission/queue wait; not GPU timestamp-query timings',cpuShadow:false};
  const destroy=(error=abortError())=>{if(dead)return;dead=true;reason=error;lossReject(error);options.signal?.removeEventListener('abort',onAbort);if(device&&onUncaptured)device.removeEventListener?.('uncapturederror',onUncaptured);for(const b of buffers){try{b.destroy();}catch{}}try{device?.destroy();}catch{}};
  const onAbort=()=>destroy(abortError());let onUncaptured;
  function check(signal){if(dead)throw reason??abortError();if(options.signal?.aborted||signal?.aborted){destroy();throw abortError();}}
  async function wait(promise,label,signal){check(signal);let timer,listener;const timeout=new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error(`Pinhole GPU ${label} timed out`)),timeoutMs);});
    const local=new Promise((_,reject)=>{if(signal){listener=()=>reject(abortError());signal.addEventListener('abort',listener,{once:true});}});
    try{const r=await Promise.race([promise,fatal,timeout,local]);check(signal);return r;}finally{clearTimeout(timer);if(listener)signal.removeEventListener('abort',listener);}}
  options.signal?.addEventListener('abort',onAbort,{once:true});
  const start=now();let backend,pipelines,uniform,ping,scalar,inputFFT,accumulator,flags,readback,flagReadback,twiddles;
  const U=globalThis.GPUBufferUsage??{MAP_READ:1,COPY_SRC:4,COPY_DST:8,UNIFORM:64,STORAGE:128};
  const MAP_READ=globalThis.GPUMapMode?.READ??1;
  try{
    check();const adapter=await wait(gpu.requestAdapter({powerPreference:'high-performance'}),'adapter request');if(!adapter)throw new Error('No WebGPU adapter available');
    const info=adapterInfo(adapter);const software=/swiftshader|llvmpipe|lavapipe|software/i.test(Object.values(info).join(' '))||info.isFallbackAdapter===true;
    backend={kind:'webgpu',label:`WebGPU${software?' software adapter':''}: ${info.description||info.device||info.architecture||info.vendor||'adapter identity unavailable'}`,adapter:info,softwareAdapter:software,physicalGPUValidated:false,browserValidated:false};
    const limits=adapter.limits??{};
    for(const name of ['maxBufferSize','maxStorageBufferBindingSize'])if(Number(limits[name])<capacity.maxSingleBufferBytes)throw new Error(`WebGPU capacity insufficient: ${name}=${limits[name]}, requires ${capacity.maxSingleBufferBytes}`);
    if(Number(limits.maxComputeWorkgroupsPerDimension)<Math.ceil(m*m/64))throw new Error('WebGPU dispatch capacity insufficient');
    const requested=Promise.resolve(adapter.requestDevice()).then(d=>{if(dead){try{d.destroy();}catch{}}else device=d;return d;});
    await wait(requested,'device request');check();
    device.lost?.then(info=>{if(!dead)destroy(new Error(`Pinhole WebGPU device lost: ${info.message||info.reason||'unknown'}`));});
    onUncaptured=event=>destroy(new Error(`Pinhole WebGPU uncaptured error: ${event.error?.message||'unknown'}`));device.addEventListener?.('uncapturederror',onUncaptured);
    let code=options.shaderSource;if(code===undefined){const response=await wait(fetch(new URL('./pinhole-compute.wgsl',import.meta.url),{signal:options.signal}),'shader fetch');if(!response.ok)throw new Error(`Cannot load pinhole shader (${response.status})`);code=await wait(response.text(),'shader text');}
    device.pushErrorScope('validation');const module=device.createShaderModule({label:'Pinhole complex-f32 FFT',code});
    const compilation=await wait(module.getCompilationInfo(),'shader compilation');const errors=compilation.messages.filter(x=>x.type==='error');const compileValidation=await wait(device.popErrorScope(),'shader validation');
    if(errors.length||compileValidation)throw new Error(`Pinhole WGSL compilation failed: ${errors.map(e=>e.message).join('; ')} ${compileValidation?.message??''}`);
    pipelines={};for(const entryPoint of ['packReal','bitReverse','butterfly','multiply','accumulate'])pipelines[entryPoint]=await wait(device.createComputePipelineAsync({label:`Pinhole ${entryPoint}`,layout:'auto',compute:{module,entryPoint}}),'pipeline creation');
    device.pushErrorScope('validation');
    const allocate=(label,size,usage)=>{const b=device.createBuffer({label,size,usage});buffers.push(b);return b;};
    uniform=allocate('Pinhole stage uniforms',capacity.uniformBytes,U.UNIFORM|U.COPY_DST);
    ping=[0,1].map(i=>allocate(`Pinhole FFT ping-pong ${i}`,capacity.complexBytes,U.STORAGE|U.COPY_SRC|U.COPY_DST));
    inputFFT=allocate('Pinhole reusable source spectrum',capacity.complexBytes,U.STORAGE|U.COPY_DST);
    scalar=allocate('Pinhole streaming real input',capacity.scalarBytes,U.STORAGE|U.COPY_DST);
    accumulator=allocate('Pinhole planar output accumulators',capacity.outputBytes,U.STORAGE|U.COPY_SRC|U.COPY_DST);
    flags=allocate('Pinhole finite/roundoff counters',32,U.STORAGE|U.COPY_SRC|U.COPY_DST);
    readback=allocate('Pinhole output readback',capacity.outputBytes,U.MAP_READ|U.COPY_DST);
    flagReadback=allocate('Pinhole safety readback',32,U.MAP_READ|U.COPY_DST);
    twiddles=allocate('Pinhole f32 FFT twiddles',capacity.twiddleBytes,U.STORAGE|U.COPY_DST);
    const twiddleValues=new Float32Array(m);for(let k=0;k<m/2;k++){twiddleValues[2*k]=Math.cos(2*Math.PI*k/m);twiddleValues[2*k+1]=-Math.sin(2*Math.PI*k/m);}device.queue.writeBuffer(twiddles,0,twiddleValues);
    const allocationError=await wait(device.popErrorScope(),'buffer validation');if(allocationError)throw new Error(`Pinhole buffer validation failed: ${allocationError.message}`);
    stats.initializationMs=now()-start;check();
  }catch(error){destroy(error);throw error;}
  function command(){device.pushErrorScope('validation');const encoder=device.createCommandEncoder({label:'Pinhole streaming FFT command'}),params=new ArrayBuffer(capacity.uniformBytes);let slots=0;
    const encode=(entry,input,output,{span=0,axis=0,inverse=false,channel=0,mode=0,weight=1,tolerance=0}={})=>{
      if(slots>=96)throw new Error('Pinhole stage uniform capacity exceeded');const offset=slots++*256;new Uint32Array(params,offset,8).set([n,m,m*m,span,axis,inverse?1:0,channel,mode]);new Float32Array(params,offset+32,4).set([weight,tolerance,0,0]);
      const entries=[{binding:0,resource:{buffer:uniform,offset,size:64}}];
      if(entry==='packReal')entries.push({binding:2,resource:{buffer:output}},{binding:3,resource:{buffer:scalar}});
      else if(entry==='accumulate')entries.push({binding:1,resource:{buffer:input}},{binding:5,resource:{buffer:accumulator}},{binding:6,resource:{buffer:flags}});
      else{entries.push({binding:1,resource:{buffer:input}},{binding:2,resource:{buffer:output}});if(entry==='multiply')entries.push({binding:4,resource:{buffer:inputFFT}});if(entry==='butterfly')entries.push({binding:7,resource:{buffer:twiddles}});}
      const pipeline=pipelines[entry],group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries});
      const pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil((entry==='accumulate'?n*n:m*m)/64));pass.end();stats.fftPasses+=entry==='butterfly'?1:0;
    };
    const transform=(index=0,inverse=false)=>{for(let axis=0;axis<2;axis++){encode('bitReverse',ping[index],ping[1-index],{axis});index=1-index;for(let span=2;span<=m;span*=2){encode('butterfly',ping[index],ping[1-index],{axis,span,inverse});index=1-index;}}return index;};
    const submit=async signal=>{check(signal);device.queue.writeBuffer(uniform,0,params,0,slots*256);device.queue.submit([encoder.finish()]);stats.submissions++;await wait(device.queue.onSubmittedWorkDone(),'queue completion',signal);const error=await wait(device.popErrorScope(),'command validation',signal);if(error)throw new Error(`Pinhole command validation failed: ${error.message}`);};
    return{encoder,encode,transform,submit};
  }
  async function exclusive(fn,signal){check(signal);if(busy)throw new Error('Pinhole GPU runner methods must be sequential');busy=true;try{return await fn();}catch(error){destroy(error);throw error;}finally{busy=false;}}
  async function mapped(buffer,signal){let mapped=false;try{await wait(buffer.mapAsync(MAP_READ),'readback map',signal);mapped=true;return buffer.getMappedRange().slice(0);}finally{if(mapped)buffer.unmap();}}
  const runner={backend,capacity,stats,
    async convolveAccumulate({source,kernels,signal,onProgress}={}){return exclusive(async()=>{
      const info=arrayInfo(source,n*n,'source');if(!Array.isArray(kernels)||kernels.length===0)fail('kernels must be a nonempty array');
      let time=now();device.queue.writeBuffer(scalar,0,source);const cmd=command();cmd.encode('packReal',null,ping[0]);const sourceIndex=cmd.transform(0);cmd.encoder.copyBufferToBuffer(ping[sourceIndex],0,inputFFT,0,capacity.complexBytes);await cmd.submit(signal);stats.sourceTransforms++;stats.sourceFFTMs+=now()-time;
      for(let k=0;k<kernels.length;k++){
        check(signal);const kernel=kernels[k];if(!kernel||!Number.isInteger(kernel.channel)||kernel.channel<0||kernel.channel>=outputChannels)fail('kernel channel outside configured accumulators');
        const ki=arrayInfo(kernel.values,m*m,`kernel ${k}`),weight=kernel.weight??1;
        if(!Number.isFinite(weight)||weight<0||!Number.isFinite(Math.fround(weight))||(weight>0&&Math.fround(weight)===0))fail('kernel weight must be nonnegative finite f32');
        const bound=info.max*ki.sum*weight;if(!Number.isFinite(bound)||bound>PINHOLE_GPU_LIMITS.maxOutputMagnitude||info.sum*ki.sum>PINHOLE_GPU_LIMITS.maxOutputMagnitude)fail('convolution magnitude bound exceeds safe f32 domain');
        const tolerance=Math.max(PINHOLE_GPU_TOLERANCE.roundoffAbsolute,PINHOLE_GPU_TOLERANCE.roundoffRelative*bound);
        if(stats.accumulations>=Math.floor(0xffffffff/(n*n)))fail('safety inspection counter capacity exceeded');
        stats.maxRoundoffTolerance=Math.max(stats.maxRoundoffTolerance,tolerance);stats.maxConvolutionBound=Math.max(stats.maxConvolutionBound,bound);
        time=now();device.queue.writeBuffer(scalar,0,kernel.values);const c=command();c.encode('packReal',null,ping[0],{mode:1});let index=c.transform(0);c.encode('multiply',ping[index],ping[1-index]);index=c.transform(1-index,true);c.encode('accumulate',ping[index],null,{channel:kernel.channel,weight,tolerance});await c.submit(signal);
        stats.kernelTransforms++;stats.inverseTransforms++;stats.accumulations++;stats.kernelConvolutionMs+=now()-time;
        const update={stage:'convolution',kernelIndex:k,kernelCount:kernels.length,sourceTransforms:stats.sourceTransforms,kernelTransforms:stats.kernelTransforms,backend};onProgress?.(update);options.onProgress?.(update);check(signal);
      }
      return{backend,stats:{...stats}};
    },signal);},
    async readResults({signal}={}){return exclusive(async()=>{
      const time=now(),encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(accumulator,0,readback,0,capacity.outputBytes);encoder.copyBufferToBuffer(flags,0,flagReadback,0,32);device.queue.submit([encoder.finish()]);stats.submissions++;await wait(device.queue.onSubmittedWorkDone(),'output completion',signal);
      const flagBytes=await mapped(flagReadback,signal),u=new Uint32Array(flagBytes),f=new Float32Array(flagBytes);
      const validation={passed:u[0]===0&&u[1]===0&&u[2]===0,mode:'finite-and-roundoff',cpuParityChecked:false,nonFinite:u[0],significantNegative:u[1],imaginaryResidual:u[2],clampedRoundoff:u[3],maxClampedRoundoff:f[4],inspected:u[5],tolerance:PINHOLE_GPU_TOLERANCE,maxAppliedTolerance:stats.maxRoundoffTolerance,roundoffPolicy:'Per-kernel tolerance = max(absolute, relative * source maximum * kernel L1 * weight); values in [-tolerance,0) are measured then clamped; larger negatives and imaginary residuals reject all output'};
      if(!validation.passed)throw Object.assign(new Error('Pinhole GPU finite/roundoff gate failed; output withheld'),{name:'PinholeGPUValidationError',validation});
      const all=new Float32Array(await mapped(readback,signal));for(let i=0;i<all.length;i++)if(!Number.isFinite(all[i])||all[i]<0)throw new Error(`Pinhole GPU invalid output ${i}; output withheld`);
      check(signal);const channels=Array.from({length:outputChannels},(_,i)=>all.slice(i*n*n,(i+1)*n*n));stats.readbackMs+=now()-time;return{channels,width:n,fftSize:m,backend,validation,stats:{...stats}};
    },signal);},
    /** Debug/offline test hook only. Interleaved complex M*M f32, exact size. */
    async fft2D(values,{inverse=false,signal}={}){return exclusive(async()=>{
      arrayInfo(values,m*m*2,'complex FFT input',{signed:true});const read=device.createBuffer({label:'Pinhole debug FFT readback',size:capacity.complexBytes,usage:U.MAP_READ|U.COPY_DST});buffers.push(read);
      try{device.queue.writeBuffer(ping[0],0,values);const c=command(),index=c.transform(0,inverse);c.encoder.copyBufferToBuffer(ping[index],0,read,0,capacity.complexBytes);await c.submit(signal);const result=new Float32Array(await mapped(read,signal));if(!result.every(Number.isFinite))throw new Error('Pinhole debug FFT nonfinite output');return result;}finally{read.destroy();const i=buffers.indexOf(read);if(i>=0)buffers.splice(i,1);}
    },signal);},
    destroy(){destroy();}
  };return runner;
}

/** Independent Float64 separable radix-2 oracle. Offline/debug and explicitly
 * labelled small CPU fallback only. It is never invoked by the GPU runner. */
export function pinholeFFT2DCPU(values,size,{inverse=false}={}){
  if(!power2(size)||size>1024||values.length!==2*size*size)fail('CPU FFT needs exact complex power-of-two array');
  const out=Float64Array.from(values);if(!out.every(Number.isFinite))fail('CPU FFT input must be finite');
  const re=new Float64Array(size),im=new Float64Array(size);
  for(let axis=0;axis<2;axis++)for(let line=0;line<size;line++){
    for(let i=0;i<size;i++){const at=2*(axis===0?line*size+i:i*size+line);re[i]=out[at];im[i]=out[at+1];}
    for(let i=1,j=0;i<size;i++){let bit=size>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;if(i<j){[re[i],re[j]]=[re[j],re[i]];[im[i],im[j]]=[im[j],im[i]];}}
    for(let length=2;length<=size;length*=2){const half=length/2;for(let base=0;base<size;base+=length)for(let k=0;k<half;k++){
      const angle=(inverse?1:-1)*2*Math.PI*k/length,c=Math.cos(angle),s=Math.sin(angle),a=base+k,b=a+half,tr=re[b]*c-im[b]*s,ti=re[b]*s+im[b]*c,ar=re[a],ai=im[a];re[a]=ar+tr;im[a]=ai+ti;re[b]=ar-tr;im[b]=ai-ti;
    }}
    for(let i=0;i<size;i++){const at=2*(axis===0?line*size+i:i*size+line),scale=inverse?1/size:1;out[at]=re[i]*scale;out[at+1]=im[i]*scale;}
  }return out;
}
export function convolvePinholeCPU({source,width,fftSize=2*width,kernels,outputChannels=6}={}){
  pinholeGPUCapacity({width,fftSize,outputChannels});if(width>64)fail('CPU fallback is limited to native 64x64; select a lower resolution explicitly');
  const start=now();arrayInfo(source,width*width,'CPU source');if(!Array.isArray(kernels)||!kernels.length)fail('CPU kernels required');
  const packed=new Float64Array(2*fftSize*fftSize);for(let y=0;y<width;y++)for(let x=0;x<width;x++)packed[2*(y*fftSize+x)]=source[y*width+x];
  const spectrum=pinholeFFT2DCPU(packed,fftSize),channels=Array.from({length:outputChannels},()=>new Float64Array(width*width));
  let roundoff=0,maxRoundoff=0;for(const kernel of kernels){arrayInfo(kernel.values,fftSize*fftSize,'CPU kernel');if(!Number.isInteger(kernel.channel)||kernel.channel<0||kernel.channel>=outputChannels)fail('CPU kernel channel invalid');const weight=kernel.weight??1;if(!Number.isFinite(weight)||weight<0)fail('CPU kernel weight invalid');
    packed.fill(0);for(let i=0;i<kernel.values.length;i++)packed[2*i]=kernel.values[i];const h=pinholeFFT2DCPU(packed,fftSize);for(let i=0;i<h.length;i+=2){const a=h[i],b=h[i+1];h[i]=a*spectrum[i]-b*spectrum[i+1];h[i+1]=a*spectrum[i+1]+b*spectrum[i];}const image=pinholeFFT2DCPU(h,fftSize,{inverse:true});
    let peak=0;for(let y=0;y<width;y++)for(let x=0;x<width;x++)peak=Math.max(peak,Math.abs(image[2*(y*fftSize+x)]*weight));const tolerance=Math.max(1e-30,peak*1e-11);
    for(let y=0;y<width;y++)for(let x=0;x<width;x++){const at=2*(y*fftSize+x),v=image[at]*weight;if(!Number.isFinite(v)||!Number.isFinite(image[at+1])||v < -tolerance||Math.abs(image[at+1]*weight)>tolerance)throw new Error('CPU convolution finite/roundoff gate failed');if(v<0){roundoff++;maxRoundoff=Math.max(maxRoundoff,-v);}channels[kernel.channel][y*width+x]+=Math.max(0,v);}
  }
  return{channels,width,fftSize,backend:{kind:'cpu',label:'CPU Float64 FFT, explicitly limited to native 64x64',physicalGPUValidated:false},validation:{passed:true,mode:'CPU finite-and-roundoff',cpuParityChecked:false,clampedRoundoff:roundoff,maxClampedRoundoff:maxRoundoff},stats:{elapsedMs:now()-start,sourceTransforms:1,kernelTransforms:kernels.length,inverseTransforms:kernels.length,cpuShadow:false}};
}
