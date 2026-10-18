// General 3D path-space transport is constructed on the CPU. WebGPU evaluates
// local coherent UTD events and independent stochastic camera contributions.
// Strict (default) admission requires exhaustive CPU parity. The opt-in mono
// finite mode checks every event record and pixel for finite-safety only.
export const PLT_GPU_BUILD_ID = 'plt-camera-gpu-2026-10-12.5';
export const PLT_GPU_SCOPE = 'CPU 3D BVH path construction and native sampling; WebGPU local coherent UTD events and stochastic camera sums; every-event and every-pixel CPU verification';
export const PLT_GPU_FINITE_SCOPE = 'CPU 3D BVH path construction and native sampling; WebGPU local coherent UTD events and stochastic camera sums; every-event-record and every-pixel finite-safety checks; CPU parity not checked';
export const PLT_GPU_LIMITS = Object.freeze({pixels:65536,events:524288,edges:2097152,edgesPerEvent:128,
  terms:1048576,termsPerPixel:65536,depth:6,samplesPerPixel:65536,maxPhase:100000,maxSpectralRawPhase:100000000,batchEvents:2048,batchPixels:1024});
// Fixed before validation. Backend parity does not establish physical accuracy.
export const PLT_GPU_TOLERANCE = Object.freeze({eventAbsolute:2e-5,eventRelative:.003,
  pixelAbsolute:2e-5,pixelRelative:.003,boundaryEpsilon:2e-6});
const now=()=>globalThis.performance?.now?.()??Date.now();
const abortError=()=>Object.assign(new Error('PLT GPU calculation cancelled or destroyed'),{name:'AbortError'});
function fail(message){throw new RangeError(`PLT GPU: ${message}`);}
function count(value,max,label,min=0){if(!Number.isInteger(value)||value<min||value>max)fail(`${label} must be an integer in ${min}..${max}`);}
function number(value,label,min=-1e30,max=1e30){if(!Number.isFinite(value)||value<min||value>max)fail(`${label} is outside the bounded finite domain [${min},${max}]`);return value;}
function checkSignal(signal){if(signal?.aborted)throw abortError();}
function fingerprint(source){let h=2166136261;for(let i=0;i<source.length;i++){h^=source.charCodeAt(i);h=Math.imul(h,16777619);}return(h>>>0).toString(16).padStart(8,'0');}
function adapterSummary(adapter){const info={};for(const key of ['vendor','architecture','device','description','isFallbackAdapter']){try{const v=adapter.info?.[key];if(typeof v==='string'||typeof v==='boolean')info[key]=v;}catch{}}return info;}
function timed(promise,ms,label,signal){let timer,listener;const competitors=[promise,new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error(label)),ms);})];
  if(signal)competitors.push(new Promise((_,reject)=>{listener=()=>reject(abortError());if(signal.aborted)listener();else signal.addEventListener('abort',listener,{once:true});}));
  return Promise.race(competitors).finally(()=>{clearTimeout(timer);if(listener)signal.removeEventListener('abort',listener);});}

/** Validate and snapshot the entire job before any await or device allocation.
 * Event ABI: {...operator, referenceF}, with real directAmplitude and edges
 * {k,n,phiI,phiO,ro,sinBeta,phaseExcess,edgeId}. The wrapped form
 * {operator,referenceF} is accepted as well. Each event is a LOCAL field sum.
 * Opt-in validationMode:'finite' supports mono jobs without reading or building
 * CPU references. It retains the same transport ABI and bounded input checks.
 */
export function packPLTGPU(job,{validationMode='strict'}={}){
  if(validationMode!=='strict'&&validationMode!=='finite')fail('validationMode must be strict or finite');
  const finite=validationMode==='finite';
  if(!job||typeof job!=='object')fail('job is required');
  if(finite&&job.spectralMode==='visible')fail('finite validation mode currently supports mono jobs only; visible spectral jobs require strict validation');
  count(job.width,PLT_GPU_LIMITS.pixels,'width',1);count(job.height,PLT_GPU_LIMITS.pixels,'height',1);
  const physicalPixels=job.width*job.height;count(physicalPixels,PLT_GPU_LIMITS.pixels,'pixel count',1);
  // Visible XYZ reuses the scalar camera kernel as three interleaved virtual
  // pixels. The event tape and hardened local UTD shader are unchanged. Apply
  // the original allocation budgets AFTER expansion, not to physical pixels.
  const channels=job.spectralMode==='visible'?3:1,pixels=physicalPixels*channels;
  count(pixels,PLT_GPU_LIMITS.pixels,'expanded camera component count',1);
  count(job.samplesPerPixel,PLT_GPU_LIMITS.samplesPerPixel,'samplesPerPixel',1);
  if(!Array.isArray(job.events)||!Array.isArray(job.terms))fail('events and terms must be arrays');
  count(job.events.length,PLT_GPU_LIMITS.events,'event count');count(job.terms.length,PLT_GPU_LIMITS.terms,'term count');
  const termCount=job.terms.length*channels;count(termCount,PLT_GPU_LIMITS.terms,'expanded term count');
  let reference=null;
  if(!finite){
    if(!(job.reference instanceof Float64Array)||job.reference.length!==physicalPixels)fail('reference must be an exact full-image Float64Array');
    for(let i=0;i<physicalPixels;i++)number(job.reference[i],`reference[${i}]`,0);
    if(channels===3&&(!(job.referenceXYZ instanceof Float64Array)||job.referenceXYZ.length!==pixels))fail('referenceXYZ must be an exact interleaved full-image Float64Array');
    reference=(channels===3?job.referenceXYZ:job.reference).slice();for(let i=0;i<pixels;i++)number(reference[i],`camera reference[${i}]`,0);
    if(channels===3)for(let i=0;i<physicalPixels;i++)if(job.reference[i]!==reference[3*i+1])fail(`reference[${i}] must equal referenceXYZ Y`);
  }
  const phaseReduction=channels===3?{mode:'f64-centered-modulo-2pi',rawPhaseLimit:PLT_GPU_LIMITS.maxSpectralRawPhase,packedPhaseLimit:PLT_GPU_LIMITS.maxPhase,maxAbsoluteRawPhase:0,maxAbsolutePackedPhase:0,reducedEdges:0}:null;
  let edgeCount=0;for(const [i,item] of job.events.entries()){
    const op=item?.operator??item;if(!op||!Array.isArray(op.edges))fail(`event ${i} needs an edges array`);
    count(op.edges.length,PLT_GPU_LIMITS.edgesPerEvent,`event ${i} edge count`);edgeCount+=op.edges.length;
  }count(edgeCount,PLT_GPU_LIMITS.edges,'total edge count');
  const eventData=new ArrayBuffer(job.events.length*32),eu=new Uint32Array(eventData),ef=new Float32Array(eventData);
  const edgeData=new Float32Array(edgeCount*8),eventReference=finite?null:new Float64Array(job.events.length),eventEdgeCounts=new Uint32Array(job.events.length);
  let at=0;for(let i=0;i<job.events.length;i++){
    const item=job.events[i],op=item.operator??item,boundary=['soft','hard','unpolarized'].indexOf(op.boundary);
    if(boundary<0)fail(`event ${i} requires soft, hard, or unpolarized boundary`);
    if(op.boundaryEpsilon!==undefined&&op.boundaryEpsilon!==PLT_GPU_TOLERANCE.boundaryEpsilon)fail(`event ${i} boundaryEpsilon differs from the hardened WGSL convention`);
    const direct=number(op.directAmplitude,`event ${i} directAmplitude`,0,1);
    if(!finite)eventReference[i]=number(item.referenceF,`event ${i} referenceF`,0);
    eventEdgeCounts[i]=op.edges.length;eu.set([at,op.edges.length,boundary,0],i*8);ef[i*8+4]=direct;
    for(let j=0;j<op.edges.length;j++,at++){
      const edge=op.edges[j];if(!edge)fail(`event ${i} edge ${j} is missing`);
      const k=number(edge.k??op.k,`event ${i} edge ${j} k`,1e-6,1e8),n=number(edge.n,'wedge index',1,2);
      const phiI=number(edge.phiI,'phiI',0,n*Math.PI),phiO=number(edge.phiO,'phiO',0,n*Math.PI);
      const ro=number(edge.ro,'ro',1e-8,1e8),sinBeta=number(edge.sinBeta,'sinBeta',1e-5,1);
      const phaseExcess=number(edge.phaseExcess,'phaseExcess',-1e8,1e8),L=ro*sinBeta*sinBeta;
      const rawPhase=k*phaseExcess;let packedPhaseExcess=phaseExcess;
      if(channels===3){
        number(rawPhase,'visible raw local phase',-PLT_GPU_LIMITS.maxSpectralRawPhase,PLT_GPU_LIMITS.maxSpectralRawPhase);
        // Only exp(-i*k*phaseExcess) uses this distance in the stable shader.
        // Reduce its phase in f64 BEFORE rounding operands to f32. This is a
        // periodic identity, not a clamp; k and every UTD coefficient stay
        // unchanged. Divide by the actual packed k to avoid adding its f32
        // quantization error back to the already reduced phase.
        const tau=2*Math.PI;let reduced=rawPhase%tau;if(reduced>Math.PI)reduced-=tau;else if(reduced< -Math.PI)reduced+=tau;
        packedPhaseExcess=reduced/Math.fround(k);
        phaseReduction.maxAbsoluteRawPhase=Math.max(phaseReduction.maxAbsoluteRawPhase,Math.abs(rawPhase));
        if(Math.abs(rawPhase)>Math.PI)phaseReduction.reducedEdges++;
      }else number(rawPhase,'local phase',-PLT_GPU_LIMITS.maxPhase,PLT_GPU_LIMITS.maxPhase);
      number(k*L,'kL',1e-12,1e12);
      edgeData.set([k,n,phiI,phiO,ro,sinBeta,packedPhaseExcess,L],at*8);
      // Verify the actual f32 operands too, including underflow and phase range.
      const f=edgeData.subarray(at*8,at*8+8);if(!f.every(Number.isFinite)||f[0]<=0||f[4]<=0||f[5]<=0||f[7]<=0)fail('edge cannot be represented safely in f32');
      number(f[0]*f[6],'packed local phase',-PLT_GPU_LIMITS.maxPhase,PLT_GPU_LIMITS.maxPhase);
      if(phaseReduction)phaseReduction.maxAbsolutePackedPhase=Math.max(phaseReduction.maxAbsolutePackedPhase,Math.abs(f[0]*f[6]));
    }
  }
  const ranges=new Uint32Array(pixels*2);
  for(let i=0;i<job.terms.length;i++){
    const term=job.terms[i];if(!term)fail(`term ${i} is missing`);count(term.pixel,physicalPixels-1,`term ${i} pixel`);
    number(term.weight,`term ${i} weight`,0,1e30);
    if(channels===3){
      if(!Array.isArray(term.xyzWeight)||term.xyzWeight.length!==3)fail(`term ${i} xyzWeight must be an exact three-component array`);
      for(let c=0;c<3;c++){number(term.xyzWeight[c],`term ${i} xyzWeight[${c}]`,0,1e30);number(term.weight*term.xyzWeight[c],`term ${i} weighted XYZ[${c}]`,0,1e30);}
    }
    if(!Array.isArray(term.eventIds))fail(`term ${i} eventIds must be an array`);
    count(term.eventIds.length,PLT_GPU_LIMITS.depth,`term ${i} event depth`);
    for(const id of term.eventIds)count(id,job.events.length-1,`term ${i} event ID`);
    for(let c=0;c<channels;c++)count(++ranges[(term.pixel*channels+c)*2+1],PLT_GPU_LIMITS.termsPerPixel,`pixel ${term.pixel} term count`);
  }
  let start=0;for(let i=0;i<pixels;i++){ranges[i*2]=start;start+=ranges[i*2+1];}
  const cursors=Uint32Array.from({length:pixels},(_,i)=>ranges[2*i]);
  const termData=new ArrayBuffer(termCount*48),tu=new Uint32Array(termData),tf=new Float32Array(termData);
  for(const term of job.terms)for(let c=0;c<channels;c++){
    const index=cursors[term.pixel*channels+c]++,base=index*12;
    tf[base]=channels===3?term.weight*term.xyzWeight[c]:term.weight;tu[base+1]=term.eventIds.length;tu.set(term.eventIds,base+4);
  }
  const savedEventCount=job.events.length,savedSamples=job.samplesPerPixel;
  const createUniforms=(first=0,batch=0)=>new Uint32Array([savedEventCount,pixels,savedSamples,0,first,batch,0,0]);
  return {width:job.width,height:job.height,pixels,physicalPixels,channels,samplesPerPixel:job.samplesPerPixel,eventCount:job.events.length,edgeCount,termCount,physicalTermCount:job.terms.length,
    eventData,edgeData,termData,ranges,reference,eventReference,eventEdgeCounts,createUniforms,...(finite?{validationMode}:{}),...(phaseReduction?{phaseReduction}: {})};
}

function gateArray(expected,actual,absolute,relative,kind){
  const gate={passed:true,coverage:`every-${kind}`,count:expected.length,nonfinite:0,negative:0,numericalFailures:0,maxAbsoluteError:0,maxToleranceRatio:0,
    tolerance:{absolute,relative},absoluteErrors:new Float64Array(expected.length),toleranceRatios:new Float64Array(expected.length),failures:[]};
  for(let i=0;i<expected.length;i++){
    const value=actual[i],error=Math.abs(value-expected[i]),ratio=error/(absolute+relative*Math.abs(expected[i]));
    gate.absoluteErrors[i]=error;gate.toleranceRatios[i]=ratio;
    const nonfinite=!Number.isFinite(value),negative=value<0,bad=nonfinite||negative||!Number.isFinite(ratio)||ratio>1;
    if(nonfinite)gate.nonfinite++;if(negative)gate.negative++;if(!Number.isFinite(ratio)||ratio>1)gate.numericalFailures++;
    if(Number.isFinite(error))gate.maxAbsoluteError=Math.max(gate.maxAbsoluteError,error);if(Number.isFinite(ratio))gate.maxToleranceRatio=Math.max(gate.maxToleranceRatio,ratio);
    if(bad){gate.passed=false;gate.failures.push({index:i,expected:expected[i],actual:value,absoluteError:error,toleranceRatio:ratio});}
  }return gate;
}
function rejectGate(gate,stage){if(!gate.passed){const error=new Error(`PLT GPU ${stage} CPU parity failed; unverified image withheld`);error.name='PLTGPUParityError';error.validation=gate;error.parity=gate;throw error;}}
/** Retain numerical errors for ALL events, including unused path events. */
export function pltEventParityGate(packed,data){
  if(!(data instanceof Float32Array)||data.length!==packed.eventCount*8)fail('event readback must be an exact stride-8 Float32Array');
  const values=Float32Array.from({length:packed.eventCount},(_,i)=>data[8*i+4]);
  const t=PLT_GPU_TOLERANCE,gate=gateArray(packed.eventReference,values,t.eventAbsolute,t.eventRelative,'event');
  gate.invalidEventRecords=0;
  for(let i=0;i<packed.eventCount;i++){
    const row=data.subarray(i*8,i*8+8);
    if(!row.every(Number.isFinite)||row[5]<0||row[6]<0||row[7]!==packed.eventEdgeCounts[i]){
      gate.invalidEventRecords++;gate.passed=false;gate.failures.push({index:i,reason:'nonfinite, negative, or inconsistent event output',actual:[...row]});
    }
  }rejectGate(gate,'every-event');return gate;
}
/** No masking or sampled-point admission: every camera pixel, and every XYZ
 * component for visible jobs, is checked against its corresponding CPU value. */
export function pltPixelParityGate(packed,data){
  if(!(data instanceof Float32Array)||data.length!==packed.pixels)fail('camera readback must be an exact Float32Array');
  const spectral=packed.channels===3,kind=spectral?'color-component':'pixel';
  const t=PLT_GPU_TOLERANCE,gate=gateArray(packed.reference,data,t.pixelAbsolute,t.pixelRelative,kind);
  if(spectral){gate.physicalPixels=packed.physicalPixels;gate.channels=3;gate.channelOrder='XYZ';
    for(const failure of gate.failures){failure.pixel=Math.floor(failure.index/3);failure.channel='XYZ'[failure.index%3];}}
  rejectGate(gate,`every-${kind}`);return gate;
}

function rejectFiniteGate(gate,stage){if(!gate.passed){const error=new Error(`PLT GPU ${stage} finite-safety validation failed; unsafe image withheld`);error.name='PLTGPUFiniteError';error.validation=gate;throw error;}}
/** Safety admission only: no CPU operator evaluation, reference comparison,
 * field-to-power consistency claim, or numerical/physical accuracy claim. */
export function pltEventFiniteGate(packed,data){
  if(!(data instanceof Float32Array)||data.length!==packed.eventCount*8)fail('event readback must be an exact stride-8 Float32Array');
  const gate={passed:true,mode:'finite',parityChecked:false,coverage:'every-event-record-finite-safety',count:packed.eventCount,
    nonfinite:0,negative:0,invalidEventRecords:0,mismatchedEdgeCounts:0,failures:[]};
  for(let i=0;i<packed.eventCount;i++){
    const row=data.subarray(i*8,i*8+8),nonfinite=!row.every(Number.isFinite),negative=row[4]<0||row[5]<0||row[6]<0,mismatchedEdgeCount=row[7]!==packed.eventEdgeCounts[i];
    if(nonfinite)gate.nonfinite++;if(negative)gate.negative++;if(mismatchedEdgeCount)gate.mismatchedEdgeCounts++;
    if(nonfinite||negative||mismatchedEdgeCount){gate.passed=false;gate.invalidEventRecords++;gate.failures.push({index:i,reason:'nonfinite, negative, or inconsistent event output',actual:[...row]});}
  }
  rejectFiniteGate(gate,'every-event-record');return gate;
}
export function pltPixelFiniteGate(packed,data){
  if(!(data instanceof Float32Array)||data.length!==packed.pixels)fail('camera readback must be an exact Float32Array');
  const gate={passed:true,mode:'finite',parityChecked:false,coverage:'every-pixel-finite-safety',count:packed.pixels,nonfinite:0,negative:0,failures:[]};
  for(let i=0;i<packed.pixels;i++){
    const value=data[i],nonfinite=!Number.isFinite(value),negative=value<0;
    if(nonfinite)gate.nonfinite++;if(negative)gate.negative++;
    if(nonfinite||negative){gate.passed=false;gate.failures.push({index:i,actual:value});}
  }
  rejectFiniteGate(gate,'every-pixel');return gate;
}

/** One device per engine. Cancel/failure invalidates the engine permanently;
 * no partially read, stale, or numerically rejected image can be returned. */
export async function makePLTGPUEngine(job,{gpu=globalThis.navigator?.gpu,shaderSource,signal:setupSignal,onLost=()=>{},timeoutMs=30000,validationMode='strict',_shared=null}={}){
  const setupStart=now();checkSignal(setupSignal);number(timeoutMs,'timeoutMs',1,120000);const packed=packPLTGPU(job,{validationMode}),finite=validationMode==='finite';
  if(!gpu)throw new Error('WebGPU unavailable; PLT image was not computed');
  const adapter=_shared?.adapter??await timed(gpu.requestAdapter({powerPreference:'low-power'}),timeoutMs,'PLT adapter request timed out',setupSignal);
  checkSignal(setupSignal);if(!adapter)throw new Error('No WebGPU adapter for PLT camera');
  const pending=_shared?.device?Promise.resolve(_shared.device):adapter.requestDevice();let device;
  try{device=await timed(pending,timeoutMs,'PLT device request timed out',setupSignal);}
  catch(error){pending.then(late=>{try{late.destroy();}catch{}},()=>{});throw error;}
  // Cancellation can win after requestDevice resolves but before this await
  // resumes. A closed batch runner must never adopt that late device.
  if(setupSignal?.aborted){try{device.destroy();}catch{}throw abortError();}
  if(_shared&&!_shared.device){_shared.adapter=adapter;_shared.device=device;
    // One listener per reused device, never one retained closure per tile.
    device.lost.then(info=>{_shared.fatal=new Error(info.message||'PLT WebGPU device lost');_shared.onLost?.(_shared.fatal.message);},error=>{_shared.fatal=error;_shared.onLost?.(error.message);});
  }
  const resources=[];let destroyed=false,running=false,fatal=null,interrupt;
  const interrupted=new Promise((_,reject)=>{interrupt=reject;});interrupted.catch(()=>{});
  const destroy=()=>{if(destroyed)return;destroyed=true;interrupt(fatal??abortError());for(const r of resources){try{r.destroy();}catch{}}
    setupSignal?.removeEventListener('abort',onAbort);device.removeEventListener?.('uncapturederror',onUncaptured);
    if(_shared){if(_shared.onLost===reportLost)_shared.onLost=null;}else try{device.destroy();}catch{}};
  const onAbort=()=>{fatal=abortError();destroy();};
  const reportLost=message=>{if(destroyed)return;fatal=new Error(message);try{onLost(message);}catch{}destroy();};
  const onUncaptured=event=>reportLost(event.error?.message||'PLT WebGPU uncaptured error');
  device.addEventListener?.('uncapturederror',onUncaptured);
  if(_shared)_shared.onLost=reportLost;else device.lost.then(info=>reportLost(info.message||'PLT WebGPU device lost'),error=>reportLost(error.message));
  setupSignal?.addEventListener('abort',onAbort,{once:true});
  const check=()=>{if(_shared?.fatal)throw _shared.fatal;if(fatal)throw fatal;if(destroyed||setupSignal?.aborted)throw abortError();};
  const wait=(promise,label)=>timed(Promise.race([promise,interrupted]),timeoutMs,label);
  const scoped=async action=>{const filters=['out-of-memory','internal','validation'];let pushed=0,result,failure;
    try{for(const filter of filters){device.pushErrorScope(filter);pushed++;}result=await action();}catch(error){failure=error;}
    while(pushed--){try{const error=await wait(device.popErrorScope(),'PLT error scope timed out');if(error)failure??=new Error(error.message);}catch(error){failure??=error;}}
    if(failure)throw failure;return result;};
  const diagnostic={buildId:PLT_GPU_BUILD_ID,stage:'setup',adapter:adapterSummary(adapter),events:packed.eventCount,pixels:packed.pixels,physicalPixels:packed.physicalPixels,channels:packed.channels};
  try{
    check();const source=_shared?.source??shaderSource??await wait((async()=>{
      const texts=await Promise.all(['utd-functions.wgsl','plt-compute.wgsl'].map(async name=>{const response=await fetch(new URL(`./${name}`,import.meta.url),{signal:setupSignal});
        if(!response.ok)throw new Error(`Cannot load ${name} (${response.status})`);return response.text();}));return texts.join('\n');})(),'PLT WGSL load timed out');
    check();diagnostic.shaderFingerprintFNV1a=fingerprint(source);
    const pipelines=_shared?.pipelines??await scoped(async()=>{
      const module=device.createShaderModule({label:'PLT local UTD and stochastic camera',code:source});
      const info=await wait(module.getCompilationInfo(),'PLT WGSL compilation timed out'),errors=info.messages.filter(x=>x.type==='error');
      if(errors.length)throw new Error(errors.map(x=>`WGSL ${x.lineNum}:${x.linePos} ${x.message}`).join('; '));
      return Promise.all(['plt_events','plt_camera'].map(entryPoint=>wait(device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint}}),'PLT pipeline compilation timed out')));
    });check();
    if(_shared){_shared.source=source;_shared.pipelines=pipelines;}
    let uniform,eventInput,edgeInput,eventOutput,cameraOutput,eventReadback,cameraReadback,groups;
    await scoped(async()=>{
      const make=(size,usage,data,label)=>{size=Math.max(16,Math.ceil(size/16)*16);
        if(size>device.limits.maxBufferSize||((usage&GPUBufferUsage.STORAGE)&&size>device.limits.maxStorageBufferBindingSize))throw new Error('PLT job exceeds adapter buffer limits');
        const buffer=device.createBuffer({size,usage,label});resources.push(buffer);if(data?.byteLength)device.queue.writeBuffer(buffer,0,data);return buffer;};
      const storage=(data,label,minSize=16)=>make(Math.max(minSize,data.byteLength),GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,data,label);
      uniform=make(32,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,packed.createUniforms(),'PLT parameters');
      eventInput=storage(packed.eventData,'PLT events',32);edgeInput=storage(packed.edgeData,'PLT edges',32);
      const termInput=storage(packed.termData,'PLT terms',48),rangeInput=storage(packed.ranges,'PLT pixel ranges');
      eventOutput=make(Math.max(32,packed.eventCount*32),GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,null,'PLT event output');
      cameraOutput=make(packed.pixels*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,null,'PLT camera output');
      eventReadback=make(packed.eventCount*32,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST,null,'PLT event readback');
      cameraReadback=make(packed.pixels*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST,null,'PLT camera readback');
      groups=[[[0,uniform],[1,eventInput],[2,edgeInput],[3,eventOutput]],[[0,uniform],[3,eventOutput],[4,termInput],[5,rangeInput],[6,cameraOutput]]].map((entries,i)=>device.createBindGroup({layout:pipelines[i].getBindGroupLayout(0),entries:entries.map(([binding,buffer])=>({binding,resource:{buffer}}))}));
    });check();const setupMs=now()-setupStart;
    return {destroy,type:'WebGPU hybrid',width:packed.width,height:packed.height,
      async run({signal,onProgress=()=>{}}={}){
        check();checkSignal(signal);if(running)throw new Error('A PLT GPU run is already in progress');running=true;
        signal?.addEventListener('abort',onAbort,{once:true});let eventValidation=null,pixelValidation=null;
        const read=async(output,buffer,count)=>{
          if(!count)return new Float32Array();let mapped=false;
          try{check();const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(output,0,buffer,0,count*4);device.queue.submit([encoder.finish()]);
            await wait(buffer.mapAsync(GPUMapMode.READ),'PLT GPU readback timed out');mapped=true;check();return new Float32Array(buffer.getMappedRange(0,count*4).slice(0));
          }finally{if(mapped){try{buffer.unmap();}catch{}}}
        };
        const dispatch=async(stage,count,batchLimit,base,scale)=>{
          for(let first=0;first<count;first+=batchLimit){check();const batch=Math.min(batchLimit,count-first);device.queue.writeBuffer(uniform,0,packed.createUniforms(first,batch));
            const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipelines[stage]);pass.setBindGroup(0,groups[stage]);pass.dispatchWorkgroups(Math.ceil(batch/64));pass.end();device.queue.submit([encoder.finish()]);
            await wait(device.queue.onSubmittedWorkDone(),'PLT GPU dispatch timed out');check();onProgress(base+scale*(first+batch)/count);check();await new Promise(resolve=>setTimeout(resolve,0));
          }
        };
        try{
          diagnostic.stage='events';const eventStart=now();
          const eventData=await scoped(async()=>{await dispatch(0,packed.eventCount,PLT_GPU_LIMITS.batchEvents,0,.6);return read(eventOutput,eventReadback,packed.eventCount*8);});
          check();const eventDispatchReadbackWallMs=now()-eventStart,gateStart=now();
          try{eventValidation=finite?pltEventFiniteGate(packed,eventData):pltEventParityGate(packed,eventData);}catch(parityError){
            if(parityError.name!=='PLTGPUParityError')throw parityError;
            // Keep the original readback immutable. This separate, bounded
            // rerun is only evidence; it can neither repair nor admit an image.
            diagnostic.failureTrace={schema:'plt-event-failure-trace-v1',status:'unavailable'};
            const traceStart=now();
            try{
              const traceWork=async()=>{
                check();checkSignal(signal);const trace=await import('./plt-failure-trace.js');check();checkSignal(signal);
                const ids=trace.selectPLTTraceEvents(parityError.validation,packed.eventCount);
                if(!ids.length)throw new Error('No bounded failed event indices');
                diagnostic.failureTrace=trace.describePLTFailureTrace(packed,eventData,ids,null);
                const out=await scoped(async()=>{
                  check();const module=device.createShaderModule({label:'PLT failure-only observational rerun',code:source+'\n'+trace.PLT_FAILURE_TRACE_WGSL});
                  const info=await wait(module.getCompilationInfo(),'PLT diagnostic compile timed out');check();
                  const errors=info.messages.filter(x=>x.type==='error');if(errors.length)throw new Error(errors.map(x=>x.message).join('; '));
                  const pipeline=await wait(device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'plt_failure_trace'}}),'PLT diagnostic pipeline timed out');check();checkSignal(signal);
                  const make=(size,usage,label)=>{const buffer=device.createBuffer({size,usage,label});resources.push(buffer);return buffer;};
                  const selector=make(Math.max(16,ids.length*4),GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,'PLT diagnostic selection');
                  const bytes=ids.length*trace.PLT_TRACE_EVENT_FLOATS*4;
                  const output=make(bytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,'PLT diagnostic output');
                  const staging=make(bytes,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ,'PLT diagnostic readback');
                  device.queue.writeBuffer(selector,0,new Uint32Array(ids));
                  const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[[1,eventInput],[2,edgeInput],[7,selector],[8,output]].map(([binding,buffer])=>({binding,resource:{buffer}}))});
                  const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(ids.length);pass.end();device.queue.submit([encoder.finish()]);
                  await wait(device.queue.onSubmittedWorkDone(),'PLT diagnostic dispatch timed out');check();checkSignal(signal);
                  return read(output,staging,ids.length*trace.PLT_TRACE_EVENT_FLOATS);
                });
                check();checkSignal(signal);return trace.describePLTFailureTrace(packed,eventData,ids,out);
              };
              diagnostic.failureTrace=await timed(traceWork(),Math.min(timeoutMs,5000),'PLT failure diagnostics exceeded 5s budget',signal);
            }catch(traceError){
              if(signal?.aborted||setupSignal?.aborted)throw abortError();
              diagnostic.failureTrace={...diagnostic.failureTrace,status:'unavailable',error:String(traceError.message).slice(0,1000)};
            }
            diagnostic.failureTrace.wallMs=now()-traceStart;
            throw parityError;
          }
          const eventGateMs=now()-gateStart;
          diagnostic.stage='camera';const cameraStart=now();
          const image=await scoped(async()=>{await dispatch(1,packed.pixels,PLT_GPU_LIMITS.batchPixels,.6,.4);return read(cameraOutput,cameraReadback,packed.pixels);});
          check();const cameraDispatchReadbackWallMs=now()-cameraStart,pixelGateStart=now();pixelValidation=finite?pltPixelFiniteGate(packed,image):pltPixelParityGate(packed,image);const pixelGateMs=now()-pixelGateStart;
          // A visible image cannot escape until every X, Y and Z component has
          // passed. Luminance is extracted from this verified GPU output, never
          // substituted with the CPU reference or post-hoc colorization.
          const spectral=packed.channels===3,resultImage=spectral?Float32Array.from({length:packed.physicalPixels},(_,i)=>image[3*i+1]):image;
          check();return {image:resultImage,...(spectral?{xyz:image,spectralMode:'visible',phaseReduction:packed.phaseReduction}:{}),width:packed.width,height:packed.height,backend:'WebGPU hybrid',label:finite?PLT_GPU_FINITE_SCOPE:spectral?`${PLT_GPU_SCOPE}; sampled XYZ camera sums and every-color-component verification`:PLT_GPU_SCOPE,
            buildId:PLT_GPU_BUILD_ID,adapter:diagnostic.adapter,shaderFingerprintFNV1a:diagnostic.shaderFingerprintFNV1a,
            timings:{gpuSetupWallMs:setupMs,eventDispatchReadbackWallMs,cameraDispatchReadbackWallMs,eventGateMs,pixelGateMs,cpuReferenceProvided:!finite},
            validation:{passed:true,...(finite?{mode:'finite',parityChecked:false,coverage:'every-event-record-and-pixel-finite-safety'}:{coverage:spectral?'every-event-and-every-color-component':'every-event-and-every-pixel'}),events:eventValidation,pixels:pixelValidation},eventPowers:Float32Array.from({length:packed.eventCount},(_,i)=>eventData[8*i+4])};
        }catch(error){error.diagnostic={...diagnostic,error:error.message,validation:{events:eventValidation,pixels:pixelValidation,rejected:error.validation??null}};destroy();throw error;
        }finally{signal?.removeEventListener('abort',onAbort);running=false;}
      }};
  }catch(error){error.diagnostic??={...diagnostic,error:error.message};destroy();throw error;}
}
export async function renderPLTGPU(job,options={}){const engine=await makePLTGPUEngine(job,options);try{return await engine.run(options);}finally{engine.destroy();}}


/** Sequential tile runner: one adapter/device and immutable pipeline pair.
 * Each tile still snapshots its tape, checks EVERY event and pixel, and frees
 * all per-tile GPU buffers. Any failure/cancel/loss poisons this session.
 * No per-tile device.lost closures or validation arrays are retained here.
 */
export function createPLTGPUBatchRunner(options={}){
  const shared={},lifetime=new AbortController(),ownerSignal=options.signal;let closed=false,active=false,engine=null;
  const onOwnerAbort=()=>destroy();
  const destroy=()=>{if(closed)return;closed=true;ownerSignal?.removeEventListener('abort',onOwnerAbort);lifetime.abort();engine?.destroy();engine=null;try{shared.device?.destroy();}catch{}shared.onLost=null;shared.pipelines=null;shared.source=null;};
  ownerSignal?.addEventListener('abort',onOwnerAbort,{once:true});if(ownerSignal?.aborted)destroy();
  return {destroy,async run(job,{signal,onProgress=()=>{}}={}){
    if(closed)throw abortError();if(active)throw new Error('PLT batch GPU runner is sequential');checkSignal(signal);active=true;
    const combined=new AbortController(),abort=()=>combined.abort();
    lifetime.signal.addEventListener('abort',abort,{once:true});signal?.addEventListener('abort',abort,{once:true});
    if(lifetime.signal.aborted||signal?.aborted)combined.abort();
    try{
      engine=await makePLTGPUEngine(job,{...options,signal:combined.signal,_shared:shared});
      const result=await engine.run({signal:combined.signal,onProgress});checkSignal(combined.signal);return result;
    }catch(error){destroy();throw error;}
    finally{engine?.destroy();engine=null;active=false;lifetime.signal.removeEventListener('abort',abort);signal?.removeEventListener('abort',abort);}
  }};
}
