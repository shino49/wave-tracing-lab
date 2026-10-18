import {createPLTScene,orbitCamera,buildPLTJob,validatePLTOptions,PLT_SCOPE,PLT_PROGRESSIVE_LIMITS,createPLTRandomStates,buildPLTBatch,accumulatePLTBatch,emptyPLTStats,mergePLTStats,measurePLTImage} from './plt-core.js';
import {SPECTRAL_SCOPE,spectralDisplaySummary,CIE_Y_INTEGRAL} from './plt-spectrum.js';
async function executeLegacyPLTTask(request,{progress=()=>{},signal,gpuRunner}={}){
 const start=performance.now(),o=validatePLTOptions(request.options),configuration={...request.scene};
 if(!['gallery','prism','baffles','double-edge','micro-gallery','slitbench-demo'].includes(configuration.preset??'gallery'))throw new Error('Unsupported scene');
 if(!['auto','cpu'].includes(request.backend??'auto'))throw new Error('Unsupported backend');
 if(configuration.objText&&typeof configuration.objText!=='string')throw new Error('Invalid OBJ input');
 const scene=createPLTScene({...configuration,spectralMode:o.spectralMode,spectralSource:o.spectralSource,spectralMaterial:o.spectralMaterial}),cam=request.camera??{},camera=orbitCamera(scene.camera,cam.yaw??0,cam.pitch??0,cam.distanceScale??1);
 const modes=request.compare!==false?[true,false]:[o.wave],results=[],warnings=[];
 for(const wave of modes){
  const name=wave?'波动 PLT':'几何光线',job=await buildPLTJob(scene,camera,{...o,wave},{signal,onProgress:p=>progress({text:`${name}：${Math.round(100*p.fraction)}% · ${p.events} 个局部算子`,fraction:(results.length+p.fraction)/modes.length})});
  let image=job.reference,backend='CPU 双精度',validation=null,gpuTimings=null;
  if((request.backend??'auto')==='auto'){
   if(!gpuRunner&&!globalThis.navigator?.gpu)warnings.push(`${name}：Worker 无可用 WebGPU，使用 CPU 双精度。`);
   else try{
    progress({text:`${name}：WebGPU 计算局部散射与相机累积，逐事件、逐像素核验…`,fraction:(results.length+.99)/modes.length});
    const runner=gpuRunner??(await import('./plt-gpu.js')).renderPLTGPU,result=await runner(job,{signal});
    if(!result.validation?.passed||result.image?.length!==job.reference.length||!Array.from(result.image).every(v=>Number.isFinite(v)&&v>=0))throw new Error('GPU 未返回通过完整准入的相机图像');
    image=result.image;validation=result.validation;gpuTimings=result.timings;backend=result.label??'CPU 遍历 + WebGPU 算子与相机';
   }catch(error){if(error.name==='AbortError')throw error;warnings.push(`${name}：GPU 未通过准入，保留 CPU 参考。${error.message}`);validation={passed:false,diagnostic:error.diagnostic??error.diagnostics??null};}
  }
  results.push({wave,width:job.width,height:job.height,image,backend,validation,stats:job.stats,paths:job.paths,multiRegionEvidence:job.multiRegionEvidence,measurement:job.measurement,timings:{...job.timings,gpu:gpuTimings},events:job.events.length,terms:job.terms.length});
 }
 const wave=results.find(r=>r.wave),ray=results.find(r=>!r.wave);let comparison=null;
 if(wave&&ray){let abs=0,norm=0;for(let i=0;i<wave.image.length;i++){abs+=Math.abs(wave.image[i]-ray.image[i]);norm+=Math.abs(ray.image[i]);}comparison={absoluteMeanDifference:abs/wave.image.length,relativeL1:norm>0?abs/norm:null};}
 return {results,warnings,comparison,scope:PLT_SCOPE,options:o,camera,scene:{preset:configuration.preset,triangles:scene.triangles.length,wedges:scene.wedges.length,unsupportedEdges:scene.unsupportedEdges.length,emitters:scene.emitters.length},totalMs:performance.now()-start,privacy:'OBJ remains in browser memory; no model upload; no server computation'};
}

const clock=()=>globalThis.performance?.now?.()??Date.now();
const cancelled=()=>Object.assign(new Error('Render cancelled'),{name:'AbortError'});
const integer=(v,min,max,name)=>{if(!Number.isInteger(v)||v<min||v>max)throw new RangeError(`${name} must be an integer within ${min}…${max}`);return v;};
export function validatePLTProgressiveOptions(value={}){
 if(value===true||value===undefined)value={};if(!value||typeof value!=='object'||Array.isArray(value))throw new RangeError('Invalid progressive configuration');
 const o={tileSize:32,batchSamples:1,maxRuntimeMs:120000,previewIntervalMs:100,...value};
 integer(o.tileSize,1,PLT_PROGRESSIVE_LIMITS.maxTileSize,'Tile size');integer(o.batchSamples,1,PLT_PROGRESSIVE_LIMITS.maxBatchSamples,'Sample batch');
 integer(o.maxRuntimeMs,1,PLT_PROGRESSIVE_LIMITS.maxRuntimeMs,'Runtime cap');integer(o.previewIntervalMs,0,5000,'Preview interval');return o;
}
// Validation may contain a full array per event/pixel; never retain those arrays
// across batches or copy them into messages/reports. Diagnostics are bounded too.
function awaitBatch(promise,signal){
 return new Promise((resolve,reject)=>{const abort=()=>reject(signal.reason?.name==='TimeoutError'?signal.reason:cancelled());signal.addEventListener('abort',abort,{once:true});if(signal.aborted)abort();Promise.resolve(promise).then(value=>{signal.removeEventListener('abort',abort);resolve(value);},error=>{signal.removeEventListener('abort',abort);reject(error);});});
}
function compactDiagnostic(value,depth=0,budget={left:256}){
 if(value===null||typeof value!=='object')return typeof value==='string'?value.slice(0,1000):value;
 if(depth>5||budget.left--<=0)return '[omitted]';
 if(ArrayBuffer.isView(value))return {type:value.constructor.name,length:value.length};
 if(Array.isArray(value))return value.slice(0,8).map(v=>compactDiagnostic(v,depth+1,budget));
 const out={};for(const [key,v] of Object.entries(value).slice(0,32)){if(['absoluteErrors','toleranceRatios','eventPowers'].includes(key))continue;
  // The trace producer has fixed 8-event / 2-edge caps. Preserve its nested
  // schema; the generic summary's depth cap would discard the useful lanes.
  if(key==='failureTrace'&&v?.schema==='plt-event-failure-trace-v1'){
   try{const encoded=JSON.stringify(v);out[key]=encoded.length<=131072?JSON.parse(encoded):{schema:v.schema,status:'unavailable',error:'Trace exceeded report byte budget'};}catch{out[key]={schema:v.schema,status:'unavailable',error:'Trace serialization failed'};}continue;
  }
  out[key]=compactDiagnostic(v,depth+1,budget);}return out;
}
function addGateSummary(target,gate,fallbackCount){
 target.count+=gate?.count??fallbackCount;
 for(const key of ['nonfinite','negative','numericalFailures','invalidEventRecords'])target[key]=(target[key]??0)+(gate?.[key]??0);
 for(const key of ['maxAbsoluteError','maxToleranceRatio'])target[key]=Math.max(target[key]??0,gate?.[key]??0);
 if(gate?.tolerance)target.tolerance={...gate.tolerance};
}
function validationSummary(state){return state.validation??= {passed:true,coverage:'every-event-and-every-pixel per accepted batch',acceptedBatches:0,rejectedBatches:0,cpuFallbackBatches:0,events:{count:0},pixels:{count:0}};}
function addTimings(target,source){for(const [key,value] of Object.entries(source??{}))if(typeof value==='number'&&Number.isFinite(value))target[key]=(target[key]??0)+value;}
function snapshotMode(state,o,gpuFailed,partial){
 const useGPU=state.gpuBatches>0&&!gpuFailed,sums=useGPU?state.gpuSums:state.cpuSums,image=new Float64Array(sums.length),sampleCounts=state.sampleCounts.slice();let min=Infinity,max=0;
 for(let i=0;i<image.length;i++){const count=sampleCounts[i];min=Math.min(min,count);max=Math.max(max,count);image[i]=count===o.samplesPerPixel?sums[i]:count?sums[i]*(o.samplesPerPixel/count):0;if(!Number.isFinite(image[i])||image[i]<0)throw new Error('Invalid accumulated camera response');}
 let xyz=null;if(o.spectralMode==='visible'){const source=useGPU?state.gpuXYZ:state.cpuXYZ;xyz=new Float64Array(source.length);for(let i=0;i<sampleCounts.length;i++){const n=sampleCounts[i];for(let c=0;c<3;c++){const v=n===o.samplesPerPixel?source[3*i+c]:n?source[3*i+c]*(o.samplesPerPixel/n):0;if(!Number.isFinite(v)||v<0)throw new Error('Invalid accumulated XYZ');xyz[3*i+c]=v;}image[i]=xyz[3*i+1];}}
 return {wave:state.wave,width:o.width,height:o.height,image,...(xyz?{xyz,spectralMode:'visible'}:{}),sampleCounts,completedSamplesPerPixel:min,minSamplesPerPixel:min,maxSamplesPerPixel:max,partial,backend:useGPU?state.gpuLabel:'CPU 双精度'};
}
function spectralNoiseSummary(state,o){
 const n=o.samplesPerPixel,meanVarianceXYZ=[0,0,0],maxVarianceXYZ=[0,0,0],pixels=o.width*o.height;
 if(n>1)for(let i=0;i<pixels;i++)for(let c=0;c<3;c++){const mean=state.cpuXYZ[3*i+c],variance=Math.max(0,(state.spectralSquares[3*i+c]-n*mean*mean)/(n-1));meanVarianceXYZ[c]+=variance/pixels;maxVarianceXYZ[c]=Math.max(maxVarianceXYZ[c],variance);}
 return {kind:'descriptive per-path sample variance; correlated wavelength sampling, not confidence interval',meanVarianceXYZ,maxVarianceXYZ,samplesPerPixel:n,reference:'CPU canonical samples',available:n>1};
}
/** Progressive execution retains at most one tile tape. Each mode uses its own
 * full-frame uint RNG state, actual completed sample counts, and image sums.
 * Preview publication occurs only AFTER exhaustive GPU admission or CPU fallback.
 * An interrupted tile never changes counts or publishes its partial contents. */
export async function executePLTTask(request,{progress=()=>{},preview=()=>{},signal,gpuRunner}={}){
 if(request.progressive===false){if(request.options?.spectralMode==='visible')throw new Error('Visible spectrum requires bounded progressive execution');if(['resident-shadow','resident-recovery'].includes(request.backend))throw new Error('Resident shadow backend requires bounded progressive execution');return executeLegacyPLTTask(request,{progress,signal,gpuRunner});}
 const start=clock(),o=validatePLTOptions(request.options,{progressive:true}),po=validatePLTProgressiveOptions(request.progressive),configuration={...request.scene};
 const check=()=>{if(signal?.aborted)throw signal.reason?.name==='TimeoutError'?signal.reason:cancelled();if(clock()-start>po.maxRuntimeMs)throw Object.assign(new Error(`Rendering reached its ${po.maxRuntimeMs/1000}s runtime cap; reduce resolution, samples, or depth. No incomplete result was finalized.`),{name:'TimeoutError'});};
 check();
 if(!['gallery','prism','baffles','double-edge','micro-gallery','slitbench-demo'].includes(configuration.preset??'gallery'))throw new Error('Unsupported scene');
 if(!['auto','cpu','resident-shadow','resident-recovery'].includes(request.backend??'auto'))throw new Error('Unsupported backend');
 if(['resident-shadow','resident-recovery'].includes(request.backend)&&o.spectralMode==='visible')throw new Error('Experimental resident shadow has no spectral admission; select the stable auto or CPU backend');
 if(configuration.objText&&typeof configuration.objText!=='string')throw new Error('Invalid OBJ input');
 const scene=createPLTScene({...configuration,spectralMode:o.spectralMode,spectralSource:o.spectralSource,spectralMaterial:o.spectralMaterial}),cam=request.camera??{},camera=orbitCamera(scene.camera,cam.yaw??0,cam.pitch??0,cam.distanceScale??1);check();
 const visible=o.spectralMode==='visible',modes=request.compare!==false?[true,false]:[o.wave],pixels=o.width*o.height,warnings=[],states=modes.map(wave=>({wave,rngStates:createPLTRandomStates(o),sampleCounts:new Uint16Array(pixels),cpuSums:new Float64Array(pixels),gpuSums:new Float64Array(pixels),...(visible?{cpuXYZ:new Float64Array(pixels*3),gpuXYZ:new Float64Array(pixels*3),spectralSquares:new Float64Array(pixels*3)}:{}),stats:emptyPLTStats(),paths:[],multiRegionEvidence:[],events:0,terms:0,validation:null,gpuBatches:0,gpuLabel:'CPU 遍历 + WebGPU 算子与相机',timings:{cpuTraversalAndReferenceMs:0,gpu:null}}));
 const tiles=[];for(let y=0;y<o.height;y+=po.tileSize)for(let x=0;x<o.width;x+=po.tileSize)tiles.push({x,y,width:Math.min(po.tileSize,o.width-x),height:Math.min(po.tileSize,o.height-y)});
 const totalSamples=pixels*o.samplesPerPixel*modes.length,totalBatches=Math.ceil(o.samplesPerPixel/po.batchSamples)*tiles.length*modes.length;
 let completedSamples=0,completedBatches=0,completedPasses=0,lastPreview=-Infinity,gpuSession=null,gpuFailure=null,firstPreviewMs=null,firstFullPassMs=null,lastProgress=-Infinity;
 const externalSignal=signal,controller=new AbortController(),onAbort=()=>controller.abort();externalSignal?.addEventListener('abort',onAbort,{once:true});signal=controller.signal;
 const deadlineTimer=setTimeout(()=>controller.abort(Object.assign(new Error(`Rendering reached its ${po.maxRuntimeMs/1000}s runtime cap; reduce resolution, samples, or depth. No incomplete result was finalized.`),{name:'TimeoutError'})),Math.max(1,po.maxRuntimeMs-(clock()-start)));if(externalSignal?.aborted)controller.abort();
 const wantsResidentRecovery=request.backend==='resident-recovery',wantsResident=wantsResidentRecovery||request.backend==='resident-shadow',wantsGPU=wantsResident||(request.backend??'auto')==='auto',gpuAvailable=!!gpuRunner||!!globalThis.navigator?.gpu;
 let residentReference=null,residentRecovery=null;
 if(wantsResidentRecovery)warnings.push('逐路径 CPU 回退实验：未建模误差的后续传播先回退 CPU；GPU 不确定路径整条重放，剩余数值漂移仍依赖完整 CPU 影子审核发现。默认关闭，尚无生产级免审核保证。');
 if(wantsResident)warnings.push('GPU 驻留实验仍逐路径、逐随机数、逐事件及逐像素执行完整 CPU 审核；总耗时包含双重计算，尚无免审核快速模式。');
 if(wantsGPU&&!gpuAvailable)for(const wave of modes)warnings.push(`${wave?'波动 PLT':'几何光线'}：Worker 无可用 WebGPU，使用 CPU 双精度。`);
 const peaks={events:0,terms:0,edgeRecords:0,tilePixels:0,batchPaths:0,retainedPaths:0,retainedEvidence:0};
 const notifyProgress=(text,fraction,force=false)=>{check();if(force||clock()-lastProgress>=100){progress({text,fraction,completedSamplesPerPixel:completedPasses,targetSamplesPerPixel:o.samplesPerPixel,completedBatches,totalBatches});lastProgress=clock();check();}};
 const notifyPreview=(batch,force=false)=>{check();if(!force&&clock()-lastPreview<po.previewIntervalMs)return;const results=states.map(s=>snapshotMode(s,o,!!gpuFailure,true));const completedSamplesPerPixel=Math.min(...results.map(r=>r.completedSamplesPerPixel));const elapsedMs=clock()-start;preview({results,width:o.width,height:o.height,completedSamplesPerPixel,targetSamplesPerPixel:o.samplesPerPixel,batch,elapsedMs,fraction:completedSamples/totalSamples,partial:true});lastPreview=clock();firstPreviewMs??=elapsedMs;check();};
 try{
  residentReference=wantsResident?await import('./plt-resident-reference.js'):null;residentRecovery=wantsResidentRecovery?await import('./plt-resident-recovery.js'):null;check();
  for(let sampleStart=0;sampleStart<o.samplesPerPixel;sampleStart+=po.batchSamples){
   const batchSamples=Math.min(po.batchSamples,o.samplesPerPixel-sampleStart);
   for(const tile of tiles)for(const state of states){
    check();const name=state.wave?'波动 PLT':'几何光线',batch={index:completedBatches+1,total:totalBatches,tile:{...tile},sampleStart,samplesPerPixel:batchSamples,wave:state.wave};
    notifyProgress(`${name}：分块 ${completedBatches+1}/${totalBatches} · 完整 ${completedPasses}/${o.samplesPerPixel} spp`,completedSamples/totalSamples,true);
    // Every pixel enters a pass with the same completed sample prefix. A tile's
    // RNG is resumed, never re-seeded at a sample-batch boundary.
    for(let y=tile.y;y<tile.y+tile.height;y++)for(let x=tile.x;x<tile.x+tile.width;x++)if(state.sampleCounts[y*o.width+x]!==sampleStart)throw new Error('Progressive sample order invariant violated');
    const residentPrefixes=wantsResident&&gpuAvailable&&!gpuFailure?state.rngStates.slice():null;
    const buildBatch=residentPrefixes?residentReference.buildResidentReferenceBatch:buildPLTBatch;
    const job=await buildBatch(scene,camera,{...o,wave:state.wave},{tile,sampleStart,samplesPerPixel:batchSamples,rngStates:state.rngStates,signal,check,onProgress:p=>notifyProgress(`${name}：分块 ${completedBatches+1}/${totalBatches} · ${Math.round(p.fraction*100)}% · 完整 ${completedPasses}/${o.samplesPerPixel} spp`,completedSamples/totalSamples)});check();
    peaks.events=Math.max(peaks.events,job.events.length);peaks.terms=Math.max(peaks.terms,job.terms.length);peaks.edgeRecords=Math.max(peaks.edgeRecords,job.edgeRecords);peaks.tilePixels=Math.max(peaks.tilePixels,job.pixelIndices.length);peaks.batchPaths=Math.max(peaks.batchPaths,job.pixelIndices.length*batchSamples);
    let acceptedImage=null,acceptedXYZ=null;
    if(wantsGPU&&gpuAvailable&&!gpuFailure){
     try{
      notifyProgress(`${name}：逐事件、逐像素核验当前分块…`,completedSamples/totalSamples,true);
      let result;
      if(wantsResident){
       if(gpuRunner)result=await awaitBatch(gpuRunner({scene,camera,options:{...o,wave:state.wave},tile,sampleStart,samplesPerPixel:batchSamples,rngStates:residentPrefixes},{signal}),signal);
       else{if(!gpuSession){const module=await import('./plt-resident-gpu.js');check();gpuSession=await module.createPLTResidentRunner(scene,camera,o,{signal,recoveryMode:wantsResidentRecovery?'selective':'strict'});check();}result=await awaitBatch(gpuSession.run({tile,sampleStart,samplesPerPixel:batchSamples,wave:state.wave,rngStates:residentPrefixes,signal}),signal);}
       check();const auditStart=clock();
       if(wantsResidentRecovery)result=residentRecovery.admitResidentWithRecovery(job,result);
       const audit=wantsResidentRecovery?result.validation:residentReference.assertResidentAudit(job,result.audit);
       // Candidate pixels cannot be substituted after the audited buffer. The
       // canonical decoder's exact image object is the only admitted image.
       if(result.image!==result.audit.reference)throw new Error('Resident candidate image is not its audited pixel buffer');
       if(result.audit.pixelIndices.length!==job.pixelIndices.length||result.audit.pixelIndices.some((id,i)=>id!==job.pixelIndices[i]))throw new Error('Resident audited pixel order differs from the committed tile');
       result={...result,label:wantsResidentRecovery?'GPU 驻留 + 逐路径 CPU 回退实验 + 完整审核':'GPU 驻留实验 + 完整 CPU 逐路径审核',validation:{passed:true,coverage:'every-path-rng-event-and-pixel',events:{count:audit.checked.events},pixels:{count:audit.checked.pixels},audit},timings:{...result.timings,cpuAuditGateMs:clock()-auditStart,fullCPUShadow:true}};
      }else{
       let runner=gpuRunner;
       if(!runner){if(!gpuSession){const module=await import('./plt-gpu.js');check();gpuSession=await module.createPLTGPUBatchRunner({signal});check();}runner=(job,options)=>gpuSession.run(job,options);}
       result=await awaitBatch(runner(job,{signal}),signal);check();
      }
      if(!result.validation?.passed||result.image?.length!==job.reference.length||!result.image.every(v=>Number.isFinite(v)&&v>=0))throw new Error('GPU 未返回通过完整准入的相机分块');
      if(visible&&(!(result.xyz instanceof Float32Array||result.xyz instanceof Float64Array)||result.xyz.length!==job.referenceXYZ.length||!result.xyz.every(v=>Number.isFinite(v)&&v>=0)||result.image.some((v,i)=>v!==result.xyz[3*i+1])))throw new Error('GPU 未返回经过逐分量核验的 XYZ 分块');
      acceptedImage=result.image;acceptedXYZ=result.xyz??null;const validation=validationSummary(state);validation.acceptedBatches++;addGateSummary(validation.events,result.validation.events,job.events.length);addGateSummary(validation.pixels,result.validation.pixels,job.reference.length);
      if(wantsResidentRecovery){validation.recovery??={gpuPredicatePaths:0,coverageGuardPaths:0,auditGeometryPaths:0,fullCPUShadow:true,productionGuard:false};const numerical=result.recovery?.numericGeometry?.count??0,guarded=result.recovery?.keys?.filter(entry=>entry.reasons?.includes('unmodelledContinuation')).length??0;validation.recovery.gpuPredicatePaths+=(result.recovery?.paths??0)-numerical-guarded;validation.recovery.coverageGuardPaths+=guarded;validation.recovery.auditGeometryPaths+=numerical;validation.recovery.continuationGuard=result.recovery?.continuationGuard??'unknown';}
      if(wantsResident){validation.coverage='every-path-rng-scatter-edge-event-and-pixel per accepted batch';validation.fullCPUShadow=true;validation.auditCounts??={};for(const [key,value] of Object.entries(result.validation.audit.checked))validation.auditCounts[key]=(validation.auditCounts[key]??0)+value;}
      if(result.phaseReduction){validation.phaseReduction??={...result.phaseReduction,maxAbsoluteRawPhase:0,maxAbsolutePackedPhase:0,reducedEdges:0};validation.phaseReduction.maxAbsoluteRawPhase=Math.max(validation.phaseReduction.maxAbsoluteRawPhase,result.phaseReduction.maxAbsoluteRawPhase);validation.phaseReduction.maxAbsolutePackedPhase=Math.max(validation.phaseReduction.maxAbsolutePackedPhase,result.phaseReduction.maxAbsolutePackedPhase);validation.phaseReduction.reducedEdges+=result.phaseReduction.reducedEdges;}
      state.gpuBatches++;state.gpuLabel=result.label??state.gpuLabel;state.timings.gpu??={};addTimings(state.timings.gpu,result.timings);
     }catch(error){
      if(error.name==='AbortError'||error.name==='TimeoutError'){check();throw error;}check();gpuFailure={message:error.message,diagnostic:compactDiagnostic(error.diagnostic??error.diagnostics??error.validation??error.audit??null)};
      const validation=validationSummary(state);validation.passed=false;validation.rejectedBatches++;validation.diagnostic=gpuFailure.diagnostic;
      try{gpuSession?.destroy();}catch{}gpuSession=null;
      warnings.push(`GPU 未通过当前分块准入，整幅图像使用 CPU 双精度参考并停止本次 GPU 尝试。${error.message}`);
     }
    }
    check();accumulatePLTBatch(job,state.cpuSums,o.samplesPerPixel,state.cpuXYZ);
    if(visible){for(let i=0;i<job.pixelIndices.length;i++)for(let sample=0;sample<batchSamples;sample++)for(let c=0;c<3;c++){const v=job.spectralSampleValues[3*(i*batchSamples+sample)+c],index=3*job.pixelIndices[i]+c;state.spectralSquares[index]+=v*v;if(!Number.isFinite(state.spectralSquares[index]))throw new Error('Spectral variance overflow');}}
    if(acceptedImage)for(let i=0;i<acceptedImage.length;i++){state.gpuSums[job.pixelIndices[i]]+=acceptedImage[i]*(batchSamples/o.samplesPerPixel);if(visible)for(let c=0;c<3;c++)state.gpuXYZ[3*job.pixelIndices[i]+c]+=acceptedXYZ[3*i+c]*(batchSamples/o.samplesPerPixel);}
    if(gpuFailure)for(const item of states){const validation=validationSummary(item);validation.passed=false;validation.diagnostic??=gpuFailure.diagnostic;}
    if(wantsGPU&&gpuAvailable&&!acceptedImage)validationSummary(state).cpuFallbackBatches++;
    for(const globalPixel of job.pixelIndices)state.sampleCounts[globalPixel]+=batchSamples;
    mergePLTStats(state.stats,job.stats);state.events+=job.events.length;state.terms+=job.terms.length;state.timings.cpuTraversalAndReferenceMs+=job.timings.cpuTraversalAndReferenceMs;
    for(const path of job.paths)if(state.paths.length<12&&(path.diffractionCount>=2||state.paths.length<3))state.paths.push(path);
    for(const evidence of job.multiRegionEvidence)if(state.multiRegionEvidence.length<8)state.multiRegionEvidence.push({...evidence,batch:completedBatches+1});
    peaks.retainedPaths=Math.max(peaks.retainedPaths,states.reduce((s,item)=>s+item.paths.length,0));peaks.retainedEvidence=Math.max(peaks.retainedEvidence,states.reduce((s,item)=>s+item.multiRegionEvidence.length,0));
    completedSamples+=job.pixelIndices.length*batchSamples;completedBatches++;
    const passComplete=tile===tiles.at(-1)&&state===states.at(-1);if(passComplete){completedPasses=sampleStart+batchSamples;firstFullPassMs??=clock()-start;}
    notifyPreview(batch,passComplete);notifyProgress(`${name}：已完成分块 ${completedBatches}/${totalBatches} · 完整 ${completedPasses}/${o.samplesPerPixel} spp`,completedSamples/totalSamples,true);
    // Yield after a committed batch, allowing cancellation/new Worker requests.
    await new Promise(resolve=>setTimeout(resolve,0));check();
   }
  }
  check();const results=states.map(state=>{const result=snapshotMode(state,o,!!gpuFailure,false);if(result.minSamplesPerPixel!==o.samplesPerPixel||result.maxSamplesPerPixel!==o.samplesPerPixel)throw new Error('Incomplete sample pass cannot finalize');return {...result,validation:state.validation,stats:state.stats,paths:state.paths,multiRegionEvidence:state.multiRegionEvidence,measurement:{...measurePLTImage(result.image),...(visible?{unit:'relative spectral PLT CIE Y',spectralNormalization:'integral over 360–830 nm divided by fixed CIE y integral; source SPD sampled at each hero wavelength; not calibrated radiometry'}:{})},...(visible?{spectralNoise:spectralNoiseSummary(state,o),displayGamut:spectralDisplaySummary(result.xyz)}:{}),timings:state.timings,events:state.events,terms:state.terms};});
  const wave=results.find(r=>r.wave),ray=results.find(r=>!r.wave);let comparison=null;
  if(wave&&ray){let abs=0,norm=0;for(let i=0;i<wave.image.length;i++){abs+=Math.abs(wave.image[i]-ray.image[i]);norm+=Math.abs(ray.image[i]);}comparison={absoluteMeanDifference:abs/wave.image.length,relativeL1:norm>0?abs/norm:null};}
  return {results,warnings,comparison,...(visible?{colorimetry:{observer:'CIE 1931 2 degree',wavelengthBandNm:[360,830],wavelengthPDFPerNm:1/470,fixedCIEYIntegralNm:CIE_Y_INTEGRAL,monochromeWavelengthInputIgnored:true,source:o.spectralSource,surfaceSpectrum:o.spectralMaterial,camera:'visible Gaussian-envelope specialization: waist floor 0.003 mm; quadrature divergence; slope guard 0.15',measurement:'relative spectral native-PLT response; no calibrated radiometry or global energy claim',display:'XYZ average then linear sRGB; display-only negative gamut clip, fixed exponential compression, standard sRGB transfer'}}:{}),scope:visible?`${PLT_SCOPE}; ${SPECTRAL_SCOPE}`:PLT_SCOPE,options:o,camera,scene:{physicalScale:scene.physicalScale,preset:configuration.preset??'gallery',triangles:scene.triangles.length,wedges:scene.wedges.length,unsupportedEdges:scene.unsupportedEdges.length,emitters:scene.emitters.length},totalMs:clock()-start,progressive:{...po,completedBatches,totalBatches,completedSamplesPerPixel:completedPasses,firstPreviewMs,firstFullPassMs,peakTape:peaks,persistentPixelBytes:modes.length*pixels*(4+2+8+8+(visible?72:0)),sampleSequence:visible?'shifted base-2 wavelength sequence and independent path RNG keyed by seed/globalPixel/absoluteSample; paired wave/ray camera draws':'v11 sequential per-global-pixel RNG state; exact CPU term order',limits:PLT_PROGRESSIVE_LIMITS},privacy:'OBJ remains in browser memory; no model upload; no server computation'};
 }finally{clearTimeout(deadlineTimer);externalSignal?.removeEventListener('abort',onAbort);try{gpuSession?.destroy();}catch{}}
}
