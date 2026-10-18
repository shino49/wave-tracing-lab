/** Practical monochrome 3D PLT mode. CPU constructs geometry and sampling tapes;
 * WebGPU evaluates local UTD powers and camera sums without a CPU shadow image.
 * Safety checks remain mandatory. DEBUG uses the unchanged strict runtime. */
import {createPLTScene,orbitCamera,validatePLTOptions,PLT_SCOPE,PLT_PROGRESSIVE_LIMITS,createPLTRandomStates,buildPLTBatch,emptyPLTStats,mergePLTStats,measurePLTImage,materializePLTBatchReference,summarizePLTBatchOperators,replayPLT} from './plt-core.js';
import {executePLTTask,validatePLTProgressiveOptions} from './plt-runtime.js';
export const PLT_DEMO_BUILD_ID='plt-demo-deferred-2026-10-13.1';
export const PLT_DEMO_SCOPE='Native 3D PLT approximation; CPU geometry and sampling; optional WebGPU local coherent UTD and positive camera sums; no global coherent field or calibrated radiance';
const now=()=>globalThis.performance?.now?.()??Date.now();
const aborted=signal=>signal?.reason?.name==='TimeoutError'?signal.reason:Object.assign(new Error('Demo render cancelled'),{name:'AbortError'});
const finiteArray=(value,length)=>((value instanceof Float32Array||value instanceof Float64Array)&&value.length===length&&value.every(v=>Number.isFinite(v)&&v>=0));
function awaitBatch(promise,signal){
 return new Promise((resolve,reject)=>{const abort=()=>reject(aborted(signal));signal.addEventListener('abort',abort,{once:true});if(signal.aborted)abort();Promise.resolve(promise).then(value=>{signal.removeEventListener('abort',abort);resolve(value);},error=>{signal.removeEventListener('abort',abort);reject(error);});});
}
function addTimings(target,source){for(const [key,value]of Object.entries(source??{}))if(typeof value==='number'&&Number.isFinite(value))target[key]=(target[key]??0)+value;}
function backend(state){if(state.gpuBatches&&state.cpuBatches)return '混合：已完成 GPU 分块 + CPU 后续分块';if(state.gpuBatches)return 'CPU 遍历与采样 + WebGPU UTD 与相机';return 'CPU 原生 3D UTD';}
function snapshot(state,o,partial){
 const image=new Float64Array(state.sums.length),sampleCounts=state.sampleCounts.slice();let min=Infinity,max=0;
 for(let i=0;i<image.length;i++){const n=sampleCounts[i];min=Math.min(min,n);max=Math.max(max,n);image[i]=n===o.samplesPerPixel?state.sums[i]:n?state.sums[i]*o.samplesPerPixel/n:0;if(!Number.isFinite(image[i])||image[i]<0)throw new Error('Invalid accumulated demo camera response');}
 let utdOffImage=null;if(state.controlSums){utdOffImage=new Float64Array(image.length);for(let i=0;i<image.length;i++){const n=sampleCounts[i];utdOffImage[i]=n===o.samplesPerPixel?state.controlSums[i]:n?state.controlSums[i]*o.samplesPerPixel/n:0;if(!Number.isFinite(utdOffImage[i])||utdOffImage[i]<0)throw new Error('Invalid direct-only control accumulation');}}
 return {wave:state.wave,width:o.width,height:o.height,image,...(utdOffImage?{utdOffImage}:{}),sampleCounts,completedSamplesPerPixel:min,minSamplesPerPixel:min,maxSamplesPerPixel:max,partial,backend:backend(state),executionMode:'demo',parityChecked:false};
}
function checkGPUResult(result,job){
 if(!result||result.width!==job.width||result.height!==job.height||!finiteArray(result.image,job.pixelIndices.length))throw new Error('GPU returned an invalid camera tile');
 if(result.validation?.passed!==true||result.validation.mode!=='finite'||result.validation.parityChecked!==false||result.validation.events?.count!==job.events.length||result.validation.pixels?.count!==job.pixelIndices.length)throw new Error('GPU did not return the required finite-safety record');
 if(!finiteArray(result.eventPowers,job.events.length))throw new Error('GPU returned invalid local operator powers');
 // Own immutable copies before a batch is committed or exposed in a preview.
 return {image:result.image.slice(),eventPowers:result.eventPowers.slice()};
}
export async function executePLTDemoTask(request={}, {progress=()=>{},preview=()=>{},signal:externalSignal,gpuRunner}={}){
 if(!request||typeof request!=='object'||Array.isArray(request))throw new TypeError('Invalid demo request');
 if(request.utdAblation!==undefined&&typeof request.utdAblation!=='boolean')throw new TypeError('utdAblation must be boolean');
 if(request.debugValidation!==undefined&&typeof request.debugValidation!=='boolean')throw new TypeError('debugValidation must be boolean');
 if(request.debugValidation===true){const result=await executePLTTask(request,{progress,preview,signal:externalSignal,gpuRunner});return {...result,executionMode:'debug-strict',demoBuildId:PLT_DEMO_BUILD_ID};}
 const started=now(),o=validatePLTOptions(request.options,{progressive:true}),po=validatePLTProgressiveOptions(request.progressive);
 if(o.spectralMode==='visible')throw new Error('Normal demo mode is monochrome; use the spectral renderer or explicit DEBUG validation');
 if(!['auto','cpu'].includes(request.backend??'auto'))throw new Error('Unsupported demo backend');
 const configuration={...request.scene};
 if(!['gallery','prism','baffles','double-edge','micro-gallery','slitbench-demo'].includes(configuration.preset??'gallery'))throw new Error('Unsupported demo scene');
 if(configuration.objText&&typeof configuration.objText!=='string')throw new Error('Invalid OBJ input');
 const controller=new AbortController(),signal=controller.signal,externalAbort=()=>controller.abort(externalSignal?.reason);
 externalSignal?.addEventListener('abort',externalAbort,{once:true});if(externalSignal?.aborted)externalAbort();
 const timeout=()=>Object.assign(new Error(`Rendering reached its ${po.maxRuntimeMs/1000}s runtime cap; reduce resolution, samples, or depth. No incomplete result was finalized.`),{name:'TimeoutError'});
 const check=()=>{if(signal.aborted)throw aborted(signal);if(now()-started>po.maxRuntimeMs)throw timeout();};
 const timer=setTimeout(()=>controller.abort(timeout()),Math.max(1,po.maxRuntimeMs-(now()-started)));
 let session=null;
 try{
  check();const scene=createPLTScene(configuration),cam=request.camera??{},camera=orbitCamera(scene.camera,cam.yaw??0,cam.pitch??0,cam.distanceScale??1);check();
  const modes=request.compare!==false?[true,false]:[o.wave],pixels=o.width*o.height,warnings=[];
  const states=modes.map(wave=>({wave,rngStates:createPLTRandomStates(o),sampleCounts:new Uint16Array(pixels),sums:new Float64Array(pixels),controlSums:request.utdAblation===true&&wave?new Float64Array(pixels):null,cpuControlImages:0,stats:emptyPLTStats(),paths:[],multiRegionEvidence:[],events:0,terms:0,cpuBatches:0,gpuBatches:0,cpuOperatorEvaluations:0,cpuReferenceImages:0,timings:{cpuTraversalMs:0,cpuOperatorAndCameraMs:0,cpuDirectOnlyControlMs:0,gpu:null},validation:{passed:true,mode:'finite',parityChecked:false,coverage:'bounded-input-and-finite-output-safety-only',acceptedBatches:0,rejectedGPUBatches:0,cpuFallbackBatches:0,events:{count:0},pixels:{count:0}}}));
  const wantsGPU=(request.backend??'auto')==='auto',gpuAvailable=!!gpuRunner||!!globalThis.navigator?.gpu;
  let gpuFailure=null,gpuDisabled=!wantsGPU||!gpuAvailable;
  if(wantsGPU&&!gpuAvailable)warnings.push('Worker 无可用 WebGPU；采用 CPU 原生 3D UTD。没有执行 GPU，也没有重复核验。');
  const tiles=[];for(let y=0;y<o.height;y+=po.tileSize)for(let x=0;x<o.width;x+=po.tileSize)tiles.push({x,y,width:Math.min(po.tileSize,o.width-x),height:Math.min(po.tileSize,o.height-y)});
  const totalSamples=pixels*o.samplesPerPixel*modes.length,totalBatches=Math.ceil(o.samplesPerPixel/po.batchSamples)*tiles.length*modes.length;
  const peaks={events:0,terms:0,edgeRecords:0,tilePixels:0,batchPaths:0,retainedPaths:0,retainedEvidence:0};
  let completedSamples=0,completedBatches=0,completedPasses=0,lastPreview=-Infinity,lastProgress=-Infinity,firstPreviewMs=null,firstFullPassMs=null;
  const notifyProgress=(text,force=false)=>{check();if(force||now()-lastProgress>=100){progress({text,fraction:completedSamples/totalSamples,completedSamplesPerPixel:completedPasses,targetSamplesPerPixel:o.samplesPerPixel,completedBatches,totalBatches});lastProgress=now();check();}};
  const notifyPreview=(batch,force=false)=>{check();if(!force&&now()-lastPreview<po.previewIntervalMs)return;const results=states.map(state=>snapshot(state,o,true)),elapsedMs=now()-started;preview({results,width:o.width,height:o.height,completedSamplesPerPixel:Math.min(...results.map(r=>r.completedSamplesPerPixel)),targetSamplesPerPixel:o.samplesPerPixel,batch,elapsedMs,fraction:completedSamples/totalSamples,partial:true,executionMode:'demo',parityChecked:false});lastPreview=now();firstPreviewMs??=elapsedMs;check();};
  for(let sampleStart=0;sampleStart<o.samplesPerPixel;sampleStart+=po.batchSamples){
   const batchSamples=Math.min(po.batchSamples,o.samplesPerPixel-sampleStart);
   for(const tile of tiles)for(const state of states){
    check();const name=state.wave?'波动 PLT':'几何光线',batch={index:completedBatches+1,total:totalBatches,tile:{...tile},sampleStart,samplesPerPixel:batchSamples,wave:state.wave};
    notifyProgress(`${name}：分块 ${completedBatches+1}/${totalBatches} · 完整 ${completedPasses}/${o.samplesPerPixel} spp`,true);
    for(let y=tile.y;y<tile.y+tile.height;y++)for(let x=tile.x;x<tile.x+tile.width;x++)if(state.sampleCounts[y*o.width+x]!==sampleStart)throw new Error('Demo sample order invariant violated');
    const job=await buildPLTBatch(scene,camera,{...o,wave:state.wave},{tile,sampleStart,samplesPerPixel:batchSamples,rngStates:state.rngStates,deferOperatorEvaluation:true,signal,check,onProgress:p=>notifyProgress(`${name}：构建 3D 路径 ${Math.round(p.fraction*100)}% · 分块 ${completedBatches+1}/${totalBatches}`)});check();
    peaks.events=Math.max(peaks.events,job.events.length);peaks.terms=Math.max(peaks.terms,job.terms.length);peaks.edgeRecords=Math.max(peaks.edgeRecords,job.edgeRecords);peaks.tilePixels=Math.max(peaks.tilePixels,job.pixelIndices.length);peaks.batchPaths=Math.max(peaks.batchPaths,job.pixelIndices.length*batchSamples);
    let image=null,usedGPU=false;
    if(!gpuDisabled){
     try{
      notifyProgress(`${name}：WebGPU 局部 UTD 与相机累积 · DEBUG 对照关闭`,true);
      let result;
      if(gpuRunner){const submitted=structuredClone(job);result=await awaitBatch(gpuRunner(submitted,{signal,validationMode:'finite'}),signal);}
      else{if(!session){const module=await import('./plt-gpu.js');check();session=module.createPLTGPUBatchRunner({signal,validationMode:'finite'});}result=await awaitBatch(session.run(job,{signal}),signal);}
      check();const accepted=checkGPUResult(result,job);summarizePLTBatchOperators(job,accepted.eventPowers);image=accepted.image;usedGPU=true;
      state.timings.gpu??={};addTimings(state.timings.gpu,result.timings);
     }catch(error){
      if(error.name==='AbortError'||error.name==='TimeoutError'){check();throw error;}check();gpuDisabled=true;gpuFailure={message:String(error.message??error).slice(0,1000),stage:typeof error.diagnostic?.stage==='string'?error.diagnostic.stage.slice(0,100):null};
      state.validation.rejectedGPUBatches++;try{session?.destroy();}catch{}session=null;
      warnings.push(`GPU 当前分块未通过基本安全检查，使用同一条路径记录在 CPU 计算本块及后续分块；此前已完成的 GPU 分块保留。${gpuFailure.message}`);
     }
    }
    check();if(!image){image=materializePLTBatchReference(job);if(wantsGPU&&gpuFailure)state.validation.cpuFallbackBatches++;}
    if(!finiteArray(image,job.pixelIndices.length))throw new Error('Invalid demo camera tile');
    let controlImage=null;if(state.controlSums){const controlStart=now(),powers=Float64Array.from(job.events,event=>event.directAmplitude**2);controlImage=replayPLT(job,powers);if(!finiteArray(controlImage,job.pixelIndices.length))throw new Error('Invalid same-path direct-only control');state.timings.cpuDirectOnlyControlMs+=now()-controlStart;}
    // Prepare the entire commit before changing sums or sample counts.
    const next=new Float64Array(image.length);for(let i=0;i<image.length;i++){next[i]=state.sums[job.pixelIndices[i]]+image[i]*(batchSamples/o.samplesPerPixel);if(!Number.isFinite(next[i])||next[i]<0)throw new Error('Demo camera accumulation overflow');}
    const nextControl=controlImage?new Float64Array(controlImage.length):null;if(nextControl)for(let i=0;i<nextControl.length;i++){nextControl[i]=state.controlSums[job.pixelIndices[i]]+controlImage[i]*(batchSamples/o.samplesPerPixel);if(!Number.isFinite(nextControl[i])||nextControl[i]<0)throw new Error('Direct-only control accumulation overflow');}
    check();for(let i=0;i<next.length;i++){state.sums[job.pixelIndices[i]]=next[i];if(nextControl)state.controlSums[job.pixelIndices[i]]=nextControl[i];state.sampleCounts[job.pixelIndices[i]]+=batchSamples;}if(nextControl)state.cpuControlImages++;
    if(usedGPU)state.gpuBatches++;else state.cpuBatches++;
    state.cpuOperatorEvaluations+=job.timings.cpuOperatorEvaluations??0;state.cpuReferenceImages+=job.timings.cpuReferenceImages??0;state.timings.cpuTraversalMs+=job.timings.cpuTraversalMs??0;state.timings.cpuOperatorAndCameraMs+=job.timings.cpuOperatorAndCameraMs??0;
    state.validation.acceptedBatches++;state.validation.events.count+=job.events.length;state.validation.pixels.count+=job.pixelIndices.length;
    mergePLTStats(state.stats,job.stats);state.events+=job.events.length;state.terms+=job.terms.length;
    for(const path of job.paths)if(state.paths.length<12&&(path.diffractionCount>=2||state.paths.length<3))state.paths.push(path);
    for(const evidence of job.multiRegionEvidence)if(state.multiRegionEvidence.length<8)state.multiRegionEvidence.push({...evidence,batch:completedBatches+1});
    peaks.retainedPaths=Math.max(peaks.retainedPaths,states.reduce((sum,s)=>sum+s.paths.length,0));peaks.retainedEvidence=Math.max(peaks.retainedEvidence,states.reduce((sum,s)=>sum+s.multiRegionEvidence.length,0));
    completedSamples+=job.pixelIndices.length*batchSamples;completedBatches++;
    const passComplete=tile===tiles.at(-1)&&state===states.at(-1);if(passComplete){completedPasses=sampleStart+batchSamples;firstFullPassMs??=now()-started;}
    notifyPreview(batch,passComplete);notifyProgress(`${name}：完成分块 ${completedBatches}/${totalBatches} · 完整 ${completedPasses}/${o.samplesPerPixel} spp`,true);
    await new Promise(resolve=>setTimeout(resolve,0));check();
   }
  }
  check();const results=states.map(state=>{const result=snapshot(state,o,false);if(result.minSamplesPerPixel!==o.samplesPerPixel||result.maxSamplesPerPixel!==o.samplesPerPixel)throw new Error('Incomplete demo samples cannot finalize');return {...result,validation:state.validation,stats:state.stats,paths:state.paths,multiRegionEvidence:state.multiRegionEvidence,measurement:measurePLTImage(result.image),timings:state.timings,events:state.events,terms:state.terms,execution:{cpuBatches:state.cpuBatches,gpuBatches:state.gpuBatches,cpuOperatorEvaluations:state.cpuOperatorEvaluations,cpuReferenceImages:state.cpuReferenceImages,cpuControlImages:state.cpuControlImages}};});
  const wave=results.find(r=>r.wave),ray=results.find(r=>!r.wave);let comparison=null;if(wave&&ray){let difference=0,norm=0;for(let i=0;i<wave.image.length;i++){difference+=Math.abs(wave.image[i]-ray.image[i]);norm+=Math.abs(ray.image[i]);}comparison={absoluteMeanDifference:difference/wave.image.length,relativeL1:norm>0?difference/norm:null};}
  return {results,warnings,comparison,scope:`${PLT_SCOPE}; ${PLT_DEMO_SCOPE}`,options:o,camera,scene:{physicalScale:scene.physicalScale,preset:configuration.preset??'gallery',triangles:scene.triangles.length,wedges:scene.wedges.length,unsupportedEdges:scene.unsupportedEdges.length,emitters:scene.emitters.length},totalMs:now()-started,executionMode:'demo',demoBuildId:PLT_DEMO_BUILD_ID,...(request.utdAblation===true?{utdControl:{kind:'same-wave-paths-direct-amplitude-squared',description:'Same sampled wave paths with local UTD edge fields removed; no CPU diffraction coefficients; not an independent geometric renderer or CPU parity reference'}}:{}),parityChecked:false,gpuFailure,execution:{cpuBatches:states.reduce((n,s)=>n+s.cpuBatches,0),gpuBatches:states.reduce((n,s)=>n+s.gpuBatches,0),cpuOperatorEvaluations:states.reduce((n,s)=>n+s.cpuOperatorEvaluations,0),cpuReferenceImages:states.reduce((n,s)=>n+s.cpuReferenceImages,0),cpuControlImages:states.reduce((n,s)=>n+s.cpuControlImages,0),cpuFullShadow:false,cpuTraversalAndSampling:true},progressive:{...po,completedBatches,totalBatches,completedSamplesPerPixel:completedPasses,firstPreviewMs,firstFullPassMs,peakTape:peaks,persistentPixelBytes:modes.length*pixels*(4+2+8)+(request.utdAblation===true&&modes.includes(true)?pixels*8:0),sampleSequence:'persistent per-global-pixel RNG; each completed tile committed once; CPU fallback reuses the same tape',limits:PLT_PROGRESSIVE_LIMITS},privacy:'OBJ remains in browser memory; no model upload; no server computation'};
 }finally{clearTimeout(timer);externalSignal?.removeEventListener('abort',externalAbort);try{session?.destroy();}catch{}}
}
