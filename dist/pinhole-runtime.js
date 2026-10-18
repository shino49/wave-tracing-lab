/** Fixed-physical-sensor 3-D pinhole demonstration; no CPU shadow rendering. */
import {createPinholeScene,buildPinholeSourceLayers} from './pinhole-scene.js';
import {generateKernels,buildPaddedKernel,renderPointDetector,PINHOLE_OPTICS_SCOPE} from './pinhole-optics.js';
import {createPinholeCPURunner} from './pinhole-cpu.js';
import {planPinholeContributions} from './pinhole-progress.js';
export const PINHOLE_BUILD_ID='pinhole-native-progress-2026-10-13.2';
export const PINHOLE_LIMITS=Object.freeze({resolutions:[16,32,64,128,256,512],diameterMinMm:.04,diameterMaxMm:.5,wavelengthMinNm:360,wavelengthMaxNm:830,depthLayers:[2,4,8,16],maxRuntimeMs:600000});
export const PINHOLE_DEFAULTS=Object.freeze({mode:'scene',diameterMm:.2,spectralMode:'broadband',wavelengthNm:550,resolution:512,depthLayers:8,backend:'auto',sensorWidthMm:8,detectorDistanceMm:50,pointDepthMm:250,pointProjectedMass:100,spectralStepNm:5,pixelQuadrature:4,maxRuntimeMs:600000});
const now=()=>globalThis.performance?.now?.()??Date.now();
const failure=(message,name='Error')=>Object.assign(new Error(message),{name});
export function validatePinholeRequest(input={}){
 if(!input||typeof input!=='object'||Array.isArray(input))throw new TypeError('Invalid pinhole request');
 const o={...PINHOLE_DEFAULTS,...input};
 for(const [key,allowed]of [['mode',['scene','point']],['spectralMode',['broadband','narrowband']],['backend',['auto','cpu']],['resolution',PINHOLE_LIMITS.resolutions],['depthLayers',PINHOLE_LIMITS.depthLayers]])if(!allowed.includes(o[key]))throw new RangeError(`Unsupported pinhole ${key}`);
 for(const [key,min,max]of [['diameterMm',.04,.5],['wavelengthNm',360,830],['maxRuntimeMs',1,600000]])if(!Number.isFinite(o[key])||o[key]<min||o[key]>max)throw new RangeError(`Invalid pinhole ${key}`);
 for(const key of ['sensorWidthMm','detectorDistanceMm','pointDepthMm','pointProjectedMass','spectralStepNm','pixelQuadrature'])if(o[key]!==PINHOLE_DEFAULTS[key])throw new RangeError(`${key} is fixed in this bounded camera`);
 return Object.fromEntries(Object.keys(PINHOLE_DEFAULTS).map(k=>[k,o[k]]));
}
function measure(xyz,pixelArea){const powerXYZ=[0,0,0];let peakY=0;for(let i=0;i<xyz.length;i+=3){for(let c=0;c<3;c++)powerXYZ[c]+=xyz[i+c]*pixelArea;peakY=Math.max(peakY,xyz[i+1]);}return {powerXYZ,peakY,meanY:powerXYZ[1]/(xyz.length/3*pixelArea)};}
function validateImage(xyz,n,label,{allowRoundoff=false}={}){
 if(!(xyz instanceof Float32Array||xyz instanceof Float64Array)||xyz.length!==3*n*n)throw new Error(`${label} has invalid XYZ shape`);
 let peak=0,minimum=0;for(const v of xyz){if(!Number.isFinite(v))throw new Error(`${label} contains nonfinite light`);peak=Math.max(peak,v);minimum=Math.min(minimum,v);}
 const tolerance=allowRoundoff?Math.max(1e-20,peak*2e-10):0;if(minimum< -tolerance)throw new Error(`${label} contains significant negative light`);
 let clamped=0;for(let i=0;i<xyz.length;i++)if(xyz[i]<0){xyz[i]=0;clamped++;}return {finite:true,minimumBeforeRoundoff:minimum,clampedRoundoff:clamped,tolerance};
}
function interleave(channels,n,start){if(!Array.isArray(channels)||channels.length!==6||channels.some(a=>!(a instanceof Float32Array||a instanceof Float64Array)||a.length!==n*n))throw new Error('Malformed convolution channel buffers');const out=new Float64Array(3*n*n);for(let i=0;i<n*n;i++)for(let c=0;c<3;c++)out[3*i+c]=channels[start+c][i];return out;}
const pointOverview=o=>({lengthUnit:'mm',objects:[],point:{positionMm:[0,0,-o.pointDepthMm],projectedMass:o.pointProjectedMass},aperture:{centerMm:[0,0,0],normal:[0,0,1],radiusMm:o.diameterMm/2},detector:{cornersMm:[[-4,-4,50],[4,-4,50],[4,4,50],[-4,4,50]],resolution:o.resolution},cameraOutline:{pupilCenterMm:[0,0,0],detectorCenterMm:[0,0,50],sourceBoundsMm:{min:[0,0,-250],max:[0,0,-250]}}});

export async function executePinholeTask(request={}, {signal:externalSignal,progress=()=>{},preview=null,reset=()=>{},gpuFactory}={}){
 const configuration=Object.freeze(validatePinholeRequest(request)),started=now(),controller=new AbortController(),signal=controller.signal;
 const externalAbort=()=>controller.abort(externalSignal?.reason??failure('Pinhole calculation cancelled','AbortError'));
 externalSignal?.addEventListener('abort',externalAbort,{once:true});if(externalSignal?.aborted)externalAbort();
 const timer=setTimeout(()=>controller.abort(failure('Pinhole rendering reached its runtime limit. No partial image was finalized.','TimeoutError')),configuration.maxRuntimeMs);
 const check=()=>{if(signal.aborted)throw signal.reason instanceof Error?signal.reason:failure('Pinhole calculation cancelled','AbortError');};
 const resolutions=[configuration.resolution],stages=[];let attempt=0;
 if(!resolutions.length)resolutions.push(configuration.resolution);
 const pending=promise=>new Promise((resolve,reject)=>{const abort=()=>reject(signal.reason instanceof Error?signal.reason:failure('Pinhole calculation cancelled','AbortError'));signal.addEventListener('abort',abort,{once:true});if(signal.aborted)abort();Promise.resolve(promise).then(value=>{signal.removeEventListener('abort',abort);resolve(value);},error=>{signal.removeEventListener('abort',abort);reject(error);});});
 const scene=configuration.mode==='scene'?createPinholeScene():null;let fallback=null,disableGPU=configuration.backend==='cpu',lastProgress=-Infinity,activeRunner=null;
 const notify=(phase,n,message,fraction,force=false)=>{check();if(force||now()-lastProgress>100){progress({phase,stageResolution:n,fraction,message,attempt});lastProgress=now();check();}};
 try{
  check();let finalResult;
  for(let stageIndex=0;stageIndex<resolutions.length;stageIndex++){
   const n=resolutions[stageIndex],stageStart=now(),o={...configuration,resolution:n},base=stageIndex/resolutions.length,span=1/resolutions.length;
   const timings={sceneMs:0,radialKernelsMs:0,pixelKernelsMs:0,convolutionMs:0,gpuSetupMs:0,readbackMs:0,schedulingMs:0,snapshotMs:0,previewCallbackMs:0},stageProgress=(phase,fraction,message)=>notify(phase,n,message,base+span*fraction);
   notify('scene',n,`正在实际计算 ${n} × ${n} 像素`,base,true);
   let source=null,sceneOverview=pointOverview(o),depths=[o.pointDepthMm],basisNames=['neutral'];
   if(scene){const t=now();source=await buildPinholeSourceLayers(scene,{resolution:n,sensorWidthMm:o.sensorWidthMm,detectorDistanceMm:o.detectorDistanceMm,depthCount:o.depthLayers,diameterMm:o.diameterMm},{signal,onProgress:p=>stageProgress('scene',.08*(p.fraction??0),`三维网格、遮挡与深度：${n}²`)});check();timings.sceneMs=now()-t;sceneOverview=source.overview??source.geometry;depths=Array.from(source.depthsMm);basisNames=source.basisNames;}
   const radialStart=now(),kernels=await generateKernels({resolution:n,sensorWidthMm:o.sensorWidthMm,detectorDistanceMm:o.detectorDistanceMm,diameterMm:o.diameterMm,depths,basisNames,mode:o.spectralMode,wavelengthNm:o.wavelengthNm,spectralStepNm:o.spectralStepNm},{signal,onProgress:p=>stageProgress('fresnel-kernels',.08+.17*(p.completed/p.total||0),'可见光谱与有限距离菲涅耳传播')});check();timings.radialKernelsMs=now()-radialStart;
   let waveXYZ,geometricXYZ,backend,convolutionStats=null,backendValidation=null,sourcePowerXYZ=[0,0,0],cpuRoundoff=null,progressive=null,contributionPlan=null;
   if(!scene){
    const pointStart=now(),hook={signal,onProgress:p=>stageProgress('point-source',.25+.7*(p.completed/p.total||0),'精确位置点光源：逐探测器像素积分')};
    waveXYZ=(await renderPointDetector(kernels,{kind:'wave',pixelQuadrature:o.pixelQuadrature,mass:o.pointProjectedMass},hook)).xyz;
    geometricXYZ=(await renderPointDetector(kernels,{kind:'geometric',pixelQuadrature:o.pixelQuadrature,mass:o.pointProjectedMass},hook)).xyz;
    timings.convolutionMs=now()-pointStart;backend={kind:'cpu-point',label:'CPU 双精度径向菲涅耳 · 原生探测器像素积分',cpuShadow:false};
    sourcePowerXYZ=Array.from(kernels.layers[0].basis[0].colorXYZ,v=>v*kernels.throughput*o.pointProjectedMass);
   }else{
    const tasks=[];
    for(let depth=0;depth<source.layers.length;depth++)for(let basis=0;basis<source.basisNames.length;basis++){const values=source.layers[depth].basis[basis];let sum=0;for(const value of values)sum+=value;if(sum>0){tasks.push({depth,basis,values,sourceY:sum*kernels.layers[depth].basis[basis].colorXYZ[1]});for(let c=0;c<3;c++)sourcePowerXYZ[c]+=sum*kernels.pixelAreaMm2*kernels.throughput*kernels.layers[depth].basis[basis].colorXYZ[c];}}
    const scheduleStart=now(),plan=await planPinholeContributions(tasks,{signal});timings.schedulingMs=now()-scheduleStart;tasks.splice(0,tasks.length,...plan.tasks);
    contributionPlan={totalContributions:tasks.length,checkpoints:plan.checkpoints,activeSourcePairs:plan.activeSourcePairs,firstFullSourceSupport:plan.firstFullSourceSupport,order:tasks.map(t=>({depth:t.depth,basis:t.basis,sourcePowerFraction:t.sourcePowerFraction,representedSourceFraction:t.representedSourceFraction}))};
    const progression=count=>({kind:'contribution-accumulation',attempt,completedContributions:count,totalContributions:tasks.length,sourcePowerFraction:count===tasks.length?1:tasks[count-1]?.sourcePowerFraction??0,representedSourceFraction:count===tasks.length?1:tasks[count-1]?.representedSourceFraction??0});
    const unpack=(result,useGPU)=>{
     if(useGPU&&result.validation?.passed!==true)throw new Error('GPU did not provide a successful finite-output record');
     const waveXYZ=interleave(result.channels,n,0),geometricXYZ=interleave(result.channels,n,3),roundoff={wave:validateImage(waveXYZ,n,'Wave accumulation',{allowRoundoff:!useGPU}),geometric:validateImage(geometricXYZ,n,'Geometric accumulation',{allowRoundoff:!useGPU})};
     return {waveXYZ,geometricXYZ,backend:typeof result.backend==='string'?{kind:'cpu',label:result.backend}:result.backend,stats:result.stats,validation:result.validation,roundoff};
    };
    const runConvolutions=async useGPU=>{
     check();if(useGPU){notify('gpu-setup',n,'正在创建 WebGPU FFT 管线；可随时取消',base+span*.25,true);const setupStart=now(),creation=(gpuFactory??(async args=>(await import('./pinhole-gpu.js')).createPinholeGPURunner(args)))({width:n,fftSize:2*n,outputChannels:6,signal,timeoutMs:Math.min(120000,o.maxRuntimeMs)});Promise.resolve(creation).then(r=>{if(signal.aborted)r?.destroy();},()=>{});activeRunner=await pending(creation);timings.gpuSetupMs+=now()-setupStart;}
     else activeRunner=createPinholeCPURunner({width:n,fftSize:2*n,outputChannels:6,signal});
     check();
     for(let i=0;i<tasks.length;i++){
      check();const task=tasks[i],kernelStart=now(),wave=await buildPaddedKernel(kernels,task.depth,task.basis,{kind:'wave',pixelQuadrature:o.pixelQuadrature},{signal}),geo=await buildPaddedKernel(kernels,task.depth,task.basis,{kind:'geometric',pixelQuadrature:o.pixelQuadrature},{signal});timings.pixelKernelsMs+=now()-kernelStart;check();
      const list=[...wave.channels.map((values,channel)=>({values,channel})),...geo.channels.map((values,channel)=>({values,channel:channel+3}))];
      stageProgress('convolution',.25+.7*i/Math.max(1,tasks.length),`${useGPU?'WebGPU':'CPU'} FFT：深度 / 材质 ${i+1}/${tasks.length} · ${n}²`);
      const convolutionStart=now();await pending(activeRunner.convolveAccumulate({source:task.values,kernels:list,signal}));timings.convolutionMs+=now()-convolutionStart;check();
      if(typeof preview==='function'&&plan.checkpoints.includes(i+1)){
       const t=now(),snapshot=unpack(await pending(useGPU?activeRunner.readResults({signal}):activeRunner.snapshotResults({signal})),useGPU);check();timings.snapshotMs+=now()-t;
       const callbackStart=now();preview({width:n,height:n,waveXYZ:snapshot.waveXYZ,geometricXYZ:snapshot.geometricXYZ,configuration:{...configuration},sceneOverview:structuredClone(sceneOverview),backend:structuredClone(snapshot.backend),timings:{...timings},diagnostics:{buildId:PINHOLE_BUILD_ID,cpuShadow:false,incomplete:true,backendValidation:structuredClone(snapshot.validation),cpuRoundoff:useGPU?null:snapshot.roundoff,previewMeaning:'Exact unscaled subset of positive depth/material contributions; source support is not detector energy or accuracy'},totalMs:now()-started,stageResolution:n,requestedResolution:n,complete:false,progressive:progression(i+1)});check();timings.previewCallbackMs+=now()-callbackStart;
      }
     }
     const readStart=now(),result=await pending(activeRunner.readResults({signal}));check();timings.readbackMs+=now()-readStart;
     return unpack(result,useGPU);
    };
    let convolution;
    try{
     const available=!!gpuFactory||!!globalThis.navigator?.gpu;
     if(!disableGPU&&!available){disableGPU=true;fallback={reason:'WebGPU unavailable',stageResolution:n,attemptedGPU:false};}
     try{convolution=await runConvolutions(!disableGPU);}catch(error){check();if(disableGPU)throw error;fallback={reason:String(error.message??error).slice(0,1000),stageResolution:n,attemptedGPU:true};disableGPU=true;activeRunner?.destroy();activeRunner=null;attempt++;reset({attempt,reason:fallback.reason});check();notify('cpu-fallback',n,`GPU 失败；清除未完成贡献，从零在 CPU 重算 ${n}² 图像`,base+span*.25,true);convolution=await runConvolutions(false);}
    }finally{activeRunner?.destroy();activeRunner=null;}
    ({waveXYZ,geometricXYZ,backend}=convolution);convolutionStats=convolution.stats;backendValidation=convolution.validation;progressive=progression(tasks.length);
    if(backend.kind==='cpu')cpuRoundoff=convolution.roundoff;
   }
   check();validateImage(waveXYZ,n,'Wave');validateImage(geometricXYZ,n,'Geometric');
   const waveMeasurement=measure(waveXYZ,kernels.pixelAreaMm2),geometricMeasurement=measure(geometricXYZ,kernels.pixelAreaMm2),collectedFractions={wave:waveMeasurement.powerXYZ.map((v,c)=>sourcePowerXYZ[c]>0?v/sourcePowerXYZ[c]:null),geometric:geometricMeasurement.powerXYZ.map((v,c)=>sourcePowerXYZ[c]>0?v/sourcePowerXYZ[c]:null)};
   const diagnostics={buildId:PINHOLE_BUILD_ID,cpuShadow:false,cpuReferenceImages:0,model:'3-D visible source meshes and a scalar paraxial circular-aperture imaging operator; not general coherent 3-D transport',spectral:{mode:o.spectralMode,samples:kernels.spectralSamples,stepNm:kernels.spectralStepNm,rangeNm:[360,830],source:'D65 or unit-line spectral weight',bases:basisNames},sensor:{widthMm:8,heightMm:8,resolution:n,pixelPitchMm:kernels.pixelWidthMm,pixelAreaMm2:kernels.pixelAreaMm2},aperture:{diameterMm:o.diameterMm,areaMm2:kernels.areaMm2,detectorDistanceMm:50},source:source?.metadata??{kind:'centered point',depthMm:o.pointDepthMm,projectedMass:o.pointProjectedMass,coherence:'Coherent across the pupil for each wavelength; intensities added across wavelengths'},sourcePowerXYZ,collectedFractions,measurements:{wave:waveMeasurement,geometric:geometricMeasurement},normalization:kernels.normalization,kernel:{radialStepMm:kernels.radialStepMm,radialCount:kernels.radialCount,spectralSamples:kernels.spectralSamples,depthsMm:depths,pixelQuadrature:o.pixelQuadrature},fallback,convolutionStats,backendValidation,cpuRoundoff,contributionPlan,progressiveWork:{gridBuilds:1,resolutions:[n],previewRequested:typeof preview==='function',extraGPUConvolutionFFTs:0,snapshotMs:timings.snapshotMs},warnings:['标量、近轴、薄圆孔；不包括完整电磁边界、任意场景多次散射或全局相干路径求和。',...(scene?['物体之间按非相干光强相加；不同深度按倒深度插值。孔中心可见性用于成像，孔径视差风险单独记录。']:['点光源图像由同一孔径模型逐原生探测器像素积分，不需要 FFT 卷积；未进行 GPU 运算。']),...(fallback?[`本次使用 CPU 回退：${fallback.reason}`]:[])]};
   timings.totalMs=now()-stageStart;const result={width:n,height:n,waveXYZ,geometricXYZ,configuration,sceneOverview,backend,timings,diagnostics,totalMs:now()-started,scope:PINHOLE_OPTICS_SCOPE,stageResolution:n,requestedResolution:configuration.resolution,complete:true,...(progressive?{progressive}: {})};
   stages.push({resolution:n,timings:{...timings},backend,measurements:diagnostics.measurements});finalResult=result;
   notify('stage-complete',n,`${n} × ${n} 原生像素计算完成`,(stageIndex+1)/resolutions.length,true);
  }
  check();return {...finalResult,stages,totalMs:now()-started,complete:true};
 }catch(error){if(signal.aborted&&signal.reason instanceof Error)error=signal.reason;error.diagnostics??={buildId:PINHOLE_BUILD_ID,configuration,completedStages:stages,fallback,cpuShadow:false,incomplete:true};error.timings??={totalMs:now()-started};throw error;}finally{clearTimeout(timer);externalSignal?.removeEventListener('abort',externalAbort);activeRunner?.destroy();}
}
