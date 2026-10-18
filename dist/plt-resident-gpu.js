/** Experimental real GPU path construction; NEVER consumes a CPU path tape.
 * Untrusted candidate readbacks require compareResidentAudit before display.
 * Selective replay is opt-in and still requires full-shadow admission.
 * No unaudited fast mode is exported. Scene/pipeline resources are reused.
 */
import {packResidentScene,RESIDENT_GEOMETRY_REASONS} from './plt-resident-geometry.js';
import {cameraFrame,validatePLTOptions,PLT_PROGRESSIVE_LIMITS,randomGenerator} from './plt-core.js';
import {packResidentCamera} from './plt-resident-camera.js';
export const PLT_RESIDENT_BUILD_ID='plt-resident-joint-camera-2026-10-13.1';
// Only explicitly uncertain predicates can be replayed. Invalid geometry/math,
// exhaustion, unsupported sampling and unknown bits still poison the batch.
export const RESIDENT_TRANSPORT_REASONS=Object.freeze({unmodelledContinuation:134217728,axisIntervalUncertain:268435456});
const isAnchorGuard=guard=>guard==='axis-mirror-anchor'||guard==='axis-mirror-tight'||guard==='axis-mirror-camera';
const isAxisGuard=guard=>guard==='axis-mirror-interval'||guard==='axis-mirror-unit'||isAnchorGuard(guard);
const replayReasons={...RESIDENT_GEOMETRY_REASONS,...RESIDENT_TRANSPORT_REASONS};
export const RESIDENT_REPLAY_MASK=2|Object.values(replayReasons).reduce((mask,bit)=>mask|bit,0);
export const isResidentReplayable=flags=>Number.isInteger(flags)&&flags>0&&flags<=0xffffffff&&(flags&2)!==0&&(flags&~RESIDENT_REPLAY_MASK)===0;
export const PLT_RESIDENT_SCOPE='Experimental resident GPU camera RNG, real triangle BVH/cones, region-clipped native UTD/FSD, BSDF/NEE/MIS and accumulation; exhaustive CPU shadow audit required';
export const PLT_RESIDENT_LIMITS=Object.freeze({tilePixels:1024,batchSamples:4,pathTerms:64,pathVertices:72,pathEvents:6,pathContributions:12,rngWords:64,auditVectors:387,laneBytes:448,dispatchLanes:32,stack:32});
const now=()=>globalThis.performance?.now?.()??Date.now();
const abortError=()=>Object.assign(new Error('Resident GPU calculation cancelled or destroyed'),{name:'AbortError'});
function assert(value,message){if(!value)throw new RangeError(`Resident GPU: ${message}`);}
function checkSignal(signal){if(signal?.aborted)throw signal.reason?.name==='TimeoutError'?signal.reason:abortError();}
function fingerprint(source){let h=2166136261;for(let i=0;i<source.length;i++)h=Math.imul(h^source.charCodeAt(i),16777619);return(h>>>0).toString(16).padStart(8,'0');}
function adapterSummary(adapter){const out={};for(const key of ['vendor','architecture','device','description','isFallbackAdapter'])try{const v=adapter.info?.[key];if(typeof v==='string'||typeof v==='boolean')out[key]=v;}catch{}return out;}
export function packResidentParameters(camera,options,packedScene,{tile,sampleStart,samplesPerPixel,first=0,count=0}={}){
 const o=validatePLTOptions(options,{progressive:true}),p=PLT_RESIDENT_LIMITS;
 assert(tile&&['x','y','width','height'].every(k=>Number.isInteger(tile[k])),'invalid tile');
 assert(tile.x>=0&&tile.y>=0&&tile.width>0&&tile.height>0&&tile.width<=32&&tile.height<=32&&tile.x+tile.width<=o.width&&tile.y+tile.height<=o.height,'tile outside frame');
 assert(Number.isInteger(sampleStart)&&sampleStart>=0&&Number.isInteger(samplesPerPixel)&&samplesPerPixel>=1&&samplesPerPixel<=p.batchSamples&&sampleStart+samplesPerPixel<=o.samplesPerPixel,'invalid sample prefix');
 assert(tile.width*tile.height*samplesPerPixel<=PLT_PROGRESSIVE_LIMITS.maxBatchPaths,'path capacity exceeded');
 const frame=cameraFrame(camera),half=Math.tan((camera.fov??42)*Math.PI/360),buffer=new ArrayBuffer(160),u=new Uint32Array(buffer),f=new Float32Array(buffer);
 u.set([o.width,o.height,o.maxDepth,['soft','hard','unpolarized'].indexOf(o.boundary)],0);
 u.set([tile.x,tile.y,tile.width,tile.height],4);u.set([sampleStart,samplesPerPixel,o.samplesPerPixel,o.wave?1:0],8);u.set([first,count,p.auditVectors,0],12);
 for(const [i,v] of [[16,packedScene.toLocal(camera.position)],[20,frame.forward],[24,frame.right],[28,frame.up]])f.set(v,i);
 f.set([half,o.width/o.height,o.wavelengthMm,2*Math.PI/o.wavelengthMm],32);u.set([o.nee?1:0,0,0,0],36);
 // Outward radii of immutable CPU camera constants. No per-path CPU camera
 // state is uploaded. Match the CPU half-FOV expression, not an algebraic rewrite.
 const ceilF32=value=>{if(value===0)return 0;const a=new Float32Array([value]);if(a[0]<value)new Uint32Array(a.buffer)[0]++;return a[0];};
 for(const [index,values] of [[16,camera.position.map((x,k)=>x-packedScene.origin[k])],[20,frame.forward],[24,frame.right],[28,frame.up]])f[index+3]=ceilF32(Math.max(...values.map((x,k)=>Math.abs(x-f[index+k])))+(index===16?2**-40*Math.max(1,...camera.position.map(Math.abs),...packedScene.origin.map(Math.abs)):0));
 const cpuHalf=Math.tan(((camera.fov??42)*Math.PI/180)/2);f[39]=ceilF32(Math.abs(cpuHalf-f[32]));
 assert(f.subarray(16,36).every(Number.isFinite)&&Number.isFinite(f[39]),'unrepresentable camera');
 return buffer;
}
/** Decode bounded candidate audit. All world-space points are un-rebased. */
export function decodeResidentAudit(buffer,{tile,sampleStart,samplesPerPixel,options,packedScene,laneData,recordData,recoveredPaths=new Map()}){
 assert(buffer instanceof ArrayBuffer&&buffer.byteLength===tile.width*tile.height*samplesPerPixel*PLT_RESIDENT_LIMITS.auditVectors*16,'invalid audit buffer length');
 assert(laneData instanceof ArrayBuffer&&laneData.byteLength===tile.width*tile.height*PLT_RESIDENT_LIMITS.laneBytes,'invalid lane buffer length');
 assert(recordData instanceof ArrayBuffer&&recordData.byteLength%64===0&&recordData.byteLength<=131072*64,'invalid edge record buffer length');
 const seenRecords=new Uint8Array(recordData.byteLength/64);let visitedRecords=0;
 const u=new Uint32Array(buffer),f=new Float32Array(buffer),paths=[],pixelIndices=new Uint32Array(tile.width*tile.height),reference=recoveredPaths.size?new Float64Array(pixelIndices.length):new Float32Array(pixelIndices.length);
 const world=(index)=>packedScene.toWorld([...f.subarray(index,index+3)]),xyz=index=>[...f.subarray(index,index+3)];
 const ru=new Uint32Array(recordData),rf=new Float32Array(recordData);
 const lu=new Uint32Array(laneData),lf=new Float32Array(laneData),laneWords=PLT_RESIDENT_LIMITS.laneBytes/4;
 let flags=0;const consumedRecoveries=new Set();
 for(let lane=0;lane<pixelIndices.length;lane++){
  const lo=lane*laneWords;pixelIndices[lane]=lu[lo];reference[lane]=lf[lo+49]/samplesPerPixel;
  assert(lu[lo+2]===sampleStart+samplesPerPixel&&lu[lo+3]===2,'incomplete candidate sample prefix');
  for(let sample=0;sample<samplesPerPixel;sample++){
   const base=(lane*samplesPerPixel+sample)*PLT_RESIDENT_LIMITS.auditVectors*4;
   const recovery=recoveredPaths.get(lane*samplesPerPixel+sample);
   if(recovery){
    const {path,flags:replayFlags}=recovery,slot=lane*samplesPerPixel+sample;
    assert(path&&path.flags===0&&isResidentReplayable(replayFlags),'invalid recovery certificate');
    assert(u[base]===path.key.pixel&&u[base+1]===path.key.sample&&u[base+2]===Number(path.key.wave)&&u[base+3]===replayFlags,'recovery path identity mismatch');
    assert(path.key.pixel===pixelIndices[lane]&&path.key.sample===sampleStart+sample,'recovery outside requested prefix');
    assert(u[base+4]===path.rng.start&&u[base+6]<=64,'invalid discarded RNG prefix');
    const rng=randomGenerator(path.rng.start);let hash=2166136261;
    for(let i=0;i<u[base+6];i++){const word=Math.floor(rng()*4294967296);assert(u[base+16+i]===word,'discarded path RNG corruption');hash=Math.imul(hash^word,16777619)>>>0;}
    assert(u[base+5]===rng.getState()&&u[base+7]===hash,'discarded path RNG endpoint/hash corruption');
    assert(Number.isFinite(path.value)&&path.value>=0,'invalid recovered contribution');
    paths.push(path);consumedRecoveries.add(slot);continue;
   }
   const key={wave:!!u[base+2],pixel:u[base],sample:u[base+1]},vertexCount=u[base+8],eventCount=u[base+9],contributionCount=u[base+10],rngCount=u[base+6];
   assert(vertexCount<=72&&eventCount<=6&&contributionCount<=12&&rngCount<=64&&Number.isInteger(f[base+15])&&f[base+15]>=0&&f[base+15]<=6,'corrupt candidate audit capacity');assert(u[base+2]<=1,'invalid candidate wave enum');flags|=u[base+3];
   const path={key,vertices:[],localEvents:[],contributions:[],scatters:[],rng:{start:u[base+4],final:u[base+5],count:rngCount,hash:u[base+7],words:[...u.subarray(base+16,base+16+rngCount)]},
    termination:{reason:['unknown','escaped','absorbed','sampling-rejected','max-depth'][u[base+11]]??'rejected',depth:f[base+13]},sampleValue:f[base+12],depth:f[base+13],nullSteps:f[base+14],scatterCount:f[base+15],flags:u[base+3]};
   for(let index=0;index<vertexCount;index++){
    const at=base+80+index*16,kind=u[at];assert(kind<=2,'invalid vertex kind');path.vertices.push({type:['escape','surface','region'][kind]??'invalid',triangleId:u[at+1]===0xffffffff?null:u[at+1],apertureCount:u[at+2],depth:u[at+3],center:kind?world(at+4):null,distance:kind?f[at+7]:null,radius:f[at+11],direction:xyz(at+8),origin:world(at+12),tanAngle:f[at+15]});
   }
   for(let index=0;index<eventCount;index++){
    const at=base+1232+index*20;assert(u[at+3]<=1,'invalid sampled enum');const records=[];let recordIndex=u[at+11];assert(u[at]<=131072,'invalid edge record count');for(let j=0;j<u[at];j++){assert(recordIndex*16+16<=ru.length,'invalid edge record range');assert(!seenRecords[recordIndex],'reused or cyclic edge record');seenRecords[recordIndex]=1;visitedRecords++;const r=recordIndex*16;assert(ru[r]===lane*samplesPerPixel+sample&&ru[r+1]===index,'invalid edge trace key');records.push({edgeId:ru[r+2],point:packedScene.toWorld([...rf.subarray(r+4,r+7)]),ri:rf[r+7],ro:rf[r+8],sinBeta:rf[r+9],phiI:rf[r+10],phiO:rf[r+11],n:rf[r+12],phaseExcess:rf[r+13],L:rf[r+14],fermatResidual:rf[r+15]});recordIndex=ru[r+3];}assert(recordIndex===0xffffffff,'invalid edge record chain end');path.localEvents.push({records,edgeCount:u[at],edgeHash:u[at+1],depth:u[at+2],sampled:!!u[at+3],F:f[at+4],directAmplitude:f[at+5],soft:[f[at+6],f[at+7]],hard:[f[at+8],f[at+9]],diagnostics:f[at+10],destination:world(at+12),source:world(at+16)});
   }
   for(let index=0;index<contributionCount;index++){
    const at=base+1352+index*8;assert(u[at]===1||u[at]===2,'invalid contribution kind');path.contributions.push({type:u[at]===1?'emission':'surface-nee',depth:u[at+1],value:f[at+4],weight:f[at+5],misWeight:f[at+6]});
   }
   for(let index=0;index<path.scatterCount;index++){
    const at=base+1448+index*16,kind=u[at];assert(kind>=1&&kind<=3&&u[at+2]<=1&&(f[at+9]===0||f[at+9]===1),'invalid scatter enum');
    if(kind===3)path.scatters.push({type:'fsd',depth:u[at+1],sample:{valid:!!u[at+2],wo:xyz(at+4),pdf:{value:f[at+7]},weight:f[at+8],edgeId:u[at+3]===0xffffffff?null:u[at+3],isDirect:!!f[at+9],point:world(at+12)}});
    else path.scatters.push({type:kind===1?'diffuse':'mirror',depth:u[at+1],direction:xyz(at+4),pdf:f[at+7],weight:f[at+8]});
   }
   const guardAt=base+1544;assert(u[guardAt]<=72&&u[guardAt+1]<=5&&u[guardAt+2]<=3,'invalid interval evidence');
   path.intervalGuard={queries:u[guardAt],active:!!u[guardAt+1],unitProvenance:u[guardAt+1]>=2,anchorProvenance:u[guardAt+1]>=3,tightProvenance:u[guardAt+1]===4,cameraProvenance:u[guardAt+1]===5,lastAxis:u[guardAt+2],reason:u[guardAt+3]};
   paths.push(path);
  }
 }
 if(recoveredPaths.size)for(let lane=0;lane<pixelIndices.length;lane++){
  let gpuSum=0,compensation=0,total=0;
  for(let sample=0;sample<samplesPerPixel;sample++){
   const slot=lane*samplesPerPixel+sample,path=paths[slot],recovered=recoveredPaths.has(slot),value=recovered?path.value:path.sampleValue;
   assert(Number.isFinite(value)&&value>=0,'invalid path sample value');total+=value;
   if(!recovered){const adjusted=Math.fround(value-compensation),next=Math.fround(gpuSum+adjusted);compensation=Math.fround(Math.fround(next-gpuSum)-adjusted);gpuSum=next;}
  }
  assert(lf[lane*laneWords+49]===gpuSum,'retained GPU subtotal mismatch');
  reference[lane]=total/samplesPerPixel;
 }
 assert(consumedRecoveries.size===recoveredPaths.size,'unconsumed recovery certificate');
 // A paused path may own an incomplete edge chain. Quarantine only records
 // bearing that exact recovered slot. They can never feed retained operators.
 const discardedNext=new Map(),incoming=new Set();
 for(let index=0;index<seenRecords.length;index++)if(!seenRecords[index]){
  const at=index*16,owner=ru[at],event=ru[at+1],next=ru[at+3];
  assert(consumedRecoveries.has(owner)&&event<6,'orphan edge records');
  if(next!==0xffffffff){assert(next<seenRecords.length&&!seenRecords[next]&&ru[next*16]===owner&&ru[next*16+1]===event,'discarded record escapes recovered ownership');assert(!incoming.has(next),'discarded record chain reuse');incoming.add(next);discardedNext.set(index,next);}
  visitedRecords++;
 }
 const doneRecords=new Set();for(const start of discardedNext.keys()){
  const active=new Set();let index=start;while(discardedNext.has(index)&&!doneRecords.has(index)){assert(!active.has(index),'discarded record cycle');active.add(index);index=discardedNext.get(index);}for(const id of active)doneRecords.add(id);
 }
 assert(visitedRecords===seenRecords.length,'orphan edge records');
 return {complete:true,paths,pixelIndices,reference,flags,width:tile.width,height:tile.height,samplesPerPixel};
}
export async function createPLTResidentRunner(scene,camera,options={}, {gpu=globalThis.navigator?.gpu,shaderSource,signal:ownerSignal,onLost=()=>{},timeoutMs=30000,dispatchLanes=PLT_RESIDENT_LIMITS.dispatchLanes,onSetupProgress=()=>{},recoveryMode='strict',continuationGuard='replay-unmodelled'}={}){
 const setupStart=now();camera=structuredClone(camera);checkSignal(ownerSignal);const o=validatePLTOptions(options,{progressive:true});
 assert(['strict','selective'].includes(recoveryMode),'invalid recovery mode');
 assert(['replay-unmodelled','audit-only','axis-mirror-interval','axis-mirror-unit','axis-mirror-anchor','axis-mirror-tight','axis-mirror-camera'].includes(continuationGuard),'invalid continuation guard');
 // CPU replay uses the same immutable scene snapshot as the uploaded geometry.
 const replayScene=recoveryMode==='selective'?structuredClone(scene):null;
 assert(Number.isInteger(dispatchLanes)&&dispatchLanes>=1&&dispatchLanes<=128,'invalid bounded dispatch size');assert(Number.isFinite(timeoutMs)&&timeoutMs>=1&&timeoutMs<=120000,'invalid timeout');
 const packingStart=now(),packedScene=packResidentScene(scene),scenePackingMs=now()-packingStart;
 if(!gpu)throw new Error('WebGPU unavailable for resident walker');
 let closed=false,active=false,device=null,deviceDestroyed=false,fatal=null,interrupt;
 const disposeDevice=(target=device)=>{if(!target)return;if(target===device){if(deviceDestroyed)return;deviceDestroyed=true;}try{target.destroy();}catch{}};const buffers=[];
 const interrupted=new Promise((_,reject)=>{interrupt=reject;});interrupted.catch(()=>{});
 const check=signal=>{if(fatal)throw fatal;if(closed)throw abortError();checkSignal(ownerSignal);checkSignal(signal);};
 const wait=async(promise,label,signal)=>{let timer,listener;try{return await Promise.race([promise,interrupted,new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error(label)),timeoutMs);}),...(signal?[new Promise((_,reject)=>{listener=()=>reject(signal.reason?.name==='TimeoutError'?signal.reason:abortError());if(signal.aborted)listener();else signal.addEventListener('abort',listener,{once:true});})]:[])]);}finally{clearTimeout(timer);if(listener)signal.removeEventListener('abort',listener);}};
 const destroy=()=>{if(closed)return;closed=true;interrupt(fatal??abortError());ownerSignal?.removeEventListener('abort',onAbort);device?.removeEventListener?.('uncapturederror',onUncaptured);for(const b of buffers)try{b.destroy();}catch{}disposeDevice();};
 const onAbort=()=>destroy();const onUncaptured=event=>{fatal=new Error(event.error?.message??'Resident WebGPU uncaptured error');destroy();};
 ownerSignal?.addEventListener('abort',onAbort,{once:true});
 const scoped=async(action,signal)=>{let pushed=0,result,failure;try{for(const kind of ['out-of-memory','internal','validation']){device.pushErrorScope(kind);pushed++;}result=await action();}catch(e){failure=e;}
  while(pushed--){try{const e=await wait(device.popErrorScope(),'Resident GPU error-scope timeout',signal);if(e)failure??=new Error(e.message);}catch(e){failure??=e;}}if(failure)throw failure;return result;};
 try{
  const adapter=await wait(gpu.requestAdapter({powerPreference:'low-power'}),'Resident adapter timeout',ownerSignal);check();if(!adapter)throw new Error('No resident WebGPU adapter');
  const pending=adapter.requestDevice();try{device=await wait(pending,'Resident device timeout',ownerSignal);}catch(error){pending.then(late=>disposeDevice(late),()=>{});throw error;}
  if(closed||ownerSignal?.aborted){disposeDevice();throw abortError();}
  device.addEventListener?.('uncapturederror',onUncaptured);device.lost.then(info=>{if(closed)return;fatal=new Error(info.message||'Resident WebGPU device lost');try{onLost(fatal.message);}catch{}destroy();},error=>{if(!closed){fatal=error;destroy();}});
  const source=shaderSource??(await wait(Promise.all(['utd-functions.wgsl','plt-resident-geometry.wgsl','plt-resident-interval.wgsl','plt-resident-axis-guard.wgsl','plt-resident-rng.wgsl','plt-resident-fsd.wgsl','plt-resident-walker.wgsl'].map(async name=>{const response=await fetch(new URL(name,import.meta.url),{signal:ownerSignal});if(!response.ok)throw new Error(`Cannot load ${name}: ${response.status}`);return response.text();})),'Resident shader fetch timeout',ownerSignal)).join('\n');check();
  const cameraStages=continuationGuard==='axis-mirror-camera',tightStages=continuationGuard==='axis-mirror-tight',anchorStages=isAnchorGuard(continuationGuard),axisStages=isAxisGuard(continuationGuard),stageNames=['rr_init','rr_traverse_step',...(axisStages?[...(anchorStages?['rr_anchor_init_step']:[]),'rr_axis_prepare_step','rr_axis_plane_step','rr_axis_scan_step',...(anchorStages?['rr_anchor_point_step']:[]),'rr_axis_finish_step']:[]),'rr_records_step','rr_operator_step','rr_scatter_step'];
  const pipelines=await scoped(async()=>{const module=device.createShaderModule({label:'Experimental resident 3D PLT shadow walker',code:source});const info=await wait(module.getCompilationInfo(),'Resident WGSL compilation timeout',ownerSignal);const errors=info.messages.filter(x=>x.type==='error');if(errors.length)throw new Error(errors.map(e=>`WGSL ${e.lineNum}:${e.linePos} ${e.message}`).join('; '));const compiled=[];for(const stageName of stageNames){const entryPoint=cameraStages?({rr_init:'rr_init_camera',rr_anchor_init_step:'rr_camera_init_step',rr_axis_prepare_step:'rr_camera_prepare_step',rr_axis_plane_step:'rr_camera_plane_step',rr_anchor_point_step:'rr_camera_point_step',rr_axis_finish_step:'rr_camera_finish_step'}[stageName]??stageName):anchorStages?({rr_init:'rr_init_anchored',rr_axis_prepare_step:'rr_anchor_prepare_step',rr_axis_plane_step:'rr_anchor_plane_step',rr_axis_finish_step:'rr_anchor_finish_step'}[stageName]??stageName):stageName;onSetupProgress({stage:'pipeline',entryPoint,stageName});check();compiled.push(await wait(device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint,constants:{RR_AXIS_INTERVAL_ENABLED:axisStages?1:0,RR_ANCHOR_ENABLED:anchorStages?1:0,RR_TIGHT_ENABLED:tightStages?1:0,RR_CAMERA_ENABLED:cameraStages?1:0}}}),'Resident pipeline compilation timeout',ownerSignal));check();}return compiled;},ownerSignal);check();
  const make=(size,usage,label,data)=>{size=Math.max(16,Math.ceil(size/16)*16);if(size>device.limits.maxBufferSize||((usage&GPUBufferUsage.STORAGE)&&size>device.limits.maxStorageBufferBindingSize))throw new Error('Resident buffer exceeds adapter limits');const buffer=device.createBuffer({size,usage,label});buffers.push(buffer);if(data?.byteLength)device.queue.writeBuffer(buffer,0,data);return buffer;};
  const capacity=PLT_RESIDENT_LIMITS.tilePixels,pathCapacity=capacity*PLT_RESIDENT_LIMITS.batchSamples,auditBytes=pathCapacity*PLT_RESIDENT_LIMITS.auditVectors*16,laneBytes=capacity*PLT_RESIDENT_LIMITS.laneBytes;
  let uniform,sceneBuffer,initial,lanes,audit,status,records,weights,axisStates,anchorStates,cameraStates,cameraParameters,auditRead,lanesRead,statusRead,recordsRead,groups;
  await scoped(async()=>{
   uniform=make(160,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,'Resident parameters');sceneBuffer=make(packedScene.data.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,'Resident immutable scene',packedScene.data);
   initial=make(capacity*16,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,'Resident initial prefixes');lanes=make(laneBytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,'Resident persistent lanes');audit=make(auditBytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,'Resident bounded audit');status=make(16,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,'Resident status');
   if(axisStages)axisStates=make(capacity*256,GPUBufferUsage.STORAGE,'Resident staged GPU interval state');
   if(anchorStages)anchorStates=make(capacity*256,GPUBufferUsage.STORAGE,'Resident anchored GPU proof state');
   if(cameraStages){cameraStates=make(capacity*128,GPUBufferUsage.STORAGE,'Resident immutable GPU camera proof state');cameraParameters=make(112,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,'Resident immutable fixed-point camera coefficients',packResidentCamera(camera,o));}
   weights=make(capacity*768,GPUBufferUsage.STORAGE,'Resident bounded path terms');
   records=make(131072*64,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,'Resident bounded edge records');recordsRead=make(131072*64,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST,'Resident edge readback');
   auditRead=make(auditBytes,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST,'Resident audit readback');lanesRead=make(laneBytes,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST,'Resident lane readback');statusRead=make(16,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST,'Resident status readback');
   const groupEntries={
    rr_init:[[0,uniform],[2,initial],[3,lanes],...(anchorStages?[[9,anchorStates]]:[]),...(cameraStages?[[10,cameraStates]]:[])],
    rr_traverse_step:[[0,uniform],[1,sceneBuffer],[3,lanes],[4,audit],[7,weights]],
    rr_anchor_init_step:[[0,uniform],[3,lanes],[9,anchorStates],...(cameraStages?[[10,cameraStates],[11,cameraParameters]]:[])],
    rr_anchor_point_step:[[0,uniform],[1,sceneBuffer],[3,lanes],[8,axisStates],[9,anchorStates],...(cameraStages?[[10,cameraStates]]:[])],
    rr_axis_prepare_step:[[0,uniform],[1,sceneBuffer],[3,lanes],[8,axisStates],...(anchorStages?[[9,anchorStates]]:[]),...(cameraStages?[[10,cameraStates]]:[])],
    rr_axis_plane_step:[[0,uniform],[1,sceneBuffer],[3,lanes],[8,axisStates],...(anchorStages?[[9,anchorStates]]:[]),...(cameraStages?[[10,cameraStates]]:[])],
    rr_axis_scan_step:[[0,uniform],[1,sceneBuffer],[3,lanes],[8,axisStates]],
    rr_axis_finish_step:[[0,uniform],[1,sceneBuffer],[3,lanes],[4,audit],[8,axisStates],...(anchorStages?[[9,anchorStates]]:[]),...(cameraStages?[[10,cameraStates]]:[])],
    rr_records_step:[[0,uniform],[1,sceneBuffer],[3,lanes],[4,audit],[5,status],[6,records]],
    rr_operator_step:[[0,uniform],[3,lanes],[4,audit],[6,records],[7,weights]],
    rr_scatter_step:[[0,uniform],[1,sceneBuffer],[3,lanes],[4,audit],[5,status],[7,weights]],
   };
   groups=stageNames.map((name,i)=>device.createBindGroup({layout:pipelines[i].getBindGroupLayout(0),entries:groupEntries[name].map(([binding,buffer])=>({binding,resource:{buffer}}))}));
  },ownerSignal);check();
  const setupMs=now()-setupStart,diagnostic={buildId:PLT_RESIDENT_BUILD_ID,adapter:adapterSummary(adapter),shaderFingerprintFNV1a:fingerprint(source),scene:packedScene.metadata,dispatchLanes};
  return {destroy,diagnostic,packedScene,async run({tile,sampleStart=0,samplesPerPixel=1,wave=o.wave,rngStates,signal,onProgress=()=>{},guard=continuationGuard,axisSubmission='staged',laneChunkSize=dispatchLanes}={}){
   check(signal);if(active)throw new Error('Resident runner requires sequential batches');active=true;const cancel=()=>destroy();signal?.addEventListener('abort',cancel,{once:true});const runStart=now();
   try{
    assert(rngStates instanceof Uint32Array&&rngStates.length===o.width*o.height,'full-frame persistent RNG prefix required');
    assert(['replay-unmodelled','audit-only','axis-mirror-interval','axis-mirror-unit','axis-mirror-anchor','axis-mirror-tight','axis-mirror-camera'].includes(guard),'invalid batch continuation guard');
    assert(!isAxisGuard(guard)||axisStages,'interval guard was not compiled for this runner');
    assert(!isAnchorGuard(guard)||anchorStages,'anchor guard was not compiled for this runner');
    assert(guard!=='axis-mirror-tight'||tightStages,'tight guard was not compiled for this runner');
    assert(guard!=='axis-mirror-camera'||cameraStages,'camera guard was not compiled for this runner');
    assert(['staged','batched','round'].includes(axisSubmission),'invalid axis submission mode');
    // Per-run throughput choice only: the constructor default and pipeline
    // state remain immutable. Validate before any upload or dispatch. Each
    // chunk still completes before its shared range uniform is rewritten.
    assert(Number.isInteger(laneChunkSize)&&laneChunkSize>=1&&laneChunkSize<=128,'invalid bounded lane chunk size');
    const params=packResidentParameters(camera,{...o,wave},packedScene,{tile,sampleStart,samplesPerPixel}),pixels=tile.width*tile.height,input=new Uint32Array(pixels*4);
    new Uint32Array(params)[37]=recoveryMode==='selective'?1:0;
    new Uint32Array(params)[38]=recoveryMode==='selective'?(guard==='axis-mirror-camera'?10:guard==='axis-mirror-tight'?8:guard==='axis-mirror-anchor'?6:guard==='axis-mirror-unit'?4:guard==='axis-mirror-interval'?2:guard==='replay-unmodelled'?1:0):0;
    const floats=new Float32Array(params);if(isAxisGuard(guard)&&(![...floats.subarray(16,36),floats[39]].every(v=>v===0||Math.abs(v)>=2**-120&&Math.abs(v)<=2**120)||camera.position.some(v=>Math.abs(v)>256)))new Uint32Array(params)[38]=3;
    const recoveredPaths=new Map();let cpuSelectiveReplayMs=0,replayReadbackWallMs=0;
    for(let i=0;i<pixels;i++){const global=(tile.y+Math.floor(i/tile.width))*o.width+tile.x+i%tile.width;input.set([global,rngStates[global],sampleStart,0],i*4);}
    const read=async(output,readback,bytes)=>{check(signal);let mapped=false;try{const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(output,0,readback,0,bytes);device.queue.submit([encoder.finish()]);await wait(readback.mapAsync(GPUMapMode.READ,0,bytes),'Resident readback timeout',signal);mapped=true;check(signal);return readback.getMappedRange(0,bytes).slice(0);}finally{if(mapped)try{readback.unmap();}catch{}}};
    const stageWallMs=stageNames.map(()=>0),stageTime=name=>stageWallMs[stageNames.indexOf(name)]??0;let statusReadbackWallMs=0;
    const dispatch=async(stage,count)=>{check(signal);const phaseStart=now();const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipelines[stage]);pass.setBindGroup(0,groups[stage]);pass.dispatchWorkgroups(Math.ceil(count/32));pass.end();device.queue.submit([encoder.finish()]);await wait(device.queue.onSubmittedWorkDone(),'Resident dispatch timeout',signal);check(signal);stageWallMs[stage]+=now()-phaseStart;};
    // At most64 supported triangles. Both dependent passes for every triangle
    // are ordered in one command buffer; no readback or partial audit admission.
    const trianglePassCount=axisStages&&packedScene.metadata.axisGuard.eligible?packedScene.metadata.triangles:0;
    let gpuAxisTriangleLoopSubmitAndWaitMs=0;
    const dispatchAxisTriangles=async count=>{check(signal);const began=now(),encoder=device.createCommandEncoder();for(let triangle=0;triangle<trianglePassCount;triangle++)for(const name of ['rr_axis_plane_step','rr_axis_scan_step']){const stage=stageNames.indexOf(name),pass=encoder.beginComputePass();pass.setPipeline(pipelines[stage]);pass.setBindGroup(0,groups[stage]);pass.dispatchWorkgroups(Math.ceil(count/32));pass.end();}device.queue.submit([encoder.finish()]);await wait(device.queue.onSubmittedWorkDone(),'Resident staged triangle-loop timeout',signal);check(signal);gpuAxisTriangleLoopSubmitAndWaitMs+=now()-began;};
    // Same pending-guard barrier and exact pass order, with one bounded submit.
    // Cancellation can suppress later submissions/admission, not passes already
    // submitted together. No CPU-visible state is committed before full audit.
    let gpuAxisGuardSubmitAndWaitMs=0;
    const dispatchAxisGuard=async count=>{check(signal);const began=now(),encoder=device.createCommandEncoder();const encode=name=>{const stage=stageNames.indexOf(name),pass=encoder.beginComputePass();pass.setPipeline(pipelines[stage]);pass.setBindGroup(0,groups[stage]);pass.dispatchWorkgroups(Math.ceil(count/32));pass.end();};if(isAnchorGuard(guard))encode('rr_anchor_init_step');encode('rr_axis_prepare_step');for(let triangle=0;triangle<trianglePassCount;triangle++){encode('rr_axis_plane_step');encode('rr_axis_scan_step');}if(isAnchorGuard(guard))encode('rr_anchor_point_step');encode('rr_axis_finish_step');device.queue.submit([encoder.finish()]);await wait(device.queue.onSubmittedWorkDone(),'Resident batched axis-guard timeout',signal);check(signal);gpuAxisGuardSubmitAndWaitMs+=now()-began;};
    // Opt-in whole lane-round grouping. Every dependency retains a distinct
    // compute pass; only queue submits/waits and host yields are reduced. One
    // chunk owns the shared dispatch uniform until its submission completes.
    // Status/replay and admission remain outside this bounded command buffer.
    let gpuRoundSubmitAndWaitMs=0;
    const dispatchRound=async count=>{
     check(signal);const began=now(),encoder=device.createCommandEncoder();let passes=0;
     const encode=name=>{const stage=stageNames.indexOf(name),pass=encoder.beginComputePass();pass.setPipeline(pipelines[stage]);pass.setBindGroup(0,groups[stage]);pass.dispatchWorkgroups(Math.ceil(count/32));pass.end();passes++;};
     encode('rr_traverse_step');
     if(isAxisGuard(guard)){
      if(isAnchorGuard(guard))encode('rr_anchor_init_step');
      encode('rr_axis_prepare_step');
      for(let triangle=0;triangle<trianglePassCount;triangle++){encode('rr_axis_plane_step');encode('rr_axis_scan_step');}
      if(isAnchorGuard(guard))encode('rr_anchor_point_step');
      encode('rr_axis_finish_step');
     }
     encode('rr_records_step');encode('rr_operator_step');encode('rr_scatter_step');
     device.queue.submit([encoder.finish()]);await wait(device.queue.onSubmittedWorkDone(),'Resident lane-round timeout',signal);check(signal);gpuRoundSubmitAndWaitMs+=now()-began;return passes;
    };
    let rounds=0,dispatches=0,dispatchReadbackMs=0,recordCount=0;
    await scoped(async()=>{device.queue.writeBuffer(uniform,0,params);device.queue.writeBuffer(initial,0,input);device.queue.writeBuffer(status,0,new Uint32Array(4));await dispatch(0,pixels);},signal);
    const maxRounds=(o.maxDepth+65)*(17+packedScene.metadata.wedges+1)*samplesPerPixel;
    for(;;){
     check(signal);const phaseStart=now();device.queue.writeBuffer(status,0,new Uint32Array(2));
     await scoped(async()=>{for(let first=0;first<pixels;first+=laneChunkSize){check(signal);const count=Math.min(laneChunkSize,pixels-first);device.queue.writeBuffer(uniform,48,new Uint32Array([first,count,PLT_RESIDENT_LIMITS.auditVectors,0]));if(axisSubmission==='round'){dispatches+=await dispatchRound(count);await new Promise(resolve=>setTimeout(resolve,0));continue;}for(let stage=1;stage<pipelines.length;stage++){if((!isAxisGuard(guard)&&stageNames[stage].startsWith('rr_axis_'))||(!isAnchorGuard(guard)&&stageNames[stage].startsWith('rr_anchor_')))continue;if(axisSubmission==='batched'&&(stageNames[stage].startsWith('rr_axis_')||stageNames[stage].startsWith('rr_anchor_'))){if(stageNames[stage]===(isAnchorGuard(guard)?'rr_anchor_init_step':'rr_axis_prepare_step')){await dispatchAxisGuard(count);dispatches+=2+2*trianglePassCount+(isAnchorGuard(guard)?2:0);await new Promise(resolve=>setTimeout(resolve,0));}continue;}if(stageNames[stage]==='rr_axis_scan_step')continue;if(stageNames[stage]==='rr_axis_plane_step'){await dispatchAxisTriangles(count);dispatches+=2*trianglePassCount;}else{await dispatch(stage,count);dispatches++;}await new Promise(resolve=>setTimeout(resolve,0));}}},signal);
     const statusStart=now(),progress=new Uint32Array(await read(status,statusRead,16));statusReadbackWallMs+=now()-statusStart;recordCount=progress[2];assert(recordCount<=131072,'edge record count overflow');rounds++;dispatchReadbackMs+=now()-phaseStart;
     if(progress[1]){
      const replayReadStart=now(),pausedData=await read(lanes,lanesRead,pixels*PLT_RESIDENT_LIMITS.laneBytes),paused=new Uint32Array(pausedData),samples=[];
      replayReadbackWallMs+=now()-replayReadStart;
      for(let lane=0;lane<pixels;lane++){const at=lane*(PLT_RESIDENT_LIMITS.laneBytes/4);if(paused[at+11])samples.push({lane,pixel:paused[at],sample:paused[at+2]-1,depth:paused[at+4],nullSteps:paused[at+5],flags:paused[at+11],intervalReason:paused[at+111]});}
      if(recoveryMode!=='selective'||!isResidentReplayable(progress[1])||!samples.length){
       const error=new Error(`Resident candidate rejected: diagnostic flags ${progress[1]} (ambiguity/unsupported range/capacity); CPU shadow retained`);error.name='PLTResidentRejected';error.diagnostic={...diagnostic,stage:'resident-path',flags:progress[1],reasons:Object.entries(replayReasons).filter(([,bit])=>progress[1]&bit).map(([name])=>name),rounds,dispatches,rejectedPaths:samples.slice(0,8),timings:{scenePackingMs,gpuSetupWallMs:setupMs,gpuResidentTotalWallMs:now()-runStart}};throw error;
      }
      const {buildResidentReferenceBatch}=await import('./plt-resident-reference.js');check(signal);
      let recoveredFlags=0,lastReplayYield=now();
      const checkReplay=()=>{check(signal);if(now()-runStart>timeoutMs)throw Object.assign(new Error('Resident batch wall-clock timeout during CPU replay'),{name:'TimeoutError'});};
      for(const info of samples){
       const {lane,pixel,sample,flags}=info,at=lane*(PLT_RESIDENT_LIMITS.laneBytes/4),slot=lane*samplesPerPixel+sample-sampleStart;
       assert(paused[at+3]===3&&isResidentReplayable(flags),'unpaused or nonrecoverable path');
       assert(pixel===input[lane*4]&&sample>=sampleStart&&sample<sampleStart+samplesPerPixel&&!recoveredPaths.has(slot),'invalid recovery sample ownership');
       const cpuStart=now(),prefixes=new Uint32Array(o.width*o.height);prefixes[pixel]=paused[at+17];
       const job=await wait(buildResidentReferenceBatch(replayScene,camera,{...o,wave},{tile:{x:pixel%o.width,y:Math.floor(pixel/o.width),width:1,height:1},sampleStart:sample,samplesPerPixel:1,rngStates:prefixes,signal,check:checkReplay}),'Resident selective CPU replay timeout',signal);checkReplay();
       const path=job.audit.paths[0];assert(job.audit.complete&&job.audit.paths.length===1&&path.key.pixel===pixel&&path.key.sample===sample&&path.rng.start===paused[at+17],'CPU replay identity mismatch');
       cpuSelectiveReplayMs+=now()-cpuStart;
       if(now()-lastReplayYield>=8){await new Promise(resolve=>setTimeout(resolve,0));checkReplay();lastReplayYield=now();}
       recoveredPaths.set(slot,{path,flags,intervalReason:info.intervalReason,triggerDepth:info.depth});recoveredFlags|=flags;
       // Restore only the next RNG prefix and lane lifecycle, never CPU geometry
       // or a path tape. Failed partial radiance was not added by rr_finish.
       paused[at+1]=path.rng.final;paused[at+3]=paused[at+2]>=sampleStart+samplesPerPixel?2:0;paused[at+11]=0;
       device.queue.writeBuffer(lanes,lane*PLT_RESIDENT_LIMITS.laneBytes,paused.subarray(at,at+PLT_RESIDENT_LIMITS.laneBytes/4));
      }
      assert(recoveredFlags===progress[1],'unowned diagnostic flags');
      // The active count was computed before CPU replay; take one more bounded
      // round to observe the resumed or completed lane, even on its last sample.
      progress[0]=Math.max(progress[0],1);
     }
     onProgress({round:rounds,active:progress[0],pixels,fraction:1-progress[0]/pixels});check(signal);if(progress[0]===0)break;
     if(rounds>=maxRounds)throw new Error('Resident continuation bound exceeded; candidate withheld');
    }
    const readStart=now(),laneData=await read(lanes,lanesRead,pixels*PLT_RESIDENT_LIMITS.laneBytes),auditData=await read(audit,auditRead,pixels*samplesPerPixel*PLT_RESIDENT_LIMITS.auditVectors*16);check(signal);
    const recordData=recordCount?await read(records,recordsRead,recordCount*64):new ArrayBuffer(0);
    const decoded=decodeResidentAudit(auditData,{tile,sampleStart,samplesPerPixel,options:o,packedScene,laneData,recordData,recoveredPaths}),finalRngStates=new Uint32Array(pixels),lu=new Uint32Array(laneData);
    for(let i=0;i<pixels;i++)finalRngStates[i]=lu[i*(PLT_RESIDENT_LIMITS.laneBytes/4)+1];
    return {candidate:true,admitted:false,audit:decoded,image:decoded.reference,pixelIndices:decoded.pixelIndices,finalRngStates,diagnostic,recovery:{mode:recoveryMode,continuationGuard:guard,axisSubmission,laneChunkSize,paths:recoveredPaths.size,keys:[...recoveredPaths.values()].map(({path,flags,intervalReason,triggerDepth})=>({key:path.key,flags,intervalReason,triggerDepth,reasons:Object.entries(replayReasons).filter(([,bit])=>flags&bit).map(([name])=>name)})),completeCPUPathReplay:true},timings:{scenePackingMs,gpuSetupWallMs:setupMs,gpuInitSubmitAndWaitMs:stageWallMs[0],gpuTraversalSubmitAndWaitMs:stageWallMs[1],gpuRecordsSubmitAndWaitMs:stageTime('rr_records_step'),gpuOperatorSubmitAndWaitMs:stageTime('rr_operator_step'),gpuScatterSubmitAndWaitMs:stageTime('rr_scatter_step'),gpuAnchorInitSubmitAndWaitMs:stageTime('rr_anchor_init_step'),gpuAnchorPointSubmitAndWaitMs:stageTime('rr_anchor_point_step'),gpuAxisPrepareSubmitAndWaitMs:stageTime('rr_axis_prepare_step'),gpuAxisTriangleLoopSubmitAndWaitMs,gpuAxisGuardSubmitAndWaitMs,gpuRoundSubmitAndWaitMs,gpuAxisFinishSubmitAndWaitMs:stageTime('rr_axis_finish_step'),gpuStatusReadbackWallMs:statusReadbackWallMs,cpuSelectiveReplayMs,replayReadbackWallMs,gpuResidentDispatchReadbackWallMs:dispatchReadbackMs,gpuFinalReadbackDecodeMs:now()-readStart,gpuResidentTotalWallMs:now()-runStart,cpuReferenceProvided:false,cpuSelectiveReplayProvided:recoveredPaths.size>0},rounds,dispatches,scope:PLT_RESIDENT_SCOPE};
   }catch(error){error.diagnostic??={...diagnostic,error:error.message};destroy();throw error;}
   finally{signal?.removeEventListener('abort',cancel);active=false;}
  }};
 }catch(error){destroy();throw error;}
}
