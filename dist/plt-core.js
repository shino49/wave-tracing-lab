/** Browser-local, monochromatic backward positive-light transport.
 * Native centre-connected/local-plane UTD approximation, finite path depth.
 * Scene/camera radiance proxy (relative units); NOT a coherent global field.
 * Geometry and sampling are CPU; the same positive path tape is GPU-replayable.
 */
import {prepareMesh,makePresetOBJ,transformMesh,add,sub,scale,dot,cross,length,normalize} from './mesh-geometry.js';
import {preparePLTScene,rayNearest,segmentVisible,firstConeContact,collectRegionGeometry} from './plt-geometry.js';
import {makeAperture,sampleFSD,evaluateFSD,buildFSDOperator,evaluateOperator} from './plt-fsd.js';
import {sampleVisibleWavelength,hashSpectralIndex,sourceSpectrum,reflectanceSpectrum,validateSpectralOptions} from './plt-spectrum.js';
import {createPLTDemoScene,PLT_DEMO_SCENE_ID} from './plt-demo-scene.js';
export const PLT_SCOPE='3D backward PLT architecture; native centre-connection/local-plane UTD and angular proposal; finite-depth positive path ensemble; relative camera response, not calibrated radiance';
export const PLT_LIMITS=Object.freeze({maxTriangles:4096,maxPixels:16384,maxSamples:512,maxDepth:6,maxEvents:250000,maxTerms:1000000,maxNullSteps:64});
// Only the tiled engine accepts the larger frame. A legacy full-frame tape keeps
// its original cap; the progressive cap never enlarges a batch's path budget.
export const PLT_PROGRESSIVE_LIMITS=Object.freeze({maxWidth:256,maxHeight:192,maxPixels:49152,maxTileSize:32,maxBatchSamples:4,maxBatchPaths:4096,maxEvents:24576,maxTerms:131072,maxEdgeRecords:131072,maxRuntimeMs:600000});
const TAU=2*Math.PI,now=()=>globalThis.performance?.now?.()??Date.now();
const num=(v,a,b,n)=>{if(!Number.isFinite(v)||v<a||v>b)throw new RangeError(`${n} must be within ${a}…${b}`);return v;};
const vec=(v,n)=>{if(!Array.isArray(v)||v.length!==3||!v.every(Number.isFinite))throw new RangeError(`Invalid ${n}`);return [...v];};
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
export function randomGenerator(seed=1){let a=seed>>>0;const random=()=>{a=(a+0x6D2B79F5)>>>0;let t=a;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296;};random.getState=()=>a;return random;}
export function cameraFrame(camera){const forward=normalize(sub(camera.target,camera.position));const right=normalize(cross(forward,camera.up??[0,1,0]));return {forward,right,up:cross(right,forward)};}
export function sampleCamera(camera,x,y,width,height,rng,wavelengthMm,wave=true,spectral=false){
 const f=cameraFrame(camera),fov=num(camera.fov??42,5,120,'Camera field of view')*Math.PI/180,half=Math.tan(fov/2),aspect=width/height;
 const u=(2*(x+rng())/width-1)*aspect*half,v=(1-2*(y+rng())/height)*half;
 const direction=normalize(add(f.forward,add(scale(f.right,u),scale(f.up,v))));
 // Native virtual perspective camera: 10 mm image-plane distance, .75 pixel
 // spatial radius and pixel angular width. Like upstream this virtual sensor
 // does not impose a physical aperture uncertainty limit.
 const sensorScale=camera.sensorScale??1,pixelWidth=20*half/height*sensorScale,k=TAU/wavelengthMm;
 let radius0=.75*pixelWidth,tanAngle=pixelWidth/(10*sensorScale);
 if(spectral){
  // Visible-only camera specialization: identify the native pixel support with
  // a Gaussian 1/e²-intensity waist, with a fixed 3 µm physical floor. Added
  // independent angular variance is an explicit approximation, not a lens model.
  radius0=Math.max(radius0,.003);tanAngle=Math.hypot(tanAngle,wavelengthMm/(Math.PI*radius0));
  if(!(Number.isFinite(tanAngle)&&tanAngle<=.15))throw new RangeError('Visible camera envelope exceeds the 0.15 paraxial-slope guard; increase resolution or reduce field of view');
 }
 const uncertaintyScale=1;
 return {origin:[...camera.position],direction,radius0:wave?radius0:0,tanAngle:wave?tanAngle:0,
  near:1e-7,far:Infinity,k,wavelengthMm,initialWeight:1,pixelMeasure:'uniform projected-pixel area; camera response divided by sampling density',uncertaintyScale};
}
export function cosineDirection(normal,rng){const a=TAU*rng(),r=Math.sqrt(rng()),z=Math.sqrt(Math.max(0,1-r*r));const u=normalize(cross(Math.abs(normal[2])<.9?[0,0,1]:[0,1,0],normal)),v=cross(normal,u);const direction=add(scale(normal,z),add(scale(u,r*Math.cos(a)),scale(v,r*Math.sin(a))));return {direction,pdf:z/Math.PI,cosine:z};}
function copyBeam(b){return {...b,origin:[...b.origin],direction:[...b.direction]};}
function radius(b,t){return b.radius0+b.tanAngle*t;}
function reprojectRegion(b,center,distance,direction){
 const r=radius(b,distance),c=clamp(dot(b.direction,direction),-1,1),s=Math.sqrt(Math.max(0,1-c*c));
 // Circular conservative outer envelope of the native (r,r,2r) ellipsoid.
 // Inherited angular extent. Never a new minimum-uncertainty waist.
 const projected=r*Math.sqrt(1+3*s*s),axial=r*Math.sqrt(1+3*c*c);
 return {...b,origin:[...center],direction:[...direction],radius0:projected+b.tanAngle*axial,near:1e-7};
}
function reprojectSurface(b,hit,direction){
 // Native static surface footprint; circumscribed circle avoids losing any
 // projected ellipse. This conservative circular specialization enlarges support.
 return {...b,origin:[...hit.point],direction:[...direction],radius0:radius(b,hit.distance),near:1e-7};
}
export function traversePLT(scene,beam,wave=true){
 if(!wave){const hit=rayNearest(scene,beam.origin,beam.direction,{minDistance:beam.near??1e-7,maxDistance:beam.far??Infinity});return hit?{center:hit.point,distance:hit.distance,hit,ballistic:true,region:null}:null;}
 let travelled=0;
 for(let segment=0;segment<=16;segment++){
  const B=segment===16?Infinity:Math.min(65536,16*4**segment)*beam.wavelengthMm;
  const hit=rayNearest(scene,beam.origin,beam.direction,{minDistance:Math.max(beam.near??1e-7,travelled),maxDistance:Math.min(beam.far??Infinity,travelled+B*1.001)});
  if(hit){const r=radius(beam,hit.distance),region={center:hit.point,direction:beam.direction,radius:r,depth:2*r};return {center:hit.point,distance:hit.distance,hit,ballistic:true,region};}
  travelled+=B;if(!Number.isFinite(travelled)||travelled>=(beam.far??Infinity))return null;
  const contact=firstConeContact(scene,{...beam,near:travelled,far:beam.far??Infinity});
  if(!contact)return null;
  if(contact.distance-travelled<radius(beam,travelled)/2)continue;
  const t=contact.distance,center=add(beam.origin,scale(beam.direction,t)),r=radius(beam,t),region={center,direction:beam.direction,radius:r,depth:2*r};
  const primary=rayNearest(scene,beam.origin,beam.direction,{minDistance:Math.max(t,beam.near??1e-7),maxDistance:t+2*r});
  return {center,distance:t,hit:primary,ballistic:false,region};
 }
 throw new Error('Traversal failed to terminate');
}
const scaled=(terms,w)=>w===0?[]:terms.map(t=>({...t,weight:t.weight*w,eventIds:[...t.eventIds]}));
const withEvent=(terms,id)=>terms.map(t=>({...t,weight:t.weight,eventIds:[...t.eventIds,id]}));
const termValue=(t,events)=>t.eventIds.reduce((s,i)=>s*events[i].referenceF,t.weight);
const termTotal=(ts,events)=>ts.reduce((s,t)=>s+termValue(t,events),0);
const normalAt=(hit,direction)=>dot(hit.geometricNormal??hit.triangle.normal,direction)<0?[...(hit.geometricNormal??hit.triangle.normal)]:scale(hit.geometricNormal??hit.triangle.normal,-1);
function spectralMaterial(scene,m,nm,source){return nm===undefined?m:{...m,albedo:m.albedo*reflectanceSpectrum(nm,m.spectralReflectance??'neutral'),emission:(m.emission??0)*sourceSpectrum(nm,source??scene.spectralSource??'D65')};}
function materialFor(scene,hit,nm,source){const m=scene.materials[hit.materialIndex??hit.triangle?.materialIndex??0];if(!m)throw new Error('Missing surface material');return spectralMaterial(scene,m,nm,source);}
function sampleEmitter(scene,point,rng,nm,source){
 if(!scene.emitters.length)return null;let r=rng()*scene.emitterArea,chosen=scene.emitters.at(-1);for(const t of scene.emitters){r-=t.area;if(r<=0){chosen=t;break;}}
 const a=Math.sqrt(rng()),b=rng(),p=add(scale(chosen.a,1-a),add(scale(chosen.b,a*(1-b)),scale(chosen.c,a*b))),delta=sub(p,point),distance=length(delta);if(distance<1e-8)return null;
 const direction=scale(delta,1/distance),m=spectralMaterial(scene,scene.materials[chosen.materialIndex],nm,source),cosLight=m.twoSidedEmission?Math.abs(dot(chosen.normal,direction)):Math.max(0,-dot(chosen.normal,direction));
 return {point:p,direction,distance,cosLight,pdf:cosLight>0?distance*distance/(scene.emitterArea*cosLight):Infinity,emission:m.emission??0,triangle:chosen};
}
function mis(a,b){return a>0?a*a/(a*a+b*b):0;}
export function traceCameraPath(scene,initialBeam,options,rng,job,pixel=0,audit=null){
 if(job.deferOperatorEvaluation&&audit)throw new Error('Deferred operators cannot produce a CPU power audit');
 let beam=copyBeam(initialBeam),terms=[{weight:1,eventIds:[]}],pending=null,previousPDF=0,previousPosition=beam.origin,previousWasDelta=true;
 let depth=0,nullSteps=0,fsdCount=0,diffractionCount=0,surfaceCount=0,termination='max-depth';const path=[];
 const addContribution=(ts,weight,type)=>{if(!(weight>0))return;const firstTerm=job.terms.length;for(const t of scaled(ts,weight)){if(!Number.isFinite(t.weight)||t.weight<0)throw new Error('Nonfinite path weight');if(job.tapeLimits&&job.terms.length>=job.tapeLimits.maxTerms)throw new RangeError('Tile path terms exceed safety budget; use a smaller tile or sample batch');job.terms.push({...t,pixel,...(initialBeam.xyzWeight?{xyzWeight:[...initialBeam.xyzWeight],spectralSample:initialBeam.spectralSample}: {})});}job.stats.contributions++;if(fsdCount>=2)job.stats.twoFSDContributions++;if(surfaceCount>=2)job.stats.twoSurfaceContributions++;path.push({type,depth,weight});audit?.onContribution?.({type,depth,weight,firstTerm,termEnd:job.terms.length});};
 while(depth<options.maxDepth){
  const vertex=traversePLT(scene,beam,options.wave);job.stats.traversals++;
  audit?.onTraversal?.({vertex,depth,nullSteps,beam});
  if(!vertex){job.stats.escaped++;termination='escaped';break;}
  if(pending){
   const deferred=job.deferOperatorEvaluation===true,e=(deferred?buildFSDOperator:evaluateFSD)(scene,pending.aperture,pending.previousBeam.origin,vertex.center,pending.previousBeam);
   if(!deferred&&(!Number.isFinite(e.F)||e.F<0))throw new Error('Invalid local diffraction power');
   if(job.tapeLimits){job.edgeRecords+=e.operator.edges.length;if(job.events.length>=job.tapeLimits.maxEvents||job.edgeRecords>job.tapeLimits.maxEdgeRecords)throw new RangeError('Tile operators exceed safety budget; use a smaller tile or sample batch');}
   const id=job.events.length;job.events.push({...e.operator,...(deferred?{}:{referenceF:e.F}),source:[...pending.previousBeam.origin],destination:[...vertex.center]});job.stats.fsdEvaluations++;job.stats.edgeCoefficients+=e.operator.edges.length;
   audit?.onLocalEvent?.({id,depth,sampled:pending.sampled,event:job.events[id],soft:e.soft,hard:e.hard,diagnostics:e.diagnostics});
   if(e.operator.edges.length){diffractionCount++;job.stats.diffractingEvaluations++;}
   for(const warning of e.diagnostics?.warnings??[])job.stats.approximationWarnings[warning]=(job.stats.approximationWarnings[warning]??0)+1;
   job.stats.endpointExclusions+=(e.diagnostics?.endpointExclusions??0);job.stats.invalidEdges+=(e.diagnostics?.invalidEdges??0);
   terms=pending.sampled?withEvent(terms,id):terms.concat(withEvent(pending.previousTerms,id).map(t=>({...t,neeMatched:false})));pending=null;fsdCount++;
  }
  const previousBeam=copyBeam(beam),previousTerms=terms.map(t=>({...t,eventIds:[...t.eventIds]}));
  let aperture=null;
  if(options.wave&&vertex.region&&vertex.region.radius>0){const region={...vertex.region,...collectRegionGeometry(scene,vertex.region)};aperture=makeAperture(region,beam.direction,beam.k,options.boundary);if(!aperture.edges.length)aperture=null;}
  path.push({type:vertex.hit?'surface':'region',depth,center:vertex.center,triangleId:vertex.hit?.triangleId??null,edgeCount:aperture?.edges.length??0,radius:vertex.region?.radius??0});
  audit?.onVertex?.({vertex,depth,nullSteps,aperture});
  if(vertex.hit){
   const hit=vertex.hit,m=materialFor(scene,hit,beam.spectralNm,options.spectralSource),n=normalAt(hit,beam.direction),emissive=(m.emission??0)>0;
   if(emissive&&(m.twoSidedEmission||dot(hit.geometricNormal??hit.triangle.normal,beam.direction)<0)){
    let w=1;if(options.nee&&!previousWasDelta&&previousPDF>0){const dist=length(sub(hit.point,previousPosition)),cosLight=Math.abs(dot(hit.geometricNormal??hit.triangle.normal,beam.direction)),p=dist*dist/(scene.emitterArea*Math.max(cosLight,1e-30));w=mis(previousPDF,p);}
    // The additive deferred FSD term has no matching preceding surface NEE.
    // Apply MIS only to terms actually sampled by that competing BSDF technique.
    addContribution(terms.map(t=>({...t,weight:t.weight*(t.neeMatched===false?1:w)})),m.emission,'emission');
   }
   if(!(m.albedo>0)){termination='absorbed';break;}
   if(options.nee&&m.type==='diffuse'&&depth+1<options.maxDepth){
    const light=sampleEmitter(scene,hit.point,rng,beam.spectralNm,options.spectralSource);if(light&&light.cosLight>0){const cosine=Math.max(0,dot(n,light.direction));
     if(cosine>0&&segmentVisible(scene,hit.point,light.point,{endpointEpsilon:1e-6})){
      const bsdfPDF=cosine/Math.PI;addContribution(terms,m.albedo*bsdfPDF/light.pdf*light.emission*mis(light.pdf,bsdfPDF),'surface-nee');
     }
    }
   }
   let next,pdf,delta;
   if(m.type==='mirror'){next=sub(beam.direction,scale(n,2*dot(beam.direction,n)));pdf=0;delta=true;}else{const s=cosineDirection(n,rng);next=s.direction;pdf=s.pdf;delta=false;}
   audit?.onScatter?.({type:m.type==='mirror'?'mirror':'diffuse',depth,direction:next,pdf,weight:m.albedo});
   terms=scaled(terms,m.albedo).map(t=>({...t,neeMatched:true}));beam=reprojectSurface(beam,hit,next);surfaceCount++;job.stats.surfaceInteractions++;
   if(aperture)pending={aperture,previousBeam,previousTerms,sampled:false};
   previousPosition=[...hit.point];previousPDF=pdf;previousWasDelta=delta;depth++;
  }else if(aperture){
   const s=sampleFSD(aperture,previousPosition,rng);audit?.onScatter?.({type:'fsd',depth,sample:s});if(!s?.valid||!(s.weight>0)||!Number.isFinite(s.weight)){job.stats.samplingRejected++;termination='sampling-rejected';break;}
   terms=scaled(terms,s.weight);beam=reprojectRegion(beam,vertex.center,vertex.distance,s.wo);pending={aperture,previousBeam,previousTerms,sampled:true};
   previousPosition=[...vertex.center];previousPDF=0;previousWasDelta=true;job.stats.regionInteractions++;depth++;
  }else{
   if(++nullSteps>PLT_LIMITS.maxNullSteps)throw new Error('Too many unresolved null regions');
   const shift=vertex.distance+Math.max(1e-6,vertex.region?.radius*.001??1e-6);beam={...beam,origin:add(beam.origin,scale(beam.direction,shift)),radius0:radius(beam,shift)};job.stats.nullRestarts++;
  }
  if(job.events.length>PLT_LIMITS.maxEvents||job.terms.length>PLT_LIMITS.maxTerms)throw new Error('Path tape exceeds browser safety budget; reduce samples or resolution');
 }
 job.stats.maxDepthReached=Math.max(job.stats.maxDepthReached,depth);job.stats.maxFSDPerPath=Math.max(job.stats.maxFSDPerPath,fsdCount);job.stats.maxSurfacePerPath=Math.max(job.stats.maxSurfacePerPath,surfaceCount);if(fsdCount>=2)job.stats.twoFSDPaths++;if(diffractionCount>=2)job.stats.twoDiffractingPaths++;job.stats.maxDiffractingPerPath=Math.max(job.stats.maxDiffractingPerPath,diffractionCount);
 if(job.paths.length<12&&((diffractionCount>=2)||(job.paths.length<3)))job.paths.push({pixel,fsdCount,diffractionCount,surfaceCount,events:path});
 audit?.onTermination?.({reason:termination,depth,nullSteps,fsdCount,diffractionCount,surfaceCount,pending:!!pending,pathEvents:path});
 return {depth,fsdCount,diffractionCount,surfaceCount};
}
export function replayPLTXYZ(job,eventPowers=null){
 const xyz=new Float64Array(job.width*job.height*3);
 for(const t of job.terms){let v=t.weight;for(const id of t.eventIds)v*=eventPowers?eventPowers[id]:job.events[id].referenceF;
  if(!Number.isFinite(v)||v<0||!Array.isArray(t.xyzWeight)||t.xyzWeight.length!==3||!t.xyzWeight.every(w=>Number.isFinite(w)&&w>=0))throw new Error('Invalid spectral reconstructed sample');
  for(let c=0;c<3;c++){const value=v*t.xyzWeight[c]/job.samplesPerPixel;if(!Number.isFinite(value))throw new Error('Spectral sample overflow');xyz[3*t.pixel+c]+=value;if(!Number.isFinite(xyz[3*t.pixel+c]))throw new Error('Spectral sum overflow');}
 }return xyz;
}
export function replayPLT(job,eventPowers=null){
 if(job.spectralMode==='visible'){const xyz=replayPLTXYZ(job,eventPowers);return Float64Array.from({length:job.width*job.height},(_,i)=>xyz[3*i+1]);}
 const image=new Float64Array(job.width*job.height);for(const t of job.terms){let v=t.weight;for(const id of t.eventIds)v*=eventPowers?eventPowers[id]:job.events[id].referenceF;if(!Number.isFinite(v)||v<0)throw new Error('Invalid reconstructed sample');image[t.pixel]+=v/job.samplesPerPixel;}return image;
}
function finishSpectralReference(job){if(job.spectralMode==='visible'){job.referenceXYZ=replayPLTXYZ(job);job.reference=Float64Array.from({length:job.width*job.height},(_,i)=>job.referenceXYZ[3*i+1]);}else job.reference=replayPLT(job);}
function spectralBeam(camera,x,y,width,height,o,pixel,sample,rng){
 const spectral=sampleVisibleWavelength(o.seed,pixel,sample),beam=sampleCamera(camera,x,y,width,height,rng,spectral.wavelengthMm,o.wave,true);
 return {...beam,spectralNm:spectral.nm,xyzWeight:spectral.xyzWeight,spectralSample:sample};
}
export function validatePLTSpectralScene(scene,camera){
 // Conservative engineering admission for fixed geometry tolerances. This is
 // not a proof that high-frequency UTD is valid at every endpoint or angle.
 for(const triangle of scene.triangles){const lengths=[length(sub(triangle.a,triangle.b)),length(sub(triangle.b,triangle.c)),length(sub(triangle.c,triangle.a))];
  if(Math.min(...lengths)<.004||2*triangle.area/Math.max(...lengths)<.002)throw new RangeError('Visible geometry is below the 4 µm edge / 2 µm altitude numeric guard');
  if([...triangle.a,...triangle.b,...triangle.c].some(x=>!Number.isFinite(x)||Math.abs(x)>100))throw new RangeError('Visible geometry exceeds the ±100 mm numeric guard');
 }
 if([...camera.position,...camera.target].some(x=>!Number.isFinite(x)||Math.abs(x)>100))throw new RangeError('Visible camera exceeds the ±100 mm numeric guard');
 return true;
}
function stats(){return {diffractingEvaluations:0,twoDiffractingPaths:0,twoDiffractingContributions:0,maxDiffractingPerPath:0,approximationWarnings:{},traversals:0,escaped:0,contributions:0,fsdEvaluations:0,edgeCoefficients:0,endpointExclusions:0,invalidEdges:0,regionInteractions:0,surfaceInteractions:0,samplingRejected:0,nullRestarts:0,twoFSDPaths:0,twoFSDContributions:0,twoSurfaceContributions:0,maxFSDPerPath:0,maxSurfacePerPath:0,maxDepthReached:0};}
export function validatePLTOptions(input={}, {progressive=false}={}){const o={width:64,height:48,samplesPerPixel:32,maxDepth:4,wavelengthMm:.025,spectralMode:'monochrome',spectralSource:'D65',spectralMaterial:'neutral',wave:true,nee:true,boundary:'unpolarized',seed:7391,...input};for(const [k,a,b] of [['width',8,progressive?PLT_PROGRESSIVE_LIMITS.maxWidth:128],['height',8,progressive?PLT_PROGRESSIVE_LIMITS.maxHeight:128],['samplesPerPixel',1,PLT_LIMITS.maxSamples],['maxDepth',1,6]]){num(o[k],a,b,k);if(!Number.isInteger(o[k]))throw new RangeError(`${k} must be integer`);}if(o.width*o.height>(progressive?PLT_PROGRESSIVE_LIMITS.maxPixels:PLT_LIMITS.maxPixels))throw new RangeError('Too many pixels');if(o.spectralMode!=='visible')num(o.wavelengthMm,.0001,.5,'Wavelength');if(!['soft','hard','unpolarized'].includes(o.boundary))throw new RangeError('Invalid boundary');validateSpectralOptions(o);return o;}
export async function buildPLTJob(scene,camera,input={}, {onProgress=()=>{},signal,yieldEvery=32}={}){
 const o=validatePLTOptions(input),start=now();if(o.spectralMode==='visible')validatePLTSpectralScene(scene,camera);const job={...o,events:[],terms:[],paths:[],stats:stats(),camera,scope:PLT_SCOPE};
 if(scene.triangles.length>PLT_LIMITS.maxTriangles)throw new RangeError(`Renderer cap: ${PLT_LIMITS.maxTriangles} triangles`);
 let done=0;const total=o.width*o.height;
 for(let pixel=0;pixel<total;pixel++){
  if(signal?.aborted)throw Object.assign(new Error('Render cancelled'),{name:'AbortError'});
  const rng=randomGenerator((o.seed^Math.imul(pixel+1,0x9E3779B1))>>>0),x=pixel%o.width,y=Math.floor(pixel/o.width);
  for(let s=0;s<o.samplesPerPixel;s++){const pathRng=o.spectralMode==='visible'?randomGenerator(hashSpectralIndex(o.seed,pixel,s,0x50415448)):rng,beam=o.spectralMode==='visible'?spectralBeam(camera,x,y,o.width,o.height,o,pixel,s,pathRng):sampleCamera(camera,x,y,o.width,o.height,pathRng,o.wavelengthMm,o.wave);traceCameraPath(scene,beam,o,pathRng,job,pixel);}
  if(++done%yieldEvery===0){onProgress({fraction:done/total,pixels:done,total,events:job.events.length});await new Promise(resolve=>setTimeout(resolve,0));}
 }
 finishSpectralReference(job);
 const multiTerms=job.terms.filter(t=>t.eventIds.filter(id=>job.events[id].edges.length>0).length>=2&&termValue(t,job.events)>0);
 job.stats.twoDiffractingContributions=multiTerms.length;
 job.multiRegionEvidence=multiTerms.slice(0,8).map(t=>({pixel:t.pixel,sampleWeight:t.weight,contribution:termValue(t,job.events)/job.samplesPerPixel,operators:t.eventIds.map(id=>({id,F:job.events[id].referenceF,directAmplitude:job.events[id].directAmplitude,source:job.events[id].source,destination:job.events[id].destination,edges:job.events[id].edges.map(e=>({edgeId:e.edgeId,point:e.point,ro:e.ro,sinBeta:e.sinBeta}))}))}));
 job.timings={cpuTraversalAndReferenceMs:now()-start};job.sceneSummary={triangles:scene.triangles.length,wedges:scene.wedges.length,unsupportedEdges:scene.unsupportedEdges?.length??0,emitterArea:scene.emitterArea};
 job.measurement={unit:'relative native-PLT camera response',normalization:'sample arithmetic mean; fixed source radiance, no image-wise normalization',mean:job.reference.reduce((s,x)=>s+x,0)/total,max:Math.max(...job.reference),nonzero:job.reference.reduce((s,x)=>s+(x>0),0)};return job;
}
/** One uint32 of state per global pixel. Resuming this exact sequential RNG
 * preserves every v11 path, including its variable number of random draws. */
export function createPLTRandomStates(input={}){
 const o=validatePLTOptions(input,{progressive:true});
 return Uint32Array.from({length:o.width*o.height},(_,pixel)=>(o.seed^Math.imul(pixel+1,0x9E3779B1))>>>0);
}
export function emptyPLTStats(){return stats();}
export function mergePLTStats(target,source){
 for(const [key,value] of Object.entries(source)){
  if(key==='approximationWarnings'){for(const [warning,count] of Object.entries(value))target.approximationWarnings[warning]=(target.approximationWarnings[warning]??0)+count;}
  else if(key.startsWith('max'))target[key]=Math.max(target[key]??0,value);
  else target[key]=(target[key]??0)+value;
 }return target;
}
export function measurePLTImage(image){
 let sum=0,max=0,nonzero=0;for(const value of image){if(!Number.isFinite(value)||value<0)throw new Error('Invalid accumulated camera response');sum+=value;max=Math.max(max,value);nonzero+=value>0;}
 return {unit:'relative native-PLT camera response',normalization:'sample arithmetic mean; fixed source radiance, no image-wise normalization',mean:sum/image.length,max,nonzero};
}
/** Bounded local tape, with full-frame sensor coordinates. Local pixel IDs are
 * solely the GPU tape ABI; pixelIndices maps them back to the full camera. */
export async function buildPLTBatch(scene,camera,input={}, {tile,sampleStart=0,samplesPerPixel=1,rngStates,onProgress=()=>{},signal,check=()=>{},yieldEvery=16,audit=null,deferOperatorEvaluation=false}={}){
 const o=validatePLTOptions(input,{progressive:true}),start=now(),limits=PLT_PROGRESSIVE_LIMITS;if(o.spectralMode==='visible')validatePLTSpectralScene(scene,camera);
 if(typeof deferOperatorEvaluation!=='boolean')throw new TypeError('deferOperatorEvaluation must be boolean');
 if(deferOperatorEvaluation&&(o.spectralMode==='visible'||audit))throw new Error('Deferred operators support monochrome unaudited tapes only');
 if(scene.triangles.length>PLT_LIMITS.maxTriangles)throw new RangeError(`Renderer cap: ${PLT_LIMITS.maxTriangles} triangles`);
 if(!tile||!['x','y','width','height'].every(k=>Number.isInteger(tile[k])))throw new RangeError('Invalid tile');
 if(tile.x<0||tile.y<0||tile.width<1||tile.height<1||tile.width>limits.maxTileSize||tile.height>limits.maxTileSize||tile.x+tile.width>o.width||tile.y+tile.height>o.height)throw new RangeError('Tile outside bounded frame');
 if(!Number.isInteger(sampleStart)||sampleStart<0||!Number.isInteger(samplesPerPixel)||samplesPerPixel<1||samplesPerPixel>limits.maxBatchSamples||sampleStart+samplesPerPixel>o.samplesPerPixel)throw new RangeError('Invalid progressive sample range');
 if(!(rngStates instanceof Uint32Array)||rngStates.length!==o.width*o.height)throw new RangeError('Full-frame persistent RNG states are required');
 if(!Number.isInteger(yieldEvery)||yieldEvery<1||yieldEvery>1024)throw new RangeError('Invalid batch yield interval');
 const total=tile.width*tile.height;
 if(total*samplesPerPixel>limits.maxBatchPaths)throw new RangeError('Too many paths in a batch');
 const job={...o,...(deferOperatorEvaluation?{deferOperatorEvaluation:true,finalSamplesPerPixel:o.samplesPerPixel}:{}),width:tile.width,height:tile.height,frameWidth:o.width,frameHeight:o.height,samplesPerPixel,sampleStart,tile:{...tile},pixelIndices:new Uint32Array(total),events:[],terms:[],paths:[],stats:stats(),edgeRecords:0,tapeLimits:limits,camera,scope:PLT_SCOPE};
 if(o.spectralMode==='visible'){job.spectralSamples=[];job.spectralSampleValues=new Float64Array(total*samplesPerPixel*3);}
 let lastYield=now();const checkpoint=()=>{check();if(signal?.aborted)throw Object.assign(new Error('Render cancelled'),{name:'AbortError'});};
 checkpoint();
 for(let pixel=0;pixel<total;pixel++){
  checkpoint();const x=tile.x+pixel%tile.width,y=tile.y+Math.floor(pixel/tile.width),globalPixel=y*o.width+x,rng=randomGenerator(rngStates[globalPixel]);job.pixelIndices[pixel]=globalPixel;
  for(let sample=sampleStart;sample<sampleStart+samplesPerPixel;sample++){
   checkpoint();const before=job.paths.length,pathAudit=audit?.beginPath?.({wave:!!o.wave,pixel:globalPixel,localPixel:pixel,sample,rngState:rng.getState(),job});
   const sampleRng=o.spectralMode==='visible'?randomGenerator(hashSpectralIndex(o.seed,globalPixel,sample,0x50415448)):rng;
   const pathRng=pathAudit?.onRandomWord?()=>{const value=sampleRng();pathAudit.onRandomWord((value*4294967296)>>>0);return value;}:sampleRng;
   const beam=o.spectralMode==='visible'?spectralBeam(camera,x,y,o.width,o.height,o,globalPixel,sample,pathRng):sampleCamera(camera,x,y,o.width,o.height,pathRng,o.wavelengthMm,o.wave);pathAudit?.onCamera?.(beam);const firstTerm=job.terms.length;
   traceCameraPath(scene,beam,o,pathRng,job,pixel,pathAudit);pathAudit?.onEnd?.({rngState:rng.getState()});
   if(o.spectralMode==='visible'){const value=job.terms.slice(firstTerm).reduce((sum,t)=>sum+termValue(t,job.events),0);const index=pixel*samplesPerPixel+sample-sampleStart;for(let c=0;c<3;c++){const v=value*beam.xyzWeight[c];if(!Number.isFinite(v)||v<0)throw new Error('Invalid spectral sample');job.spectralSampleValues[3*index+c]=v;}job.spectralSamples.push({pixel:globalPixel,sample,nm:beam.spectralNm});}
   for(let i=before;i<job.paths.length;i++){job.paths[i].pixel=globalPixel;job.paths[i].sample=sample;}
  }
  rngStates[globalPixel]=rng.getState();
  if((pixel+1)%yieldEvery===0){onProgress({fraction:(pixel+1)/total,pixels:pixel+1,total,events:job.events.length});checkpoint();if(now()-lastYield>=8){await new Promise(resolve=>setTimeout(resolve,0));checkpoint();lastYield=now();}}
 }
 checkpoint();if(!deferOperatorEvaluation)finishSpectralReference(job);job.multiRegionEvidence=[];
 if(!deferOperatorEvaluation)for(const term of job.terms){if(term.eventIds.filter(id=>job.events[id].edges.length>0).length<2)continue;const contribution=termValue(term,job.events);if(!(contribution>0))continue;job.stats.twoDiffractingContributions++;
  if(job.multiRegionEvidence.length<8)job.multiRegionEvidence.push({pixel:job.pixelIndices[term.pixel],sampleStart,sampleEnd:sampleStart+samplesPerPixel,sampleWeight:term.weight,contribution:contribution/o.samplesPerPixel,normalizationSamples:o.samplesPerPixel,operators:term.eventIds.map(id=>({id,F:job.events[id].referenceF,directAmplitude:job.events[id].directAmplitude,source:job.events[id].source,destination:job.events[id].destination,edges:job.events[id].edges.slice(0,16).map(e=>({edgeId:e.edgeId,point:e.point,ro:e.ro,sinBeta:e.sinBeta})),totalEdges:job.events[id].edges.length}))});
 }
 job.timings=deferOperatorEvaluation?{cpuTraversalMs:now()-start,cpuOperatorEvaluations:0,cpuReferenceImages:0}:{cpuTraversalAndReferenceMs:now()-start};return job;
}
/** Bounded contribution evidence from the processor that evaluated the events.
 * This does not construct a CPU reference image. Finiteness of the published
 * camera image remains the execution backend's responsibility. */
export function summarizePLTBatchOperators(job,eventPowers=null){
 if(eventPowers!==null&&(!(eventPowers instanceof Float32Array||eventPowers instanceof Float64Array)||eventPowers.length!==job.events.length||!eventPowers.every(v=>Number.isFinite(v)&&v>=0)))throw new Error('Invalid local operator powers');
 const power=id=>eventPowers?eventPowers[id]:job.events[id].referenceF;
 job.multiRegionEvidence=[];job.stats.twoDiffractingContributions=0;
 for(const term of job.terms){
  if(term.eventIds.filter(id=>job.events[id].edges.length>0).length<2||!(term.weight>0)||!term.eventIds.every(id=>power(id)>0))continue;
  job.stats.twoDiffractingContributions++;
  if(job.multiRegionEvidence.length>=8)continue;
  const contribution=term.eventIds.reduce((v,id)=>v*power(id),term.weight);
  if(!Number.isFinite(contribution)||contribution<0)throw new Error('Invalid local contribution evidence');
  job.multiRegionEvidence.push({pixel:job.pixelIndices[term.pixel],sampleStart:job.sampleStart,sampleEnd:job.sampleStart+job.samplesPerPixel,sampleWeight:term.weight,contribution:contribution/(job.finalSamplesPerPixel??job.samplesPerPixel),normalizationSamples:job.finalSamplesPerPixel??job.samplesPerPixel,operators:term.eventIds.map(id=>({id,F:power(id),directAmplitude:job.events[id].directAmplitude,source:job.events[id].source,destination:job.events[id].destination,edges:job.events[id].edges.slice(0,16).map(e=>({edgeId:e.edgeId,point:e.point,ro:e.ro,sinBeta:e.sinBeta})),totalEdges:job.events[id].edges.length}))});
 }
 return job;
}
/** Lazy, idempotent CPU evaluation of an already constructed monochrome tape.
 * No geometry, RNG or sampling is repeated. Used only by the demo CPU backend
 * or after a failed GPU batch; successful GPU batches never call this. */
export function materializePLTBatchReference(job){
 if(job.spectralMode==='visible')throw new Error('Lazy demo CPU evaluation is monochrome only');
 if(job.reference!==undefined){if(!(job.reference instanceof Float64Array)||job.reference.length!==job.width*job.height||!job.reference.every(v=>Number.isFinite(v)&&v>=0))throw new Error('Invalid cached CPU reference');return job.reference;}
 const start=now();let count=0;
 for(const event of job.events){
  if(event.referenceF===undefined){const value=evaluateOperator(event);if(!Number.isFinite(value.F)||value.F<0||!value.soft.every(Number.isFinite)||!value.hard.every(Number.isFinite))throw new Error('Invalid local diffraction power');event.referenceF=value.F;count++;}
  else if(!Number.isFinite(event.referenceF)||event.referenceF<0)throw new Error('Invalid cached local diffraction power');
 }
 const reference=replayPLT(job);if(!reference.every(v=>Number.isFinite(v)&&v>=0))throw new Error('Invalid CPU camera response');
 job.reference=reference;summarizePLTBatchOperators(job);job.timings??={};job.timings.cpuOperatorAndCameraMs=(job.timings.cpuOperatorAndCameraMs??0)+now()-start;job.timings.cpuOperatorEvaluations=(job.timings.cpuOperatorEvaluations??0)+count;job.timings.cpuReferenceImages=(job.timings.cpuReferenceImages??0)+1;
 return reference;
}
/** Add CPU terms in their original per-pixel order using the FINAL denominator.
 * That makes the final CPU image bit-identical across tile/batch schedules and
 * to buildPLTJob. A preview divides by actual committed per-pixel sample counts. */
export function accumulatePLTBatch(job,targetSums,targetSamplesPerPixel,targetXYZ=null){
 if(job.spectralMode==='visible'){if(!(targetXYZ instanceof Float64Array)||targetXYZ.length!==targetSums.length*3)throw new Error('Missing full-frame XYZ accumulator');for(const term of job.terms){const value=termValue(term,job.events),pixel=job.pixelIndices[term.pixel];if(!Number.isFinite(value)||value<0)throw new Error('Invalid spectral progressive sample');for(let c=0;c<3;c++){targetXYZ[pixel*3+c]+=value*term.xyzWeight[c]/targetSamplesPerPixel;if(!Number.isFinite(targetXYZ[pixel*3+c]))throw new Error('Spectral progressive overflow');}targetSums[pixel]=targetXYZ[pixel*3+1];}return;}
 for(const term of job.terms){const value=termValue(term,job.events);if(!Number.isFinite(value)||value<0)throw new Error('Invalid progressive sample');targetSums[job.pixelIndices[term.pixel]]+=value/targetSamplesPerPixel;}
}

function patch(vertices,materialIndex){return {mesh:prepareMesh({vertices,faces:[[0,1,2],[0,2,3]]}),materialIndex,surface:'open-plate'};}
export const PLT_MATERIALS=Object.freeze([{type:'diffuse',albedo:.65,emission:0},{type:'diffuse',albedo:.05,emission:0},{type:'mirror',albedo:.88,emission:0},{type:'diffuse',albedo:0,emission:5,twoSidedEmission:true},{type:'diffuse',albedo:.85,emission:0}]);
export function createPLTScene(configuration={}){
 const {preset='gallery',objText=null,modelRotation=[0,0,0],modelScale=1,modelTranslation=[0,0,0],modelMaterial=0,emission=5,albedo=.65,spectralMode='monochrome',spectralSource='D65',spectralMaterial='neutral'}=configuration;
 if(preset===PLT_DEMO_SCENE_ID){if(objText)throw new Error('The bounded demo fixture does not accept an imported model');if(spectralMode==='visible')throw new Error('The bounded demo fixture is monochrome only');return createPLTDemoScene(configuration);}
 const materials=PLT_MATERIALS.map(m=>({...m}));materials[0].albedo=num(albedo,0,1,'Albedo');materials[3].emission=num(emission,0,100,'Emitter radiance');
 if(spectralMode==='visible'){validateSpectralOptions({spectralMode,spectralSource,spectralMaterial});materials.push({...materials[0],spectralReflectance:spectralMaterial});}
 const modelIndex=spectralMode==='visible'&&modelMaterial===0?5:modelMaterial;let meshes=[],camera;
 if(preset==='double-edge'){
  // Finite 3D screens and finite source. ALL edges and surfaces participate.
  meshes=[patch([[0,-12,-10],[0,1,-10],[0,1,10],[0,-12,10]],1),patch([[8,-12,-10],[8,2,-10],[8,2,10],[8,-12,10]],1),patch([[16,-7,-10],[16,7,-10],[16,7,10],[16,-7,10]],3)];camera={position:[-2,.8,0],target:[8,1.8,0],up:[0,1,0],fov:18};
 }else{
  meshes=[patch([[-6,-1.5,5],[6,-1.5,5],[6,-1.5,-6],[-6,-1.5,-6]],0),patch([[-6,-1.5,-6],[6,-1.5,-6],[6,5,-6],[-6,5,-6]],0),patch([[-4,3,-4],[-4,3,1],[1,3,1],[1,3,-4]],3)];
  if(preset==='baffles'){
   meshes.push(patch([[-4,-1.5,-1],[.2,-1.5,-1],[.2,2,-1],[-4,2,-1]],1),patch([[1,-1.5,-3],[4,-1.5,-3],[4,2,-3],[1,2,-3]],1));
   meshes.push(patch([[-4,-1,-5.9],[4,-1,-5.9],[4,2.8,-5.9],[-4,2.8,-5.9]],3));
  }else{
   const base=objText??makePresetOBJ(preset==='prism'?'triangular-prism':'box',{width:2.6,height:2.6,depth:2.6});
   const mesh=prepareMesh(base,{unitScale:num(modelScale,.001,1000,'Model scale'),rotationDeg:vec(modelRotation,'Model rotation'),translation:vec(modelTranslation,'Model translation')});
   meshes.push({mesh,materialIndex:spectralMode==='visible'&&modelMaterial===0?5:num(modelMaterial,0,2,'Material index'),surface:'auto'});
  }
  camera={position:[6,3.6,10],target:[0,0,-.8],up:[0,1,0],fov:42};
 }
 if(objText&&(preset==='double-edge'||preset==='baffles')){meshes.push({mesh:prepareMesh(objText,{unitScale:modelScale,rotationDeg:modelRotation,translation:modelTranslation}),materialIndex:modelIndex,surface:'auto'});}
 if(preset==='micro-gallery'){const factor=.02;meshes=meshes.map(item=>({...item,mesh:prepareMesh({vertices:item.mesh.vertices.map(p=>scale(p,factor)),faces:item.mesh.authoredFaces})}));camera={...camera,position:scale(camera.position,factor),target:scale(camera.target,factor),sensorScale:factor};}
 const scene=preparePLTScene(meshes);scene.materials=materials;scene.spectralSource=spectralSource;scene.physicalScale=preset==='micro-gallery'?{worldScale:.02,roomWidthMm:.24,defaultBoxWidthMm:.052}:null;scene.emitters=scene.triangles.filter(t=>materials[t.materialIndex]?.emission>0);scene.emitterArea=scene.emitters.reduce((s,t)=>s+t.area,0);scene.camera=camera;scene.preset=preset;return scene;
}
export function orbitCamera(camera,yawDeg=0,pitchDeg=0,distanceScale=1){num(yawDeg,-36000,36000,'Camera yaw');num(pitchDeg,-36000,36000,'Camera pitch');num(distanceScale,.001,1000,'Camera distance scale');const p=sub(camera.position,camera.target),r=length(p)*distanceScale,az=Math.atan2(p[0],p[2])+yawDeg*Math.PI/180,el=clamp(Math.asin(p[1]/length(p))+pitchDeg*Math.PI/180,-1.45,1.45);return {...camera,position:add(camera.target,[r*Math.cos(el)*Math.sin(az),r*Math.sin(el),r*Math.cos(el)*Math.cos(az)])};}
