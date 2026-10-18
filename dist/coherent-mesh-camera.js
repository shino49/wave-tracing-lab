/** Perspective viewing of diffuse receivers illuminated by the existing coherent
 * route solver. The sqrt(cos) envelope is a local receiver approximation, not
 * exact electromagnetic normal flux or a coherent diffuse-scattering model. */
import {hashSpectralIndex} from './plt-spectrum.js';
import {slabIsInvisible,slabContainsPoint,intersectSlabRay,fresnelS,refractSlabDirection} from './coherent-mesh-slab.js';
import {rayNearest} from './plt-geometry.js';
import {add,sub,scale,dot,cross,normalize} from './mesh-geometry.js';
const vec=(v,name)=>{if(!Array.isArray(v)||v.length!==3||v.some(x=>typeof x!=='number'||!Number.isFinite(x)||Math.abs(x)>10000))throw new TypeError(`${name} must be a finite 3-vector within ±10000 mm`);return [...v];};
const unit=(v,name)=>{if(!Array.isArray(v)||v.length!==3||Array.from(v).some(x=>!Number.isFinite(x)))throw new TypeError(`${name} must be a finite vector`);const p=[...v],n=Math.hypot(...p);if(!(n>1e-12))throw new RangeError(`${name} must be nonzero`);return scale(p,1/n);};
const cameraFrames=new WeakMap(),validatedCameras=new WeakSet();
export function validateCoherentCamera(value){
 if(value&&validatedCameras.has(value))return value;
 if(!value||typeof value!=='object'||Array.isArray(value))throw new TypeError('Camera must be an object');
 for(const key of Object.keys(value))if(!['position','target','up','fovDeg','mirrorBounces'].includes(key))throw new Error(`Unsupported camera field ${key}`);
 const position=vec(value.position,'camera.position'),target=vec(value.target,'camera.target'),up=unit(value.up,'camera.up'),forward=unit(sub(target,position),'camera direction');
 if(Math.hypot(...cross(forward,up))<1e-8)throw new RangeError('Camera up must not be parallel to viewing direction');
 const fovDeg=value.fovDeg;if(typeof fovDeg!=='number'||!Number.isFinite(fovDeg)||fovDeg<=0||fovDeg>=179)throw new RangeError('Camera vertical fovDeg must be between 0 and 179');
 const hasMirrorBounces=Object.prototype.hasOwnProperty.call(value,'mirrorBounces');
 if(hasMirrorBounces&&(!Number.isInteger(value.mirrorBounces)||value.mirrorBounces<0||value.mirrorBounces>2))throw new RangeError('camera.mirrorBounces must be an integer from 0 to 2');
 const camera=Object.freeze({position:Object.freeze(position),target:Object.freeze(target),up:Object.freeze(up),fovDeg,...(hasMirrorBounces?{mirrorBounces:value.mirrorBounces}:{})});validatedCameras.add(camera);return camera;
}
function cameraRay(camera,x,y,width,height,sx,sy){
 if(!Number.isSafeInteger(width)||width<1||!Number.isSafeInteger(height)||height<1||!Number.isInteger(x)||x<0||x>=width||!Number.isInteger(y)||y<0||y>=height||![sx,sy].every(v=>Number.isFinite(v)&&v>=0&&v<=1))throw new RangeError('Invalid camera pixel/sample');
 const c=validateCoherentCamera(camera);let frame=cameraFrames.get(c);if(!frame){const forward=normalize(sub(c.target,c.position)),right=normalize(cross(forward,c.up));frame={forward,right,up:cross(right,forward),h:Math.tan(c.fovDeg*Math.PI/360)};cameraFrames.set(c,frame);}const {forward,right,up,h}=frame;
 const direction=normalize(add(forward,add(scale(right,(2*(x+sx)/width-1)*h*width/height),scale(up,(1-2*(y+sy)/height)*h))));
 return {camera:c,origin:c.position,direction};
}
/** Nearest opaque geometric hit, including backfaces; winding is never flipped. */
export function cameraSample(scene,camera,x,y,width,height,sx=.5,sy=.5){
 const {origin,direction}=cameraRay(camera,x,y,width,height,sx,sy);
 const hit=rayNearest(scene,origin,direction);
 return hit?{point:[...hit.point],normal:[...hit.geometricNormal],triangleId:hit.triangleId,materialIndex:hit.materialIndex,distance:hit.distance}:null;
}
/** Bounded geometric camera transport. Mirror reflectance is intensity throughput;
 * camera paths add neither inverse-distance loss nor coherent wave phase. */
export function traceCoherentCamera(scene,camera,x,y,width,height,sx=.5,sy=.5,sampleSeed=0){
 const ray=cameraRay(camera,x,y,width,height,sx,sy);
 return traceCoherentRay(scene,ray.origin,ray.direction,ray.camera.mirrorBounces??2,null,hashSpectralIndex(sampleSeed,y*width+x,Math.floor(sx*65535)^Math.imul(Math.floor(sy*65535),65537),0x43414d53));
}
/** Geometric radiance transport, including exact-point diffuse launches.
 * A miss retains its final direction for an infinite environment lookup.
 * Slab radiance uses a fixed-s scalar interface approximation. One bounded
 * incoherent branch (R, TT, TRT, TRRT) is sampled by unabsorbed Fresnel power;
 * throughput is the admitted power sum, preserving the omitted-order loss.
 * Beer attenuation is spectral and applied by the caller using glassLength. */
export function traceCoherentRay(scene,origin,direction,budget=2,initialTriangle=null,sampleSeed=0){
 origin=vec(origin,'ray origin');direction=unit(direction,'ray direction');
 if(!Number.isInteger(budget)||budget<0||budget>2)throw new RangeError('Ray mirror budget must be 0–2');
 if(initialTriangle&&scene.triangles[initialTriangle.id]!==initialTriangle)throw new Error('Initial triangle must belong to the scene');
 const slab=slabIsInvisible(scene.slab)?null:scene.slab;
 let glassLength=0,slabEvents=[],slabTruncatedPower=0,slabUnsupported=null,minSlabT=Number.MIN_VALUE;
 const interfaceTolerance=(p,d,n)=>64*Number.EPSILON*Math.max(1,...p.map(Math.abs))/Math.max(1e-12,Math.abs(dot(d,n)));
 const corner=p=>{const d=sub(p,slab.center),e=64*Number.EPSILON*Math.max(1,...p.map(Math.abs),slab.width,slab.height);return Math.abs(dot(d,slab.u))>=slab.width/2-e||Math.abs(dot(d,slab.v))>=slab.height/2-e;};
 let minDistance=initialTriangle?Number.MIN_VALUE:1e-9,throughput=1,mirrorBounces=0,primaryMaterialType=null,previousMirror=initialTriangle,selfHitDistance=0,planeTolerance=0;
 if(initialTriangle){const t=initialTriangle,coordinateScale=Math.max(1,...origin.map(Math.abs),...t.a.map(Math.abs),...t.b.map(Math.abs),...t.c.map(Math.abs));planeTolerance=64*Number.EPSILON*coordinateScale;selfHitDistance=planeTolerance/Math.max(1e-12,Math.abs(dot(direction,t.normal)));}
 const result=(status,hit=null)=>({status,point:hit?[...hit.point]:null,normal:hit?[...hit.geometricNormal]:null,triangleId:hit?.triangleId??-1,materialIndex:hit?.materialIndex??-1,viewDirection:hit?scale(direction,-1):null,throughput,mirrorBounces,primaryMaterialType,...(slab?{glassLength,slabEvents,slabTruncatedPower,slabUnsupported}:{}),...(status==='miss'?{escapeDirection:[...direction]}:{})});
 if(slab&&slabContainsPoint(slab,origin)){slabUnsupported='Camera/ray origin inside or on dielectric slab';return result('slab-unsupported');}
 for(;;){
  let hit=rayNearest(scene,origin,direction,{minDistance});
  // Only reject an actual nearest candidate on the just-reflected plane. A
  // grazing ray can amplify plane-roundoff into a large apparent distance;
  // using that distance as a blanket near clip would tunnel through blockers.
  while(hit&&previousMirror&&hit.distance<=selfHitDistance){
   const t=hit.triangle,n=previousMirror.normal;
   const coplanar=t===previousMirror||(Math.hypot(...cross(t.normal,n))<=32*Number.EPSILON&&[t.a,t.b,t.c].every(p=>Math.abs(dot(sub(p,previousMirror.a),n))<=planeTolerance));
   if(!coplanar)break;
   // Advance just beyond this numerical intersection, never beyond a whole
   // triangle/object or a grazing uncertainty interval.
   minDistance=hit.distance+Math.max(Number.MIN_VALUE,Math.abs(hit.distance)*Number.EPSILON);
   hit=rayNearest(scene,origin,direction,{minDistance});
  }
  const slabHit=slab?intersectSlabRay(slab,origin,direction,minSlabT,hit?.distance??Infinity):null;
  if(slabHit){
   if(slabEvents.length){slabUnsupported='Repeated slab encounter after external reflection is unsupported';return result('slab-unsupported');}
   if(!slabHit.entering||!slabHit.face.startsWith('normal')||corner(slabHit.point)){slabUnsupported='Finite slab side/corner entry is unsupported';return result('slab-unsupported');}
   const f=fresnelS(1,slab.index,Math.abs(dot(direction,slabHit.normal))),weights=[f.R];
   for(let k=0;k<=slab.internalReflections;k++)weights.push(f.T*f.T*f.R**k);
   const admitted=weights.reduce((a,b)=>a+b,0),draw=(hashSpectralIndex(sampleSeed,0,0,0x534c4142)+.5)/4294967296*admitted;
   let branch=0,cumulative=weights[0];while(branch<weights.length-1&&draw>=cumulative)cumulative+=weights[++branch];
   slabTruncatedPower=throughput*Math.max(0,1-admitted);throughput*=admitted;
   if(primaryMaterialType===null)primaryMaterialType='slab';
   const event={branch:branch===0?'R':`T${'R'.repeat(branch-1)}T`,internalReflections:Math.max(0,branch-1),probability:weights[branch]/admitted,power:weights[branch],admittedPower:admitted,entry:[...slabHit.point]};slabEvents.push(event);
   origin=slabHit.point;previousMirror=null;minDistance=Number.MIN_VALUE;
   if(branch===0){direction=normalize(sub(direction,scale(slabHit.normal,2*dot(direction,slabHit.normal))));minSlabT=interfaceTolerance(origin,direction,slabHit.normal);event.exit=[...origin];continue;}
   direction=refractSlabDirection(direction,slabHit.normal,1,slab.index);
   for(let k=0;k<branch;k++){
    const exit=intersectSlabRay(slab,origin,direction,interfaceTolerance(origin,direction,slabHit.normal));
    if(!exit||!exit.face.startsWith('normal')||corner(exit.point)){slabUnsupported='Finite slab side/corner exit is unsupported';return result('slab-unsupported');}
    const obstruction=rayNearest(scene,origin,direction,{minDistance:Number.MIN_VALUE,maxDistance:exit.t});
    if(obstruction&&obstruction.distance<exit.t){slabUnsupported='Mesh surface inside dielectric slab is unsupported';return result('slab-unsupported');}
    glassLength+=exit.t;origin=exit.point;
    if(k===branch-1){const transmitted=refractSlabDirection(direction,exit.normal,slab.index,1);if(!transmitted){slabUnsupported='Internal total reflection has no admitted escape branch';return result('slab-unsupported');}direction=transmitted;minSlabT=interfaceTolerance(origin,direction,exit.normal);event.exit=[...origin];event.glassLength=glassLength;}
    else direction=normalize(sub(direction,scale(exit.normal,2*dot(direction,exit.normal))));
   }
   continue;
  }
  if(!hit)return result('miss');
  const material=scene.materials[hit.materialIndex];
  if(primaryMaterialType===null)primaryMaterialType=material.type;
  if(material.type!=='mirror')return result(material.type==='diffuse'?'diffuse':'opaque',hit);
  if(material.reflectance===0){throughput=0;return result('absorbed',hit);}
  if(mirrorBounces>=budget)return result('mirror-limit',hit);
  throughput*=material.reflectance;
  mirrorBounces++;
  const n=hit.geometricNormal;
  direction=normalize(sub(direction,scale(n,2*dot(direction,n))));
  // Start at the exact physical endpoint: moving it biases the receiver's wave
  // phase. Coplanar self-intersections are classified individually above;
  // nearby distinct surfaces are never hidden by a grazing-angle near clip.
  origin=hit.point;minSlabT=Number.MIN_VALUE;
  const t=hit.triangle,coordinateScale=Math.max(1,...origin.map(Math.abs),...t.a.map(Math.abs),...t.b.map(Math.abs),...t.c.map(Math.abs));
  previousMirror=t;planeTolerance=64*Number.EPSILON*coordinateScale;
  selfHitDistance=planeTolerance/Math.max(1e-12,Math.abs(dot(direction,n)));
  minDistance=Number.MIN_VALUE;
 }
}
export function validateDiffuseReflectance(array){
 if(!Array.isArray(array)||array.length<2||array.length>32)throw new RangeError('diffuse reflectance requires 2–32 knots');
 const knots=Array.from(array,(k,i)=>{
  if(!k||typeof k!=='object'||Array.isArray(k)||Object.keys(k).some(x=>!['wavelengthNm','value'].includes(x)))throw new TypeError('Invalid diffuse reflectance knot');
  if(!Number.isFinite(k.wavelengthNm)||k.wavelengthNm<360||k.wavelengthNm>830||!Number.isFinite(k.value)||k.value<0||k.value>1)throw new RangeError('Reflectance knots require 360–830 nm and values 0–1');
  if(i&&k.wavelengthNm<=array[i-1].wavelengthNm)throw new RangeError('Reflectance wavelengths must strictly increase');
  return Object.freeze({wavelengthNm:k.wavelengthNm,value:k.value});
 });
 if(knots[0].wavelengthNm!==360||knots.at(-1).wavelengthNm!==830)throw new RangeError('Reflectance must span the full 360–830 nm range');
 return Object.freeze(knots);
}
export function diffuseReflectanceAt(knots,nm){
 if(!Number.isFinite(nm)||nm<360||nm>830)throw new RangeError('Reflectance wavelength must be 360–830 nm');
 if(nm<=knots[0].wavelengthNm)return knots[0].value;
 for(let i=1;i<knots.length;i++)if(nm<=knots[i].wavelengthNm){const a=knots[i-1],b=knots[i],t=(nm-a.wavelengthNm)/(b.wavelengthNm-a.wavelengthNm);return a.value+(b.value-a.value)*t;}
 return knots.at(-1).value;
}
export function projectCoherentSurfaceRoutes(transport,point,normal){
 point=vec(point,'surface point');normal=unit(normal,'surface normal');
 if(transport.templates&&transport.templates.length!==transport.routes.length)throw new Error('Surface projection requires one spectral template per route');
 const directions=[],cosines=[],factors=[];
 const routes=transport.routes.map(r=>{
  const last=r.kind==='diffraction'?r.point:r.vertices?.at(-2);
  if(!last)throw new Error(`Missing final physical vertex for ${r.kind}`);
  const direction=unit(sub(last,point),'incident route direction'),cosine=Math.max(-1,Math.min(1,dot(normal,direction))),factor=Math.sqrt(Math.max(0,cosine));
  directions.push(direction);cosines.push(cosine);factors.push(factor);return {...r,amplitude:r.amplitude*factor};
 });
 let spread=0;for(let i=0;i<directions.length;i++)for(let j=0;j<i;j++)spread=Math.max(spread,Math.acos(Math.max(-1,Math.min(1,dot(directions[i],directions[j]))))*180/Math.PI);
 const angularDiagnostics={routeCount:routes.length,grazingCount:cosines.filter(c=>Math.abs(c)<.05).length,backfaceCount:cosines.filter(c=>c<=0).length,minCosine:cosines.length?Math.min(...cosines):null,maxCosine:cosines.length?Math.max(...cosines):null,maxAngularSpreadDeg:spread,approximation:'sqrt-positive-incidence-cosine local receiver envelope; not exact normal flux'};
 return {...transport,routes,...(transport.templates?{templates:transport.templates.map((t,i)=>({...t,amplitudeScale:t.amplitudeScale*factors[i],...(t.route?{route:routes[i]}:{})}))}:{}),geometricPower:routes.reduce((s,r)=>s+(['direct','reflection','transmission'].includes(r.kind)?r.amplitude*r.amplitude:0),0),angularDiagnostics};
}
const quad=(vertices,m)=>({mesh:{vertices,faces:[[0,1,2],[0,2,3]]},materialIndex:m,surface:'open-plate'});
const rho=values=>values.map(([wavelengthNm,value])=>({wavelengthNm,value}));
export function makeCoherentCameraPreset(){
 const materials=[{type:'mirror',reflectance:1,phaseRad:Math.PI},
  {type:'diffuse',reflectance:rho([[360,.62],[830,.62]])},
  {type:'diffuse',reflectance:rho([[360,.06],[480,.06],[540,.1],[610,.74],[830,.74]])},
  {type:'diffuse',reflectance:rho([[360,.06],[440,.1],[500,.7],[560,.76],[620,.08],[830,.06]])},
  {type:'diffuse',reflectance:rho([[360,.38],[440,.8],[490,.72],[560,.07],[830,.05]])}];
 const entries=[quad([[-10,0,-49],[10,0,-49],[10,0,53],[-10,0,53]],0),quad([[-1.8,.015,52],[-1.8,1.7,52],[1.8,1.7,52],[1.8,.015,52]],1)];
 // Closed extruded bevel-front solids. All depth, occlusion and incident-angle
 // variation come from their actual triangles, never painted shading/stripes.
 const solid=(x,y,w,h,z,depth,materialIndex,bevel)=>{
  const vertices=[[x,y,z],[x+w,y,z+bevel],[x+w,y+h,z+bevel],[x,y+h,z],[x,y,z+depth],[x+w,y,z+depth],[x+w,y+h,z+depth],[x,y+h,z+depth]];
  entries.push({mesh:{vertices,faces:[[0,3,2],[0,2,1],[4,5,6],[4,6,7],[0,4,7],[0,7,3],[1,2,6],[1,6,5],[0,1,5],[0,5,4],[3,7,6],[3,6,2]]},materialIndex,surface:'closed-solid'});
 };
 solid(-.95,.12,.5,.73,50,.7,2,.16);solid(-.23,.14,.46,1.03,50.25,.65,3,-.12);solid(.44,.13,.48,.53,49.85,.8,4,.2);
 return {name:'Colored solids in Lloyd-mirror wave illumination',materials,entries,source:{position:[0,.08,-50],power:1,phaseRad:0,wavelengthMm:.00055},detector:{center:[0,.65,50],u:[1,0,0],v:[0,1,0],width:2.8,height:1.8},camera:{position:[1.45,1.18,45],target:[0,.65,50.4],up:[0,1,0],fovDeg:21}};
}
