/** Geometry-derived, bounded 3-D source for the incoherent pinhole operator.
 * Lengths: mm. Camera at the pupil centre, source z < 0, detector z > 0.
 * The source layers contain L(q) cos^4(theta), NOT aperture throughput, pixel
 * area, exposure or colourimetric weights. The optics kernels own Ap / z^2.
 * Materials use the existing declared synthetic reflectance spectra.
 */
import {SPECTRAL_MATERIALS,reflectanceSpectrum,sourceSpectrum,cieXYZ,CIE_Y_INTEGRAL,xyzToLinearSRGB,linearToSRGB} from './plt-spectrum.js';

export const PINHOLE_SCENE_ID='bounded-pinhole-colored-meshes';
export const PINHOLE_BASIS_NAMES=Object.freeze([...SPECTRAL_MATERIALS]);
export const PINHOLE_SCENE_DEFAULTS=Object.freeze({resolution:512,sensorWidthMm:8,detectorDistanceMm:50,depthCount:8,diameterMm:.24,depthMinMm:200,depthMaxMm:300});
const EPS=1e-7;
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const normalize=v=>{const l=Math.hypot(...v);if(!(l>0))throw new RangeError('Zero direction');return v.map(x=>x/l);};
const finitePositive=(x,name)=>{if(!Number.isFinite(x)||x<=0)throw new RangeError(`${name} must be positive and finite`);return x;};
function cancelled(hooks){if(hooks.signal?.aborted||hooks.isCancelled?.()||hooks.shouldCancel?.()){const e=new Error('Pinhole scene build cancelled');e.name='AbortError';throw e;}}
async function checkpoint(hooks,stage,fraction){cancelled(hooks);hooks.onProgress?.({stage,fraction});if(hooks.yield)await hooks.yield();else if(hooks.yieldControl)await hooks.yieldControl();else await new Promise(r=>setTimeout(r,0));cancelled(hooks);}

function transformPoint(p,translation,rotation){
 let [x,y,z]=p;const [rx,ry,rz]=rotation.map(a=>a*Math.PI/180);
 [y,z]=[y*Math.cos(rx)-z*Math.sin(rx),y*Math.sin(rx)+z*Math.cos(rx)];
 [x,z]=[x*Math.cos(ry)+z*Math.sin(ry),-x*Math.sin(ry)+z*Math.cos(ry)];
 [x,y]=[x*Math.cos(rz)-y*Math.sin(rz),x*Math.sin(rz)+y*Math.cos(rz)];
 return [x+translation[0],y+translation[1],z+translation[2]];
}
function box(size,position,rotation=[0,0,0]){
 const [x,y,z]=size.map(v=>v/2),vertices=[[-x,-y,-z],[x,-y,-z],[x,y,-z],[-x,y,-z],[-x,-y,z],[x,-y,z],[x,y,z],[-x,y,z]].map(p=>transformPoint(p,position,rotation));
 return {vertices,faces:[[0,3,2],[0,2,1],[4,5,6],[4,6,7],[0,4,7],[0,7,3],[1,2,6],[1,6,5],[0,1,5],[0,5,4],[3,7,6],[3,6,2]]};
}
function mergeMeshes(parts){const vertices=[],faces=[];for(const m of parts){const base=vertices.length;vertices.push(...m.vertices);faces.push(...m.faces.map(f=>f.map(i=>i+base)));}return {vertices,faces};}
function sphere(radius,position,segments=32,rings=16){
 const vertices=[],normals=[],faces=[];
 for(let j=0;j<=rings;j++)for(let i=0;i<segments;i++){const t=Math.PI*j/rings,p=2*Math.PI*i/segments,n=[Math.sin(t)*Math.cos(p),Math.cos(t),Math.sin(t)*Math.sin(p)];normals.push(n);vertices.push(n.map((v,k)=>position[k]+radius*v));}
 for(let j=0;j<rings;j++)for(let i=0;i<segments;i++){
  const a=j*segments+i,b=j*segments+(i+1)%segments,c=(j+1)*segments+i,d=(j+1)*segments+(i+1)%segments;
  for(const face of (j===0?[[a,c,d]]:j===rings-1?[[a,c,b]]:[[a,c,d],[a,d,b]])){
   const n=cross(sub(vertices[face[1]],vertices[face[0]]),sub(vertices[face[2]],vertices[face[0]]));
   if(dot(n,sub(vertices[face[0]],position))<0)face.reverse();faces.push(face);
  }
 }
 return {vertices,faces,normals};
}
function boundsOf(points){const min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];for(const p of points)for(let k=0;k<3;k++){min[k]=Math.min(min[k],p[k]);max[k]=Math.max(max[k],p[k]);}return {min,max};}
function buildBVH(triangles,indices=triangles.map((_,i)=>i)){
 const bounds=boundsOf(indices.flatMap(i=>[triangles[i].bounds.min,triangles[i].bounds.max]));
 if(indices.length<=8)return {...bounds,indices};
 const axis=bounds.max.map((v,k)=>v-bounds.min[k]).reduce((a,v,k,arr)=>v>arr[a]?k:a,0);
 indices.sort((a,b)=>triangles[a].centroid[axis]-triangles[b].centroid[axis]);const mid=indices.length>>1;
 return {...bounds,left:buildBVH(triangles,indices.slice(0,mid)),right:buildBVH(triangles,indices.slice(mid))};
}
function compileScene(scene){
 const triangles=[];
 scene.meshes.forEach((mesh,objectIndex)=>{
  if(!PINHOLE_BASIS_NAMES.includes(mesh.basis))throw new RangeError('Unknown scene spectral basis');
  if(!Number.isFinite(mesh.albedo)||mesh.albedo<0||mesh.albedo>1)throw new RangeError('Material albedo must be within 0…1');
  mesh.vertices.forEach(p=>{if(p.length!==3||!p.every(Number.isFinite)||p[2]>=0)throw new RangeError('Source vertices must be finite and strictly before the pupil');});
  mesh.faces.forEach(face=>{
   if(face.length!==3||!face.every(i=>Number.isInteger(i)&&i>=0&&i<mesh.vertices.length))throw new RangeError('Invalid triangle indices');
   const vertices=face.map(i=>mesh.vertices[i]),e1=sub(vertices[1],vertices[0]),e2=sub(vertices[2],vertices[0]),n=cross(e1,e2),twiceArea=Math.hypot(...n);
   if(twiceArea<1e-10)return;
   triangles.push({vertices,e1,e2,normal:n.map(v=>v/twiceArea),normals:mesh.normals?face.map(i=>mesh.normals[i]):null,centroid:[0,1,2].map(k=>vertices.reduce((s,p)=>s+p[k],0)/3),bounds:boundsOf(vertices),basisIndex:PINHOLE_BASIS_NAMES.indexOf(mesh.basis),albedo:mesh.albedo,objectIndex});
  });
 });
 if(!triangles.length)throw new RangeError('Scene must contain nondegenerate triangles');
 scene.triangles=triangles;scene.bvh=buildBVH(triangles);scene.bounds=boundsOf(scene.meshes.flatMap(m=>m.vertices));
 scene.metadata={...scene.metadata,triangleCount:triangles.length,objectCount:scene.meshes.length,boundsMm:scene.bounds,sourceDepthRangeMm:[-scene.bounds.max[2],-scene.bounds.min[2]],spectralBasis:[...PINHOLE_BASIS_NAMES],materialModel:'Existing synthetic passive reflectance: neutral=1; coloured=.04+.92 Gaussian, not measured spectra',lighting:'Achromatic directional Lambertian key and fill; triangle visibility, no indirect light',background:'Empty space has exactly zero radiance',sourceBrightnessApplied:false};
 return scene;
}

/** An F assembled from solid boxes, a smooth-shaded triangle sphere, a rotated
 * green cuboid, a narrow opaque neutral blade and a solid pedestal. All opaque
 * occluders stay in the remote 200–300 mm source region. */
export function createPinholeScene(options={}){
 const meshes=options.meshes??(()=>{
  const fPosition=[-8,1,-250],fRotation=[6,-12,0];
  const fParts=[box([1.7,12,3],[0,0,0]),box([5.8,1.7,3],[2.05,5.15,0]),box([4.4,1.6,3],[1.35,1.05,0])];
  const f=mergeMeshes(fParts);f.vertices=f.vertices.map(p=>transformPoint(p,fPosition,fRotation));
  return [
   {...f,id:'red-f',name:'Red solid F',basis:'red',albedo:.90},
   {...sphere(5.3,[1.5,1.2,-264]),id:'blue-sphere',name:'Blue triangle sphere',basis:'blue',albedo:.92},
   {...box([5.8,7.3,6],[8.0,-2.6,-239],[12,-25,7]),id:'green-block',name:'Green rotated block',basis:'green',albedo:.88},
   {...box([1.15,12.5,2.6],[-.15,1.2,-214],[0,12,-5]),id:'neutral-occluder',name:'Near neutral occluder',basis:'neutral',albedo:.66},
   {...box([25.5,1.4,14],[.1,-6.9,-265],[0,0,0]),id:'neutral-pedestal',name:'Solid neutral pedestal',basis:'neutral',albedo:.25},
  ];
 })();
 const lights=options.lights??[{direction:[-.45,.70,1],strength:.82},{direction:[.7,.05,.8],strength:.24}];
 return compileScene({id:PINHOLE_SCENE_ID,meshes,lights:lights.map(l=>({direction:normalize(l.direction),strength:finitePositive(l.strength,'Light strength')})),metadata:{lengthUnit:'mm',description:'True 3-D coloured triangle objects with distant opaque occluder',aperturePlaneZMm:0,defaultDetectorZMm:50,nearPupilOccluders:false}});
}

export function projectPinholePoint(point,detectorDistanceMm=50){
 finitePositive(detectorDistanceMm,'Detector distance');if(!point||point.length!==3||!point.every(Number.isFinite)||point[2]>=0)throw new RangeError('Point must be finite with z < 0');
 const depthMm=-point[2];return {xMm:-detectorDistanceMm*point[0]/depthMm,yMm:-detectorDistanceMm*point[1]/depthMm,depthMm,magnification:-detectorDistanceMm/depthMm};
}
export function createPinholePointCalibration({positionMm=[0,0,-250],detectorDistanceMm=50,basis='neutral',projectedMass=1}={}){
 if(!PINHOLE_BASIS_NAMES.includes(basis))throw new RangeError('Unknown spectral basis');finitePositive(projectedMass,'Projected point mass');
 return {kind:'exact-point',positionMm:[...positionMm],...projectPinholePoint(positionMm,detectorDistanceMm),basis,basisIndex:PINHOLE_BASIS_NAMES.indexOf(basis),projectedMass,
  normalization:'Continuous delta in projected radiance density; use shifted finite-pixel PSF directly. No image-map splat or aperture throughput here.'};
}
export function pinholeDepthLayers(depthCount=8,depthMinMm=200,depthMaxMm=300){
 if(!Number.isInteger(depthCount)||depthCount<2||depthCount>64)throw new RangeError('Depth count must be an integer within 2…64');
 finitePositive(depthMinMm,'Minimum depth');finitePositive(depthMaxMm,'Maximum depth');if(depthMaxMm<=depthMinMm)throw new RangeError('Depth interval must be increasing');
 return Float64Array.from({length:depthCount},(_,i)=>1/(1/depthMinMm+(1/depthMaxMm-1/depthMinMm)*i/(depthCount-1)));
}
export function pinholeDepthWeights(depthMm,depthsMm){
 finitePositive(depthMm,'Depth');if(depthMm<depthsMm[0]-1e-6||depthMm>depthsMm.at(-1)+1e-6)throw new RangeError('Source depth is outside the declared layer interval');
 const t=Math.max(0,Math.min(depthsMm.length-1,(1/depthMm-1/depthsMm[0])/(1/depthsMm.at(-1)-1/depthsMm[0])*(depthsMm.length-1))),lo=Math.min(depthsMm.length-2,Math.floor(t)),f=t-lo;
 return {indices:[lo,lo+1],weights:[1-f,f]};
}
function boxHit(node,o,d,maxT){
 let lo=0,hi=maxT;
 for(let k=0;k<3;k++){
  if(Math.abs(d[k])<1e-15){if(o[k]<node.min[k]||o[k]>node.max[k])return false;continue;}
  let a=(node.min[k]-o[k])/d[k],b=(node.max[k]-o[k])/d[k];if(a>b)[a,b]=[b,a];lo=Math.max(lo,a);hi=Math.min(hi,b);if(lo>hi)return false;
 }
 return true;
}
function triangleT(t,o,d){
 const p=cross(d,t.e2),det=dot(t.e1,p);if(Math.abs(det)<1e-14)return Infinity;
 const inv=1/det,s=sub(o,t.vertices[0]),u=dot(s,p)*inv;if(u<-EPS||u>1+EPS)return Infinity;
 const q=cross(s,t.e1),v=dot(d,q)*inv;if(v<-EPS||u+v>1+EPS)return Infinity;
 const distance=dot(t.e2,q)*inv;return distance>EPS?distance:Infinity;
}
function blocked(scene,o,d,maxT,ignoreTriangle=-1){
 const stack=[scene.bvh];while(stack.length){const n=stack.pop();if(!boxHit(n,o,d,maxT))continue;
  if(n.indices){for(const i of n.indices)if(i!==ignoreTriangle&&triangleT(scene.triangles[i],o,d)<maxT)return true;}
  else stack.push(n.left,n.right);
 }return false;
}
/** Independent exact first-hit query, also used by tests and visibility audit. */
export function tracePinholeRay(scene,origin,direction,maxDistance=Infinity){
 if(origin.length!==3||direction.length!==3||!origin.every(Number.isFinite)||!direction.every(Number.isFinite)||!(Math.hypot(...direction)>0))throw new RangeError('Invalid ray');
 let distance=maxDistance,triangleIndex=-1;const stack=[scene.bvh];while(stack.length){const n=stack.pop();if(!boxHit(n,origin,direction,distance))continue;if(n.indices){for(const i of n.indices){const t=triangleT(scene.triangles[i],origin,direction);if(t<distance){distance=t;triangleIndex=i;}}}else stack.push(n.left,n.right);}
 return triangleIndex<0?null:{distance,triangleIndex,objectIndex:scene.triangles[triangleIndex].objectIndex,positionMm:origin.map((v,k)=>v+direction[k]*distance)};
}
function triangleProjection(triangle,n,width,z,pupil=[0,0]){
 return triangle.vertices.map(p=>{const s=-p[2],qx=-z*(p[0]-pupil[0])/s,qy=-z*(p[1]-pupil[1])/s;return [(qx/width+.5)*n-.5,(.5-qy/width)*n-.5,1/s];});
}
async function rasterGeometry(scene,{resolution:n,sensorWidthMm:width,detectorDistanceMm:z},hooks,pupil=[0,0],start=0,span=.2,stage='geometry'){
 const count=n*n,invDepth=new Float64Array(count),triangleIndex=new Int32Array(count).fill(-1),baryU=new Float32Array(count),baryV=new Float32Array(count);
 for(let ti=0;ti<scene.triangles.length;ti++){
  const t=scene.triangles[ti],v=triangleProjection(t,n,width,z,pupil),[a,b,c]=v,den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1]);if(Math.abs(den)<1e-12)continue;
  const minX=Math.max(0,Math.ceil(Math.min(...v.map(p=>p[0])))),maxX=Math.min(n-1,Math.floor(Math.max(...v.map(p=>p[0]))));
  const minY=Math.max(0,Math.ceil(Math.min(...v.map(p=>p[1])))),maxY=Math.min(n-1,Math.floor(Math.max(...v.map(p=>p[1]))));
  for(let y=minY;y<=maxY;y++)for(let x=minX;x<=maxX;x++){
   const wa=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/den,wb=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/den,wc=1-wa-wb;
   if(wa< -1e-9||wb< -1e-9||wc< -1e-9)continue;
   const iz=wa*a[2]+wb*b[2]+wc*c[2],idx=y*n+x;if(iz<=invDepth[idx]+1e-14)continue;
   invDepth[idx]=iz;triangleIndex[idx]=ti;baryU[idx]=wa*a[2]/iz;baryV[idx]=wb*b[2]/iz;
  }
  if((ti&63)===63)await checkpoint(hooks,stage,start+span*(ti+1)/scene.triangles.length);
 }
 await checkpoint(hooks,stage,start+span);return {invDepth,triangleIndex,baryU,baryV};
}
export function pinholeBasisLinearRGB(source='D65'){
 return PINHOLE_BASIS_NAMES.map(basis=>{
  const xyz=[0,0,0];for(let nm=360;nm<=830;nm+=2){const w=(nm===360||nm===830?1:2)*sourceSpectrum(nm,source)*reflectanceSpectrum(nm,basis)/CIE_Y_INTEGRAL,c=cieXYZ(nm);for(let k=0;k<3;k++)xyz[k]+=w*c[k];}
  return xyzToLinearSRGB(xyz).map(v=>Math.max(0,v));
 });
}
/** Pure diagnostic preview, not the propagated detector image. No auto exposure. */
export function makePinholeSourcePreview(source,{gain=1,sourceSpectrumName='D65'}={}){
 finitePositive(gain,'Preview gain');const colors=pinholeBasisLinearRGB(sourceSpectrumName),rgba=new Uint8ClampedArray(source.resolution**2*4);
 for(let i=0;i<source.radiance.length;i++){
  const color=colors[source.basisIndex[i]]??[0,0,0];for(let k=0;k<3;k++)rgba[4*i+k]=Math.round(255*linearToSRGB(-Math.expm1(-gain*source.radiance[i]*color[k])));rgba[4*i+3]=255;
 }return rgba;
}
export function pinholeSceneOverview(scene,options={}){
 options={...PINHOLE_SCENE_DEFAULTS,...options};
 const z=options.detectorDistanceMm,w=options.sensorWidthMm,a=options.diameterMm/2;
 return {lengthUnit:'mm',objects:scene.meshes.map(({id,name,basis,albedo,vertices,faces})=>({id,name,basis,albedo,vertices,faces})),aperture:{centerMm:[0,0,0],normal:[0,0,1],radiusMm:a},detector:{cornersMm:[[-w/2,-w/2,z],[w/2,-w/2,z],[w/2,w/2,z],[-w/2,w/2,z]],resolution:options.resolution},cameraOutline:{pupilCenterMm:[0,0,0],detectorCenterMm:[0,0,z],sourceBoundsMm:scene.bounds},pointProjectionEquation:'q = - detectorDistanceMm * sourceXY / sourceDepthMm'};
}

/** Build first-surface source density maps. Deterministic centre sampling with
 * true opaque triangle visibility, reciprocal-depth interpolation, and four
 * exact pupil-rim line-of-sight checks for every centre-visible source sample.
 * This does not integrate a source-dependent coherent pupil transmission. */
export async function buildPinholeSourceLayers(scene,options={},hooks={}){
 const started=performance.now(),o={...PINHOLE_SCENE_DEFAULTS,...options},n=o.resolution;
 if(!Number.isInteger(n)||n<16||n>1024)throw new RangeError('Resolution must be an integer within 16…1024');
 finitePositive(o.sensorWidthMm,'Sensor width');finitePositive(o.detectorDistanceMm,'Detector distance');finitePositive(o.diameterMm,'Aperture diameter');
 if(o.diameterMm>.6)throw new RangeError('This bounded scene supports aperture diameters at most 0.60 mm');
 const depths=pinholeDepthLayers(o.depthCount,o.depthMinMm,o.depthMaxMm),count=n*n,pitch=o.sensorWidthMm/n;
 if(count*o.depthCount*4>64*1024*1024)throw new RangeError('Source layer allocation exceeds the bounded 256 MiB map budget');
 const projected=scene.meshes.flatMap(m=>m.vertices.map(p=>projectPinholePoint(p,o.detectorDistanceMm))),maxExtent=Math.max(...projected.flatMap(p=>[Math.abs(p.xMm),Math.abs(p.yMm)]));
 if(maxExtent>=o.sensorWidthMm/2-pitch)throw new RangeError('Whole source geometry must fit inside the sensor-mapped source grid, with at least one pixel guard');
 if(scene.metadata.sourceDepthRangeMm[0]<o.depthMinMm||scene.metadata.sourceDepthRangeMm[1]>o.depthMaxMm)throw new RangeError('Scene geometry falls outside declared depth layers');
 cancelled(hooks);const raster=await rasterGeometry(scene,o,hooks),geometryFinished=performance.now(),layers=Array.from(depths,depthMm=>({depthMm,basis:PINHOLE_BASIS_NAMES.map(()=>new Float32Array(count))}));
 const depthMm=new Float32Array(count),radiance=new Float32Array(count),density=new Float32Array(count),basisIndex=new Int8Array(count).fill(-1),objectIndex=new Int16Array(count).fill(-1),normal=new Float32Array(count*3),visibilityMask=new Uint8Array(count),boundaryMask=new Uint8Array(count);
 let foreground=0,affected=0,blockedTests=0,geometricBoundary=0,sourceDensitySum=0,affectedDensitySum=0;const rim=[[o.diameterMm/2,0,0],[-o.diameterMm/2,0,0],[0,o.diameterMm/2,0],[0,-o.diameterMm/2,0]],perRim=[0,0,0,0];
 for(let y=0;y<n;y++){
  const qy=o.sensorWidthMm/2-(y+.5)*pitch;
  for(let x=0;x<n;x++){
   const i=y*n+x,ti=raster.triangleIndex[i];if(ti<0)continue;foreground++;
   const t=scene.triangles[ti],s=1/raster.invDepth[i],qx=(x+.5)*pitch-o.sensorWidthMm/2,p=[-qx*s/o.detectorDistanceMm,-qy*s/o.detectorDistanceMm,-s],wa=raster.baryU[i],wb=raster.baryV[i],wc=1-wa-wb;
   const nn=t.normals?normalize([0,1,2].map(k=>wa*t.normals[0][k]+wb*t.normals[1][k]+wc*t.normals[2][k])):t.normal;
   let l=0;for(const light of scene.lights){const cosine=Math.max(0,dot(nn,light.direction));if(cosine>0&&!blocked(scene,p.map((v,k)=>v+t.normal[k]*1e-5),light.direction,1000,ti))l+=t.albedo*light.strength*cosine;}
   const cos4=1/(1+(qx*qx+qy*qy)/o.detectorDistanceMm**2)**2,value=l*cos4;
   depthMm[i]=s;radiance[i]=l;density[i]=value;basisIndex[i]=t.basisIndex;objectIndex[i]=t.objectIndex;normal.set(nn,3*i);
   const {indices,weights}=pinholeDepthWeights(s,depths);for(let k=0;k<2;k++)layers[indices[k]].basis[t.basisIndex][i]=value*weights[k];
   let mask=0;for(let k=0;k<rim.length;k++){const d=sub(p,rim[k]);if(blocked(scene,rim[k],d,1-1e-6,ti)){mask|=1<<k;blockedTests++;perRim[k]++;}}
   visibilityMask[i]=mask;sourceDensitySum+=value;if(mask){affected++;affectedDensitySum+=value;}
  }
  if((y&15)===15)await checkpoint(hooks,'shading-and-pupil-visibility',.2+.35*(y+1)/n);
 }
 for(let y=0;y<n;y++)for(let x=0;x<n;x++){
  const i=y*n+x,id=objectIndex[i];if(id<0)continue;
  let boundary=false;for(const [dx,dy] of [[-1,0],[1,0],[0,-1],[0,1]]){const xx=x+dx,yy=y+dy;if(xx<0||xx>=n||yy<0||yy>=n){boundary=true;continue;}const j=yy*n+xx;if(objectIndex[j]!==id||Math.abs(depthMm[j]-depthMm[i])>2)boundary=true;}
  if(boundary){boundaryMask[i]=1;geometricBoundary++;}
 }
 const shadingFinished=performance.now(),revealedByRim=[],revealedDensityByRim=[],rimVisiblePixels=[],angularDisagreementByRim=[],revealedCenterPixels=new Set();
 // A second, independent set of complete rim-view z-buffers catches surfaces
 // missing from the centre image. Each rim-visible point is tested against the
 // centre-pupil BVH, including pixels where the centre z-buffer is empty.
 for(let k=0;k<rim.length;k++){
  const start=.55+.10*k,edge=await rasterGeometry(scene,o,hooks,rim[k],start,.04,'rim-view-visibility');
  let revealed=0,revealedDensity=0,rimForeground=0,angularDisagreement=0;
  for(let y=0;y<n;y++){
   const qy=o.sensorWidthMm/2-(y+.5)*pitch;
   for(let x=0;x<n;x++){
    const i=y*n+x,ti=edge.triangleIndex[i],centerTi=raster.triangleIndex[i];
    if((ti<0)!==(centerTi<0)||(ti>=0&&centerTi>=0&&(scene.triangles[ti].objectIndex!==scene.triangles[centerTi].objectIndex||Math.abs(1/edge.invDepth[i]-1/raster.invDepth[i])>2)))angularDisagreement++;
    if(ti<0)continue;rimForeground++;const s=1/edge.invDepth[i],qx=(x+.5)*pitch-o.sensorWidthMm/2,p=[rim[k][0]-qx*s/o.detectorDistanceMm,rim[k][1]-qy*s/o.detectorDistanceMm,-s];
    if(!blocked(scene,[0,0,0],p,1-1e-6,ti))continue;
    revealed++;const t=scene.triangles[ti],wa=edge.baryU[i],wb=edge.baryV[i],wc=1-wa-wb,nn=t.normals?normalize([0,1,2].map(j=>wa*t.normals[0][j]+wb*t.normals[1][j]+wc*t.normals[2][j])):t.normal;
    let l=0;for(const light of scene.lights){const cosine=Math.max(0,dot(nn,light.direction));if(cosine>0&&!blocked(scene,p.map((v,j)=>v+t.normal[j]*1e-5),light.direction,1000,ti))l+=t.albedo*light.strength*cosine;}
    const q0=projectPinholePoint(p,o.detectorDistanceMm),cos4=1/(1+(q0.xMm*q0.xMm+q0.yMm*q0.yMm)/o.detectorDistanceMm**2)**2;revealedDensity+=l*cos4;
    const cx=Math.floor((q0.xMm/o.sensorWidthMm+.5)*n),cy=Math.floor((.5-q0.yMm/o.sensorWidthMm)*n);if(cx>=0&&cx<n&&cy>=0&&cy<n)revealedCenterPixels.add(cy*n+cx);
   }
   if((y&31)===31)await checkpoint(hooks,'rim-view-visibility',start+.04+.06*(y+1)/n);
  }
  revealedByRim.push(revealed);revealedDensityByRim.push(revealedDensity);rimVisiblePixels.push(rimForeground);angularDisagreementByRim.push(angularDisagreement);
 }
 const audited=performance.now();let projectedSourceMass=0;for(const v of density)projectedSourceMass+=v*pitch*pitch;

 const result={sceneId:scene.id,resolution:n,width:n,height:n,sensorWidthMm:o.sensorWidthMm,detectorDistanceMm:o.detectorDistanceMm,pixelPitchMm:pitch,depthsMm:depths,basisNames:[...PINHOLE_BASIS_NAMES],layers,depthMm,radiance,density,basisIndex,objectIndex,normal,visibilityMask,boundaryMask,
  geometry:pinholeSceneOverview(scene,o),metadata:{...scene.metadata,sourceMapQuantity:'L(q) cos^4(theta)',sourceMapUnits:'relative spectral-basis radiance',throughputFactorApplied:false,pixelAreaApplied:false,depthInterpolation:'linear in reciprocal depth',rasterSampling:'One exact triangle-centre ray per detector pixel; subpixel boundary quadrature is not included',projection:'x increases right; row zero is +detector-y; q=-z*sourceXY/sourceDepth',maxProjectedExtentMm:maxExtent,sourceGuardMm:o.sensorWidthMm/2-maxExtent,projectedSourceMass,emptyPixelCount:count-foreground,
   pupilVisibility:{sampleLocationsMm:[[0,0,0],...rim],centerVisiblePixelCount:foreground,affectedPixelCount:affected,affectedImagePixelShare:affected/count,affectedVisiblePixelShare:foreground?affected/foreground:0,affectedSourceSignalShare:sourceDensitySum?affectedDensitySum/sourceDensitySum:0,geometricBoundaryPixelCount:geometricBoundary,blockedRimRayCount:blockedTests,blockedByRim:perRim,newlyRevealedByRim:revealedByRim,rimVisiblePixelCounts:rimVisiblePixels,newlyRevealedSourceSignalShareByRim:revealedDensityByRim.map(v=>sourceDensitySum?v/sourceDensitySum:0),newlyRevealedCenterPixelCount:revealedCenterPixels.size,angularViewDisagreementByRim:angularDisagreementByRim,maxAngularViewDisagreementShare:Math.max(...angularDisagreementByRim)/count,angularViewDisagreementDefinition:'Different first material/object or >2 mm depth at the same angular ray direction; includes ordinary camera parallax and is not the same-source visibility error metric',parallaxBoundMm:o.detectorDistanceMm*o.diameterMm/2*(1/o.depthMinMm-1/o.depthMaxMm),parallaxBoundPixels:o.detectorDistanceMm*o.diameterMm/2*(1/o.depthMinMm-1/o.depthMaxMm)/pitch,method:'Exact BVH segments from centre-visible sources to four rim points, plus complete rim-view z-buffers and reverse centre-visibility checks',scope:'Finite diagnostic samples include centre-hidden newly revealed surfaces and silhouettes; continuous coherent pupil masks are not integrated',approximation:'Depth-wise circular-pupil convolution uses centre-pupil visibility; source-dependent partial pupil visibility is not applied'}}};
 result.overview=result.geometry;result.diagnostics=result.metadata;result.previewRGBA=makePinholeSourcePreview(result);result.timings={geometryMs:geometryFinished-started,shadingMs:shadingFinished-geometryFinished,visibilityAuditMs:audited-shadingFinished,totalMs:performance.now()-started};await checkpoint(hooks,'complete',1);return result;
}
