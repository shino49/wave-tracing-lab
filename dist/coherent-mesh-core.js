import {validateSlabMeshSeparation,validateCoherentSlab,slabIsInvisible,slabContainsPoint,segmentIntersectsSlab,solveSlabTransmission,slabAbsorptionAt,assertSlabRouteSupported} from './coherent-mesh-slab.js';
/** Deterministic coherent transport through a shared 3-D triangle scene.
 * exp(+i omega t), outgoing exp(-ik r). Prescribed monopole amplitude is
 * sqrt(P_reference/4pi)/r; P_reference is its FREE-SPACE scalar source power.
 * The detector is an ideal local square-law response, not normal flux/watts.
 * No radiometric BSDF/PDF is square-rooted and no noisy field is squared.
 */
import {validateDiffuseReflectance,makeCoherentCameraPreset} from './coherent-mesh-camera.js';
import {preparePLTScene,segmentVisible} from './plt-geometry.js';
import {add,sub,scale,dot,cross,length,distance,normalize,prepareMesh,makePresetOBJ} from './mesh-geometry.js';

export const COHERENT_MESH_SCOPE=Object.freeze({
 field:'scalar exp(+i omega t), exp(-ik r); deterministic complete-route amplitudes',
 source:'point or uniform disk of independent isotropic monopoles (not Lambertian); fixed total prescribed strength; free-space reference power, not fixed delivered power in a reflecting environment',
 detector:'pixel-area average of local square-law scalar response; area integral is not exact normal flux or calibrated watts',
 interactions:'direct, up to two planar reflections, first-order UTD, optional one external reflection before or after diffraction; optional bounded dielectric first transmission and one-mirror transmission in either order with s-Fresnel, ray-tube spreading and Beer loss; no R-D-R',
 surfaces:'opaque triangles and ideal planar mirror patches; optional passive spectral diffuse coating, geometric environment and one diffuse indirect bounce in geometric-camera mode; high-frequency truncation at patch edges',
 excluded:'coherent rough/diffuse field scattering, more than one indirect diffuse bounce, camera diffraction, general/nested refraction, vector polarization, coherent slab internal-reflection sums, slab-edge diffraction, unrestricted reflection orders, corner diffraction, multiple diffraction, boundary feedback',
 coherence:'gamma mixes fully coherent routes with an independently phase-randomized route control; it is not a general source coherence model',
});
export const COHERENT_MESH_LIMITS=Object.freeze({maxMirrorPlanes:8,maxRoutes:65,maxTriangles:12000,maxCoordinateMm:1e4,maxReflections:2});
const TAU=2*Math.PI;
const finite=(x,name,min,max)=>{if(!Number.isFinite(x)||x<min||x>max)throw new RangeError(`${name} outside ${min}…${max}`);return x;};
const vec=(v,name)=>{if(!Array.isArray(v)||v.length!==3||v.some(x=>!Number.isFinite(x)||Math.abs(x)>1e4))throw new TypeError(`${name} must be a finite 3-vector within ±10000 mm`);return [...v];};
export function reducedPhase(x){return ((x+Math.PI)%TAU+TAU)%TAU-Math.PI;}
const reflected=(p,plane)=>sub(p,scale(plane.normal,2*dot(sub(p,plane.point),plane.normal)));
/** Planar unfolding for one physical bounce; no spreading factor per leg.
 * Rims are returned explicitly so mixed UTD may exclude unmodeled truncation. */
export function reflectCoherentPoint(p,plane){return reflected(p,plane);}
export function coherentMirrorBounce(scene,plane,a,b){
 const da=dot(sub(a,plane.point),plane.normal),db=dot(sub(b,plane.point),plane.normal);
 if(da*db<=0)return null;
 const image=reflected(a,plane),direction=sub(image,b),den=dot(direction,plane.normal);
 if(Math.abs(den)<1e-14*length(direction))return null;
 const t=dot(sub(plane.point,b),plane.normal)/den;if(!(t>1e-12&&t<1-1e-12))return null;
 const p=add(b,scale(direction,t)),hit=pointOnPatch(plane,p,scene);if(!hit)return null;
 if(distance(a,p)<1e-8||distance(p,b)<1e-8)return null;
 return {point:p,...hit};
}
function material(input={type:'opaque'}){
 if(input.type==='diffuse')return {type:'diffuse',reflectance:validateDiffuseReflectance(input.reflectance)};
 if(!['opaque','mirror'].includes(input.type))throw new Error('Only opaque blockers and ideal planar mirrors are supported; diffuse/rough materials need a field scattering model');
 if(input.type==='opaque')return {type:'opaque'};
 return {type:'mirror',reflectance:finite(input.reflectance??1,'mirror reflectance',0,1),phaseRad:finite(input.phaseRad??Math.PI,'mirror phase',-TAU,TAU)};
}
function barycentric(p,t){
 const a=sub(t.b,t.a),b=sub(t.c,t.a),q=sub(p,t.a),den=2*t.area;
 const u=dot(cross(q,b),t.normal)/den,v=dot(cross(a,q),t.normal)/den;return [1-u-v,u,v];
}
function sameMaterial(a,b){return a.type===b.type&&a.reflectance===b.reflectance&&reducedPhase(a.phaseRad)===reducedPhase(b.phaseRad);}
function patchBoundary(hits,p,plane){
 // Union of the exact local tangent cones. This also joins T-junctions and
 // separately indexed coplanar meshes; triangle/mesh boundaries are not rims.
 const cones=[];for(const t of hits){const b=barycentric(p,t);if(b.every(x=>x>1e-14))return false;const points=[t.a,t.b,t.c],normals=[];for(let j=0;j<3;j++)if(b[(j+2)%3]<=1e-14)normals.push(normalize(cross(t.normal,sub(points[(j+1)%3],points[j]))));if(!normals.length)return false;cones.push(normals);}
 const u=normalize(sub(hits[0].b,hits[0].a)),v=cross(plane.normal,u),cuts=[0,TAU];
 for(const cone of cones)for(const normal of cone){const a=Math.atan2(dot(normal,v),dot(normal,u));cuts.push((a+Math.PI/2+TAU)%TAU,(a+3*Math.PI/2+TAU)%TAU);}cuts.sort((a,b)=>a-b);
 for(let j=0;j<cuts.length-1;j++){if(cuts[j+1]-cuts[j]<1e-12)continue;const a=(cuts[j]+cuts[j+1])/2,d=add(scale(u,Math.cos(a)),scale(v,Math.sin(a)));if(!cones.some(cone=>cone.every(n=>dot(n,d)>=-1e-12)))return true;}return false;
}
/** Geometry-only union boundary query for thin coplanar scattering patches.
 * Covers separately indexed meshes and T-junctions; material changes alone
 * do not create an open physical knife edge. Used during wedge preparation. */
export function coplanarPatchBoundary(scene,p,normal){
 const hits=scene.triangles.filter(t=>Math.abs(Math.abs(dot(t.normal,normal))-1)<1e-11&&Math.abs(dot(sub(t.a,p),normal))<1e-9&&barycentric(p,t).every(x=>x>=-2e-10));
 return hits.length?patchBoundary(hits,p,{normal}):true;
}
function pointOnPatch(plane,p,scene){
 const hits=plane.triangles.filter(t=>barycentric(p,t).every(x=>x>=-1e-14));if(!hits.length)return null;
 const m=scene.materials[hits[0].materialIndex];if(hits.some(t=>!sameMaterial(m,scene.materials[t.materialIndex])))throw new Error('Overlapping mirror patches with different complex coefficients are ambiguous');
 const opaqueHits=plane.coincidentOtherTriangles?.filter(t=>barycentric(p,t).every(x=>x>=-1e-14))??[];
 if(opaqueHits.length&&!patchBoundary(opaqueHits,p,plane))throw new Error('Coplanar overlapping mirror and opaque materials do not define a unique boundary');
 // Triangle tessellation seams are not physical aperture boundaries.
 const boundary=patchBoundary(hits,p,plane);
 return {material:m,boundary,triangleIds:hits.map(t=>t.id)};
}
/** Reuses the scene BVH with plane-distance intersections. Narrow accepted
 * triangles must not turn an endpoint into an interior self-occluder through
 * a poorly conditioned triple-product distance quotient. No face is ignored. */
export function coherentSegmentVisible(scene,a,b){
 const delta=sub(b,a),d=length(delta);if(d===0)return true;const direction=scale(delta,1/d),e=Math.min(d/4,Math.max(1e-10,128*Number.EPSILON*Math.max(1,...a.map(Math.abs),...b.map(Math.abs))));
 const stack=[scene.bvh.root];while(stack.length){const node=scene.bvh.nodes[stack.pop()];let lo=e,hi=d-e;for(let k=0;k<3;k++){const mn=node.min[k]-e,mx=node.max[k]+e;if(direction[k]===0){if(a[k]<mn||a[k]>mx){hi=-1;break;}}else{let x=(mn-a[k])/direction[k],y=(mx-a[k])/direction[k];if(x>y)[x,y]=[y,x];lo=Math.max(lo,x);hi=Math.min(hi,y);}}if(lo>hi)continue;
  if(node.triangleIds){for(const id of node.triangleIds){const t=scene.triangles[id],den=dot(t.normal,direction);if(Math.abs(den)<1e-14)continue;const s=dot(t.normal,sub(t.a,a))/den;if(s<e||s>d-e)continue;const p=add(a,scale(direction,s));if(barycentric(p,t).every(x=>x>=-1e-14))return false;}}
  else stack.push(node.left,node.right);
 }return true;
}
/** Retains general opaque geometry; mirror routes use unique physical planes.
 * Coplanar triangles across a quad are one route, not two field samples. */
export function prepareCoherentMeshScene(entries,materials=[{type:'opaque'}],{maxReflections=2,slab=null}={}){
 if(!Number.isInteger(maxReflections)||maxReflections<0||maxReflections>2)throw new RangeError('Reflection order must be 0, 1 or 2');
 const ms=materials.map(material),scene=preparePLTScene(entries,{materials:ms});slab=validateCoherentSlab(slab);validateSlabMeshSeparation(slab,scene.triangles);
 if(scene.triangles.length>COHERENT_MESH_LIMITS.maxTriangles)throw new Error('Coherent mesh triangle capacity exceeded');
 if(scene.triangles.some(t=>[t.a,t.b,t.c].some(v=>v.some(x=>Math.abs(x)>1e4))))throw new Error('Coherent mesh world-coordinate capacity exceeded');
 const planes=[];
 for(const triangle of scene.triangles){if(ms[triangle.materialIndex].type!=='mirror')continue;
  let n=[...triangle.normal];const axis=n.map(Math.abs).indexOf(Math.max(...n.map(Math.abs)));if(n[axis]<0)n=scale(n,-1);
  const tol=Math.max(1e-10,128*Number.EPSILON*Math.max(1,...triangle.a.map(Math.abs),...triangle.b.map(Math.abs),...triangle.c.map(Math.abs)));
  let plane=planes.find(p=>distance(p.normal,n)<1e-11&&[triangle.a,triangle.b,triangle.c].every(v=>Math.abs(dot(sub(v,p.point),p.normal))<=Math.max(tol,p.tolerance)));
  if(!plane){plane={id:planes.length,normal:n,point:triangle.a,triangles:[],boundaryEdges:[],tolerance:tol};planes.push(plane);}
  plane.triangles.push(triangle);
 }
 if(planes.length>8)throw new Error('At most 8 distinct mirror planes are supported; arbitrary additional opaque meshes remain allowed');
 for(const p of planes){const ids=new Set(p.triangles.map(t=>t.id));p.boundaryEdges=scene.edges.filter(e=>e.triangleIds.filter(id=>ids.has(id)).length===1);p.coincidentOtherTriangles=scene.triangles.filter(t=>ms[t.materialIndex].type!=='mirror'&&Math.abs(Math.abs(dot(t.normal,p.normal))-1)<1e-11&&[t.a,t.b,t.c].every(v=>Math.abs(dot(sub(v,p.point),p.normal))<=p.tolerance));}
 const sequences=[[]];if(maxReflections>=1)for(const a of planes)sequences.push([a.id]);if(maxReflections>=2)for(const a of planes)for(const b of planes)if(a!==b)sequences.push([a.id,b.id]);
 if(sequences.length>65)throw new Error('Coherent route capacity exceeded');
 return Object.assign(scene,{mirrorPlanes:planes,routeSequences:sequences,maxReflections,scope:COHERENT_MESH_SCOPE,slab});
}
export function compileCoherentSource(scene,source){
 const s={position:vec(source.position,'source position'),power:finite(source.power??1,'source reference power',0,1e6),wavelengthMm:finite(source.wavelengthMm??.00055,'wavelength mm',1e-5,10),phaseRad:finite(source.phaseRad??0,'source phase',-TAU,TAU)};
 if(!slabIsInvisible(scene.slab)&&slabContainsPoint(scene.slab,s.position))throw new Error('Source inside or on dielectric slab is unsupported');
 const families=scene.routeSequences.map(sequence=>{const images=[s.position];for(const i of sequence)images.push(reflected(images.at(-1),scene.mirrorPlanes[i]));return {sequence,images};});
 return {source:s,families,amplitudeScale:Math.sqrt(s.power/(4*Math.PI)),k:TAU/s.wavelengthMm};
}
function orthogonalCornerLimit(scene,compiled,receiver,family){
 if(family.sequence.length!==2)return undefined;const [ia,ib]=family.sequence,a=scene.mirrorPlanes[ia],b=scene.mirrorPlanes[ib];if(Math.abs(dot(a.normal,b.normal))>1e-12)return undefined;
 const image=family.images[2],direction=sub(image,receiver),l=length(direction),da=dot(direction,a.normal),db=dot(direction,b.normal);if(Math.abs(da)<1e-14*l||Math.abs(db)<1e-14*l)return undefined;
 const ta=dot(sub(a.point,receiver),a.normal)/da,tb=dot(sub(b.point,receiver),b.normal)/db;
 if(!(ta>0&&ta<1&&tb>0&&tb<1)||Math.abs(ta-tb)*l>1e-7)return undefined;
 // At a perpendicular reflecting corner the two ordering families coalesce.
 // The exact ±1 image solution has ONE full double image, not two half rims.
 // Other constant coefficients retain this geometrical specular limit only.
 if(ia>ib)return null;let p=add(receiver,scale(direction,(ta+tb)/2));p=sub(p,scale(a.normal,dot(sub(p,a.point),a.normal)));p=sub(p,scale(b.normal,dot(sub(p,b.point),b.normal)));const ha=pointOnPatch(a,p,scene),hb=pointOnPatch(b,p,scene);if(!ha||!hb)return null;
 const tangent=normalize(cross(a.normal,b.normal));for(const s of [-1,1]){const q=add(p,scale(tangent,s*1e-6));if(!pointOnPatch(a,q,scene)||!pointOnPatch(b,q,scene))return null;}
 const source=compiled.source.position;for(const plane of [a,b])if(dot(sub(source,plane.point),plane.normal)*dot(sub(receiver,plane.point),plane.normal)<=0)return null;
 if(!coherentSegmentVisible(scene,source,p)||!coherentSegmentVisible(scene,p,receiver))return null;
 const directLength=distance(receiver,source),ds=sub(source,image),mid=add(sub(receiver,source),sub(receiver,image)),excess=dot(ds,mid)/(l+directLength);
 return {kind:'reflection',sequence:[ia,ib],vertices:[source,p,p,receiver],pathLength:l,excess,amplitude:compiled.amplitudeScale*Math.sqrt(ha.material.reflectance*hb.material.reflectance)/l,interactionPhaseRad:compiled.source.phaseRad+ha.material.phaseRad+hb.material.phaseRad,phaseRadReduced:reducedPhase(compiled.source.phaseRad+ha.material.phaseRad+hb.material.phaseRad-compiled.k*excess),boundary:false,orthogonalCornerImageLimit:true};
}
/** Return one unique visible geometric route per plane sequence. No beam/PDF. */
export function geometricRoute(scene,compiled,receiver,family){
 const corner=orthogonalCornerLimit(scene,compiled,receiver,family);if(corner!==undefined){if(corner)assertSlabRouteSupported(scene.slab,corner.vertices,'reflection');return corner;}
 const {sequence,images}=family,source=compiled.source.position,n=sequence.length,vertices=new Array(n+2);vertices[0]=source;vertices[n+1]=receiver;
 let reflectance=1,phase=compiled.source.phaseRad,apertureWeight=1;
 for(let j=n-1;j>=0;j--){const plane=scene.mirrorPlanes[sequence[j]],target=vertices[j+2],image=images[j+1],direction=sub(image,target),den=dot(direction,plane.normal);
  if(Math.abs(den)<1e-14*length(direction))return null;const t=dot(sub(plane.point,target),plane.normal)/den;if(!(t>1e-12&&t<1-1e-12))return null;
  const p=add(target,scale(direction,t)),hit=pointOnPatch(plane,p,scene);if(!hit)return null;vertices[j+1]=p;reflectance*=hit.material.reflectance;phase+=hit.material.phaseRad;if(hit.boundary)apertureWeight*=.5;
 }
 let pathLength=0;for(let j=0;j<vertices.length-1;j++){const d=distance(vertices[j],vertices[j+1]);if(d<1e-8)return null;pathLength+=d;if(!coherentSegmentVisible(scene,vertices[j],vertices[j+1]))return null;}
 for(let j=0;j<n;j++){const plane=scene.mirrorPlanes[sequence[j]],a=dot(sub(vertices[j],vertices[j+1]),plane.normal),b=dot(sub(vertices[j+2],vertices[j+1]),plane.normal);if(a*b<=0)return null;}
 if(n===1&&vertices.slice(1).some((b,i)=>segmentIntersectsSlab(scene.slab,vertices[i],b)))return null;
 if(n>1)assertSlabRouteSupported(scene.slab,vertices,'reflection');else if(!n&&segmentIntersectsSlab(scene.slab,source,receiver))return null;
 const amplitude=compiled.amplitudeScale*Math.sqrt(reflectance)*apertureWeight/pathLength;
 // Image distance is equal to the broken path. Stable difference of squared
 // distances avoids subtracting two ~100 mm numbers to recover micron phase.
 const directVector=sub(receiver,source),imageVector=sub(receiver,images.at(-1)),directLength=length(directVector),imageLength=length(imageVector);
 const ds=sub(source,images.at(-1)),mid=add(sub(receiver,source),sub(receiver,images.at(-1)));
 const excess=dot(ds,mid)/(imageLength+directLength);
 return {kind:n?'reflection':'direct',sequence:[...sequence],vertices,pathLength,excess,amplitude,interactionPhaseRad:phase,phaseRadReduced:reducedPhase(phase-compiled.k*excess),boundary:apertureWeight!==1};
}
/** One planar reflection and exactly one broad-face slab transmission.
 * Unfold the mirror in the air leg, solve the existing Fermat slab problem,
 * then fold that leg back. An isometry preserves the ray-tube Jacobian.
 * Enumerate both orders, including paths absent from the all-air solution. */
export function reflectedSlabRoutes(scene,compiled,receiver,plane){
 if(slabIsInvisible(scene.slab))return [];
 const source=compiled.source.position,results=[];
 for(const order of ['R-TT','TT-R']){
  const first=order==='R-TT',a=first?reflected(source,plane):source,b=first?receiver:reflected(receiver,plane);
  // A virtual endpoint may lie inside the real box; that unfolding cannot
  // place the required air-leg reflection before/after this transmission.
  if(slabContainsPoint(scene.slab,a)||slabContainsPoint(scene.slab,b))continue;
  const t=solveSlabTransmission(scene.slab,a,b);if(!t)continue;
  const x=first?a:t.exit,y=first?t.entry:b,d=sub(y,x),den=dot(d,plane.normal);
  if(Math.abs(den)<1e-14*length(d))continue;
  const u=dot(sub(plane.point,x),plane.normal)/den;if(!(u>1e-12&&u<1-1e-12))continue;
  const bounce=add(x,scale(d,u)),hit=pointOnPatch(plane,bounce,scene);if(!hit)continue;
  const vertices=first?[source,bounce,t.entry,t.exit,receiver]:[source,t.entry,t.exit,bounce,receiver];
  // The other physical air legs must not reenter the finite slab. This
  // bounded family excludes repeated passages and side-face transmission.
  const glassLeg=first?2:1;let valid=true;
  for(let i=0;i<vertices.length-1;i++){
   if(distance(vertices[i],vertices[i+1])<1e-8||!coherentSegmentVisible(scene,vertices[i],vertices[i+1])){valid=false;break;}
   if(i!==glassLeg&&segmentIntersectsSlab(scene.slab,vertices[i],vertices[i+1])){valid=false;break;}
  }
  if(!valid)continue;
  const directDistance=distance(source,receiver),imageDistance=distance(a,b),image=first?a:b,real=first?source:receiver,target=first?receiver:source;
  const imageExcess=dot(sub(real,image),add(sub(target,real),sub(target,image)))/(imageDistance+directDistance);
  const excess=t.excess+imageExcess,phase=compiled.source.phaseRad+hit.material.phaseRad;
  const unabsorbedAmplitude=compiled.amplitudeScale*t.spreading*t.fieldTransmission*Math.sqrt(hit.material.reflectance)*(hit.boundary?.5:1);
  const amplitude=unabsorbedAmplitude*Math.exp(-.5*slabAbsorptionAt(scene.slab,compiled.source.wavelengthMm*1e6)*t.glassLength);
  results.push({...t,kind:'transmission',interactionOrder:order,sequence:[plane.id],vertices,directDistance,excess,unabsorbedAmplitude,amplitude,interactionPhaseRad:phase,phaseRadReduced:reducedPhase(phase-compiled.k*excess),boundary:hit.boundary});
 }
 return results;
}
export function evaluateGeometricRoutes(scene,compiled,receiver,{blockReflected=false}={}){
 receiver=vec(receiver,'receiver');const directDistance=distance(receiver,compiled.source.position);if(directDistance<1e-6)throw new Error('Detector must not contain the point source');
 if(!slabIsInvisible(scene.slab)&&slabContainsPoint(scene.slab,receiver))throw new Error('Receiver inside or on dielectric slab is unsupported');
 let possibleUnsupportedReflectedSlab=false;const routes=[];for(const family of compiled.families){if(blockReflected&&family.sequence.length)continue;const r=geometricRoute(scene,compiled,receiver,family);if(r)routes.push(r);if(family.sequence.length===1&&!slabIsInvisible(scene.slab)){const transmitted=reflectedSlabRoutes(scene,compiled,receiver,scene.mirrorPlanes[family.sequence[0]]);routes.push(...transmitted);if(!r&&!transmitted.length){const vacuum=geometricRoute({...scene,slab:null},compiled,receiver,family);if(vacuum&&vacuum.vertices.slice(1).some((b,i)=>segmentIntersectsSlab(scene.slab,vacuum.vertices[i],b)))possibleUnsupportedReflectedSlab=true;}}}
 const transmission=solveSlabTransmission(scene.slab,compiled.source.position,receiver),unsupportedSide=!transmission&&segmentIntersectsSlab(scene.slab,compiled.source.position,receiver);if(transmission&&transmission.vertices.slice(1).every((b,i)=>coherentSegmentVisible(scene,transmission.vertices[i],b))){const unabsorbedAmplitude=compiled.amplitudeScale*transmission.spreading*transmission.fieldTransmission,amplitude=unabsorbedAmplitude*Math.exp(-.5*slabAbsorptionAt(scene.slab,compiled.source.wavelengthMm*1e6)*transmission.glassLength);routes.push({...transmission,kind:'transmission',sequence:[],unabsorbedAmplitude,amplitude,interactionPhaseRad:compiled.source.phaseRad,phaseRadReduced:reducedPhase(compiled.source.phaseRad-compiled.k*transmission.excess),boundary:false});}
 return {routes,directDistance,slabDiagnostics:slabIsInvisible(scene.slab)?null:{sourceOrders:'first-transmission-and-one-mirror',unsupportedSide,possibleUnsupportedReflectedSlab,reflectedOmissionReason:possibleUnsupportedReflectedSlab?'air mirror candidate intersects slab; no supported one-pass broad-face route admitted':null,excluded:['coherent-internal-reflections','slab-edge-diffraction','slab-reflection-source-paths','refracted-diffraction','refracted-two-mirror-families','repeated-slab-passages']}};
}
export function sumCoherentRoutes(routes,{coherence=1}={}){
 finite(coherence,'route phase correlation',0,1);let re=0,im=0,power=0;for(const r of routes){if(!Number.isFinite(r.amplitude)||r.amplitude<0||!Number.isFinite(r.phaseRadReduced))throw new Error('Nonfinite coherent route');re+=r.amplitude*Math.cos(r.phaseRadReduced);im+=r.amplitude*Math.sin(r.phaseRadReduced);power+=r.amplitude*r.amplitude;}
 const coherent=re*re+im*im;return {real:re,imag:im,coherent:coherence*coherent+(1-coherence)*power,fullyCoherent:coherent,incoherent:power,crossTerm:coherence*(coherent-power)};
}
export function detectorPoint(detector,x,y,width,height,dx=.5,dy=.5){return add(detector.center,add(scale(detector.u,((x+dx)/width-.5)*detector.width),scale(detector.v,(.5-(y+dy)/height)*detector.height)));}
export function validateDetector(input){const center=vec(input.center,'detector center'),u=normalize(vec(input.u,'detector u')),v=normalize(vec(input.v,'detector v'));if(Math.abs(dot(u,v))>1e-10)throw new Error('Detector axes must be orthogonal');return {center,u,v,normal:normalize(cross(u,v)),width:finite(input.width,'detector width',1e-4,1000),height:finite(input.height,'detector height',1e-4,1000)};}

const quad=(v,m)=>({mesh:{vertices:v,faces:[[0,1,2],[0,2,3]]},materialIndex:m,surface:'open-plate'});
/** Presets are ordinary scene data, processed by the same route constructor. */
export function makeCoherentMeshPreset(name='lloyd',{phaseShiftRad=0,customOBJ=null,customMaterial='opaque'}={}){
 if(name==='object-imaging'){const data=makeCoherentCameraPreset();for(const m of data.materials)if(m.type==='mirror')m.phaseRad+=phaseShiftRad;if(customOBJ!==null){data.materials.push(customMaterial==='mirror'?{type:'mirror',reflectance:1,phaseRad:Math.PI+phaseShiftRad}:{type:'opaque'});data.entries.push({mesh:customOBJ,materialIndex:data.materials.length-1});}return data;}
 const materials=[{type:'opaque'},{type:'mirror',reflectance:1,phaseRad:Math.PI+phaseShiftRad}];let entries,position,center;
 if(name==='lloyd'){
  entries=[quad([[-10,0,-49],[10,0,-49],[10,0,49],[-10,0,49]],1)];position=[0,.08,-50];center=[0,3.05,50];
 }else if(name==='two-mirror'){
  entries=[quad([[0,0,-49],[10,0,-49],[10,0,49],[0,0,49]],1),quad([[0,0,-49],[0,10,-49],[0,10,49],[0,0,49]],1)];position=[.08,.08,-50];center=[3.05,3.05,50];
 }else if(name==='double-slit'){
  const halfWidth=10,halfHeight=10,width=.02,separation=.15;
  const intervals=[[-halfWidth,-separation/2-width/2],[-separation/2+width/2,separation/2-width/2],[separation/2+width/2,halfWidth]];
  entries=intervals.map(([a,b])=>quad([[a,-halfHeight,0],[b,-halfHeight,0],[b,halfHeight,0],[a,halfHeight,0]],1));position=[0,0,-50];center=[0,0,50];
 }else if(name==='reflected-edge'){
  materials[1]={type:'mirror',reflectance:1,phaseRad:Math.PI};materials.push({type:'mirror',reflectance:.8,phaseRad:.3+phaseShiftRad});
  entries=[quad([[0,-20,0],[30,-20,0],[30,20,0],[0,20,0]],1),quad([[-5,-20,-3.9],[-5,20,-3.9],[-5,20,-.1],[-5,-20,-.1]],2),quad([[-3,-20,-3],[-1,-20,-3],[-1,20,-3],[-3,20,-3]],0)];position=[-3,0,-4];center=[17.5,0,10];
 }else throw new Error('Unknown coherent mesh preset');
 if(customOBJ){materials.push(customMaterial==='mirror'?{type:'mirror',reflectance:1,phaseRad:Math.PI+phaseShiftRad}:{type:'opaque'});entries.push({mesh:prepareMesh(customOBJ,{rejectNonManifold:false}),materialIndex:materials.length-1,surface:'auto'});}
 return {entries,materials,source:{position},detector:{center,u:[1,0,0],v:[0,1,0],width:name==='reflected-edge'?2:name==='double-slit'?3:6,height:name==='reflected-edge'?2:name==='double-slit'?3:6},name};
}
