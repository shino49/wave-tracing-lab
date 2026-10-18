/**
 * Single-region, deterministic 3-D scalar UTD transport. Units are millimetres.
 * Gaussian cones select a finite interaction region; they do NOT multiply edge
 * amplitudes or turn conditional f into irradiance. exp(+i omega t) convention.
 * A locally collimated incident plane wave is used inside the selected region.
 * End/corner diffraction, repeated interactions and coherent vector polarization
 * transport are not implemented. Invalid samples must be masked by the caller.
 */
import {
  add,sub,scale,dot,cross,length,distance,normalize,prepareMesh,makePresetOBJ,
  extractWedges,orientWedge,diffractionPointPlane,planeDiffractionVisibility,
  gatherConeTriangles,rayTriangleIntersect,rayOccluded,
} from './mesh-geometry.js';
import {coefficient,add as cadd,mul as cmul,scale as cscale,cis,abs2,BOUNDARY_EPS} from './utd-math.js';
import {validateUTDGeometry} from './utd-geometry-check.js';

export const UTD_LIMITS=Object.freeze({maxTriangles:128,maxEdges:256});
const PI=Math.PI,TAU=2*PI;
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const finite=(v,name,min=-Infinity,max=Infinity)=>{
  if(!Number.isFinite(v)||v<min||v>max)throw new Error(`${name} must be finite in [${min}, ${max}].`);
  return v;
};
const vector=(v,name)=>{
  if(!Array.isArray(v)||v.length!==3||v.some(x=>!Number.isFinite(x)))throw new Error(`${name} requires three finite coordinates.`);
  return [...v];
};
const pointMagnitude=p=>Math.max(Math.abs(p[0]),Math.abs(p[1]),Math.abs(p[2]));
// All callers supply at most six 3-vectors. Fixed arity avoids rest/flat/map
// arrays on ray, cone and region hot paths; nested maxima retain exact epsilon.
const epsilonAt=(a,b,c,d,e,f)=>{
  let magnitude=1;
  if(a!==undefined)magnitude=Math.max(magnitude,pointMagnitude(a));
  if(b!==undefined)magnitude=Math.max(magnitude,pointMagnitude(b));
  if(c!==undefined)magnitude=Math.max(magnitude,pointMagnitude(c));
  if(d!==undefined)magnitude=Math.max(magnitude,pointMagnitude(d));
  if(e!==undefined)magnitude=Math.max(magnitude,pointMagnitude(e));
  if(f!==undefined)magnitude=Math.max(magnitude,pointMagnitude(f));
  return Math.max(1e-10,128*Number.EPSILON*magnitude);
};
const within=(p,c)=>{
  const d=sub(p,c.origin),t=dot(d,c.direction);
  return t>=c.near&&t<=c.far&&length(sub(d,scale(c.direction,t)))<=c.radius0+c.tanAngle*t+epsilonAt(p);
};
const rayMin=(mesh,o,d,options={})=>{
  let best=null;
  for(const t of mesh.triangles){const h=rayTriangleIntersect(o,d,t,options);if(h&&(!best||h.distance<best.distance))best=h;}
  return best;
};

/** Convex cross-section of a triangle at a beam-axis depth, including flat faces.
 * The query owns all scratch storage: no per-section vectors or projections.
 * Keep the original arithmetic order (including Math.hypot) and endpoint order;
 * changing either can move a tolerance-boundary contact to another triangle.
 */
function sectionAt(query,t) {
  const {points:p,depths:z,edges,section:out,refs,tol,cone}=query;
  let count=0;
  for(let i=0;i<3;i++) {
    const j=(i+1)%3,k=i*3;
    if(Math.abs(z[i]-t)<=tol) {
      const at=count*3;
      out[at]=p[i][0];out[at+1]=p[i][1];out[at+2]=p[i][2];refs[count++]=p[i];
    }
    if((z[i]<t&&z[j]>t)||(z[i]>t&&z[j]<t)) {
      const u=(t-z[i])/(z[j]-z[i]),at=count*3;
      out[at]=p[i][0]+edges[k]*u;
      out[at+1]=p[i][1]+edges[k+1]*u;
      out[at+2]=p[i][2]+edges[k+2]*u;
      refs[count++]=null;
    }
  }
  if(!count)return null;
  const ax=cone.origin[0]+cone.direction[0]*t,
    ay=cone.origin[1]+cone.direction[1]*t,az=cone.origin[2]+cone.direction[2]*t;
  if(count>=3&&query.flat) {
    // Tolerance may duplicate endpoints without making a section coplanar.
    const n=query.triangle.normal;
    let inside=true;
    for(let i=0;i<3;i++) {
      const k=i*3,dx=ax-p[i][0],dy=ay-p[i][1],dz=az-p[i][2];
      const cx=edges[k+1]*dz-edges[k+2]*dy,
        cy=edges[k+2]*dx-edges[k]*dz,cz=edges[k]*dy-edges[k+1]*dx;
      if(!(cx*n[0]+cy*n[1]+cz*n[2]>=-tol)){inside=false;break;}
    }
    if(inside){query.x=ax;query.y=ay;query.z=az;query.pointRef=null;return 0;}
  }
  let px=out[0],py=out[1],pz=out[2],pointRef=refs[0];
  let radial=Math.hypot(px-ax,py-ay,pz-az);
  for(let i=0;i<count;i++)for(let j=i+1;j<count;j++) {
    const a=i*3,b=j*3,dx=out[b]-out[a],dy=out[b+1]-out[a+1],dz=out[b+2]-out[a+2];
    const ll=dx*dx+dy*dy+dz*dz;
    const u=ll?clamp(((ax-out[a])*dx+(ay-out[a+1])*dy+(az-out[a+2])*dz)/ll,0,1):0;
    const qx=out[a]+dx*u,qy=out[a+1]+dy*u,qz=out[a+2]+dz*u;
    const r=Math.hypot(qx-ax,qy-ay,qz-az);
    if(r<radial){radial=r;px=qx;py=qy;pz=qz;pointRef=null;}
  }
  query.x=px;query.y=py;query.z=pz;query.pointRef=pointRef;
  return radial;
}
const sectionPoint=query=>query.pointRef??[query.x,query.y,query.z];

/** Earliest actual cone/triangle contact, after conservative broad-phase gather.
 * At depth t the nearest triangle-section point gives a convex distance-minus-
 * radius function. Golden-section minimization brackets feasibility, followed
 * by bisection of its first zero. This never substitutes the triangle centroid.
 */
export function firstConeTriangleContact(triangle,cone) {
  const points=[triangle.a,triangle.b,triangle.c],depths=new Float64Array(3);
  for(let i=0;i<3;i++) {
    const p=points[i];
    depths[i]=(p[0]-cone.origin[0])*cone.direction[0]+(p[1]-cone.origin[1])*cone.direction[1]+(p[2]-cone.origin[2])*cone.direction[2];
  }
  const minDepth=Math.min(depths[0],depths[1],depths[2]),maxDepth=Math.max(depths[0],depths[1],depths[2]);
  const lo=Math.max(cone.near,minDepth),hi=Math.min(cone.far,maxDepth);
  if(lo>hi)return null;
  const tol=epsilonAt(triangle.a,triangle.b,triangle.c,cone.origin),edges=new Float64Array(9);
  for(let i=0;i<3;i++) {
    const a=points[i],b=points[(i+1)%3],k=i*3;
    edges[k]=b[0]-a[0];edges[k+1]=b[1]-a[1];edges[k+2]=b[2]-a[2];
  }
  const query={triangle,cone,points,depths,edges,tol,flat:maxDepth-minDepth<=tol,
    section:new Float64Array(18),refs:new Array(6),x:0,y:0,z:0,pointRef:null};
  const test=t=>{const radial=sectionAt(query,t);return radial===null?null:radial-cone.radius0-cone.tanAngle*t;};
  const left=test(lo);if(left!==null&&left<=tol)return {distance:lo,point:sectionPoint(query),triangleId:triangle.id};
  if(hi-lo<=tol)return null;
  let a=lo,b=hi;
  const gr=(Math.sqrt(5)-1)/2;
  let c=b-gr*(b-a),d=a+gr*(b-a),fc=test(c),fd=test(d);
  for(let i=0;i<65;i++) {
    if((fc??Infinity)<(fd??Infinity)){b=d;d=c;fd=fc;c=b-gr*(b-a);fc=test(c);}
    else {a=c;c=d;fc=fd;d=a+gr*(b-a);fd=test(d);}
  }
  const candidates=[[(a+b)/2,test((a+b)/2)],[hi,test(hi)],[c,fc],[d,fd]].filter(v=>v[1]!==null);
  candidates.sort((x,y)=>x[1]-y[1]);
  if(!candidates.length||candidates[0][1]>tol)return null;
  let lower=lo,upper=candidates[0][0];
  for(let i=0;i<65;i++){const mid=(lower+upper)/2;if((test(mid)??Infinity)<=tol)upper=mid;else lower=mid;}
  test(upper);
  return {distance:upper,point:sectionPoint(query),triangleId:triangle.id};
}

/** Clip the actual 3-D edge to x²/a²+y²/a²+z²/(2a)² <= 1. */
export function clipEdgeToRegion(wedge,region) {
  const metric=p=>{
    const q=sub(p,region.center),z=dot(q,region.direction),perp=sub(q,scale(region.direction,z));
    return add(scale(perp,1/region.radius),scale(region.direction,z/(2*region.radius)));
  };
  const a=metric(wedge.a),b=metric(wedge.b),d=sub(b,a),A=dot(d,d),B=2*dot(a,d),C=dot(a,a)-1;
  let disc=B*B-4*A*C;
  if(disc<0&&disc>-1e-12*(B*B+Math.abs(4*A*C)))disc=0;
  if(A===0||disc<0)return null;
  const root=Math.sqrt(disc);let u0=Math.max(0,(-B-root)/(2*A)),u1=Math.min(1,(-B+root)/(2*A));
  if(Number.isFinite(region.depth)) {
    const za=dot(sub(wedge.a,region.center),region.direction),dz=dot(sub(wedge.b,wedge.a),region.direction);
    if(Math.abs(dz)<1e-14) {if(za< -epsilonAt(wedge.a)||za>region.depth+epsilonAt(wedge.a))return null;}
    else {const v0=-za/dz,v1=(region.depth-za)/dz;u0=Math.max(u0,Math.min(v0,v1));u1=Math.min(u1,Math.max(v0,v1));}
  }
  if(u1-u0<=1e-12)return null;
  const delta=sub(wedge.b,wedge.a),pa=add(wedge.a,scale(delta,u0)),pb=add(wedge.a,scale(delta,u1));
  return {...wedge,a:pa,b:pb,v:scale(add(pa,pb),.5),l:distance(pa,pb),
    originalA:[...wedge.a],originalB:[...wedge.b],originalLength:wedge.l,
    clippedStart:u0>1e-10,clippedEnd:u1<1-1e-10,clipInterval:[u0,u1]};
}
function regionValue(region,p) {
  const d=sub(p,region.center),z=dot(d,region.direction);
  return (dot(d,d)-.75*z*z)/(region.radius*region.radius);
}

function buildFacets(mesh,triangleIds) {
  const selected=new Set(triangleIds),seen=new Set(),adjacent=new Map(mesh.triangles.map(t=>[t.id,[]]));
  for(const edge of mesh.internalEdges)if(edge.valence===2){const [a,b]=edge.triangleIds;adjacent.get(a).push(b);adjacent.get(b).push(a);}
  const facets=[];
  for(const id of triangleIds) {
    if(seen.has(id))continue;
    const ids=[id];seen.add(id);
    for(let i=0;i<ids.length;i++)for(const j of adjacent.get(ids[i]))if(selected.has(j)&&!seen.has(j)){seen.add(j);ids.push(j);}
    const triangles=ids.map(i=>mesh.triangles[i]);
    const normal=triangles[0].normal;
    // Absolute geometric normals matter. Coplanar, opposing duplicate facets
    // were already rejected by the geometry topology gate.
    facets.push({id:facets.length,triangleIds:ids,triangles,normal,point:triangles[0].a});
  }
  return facets;
}

export function builtInSceneOBJ(name='screen') {
  if(['screen','opaque-plate','plate'].includes(name))return makePresetOBJ('finite-square',{width:2,height:20});
  if(['box','convex-box'].includes(name))return makePresetOBJ('box',{width:2,height:20,depth:2});
  if(['wedge','wedge-prism','triangular-prism'].includes(name))return makePresetOBJ('triangular-prism',{width:2,height:20,depth:2});
  if(name==='opaque-double-slit')return makePresetOBJ(name);
  throw new Error(`Unknown UTD preset '${name}'.`);
}

export function prepareUTDScene(input,state={}) {
  const mesh=input?.triangles?input:prepareMesh(input);
  const wavelengthMm=finite(state.wavelengthMm??state.lambdaMm??.05,'Wavelength',1e-6,1e4);
  const sigma0Mm=finite(state.sigma0Mm??state.beamSigmaMm??1,'Beam sigma',1e-5,1e4),k=TAU/wavelengthMm;
  const sourceOrigin=vector(state.sourceOrigin??[0,0,-5],'Source origin'),incidentDirection=normalize(vector(state.incidentDirection??[0,0,1],'Incident direction'));
  const detectorCenter=vector(state.detectorCenter??[0,0,20],'Detector center');
  const detectorU=normalize(vector(state.detectorU??[1,0,0],'Detector u')),detectorV=normalize(vector(state.detectorV??[0,1,0],'Detector v'));
  if(Math.abs(dot(detectorU,detectorV))>1e-8)throw new Error('Detector axes must be perpendicular.');
  const params={wavelengthMm,k,sigma0Mm,sourceOrigin,incidentDirection,detectorCenter,detectorU,detectorV,
    detectorWidthMm:finite(state.detectorWidthMm??6,'Detector width',1e-6,1e6),detectorHeightMm:finite(state.detectorHeightMm??6,'Detector height',1e-6,1e6),
    boundary:state.boundary??'unpolarized',surface:state.surface??(mesh.isClosed?'closed-solid':'open-plate'),
    minKr:finite(state.minKr??10,'Minimum k r sin² beta',1,1e8),minSinBeta:finite(state.minSinBeta??1e-3,'Minimum sin beta',1e-6,1),
    endpointFresnelGuard:finite(state.endpointFresnelGuard??2,'Endpoint Fresnel guard',1,10),
    phaseCurvatureLimit:finite(state.phaseCurvatureLimit??.25,'Plane-incidence curvature guard',1e-4,PI),
    boundaryEpsilon:finite(state.boundaryEpsilon??BOUNDARY_EPS,'Angular boundary epsilon',0,.001)};
  if(!['soft','hard','unpolarized'].includes(params.boundary))throw new Error('Boundary must be soft, hard, or unpolarized.');
  const incomingCone={origin:sourceOrigin,direction:incidentDirection,radius0:3*sigma0Mm,tanAngle:3/(2*k*sigma0Mm),near:0,far:Infinity};
  // Stop before expensive topology/admission work when a parsed OBJ exceeds
  // the solver budget. The parser's larger limit is not a rendering promise.
  const extracted=mesh.triangles.length>UTD_LIMITS.maxTriangles?
    {supported:false,surface:params.surface,wedges:[],unsupported:[],warnings:[],components:[]}:
    extractWedges(mesh,{surface:params.surface});
  const unsupported=[...extracted.unsupported],warnings=[...mesh.warnings,...extracted.warnings];
  const addIssue=(code,message)=>unsupported.push({code,message});
  if(mesh.triangles.length>UTD_LIMITS.maxTriangles)addIssue('triangle-safety-cap',`CPU/GPU safety cap: ${UTD_LIMITS.maxTriangles} triangles; no performance claim is implied.`);
  if(extracted.wedges.length>UTD_LIMITS.maxEdges)addIssue('edge-safety-cap',`CPU/GPU safety cap: ${UTD_LIMITS.maxEdges} physical edges.`);
  if(unsupported.length===0&&extracted.supported){const checked=validateUTDGeometry(mesh,extracted);unsupported.push(...checked.issues);warnings.push(...checked.warnings);}
  const transportAllowed=unsupported.length===0;
  const candidates=transportAllowed?gatherConeTriangles(mesh,incomingCone):[],contacts=candidates.map(t=>firstConeTriangleContact(t,incomingCone)).filter(Boolean).sort((a,b)=>a.distance-b.distance);
  const first=contacts[0]??null;
  let region=null,triangles=[],wedges=[],facets=[],phaseCurvature=0;
  const fullWedges=extracted.wedges.map(w=>orientWedge(w,scale(incidentDirection,-1))).filter(Boolean);
  if(first) {
    const a=incomingCone.radius0+incomingCone.tanAngle*first.distance;
    region={center:add(sourceOrigin,scale(incidentDirection,first.distance)),direction:incidentDirection,
      radius:a,axes:[a,a,2*a],distance:first.distance,depth:2*a,firstSurfacePoint:first.point,firstTriangleId:first.triangleId,
      near:first.distance,far:first.distance+2*a};
    const frustum={...incomingCone,near:region.near,far:region.far};
    triangles=gatherConeTriangles(mesh,frustum);
    const ids=new Set(triangles.map(t=>t.id));
    wedges=fullWedges.filter(w=>w.triangleIds.some(id=>ids.has(id))).map(w=>clipEdgeToRegion(w,region)).filter(Boolean);
    facets=buildFacets(mesh,[...ids]);
    // Mesh display/topology uses a small crease-angle grouping tolerance. A
    // reflection facet, however, is represented by one mathematical plane.
    // Reject a shallow real crease rather than flatten it and omit diffraction.
    if(facets.some(f=>f.triangles.some(t=>[t.a,t.b,t.c].some(v=>Math.abs(dot(sub(v,f.point),f.normal))>epsilonAt(v,f.point)))))
      addIssue('nonplanar-reflection-facet','A grouped reflection facet is not coplanar. Shallow unresolved creases are unsupported; geometry is not flattened.');
    // Separated interactions beyond this region cannot be represented by one
    // coherent region coefficient. Never silently flatten or merge their depth.
    const deeper=contacts.filter(c=>c.distance>region.far+epsilonAt(c.point));
    if(deeper.length)addIssue('multiple-depth-regions','Geometry intersects the incoming cone beyond the single interaction-depth interval; multiple regions are unsupported.');
    const z=first.distance,zR=2*k*sigma0Mm*sigma0Mm;
    const invCurvature=z===0?0:z/(z*z+zR*zR);
    phaseCurvature=.5*k*a*a*Math.abs(invCurvature);
    if(phaseCurvature>params.phaseCurvatureLimit)addIssue('plane-curvature-guard',`Locally collimated phase guard exceeded (${phaseCurvature.toFixed(3)} rad). This is a heuristic guard, not an error certificate.`);
  } else if(transportAllowed)warnings.push('The finite 3σ beam envelope misses the mesh; only the direct conditional plane-wave term is available.');
  warnings.push('Conditional directional UTD power transfer f; not irradiance or a calibrated Gaussian-mode solution.',
    'The 3σ envelope truncation, two-Fresnel-width endpoint screen and local-plane phase-curvature screen are disclosed approximations.',
    'Finite-edge endpoint/corner diffraction, multiple interaction regions and repeated reflections/diffractions are unsupported.');
  const packedTriangles=new Float32Array(mesh.triangles.length*12);
  mesh.triangles.forEach((t,i)=>{packedTriangles.set([...t.a,0,...t.b,0,...t.c,0],12*i);});
  const rebaseOrigin=[...mesh.bounds.center],detectorCorners=[-1,1].flatMap(u=>[-1,1].map(v=>add(detectorCenter,add(scale(detectorU,u*params.detectorWidthMm/2),scale(detectorV,v*params.detectorHeightMm/2)))));
  const maxRange=Math.max(...detectorCorners.map(p=>distance(p,rebaseOrigin)))+mesh.bounds.radius;
  const maxPhaseArgumentBound=2*k*maxRange,maxLocalCoordinate=Math.max(maxRange,distance(sourceOrigin,rebaseOrigin));
  const gpuReasons=[];
  if(maxPhaseArgumentBound>1e5)gpuReasons.push('Float32 phase argument bound exceeds 100000 radians; use CPU.');
  if(maxLocalCoordinate/Math.max(wavelengthMm,1e-30)>1e7)gpuReasons.push('Float32 local coordinate dynamic range is too large relative to wavelength; use CPU.');
  const gpuSafety={safe:gpuReasons.length===0,rebaseOrigin,maxPhaseArgumentBound,maxLocalCoordinate,reasons:gpuReasons};
  const adjacencyWords=Math.ceil(UTD_LIMITS.maxTriangles/32),packedEdges=new Float32Array(wedges.length*24),packedEdgeAdjacency=new Uint32Array(wedges.length*adjacencyWords);
  wedges.forEach((w,i)=>{
    packedEdges.set([...w.a,0,...w.b,0,...w.nff,0,...w.tff,0,...w.nbf,0,w.alpha,w.originalLength,w.clippedStart?1:0,w.clippedEnd?1:0],24*i);
    for(const id of w.triangleIds)if(id<UTD_LIMITS.maxTriangles)packedEdgeAdjacency[i*adjacencyWords+(id>>>5)]|=(1<<(id&31));
  });
  return {mesh,params,incomingCone,region,triangles,wedges,fullWedges,physicalWedges:extracted.wedges,facets,phaseCurvature,
    packedTriangles,packedEdges,packedEdgeAdjacency,adjacencyWords,
    supported:unsupported.length===0,unsupported,warnings,gpuSafety,
    counts:{triangles:mesh.triangles.length,physicalEdges:extracted.wedges.length,orientedEdges:fullWedges.length,gatheredTriangles:triangles.length,clippedEdges:wedges.length},
    model:'single-region-plane-incidence-utd',quantity:'conditional directional power transfer f'};
}

export function computeReceiverPoint(scene,x,y,w,h) {
  if(!(w>0&&h>0))throw new Error('Receiver resolution must be positive.');
  const p=scene.params;
  return add(p.detectorCenter,add(scale(p.detectorU,((x+.5)/w-.5)*p.detectorWidthMm),scale(p.detectorV,(.5-(y+.5)/h)*p.detectorHeightMm)));
}
export function outgoingConeForDirection(scene,direction) {
  if(!scene.region)return null;
  // Sphere bound encloses the complete (a,a,2a) ellipsoid for every direction.
  // This is deliberately conservative rather than an exact elliptical cone.
  return {origin:[...scene.region.center],direction:normalize(direction),radius0:2*scene.region.radius,
    tanAngle:scene.incomingCone.tanAngle,near:0,far:Infinity,conservative:true,construction:'sphere enclosing interaction ellipsoid'};
}
export function stablePlaneExcess(incidentDirection,point,destination) {
  const d=sub(destination,point),parallel=dot(d,incidentDirection),r=length(d);
  if(parallel>0){const perp=sub(d,scale(incidentDirection,parallel));return dot(perp,perp)/(r+parallel);}
  return r-parallel;
}
function wedgeAngles(wedge,wi,wo) {
  const phi_i=Math.atan2(dot(wi,wedge.nff),dot(wi,wedge.tff));
  let phi_o=Math.atan2(dot(wo,wedge.nff),dot(wo,wedge.tff));if(phi_o<0)phi_o+=TAU;
  return {phi_i,phi_o};
}
function stationaryUnbounded(w,di,dst) {
  const e=normalize(sub(w.b,w.a)),d=sub(dst,w.v),c=dot(di,e),sb=Math.sqrt(Math.max(0,1-c*c));
  if(sb<1e-12)return null;
  const da=dot(d,e),rho=length(sub(d,scale(e,da))),axial=da-rho*c/sb;
  const point=add(w.v,scale(e,axial)),ro=distance(point,dst);
  return {point,axial,ro,sinBeta:sb,endpointDistance:Math.min(axial+w.l/2,w.l/2-axial)};
}
const physicalBoundaryAt=(scene,point,eps,facetIds=null)=>scene.physicalWedges.filter(w=>
  (!facetIds||w.triangleIds.some(id=>facetIds.has(id)))&&
  pointSegmentDistance(point,w.a,w.b)<=eps);
function pointSegmentDistance(p,a,b){const d=sub(b,a),ll=dot(d,d);return distance(p,add(a,scale(d,ll?clamp(dot(sub(p,a),d)/ll,0,1):0)));}
function directWeight(scene,dst,epsilon,reasons) {
  if(!within(dst,scene.incomingCone))return 0;
  const di=scene.params.incidentDirection,wi=scale(di,-1),tol=epsilonAt(dst);
  const hit=rayMin(scene.mesh,dst,wi,{minDistance:tol});
  const boundaries=[];
  for(const w of scene.fullWedges) {
    const h=diffractionPointPlane(w,di,dst);if(!h)continue;
    const a=wedgeAngles(w,h.wi,h.wo);
    if(Math.abs(PI-Math.abs(a.phi_o-a.phi_i))<=epsilon)boundaries.push(w);
  }
  if(hit) {
    const border=physicalBoundaryAt(scene,hit.point,Math.max(tol,epsilon*hit.distance));
    if(border.length>1)reasons.add('corner-go-boundary');
    if(border.length&&!boundaries.length)boundaries.push(border[0]);
  }
  if(boundaries.length>1)reasons.add('corner-go-boundary');
  if(boundaries.length) {
    // A separate blocker makes the GO path dark even if another physical edge
    // happens to lie on the same unoccluded infinite-plane shadow seam.
    for(const t of scene.mesh.triangles) {
      const h=rayTriangleIntersect(dst,wi,t,{minDistance:tol});
      if(h&&physicalBoundaryAt(scene,h.point,Math.max(tol,epsilon*h.distance)).length===0)return 0;
    }
    const hasRegionSaddle=boundaries.some(w=>scene.wedges.some(v=>v.edgeId===w.edgeId&&diffractionPointPlane(v,di,dst)));
    if(!hasRegionSaddle)reasons.add('shadow-boundary-outside-region');
    return .5;
  }
  return hit?0:1;
}
function reflectionFields(scene,dst,epsilon,reasons) {
  let soft=[0,0],hard=[0,0],count=0;
  const di=scene.params.incidentDirection,k=scene.params.k,tol=epsilonAt(dst);
  for(const facet of scene.facets) {
    let normal=facet.normal,inc=dot(di,normal);
    if(scene.params.surface==='open-plate'&&inc>0){normal=scale(normal,-1);inc=-inc;}
    if(inc>=-scene.params.minSinBeta)continue;
    const dr=sub(di,scale(normal,2*inc)),den=dot(dr,normal);
    const r=dot(sub(dst,facet.point),normal)/den;
    if(r<=tol)continue;
    const point=sub(dst,scale(dr,r));
    const border=physicalBoundaryAt(scene,point,Math.max(tol,epsilon*r),new Set(facet.triangleIds));
    // The existing symmetric GO half-weight band straddles the physical edge.
    // Barycentric membership must not remove its slightly outside half first.
    const onFacet=facet.triangles.some(t=>rayTriangleIntersect(dst,scale(dr,-1),t,{minDistance:tol,maxDistance:r+tol}));
    if(!onFacet&&!border.length)continue;
    if(!within(point,scene.incomingCone)||regionValue(scene.region,point)>1+tol)continue;
    const ignoreTriangleIds=facet.triangleIds;
    if(rayOccluded(scene.mesh,point,scale(di,-1),{minDistance:tol,ignoreTriangleIds})||rayOccluded(scene.mesh,point,dr,{minDistance:tol,maxDistance:Math.max(tol,r-tol),ignoreTriangleIds}))continue;
    if(border.length>1)reasons.add('corner-reflection-boundary');
    const weight=border.length?.5:1;
    const phase=cis(-k*stablePlaneExcess(di,point,dst));
    soft=cadd(soft,cscale(phase,-weight));hard=cadd(hard,cscale(phase,weight));count++;
  }
  return {soft,hard,count};
}

export function evaluatePoint(scene,destination,options={}) {
  const dst=vector(destination,'Receiver point'),p=scene.params,boundary=options.boundary??p.boundary;
  if(!['soft','hard','unpolarized'].includes(boundary))throw new Error('Boundary must be soft, hard, or unpolarized.');
  const epsilon=finite(options.boundaryEpsilon??p.boundaryEpsilon,'Angular boundary epsilon',0,.001);
  const reasons=new Set(scene.unsupported.map(e=>e.code));
  if(!scene.supported)return {soft:[NaN,NaN],hard:[NaN,NaN],f:NaN,power:NaN,geometricPower:NaN,valid:false,invalidReasons:[...reasons],directWeight:0,reflectionCount:0,edgeCount:0,omittedStationary:0,quantity:'conditional directional power transfer f'};
  const direct=directWeight(scene,dst,epsilon,reasons),reflection=reflectionFields(scene,dst,epsilon,reasons);
  let soft=cadd([direct,0],reflection.soft),hard=cadd([direct,0],reflection.hard);
  const goSoft=[...soft],goHard=[...hard];
  let edgeCount=0,omittedStationary=0;
  const contributions=[];
  for(const w of scene.wedges) {
    const unlimited=stationaryUnbounded(w,p.incidentDirection,dst);
    if(!unlimited){reasons.add('edge-parallel-incidence');continue;}
    const width=Math.sqrt(unlimited.ro/(p.k*unlimited.sinBeta*unlimited.sinBeta));
    const h=diffractionPointPlane(w,p.incidentDirection,dst);
    if(!h) {
      omittedStationary++;
      const endpoint=distance(unlimited.point,w.a)<distance(unlimited.point,w.b)?w.a:w.b;
      if(unlimited.endpointDistance>=-p.endpointFresnelGuard*width&&planeDiffractionVisibility(scene.mesh,w,p.incidentDirection,dst,endpoint).visible)
        reasons.add(w.clippedStart||w.clippedEnd?'region-truncated-stationary-neighborhood':'endpoint-dominated');
      continue;
    }
    const a=wedgeAngles(w,h.wi,h.wo),n=2-w.alpha/PI;
    // The solid occupies the angular sector outside [0,n*pi]. At exact face
    // tangency we mask the unsupported grazing asymptotics instead of clipping D.
    if(a.phi_o>n*PI+epsilon)continue;
    if(!planeDiffractionVisibility(scene.mesh,w,p.incidentDirection,dst,h).visible)continue;
    if(h.sinBeta<p.minSinBeta||p.k*h.ro*h.sinBeta*h.sinBeta<p.minKr){reasons.add('grazing-or-low-kr');continue;}
    if(Math.min(a.phi_o,n*PI-a.phi_o)<p.minSinBeta){reasons.add('surface-grazing');continue;}
    const originalMargin=Math.min(distance(h.point,w.originalA),distance(h.point,w.originalB));
    if(originalMargin<p.endpointFresnelGuard*width)reasons.add('endpoint-dominated');
    if((w.clippedStart||w.clippedEnd)&&unlimited.endpointDistance<p.endpointFresnelGuard*width)reasons.add('region-truncated-stationary-neighborhood');
    const L=h.ro*h.sinBeta*h.sinBeta,phase=cscale(cis(-p.k*stablePlaneExcess(p.incidentDirection,h.point,dst)),1/Math.sqrt(h.ro));
    const ds=coefficient(p.k,n,a.phi_i,a.phi_o,L,h.sinBeta,'soft',epsilon),dh=coefficient(p.k,n,a.phi_i,a.phi_o,L,h.sinBeta,'hard',epsilon);
    const es=cmul(ds,phase),eh=cmul(dh,phase);
    soft=cadd(soft,es);hard=cadd(hard,eh);edgeCount++;
    if(options.details)contributions.push({edgeId:w.edgeId,point:h.point,ro:h.ro,sinBeta:h.sinBeta,L,...a,soft:es,hard:eh,endpointFresnelWidth:width});
  }
  const f=boundary==='soft'?abs2(soft):boundary==='hard'?abs2(hard):(abs2(soft)+abs2(hard))/2;
  const geometricPower=boundary==='soft'?abs2(goSoft):boundary==='hard'?abs2(goHard):(abs2(goSoft)+abs2(goHard))/2;
  if(!Number.isFinite(f))reasons.add('nonfinite-field');
  if(direct===0&&reflection.count===0&&edgeCount===0)reasons.add('no-first-order-path');
  if(direct===.5&&edgeCount===0)reasons.add('unsupported-shadow-boundary');
  if(scene.region&&distance(dst,scene.region.center)<=2*scene.region.radius)reasons.add('receiver-inside-interaction-region');
  const invalidReasons=[...reasons],valid=invalidReasons.length===0;
  return {soft,hard,f,power:valid?f:NaN,geometricPower,valid,invalidReasons,
    directWeight:direct,reflectionCount:reflection.count,edgeCount,omittedStationary,
    ...(options.details?{contributions}:{}),quantity:'conditional directional power transfer f'};
}
