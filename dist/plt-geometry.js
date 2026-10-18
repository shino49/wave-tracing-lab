/**
 * General, browser-safe triangle scene queries for 3-D path tracing (mm).
 * Safe OBJ parsing and physical wedge frames come from mesh-geometry.js.
 * A scene may contain disconnected plates, solids and nonconvex components.
 * Invalid local UTD topology is reported; it never makes opaque triangles vanish.
 *
 * preparePLTScene(mesh | OBJ | indexedSource | [{mesh, materialIndex, surface}])
 * Scene geometry is immutable after preparation; rebuild to change a mesh.
 * All ray/cone distances are along a normalized world-space direction. Queries
 * honor only distance intervals, never an automatic previous-face exclusion.
 */
import {
  prepareMesh, extractWedges, mergeCollinearWedges,
  add, sub, scale, dot, cross, length, distance, normalize,
} from './mesh-geometry.js';
import {firstConeTriangleContact, clipEdgeToRegion} from './utd-scene.js';

export const PLT_GEOMETRY_LIMITS=Object.freeze({maxTriangles:48000,maxMeshes:256,leafSize:8});
const vec=(v,name)=>{
  if(!Array.isArray(v)||v.length!==3||v.some(x=>!Number.isFinite(x)||Math.abs(x)>1e9))
    throw new Error(`${name} must have three finite coordinates within ±1e9.`);
  return [...v];
};
const unit=(v,name)=>normalize(vec(v,name));
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
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
const index=(x,name)=>{
  if(!Number.isSafeInteger(x)||x<0)throw new Error(`${name} must be a nonnegative integer.`);
  return x;
};
const emptyBounds=()=>({min:[Infinity,Infinity,Infinity],max:[-Infinity,-Infinity,-Infinity]});
const expand=(b,p)=>{for(let k=0;k<3;k++){b.min[k]=Math.min(b.min[k],p[k]);b.max[k]=Math.max(b.max[k],p[k]);}return b;};
function finishBounds(b) {
  const center=scale(add(b.min,b.max),.5),size=sub(b.max,b.min);
  return {...b,center,size,radius:length(size)/2};
}
const triangleBounds=t=>finishBounds([t.a,t.b,t.c].reduce(expand,emptyBounds()));
const combineBounds=items=>{
  const b=emptyBounds();for(const item of items){expand(b,item.bounds.min);expand(b,item.bounds.max);}return finishBounds(b);
};

function prepared(input,options) {
  if(input?.authoredVertices&&input?.authoredFaces) {
    // Re-validate even prepared inputs, retaining their absolute world transform.
    return prepareMesh({vertices:input.authoredVertices,faces:input.authoredFaces,warnings:input.sourceWarnings},
      {...input.options,...input.transform,...options,rejectNonManifold:false});
  }
  return prepareMesh(input,{...options,rejectNonManifold:false});
}
function componentViews(mesh) {
  const adjacent=mesh.triangles.map(()=>[]);
  for(const e of mesh.edges)for(let i=1;i<e.triangleIds.length;i++) {
    adjacent[e.triangleIds[0]].push(e.triangleIds[i]);adjacent[e.triangleIds[i]].push(e.triangleIds[0]);
  }
  const seen=new Set(),out=[];
  for(const triangle of mesh.triangles) {
    if(seen.has(triangle.id))continue;
    const ids=[triangle.id];seen.add(triangle.id);
    for(let i=0;i<ids.length;i++)for(const id of adjacent[ids[i]])if(!seen.has(id)){seen.add(id);ids.push(id);}
    ids.sort((a,b)=>a-b);
    const map=new Map(ids.map((id,i)=>[id,i]));
    const triangles=ids.map((id,i)=>({...mesh.triangles[id],id:i}));
    const edges=mesh.edges.filter(e=>map.has(e.triangleIds[0])).map(e=>({...e,triangleIds:e.triangleIds.map(id=>map.get(id))}));
    const boundaryEdges=edges.filter(e=>e.kind==='boundary'),nonManifoldEdges=edges.filter(e=>e.kind==='nonmanifold');
    out.push({ids,mesh:{...mesh,triangles,edges,boundaryEdges,nonManifoldEdges,
      internalEdges:edges.filter(e=>e.kind==='internal'),creaseEdges:edges.filter(e=>e.kind==='crease'),
      diffractionEdges:edges.filter(e=>e.kind==='boundary'||e.kind==='crease'),
      windingConflicts:edges.filter(e=>!e.orientationConsistent).length,
      isClosed:!boundaryEdges.length&&!nonManifoldEdges.length}});
  }
  return out;
}
function singleBoundaryView(mesh,edge) {
  const triangle=mesh.triangles[edge.triangleIds[0]];
  const edges=mesh.edges.filter(e=>e.triangleIds.includes(triangle.id)).map(e=>({...e,triangleIds:[0],valence:1,kind:'boundary',orientationConsistent:true}));
  return {...mesh,triangles:[{...triangle,id:0}],edges,boundaryEdges:edges,internalEdges:[],creaseEdges:[],nonManifoldEdges:[],
    diffractionEdges:edges.filter(e=>e.id===edge.id),windingConflicts:0,isClosed:false};
}
function localWedges(mesh,surface,meshId,triangleOffset,edgeOffset,componentOffset) {
  const wedges=[],unsupportedEdges=[],components=[],warnings=[];
  const report=(edge,code,message,componentId)=>unsupportedEdges.push({code,message,meshId,componentId,
    edgeId:edgeOffset+edge.id,triangleIds:edge.triangleIds.map(id=>triangleOffset+id),a:[...edge.a],b:[...edge.b]});
  for(const [localComponentId,view] of componentViews(mesh).entries()) {
    const componentId=componentOffset+localComponentId,local=view.mesh;
    const mode=surface==='auto'?(local.isClosed?'closed-solid':'open-plate'):surface;
    let extraction=extractWedges(local,{surface:mode,mergeCollinear:false});
    const addExtracted=(w,ids=view.ids)=>{
      const triangleIds=w.triangleIds.map(id=>triangleOffset+ids[id]);
      const mapped={...w,id:edgeOffset+w.id,edgeId:edgeOffset+w.edgeId,meshId,componentId,
        segmentIds:w.segmentIds.map(id=>edgeOffset+id),vertexIds:w.vertexIds.map(id=>`${meshId}:${id}`),
        triangleIds,faceTriangleIds:w.faceTriangleIds.map(face=>face.map(id=>triangleOffset+ids[id])),
        e:normalize(sub(w.b,w.a)),n:2-w.alpha/Math.PI,utdSupported:true};
      wedges.push(mapped);
    };
    components.push({id:componentId,meshId,surface:mode,closed:local.isClosed,
      triangleIds:view.ids.map(id=>triangleOffset+id),utdSupported:extraction.supported,
      reasons:extraction.unsupported.map(e=>({code:e.code,message:e.message}))});
    warnings.push(...extraction.warnings);
    if(extraction.supported){extraction.wedges.forEach(w=>addExtracted(w));continue;}
    // Concavity is a property of one edge, not grounds to disable another solid
    // or a convex edge elsewhere on this same nonconvex solid.
    const edgeOnly=extraction.unsupported.every(p=>p.edgeId!==undefined);
    if(mode==='closed-solid'&&edgeOnly) {
      const bad=new Set(extraction.unsupported.map(p=>p.edgeId));
      const filtered={...local,diffractionEdges:local.diffractionEdges.filter(e=>!bad.has(e.id))};
      const recovered=extractWedges(filtered,{surface:mode,mergeCollinear:false});
      if(recovered.supported)recovered.wedges.forEach(w=>addExtracted(w));
      for(const problem of extraction.unsupported) {
        const e=mesh.edges.find(e=>e.id===problem.edgeId);
        report(e,problem.code,problem.message,componentId);
      }
      continue;
    }
    // A genuine free boundary has a local two-sided plate wedge even when a
    // remote piece of an open shell folds. Interior open-shell creases have no
    // declared solid side, and are explicitly excluded rather than flipped.
    if(surface!=='closed-solid'&&!local.isClosed) {
      for(const edge of local.boundaryEdges) {
        const single=singleBoundaryView(local,edge),result=extractWedges(single,{surface:'open-plate',mergeCollinear:false});
        result.wedges.forEach(w=>addExtracted(w,[view.ids[edge.triangleIds[0]]]));
      }
      for(const e of local.edges.filter(e=>e.kind==='crease'||e.kind==='nonmanifold'||!e.orientationConsistent)) {
        const original=mesh.edges.find(o=>o.id===e.id);
        const code=e.kind==='nonmanifold'?'nonmanifold':!e.orientationConsistent?'inconsistent-winding':'ambiguous-open-crease';
        report(original,code,'The local faces do not define an unambiguous consistently oriented opaque solid wedge. Triangles remain opaque.',componentId);
      }
    } else {
      const reason=extraction.unsupported[0];
      for(const e of local.edges.filter(e=>e.kind!=='internal'||!e.orientationConsistent))
        report(mesh.edges.find(o=>o.id===e.id),reason.code,reason.message,componentId);
    }
  }
  return {wedges:mergeCollinearWedges(wedges),unsupportedEdges,components,warnings};
}

/** Build one immutable, material-indexed scene. OBJ directives never fetch. */
export function preparePLTScene(input,options={}) {
  const entries=Array.isArray(input)?input:[input];
  if(!entries.length||entries.length>PLT_GEOMETRY_LIMITS.maxMeshes)throw new Error('Scene requires 1–256 mesh objects.');
  const leafSize=options.leafSize??PLT_GEOMETRY_LIMITS.leafSize;
  if(!Number.isInteger(leafSize)||leafSize<1||leafSize>64)throw new Error('BVH leafSize must be an integer from 1 to 64.');
  const triangles=[],edges=[],wedges=[],unsupportedEdges=[],components=[],meshes=[],warnings=[],duplicateKeys=new Set();
  for(const [meshId,entry] of entries.entries()) {
    const descriptor=entry?.mesh!==undefined?entry:{mesh:entry};
    const surface=descriptor.surface??options.surface??'auto';
    if(!['auto','open-plate','closed-solid'].includes(surface))throw new Error('Surface must be auto, open-plate, or closed-solid.');
    const mesh=prepared(descriptor.mesh,{...(options.meshOptions??{}),...(descriptor.meshOptions??{})});
    const materialIndex=index(descriptor.materialIndex??options.materialIndex??0,'Material index');
    const materialIndices=descriptor.materialIndices??descriptor.mesh?.materialIndices??options.materialIndices;
    if(materialIndices!==undefined&&(!Array.isArray(materialIndices)||materialIndices.length!==mesh.triangles.length))
      throw new Error('materialIndices must have one index per mesh triangle.');
    const triangleOffset=triangles.length,edgeOffset=edges.length;
    if(triangleOffset+mesh.triangles.length>PLT_GEOMETRY_LIMITS.maxTriangles)throw new Error('Scene exceeds the 48000 triangle limit.');
    for(const triangle of mesh.triangles) {
      const key=[triangle.a,triangle.b,triangle.c].map(p=>p.join(',')).sort().join(';');
      if(duplicateKeys.has(key))throw new Error('Scene has coincident duplicate triangle faces, which would double-count a surface.');
      duplicateKeys.add(key);
      const mi=index(materialIndices?.[triangle.id]??descriptor.mesh?.triangles?.[triangle.id]?.materialIndex??materialIndex,'Triangle material index');
      if(options.materials&&mi>=options.materials.length)throw new Error(`Triangle material index ${mi} is outside the material array.`);
      triangles.push({...triangle,id:triangleOffset+triangle.id,localTriangleId:triangle.id,meshId,materialIndex:mi,bounds:triangleBounds(triangle)});
    }
    for(const edge of mesh.edges)edges.push({...edge,id:edgeOffset+edge.id,localEdgeId:edge.id,meshId,
      triangleIds:edge.triangleIds.map(id=>triangleOffset+id)});
    const extracted=localWedges(mesh,surface,meshId,triangleOffset,edgeOffset,components.length);
    wedges.push(...extracted.wedges);unsupportedEdges.push(...extracted.unsupportedEdges);components.push(...extracted.components);
    warnings.push(...mesh.warnings,...extracted.warnings);
    meshes.push({id:meshId,mesh,materialIndex,surface,triangleOffset,edgeOffset});
  }
  const bounds=combineBounds(triangles),bvh=buildBVH(triangles,leafSize);
  const scene={type:'plt-triangle-scene',triangles,edges,wedges,unsupportedEdges,components,meshes,bounds,bvh,
    materials:options.materials??[],warnings:[...new Set(warnings)],supported:true,
    counts:{triangles:triangles.length,edges:edges.length,wedges:wedges.length,unsupportedEdges:unsupportedEdges.length,
      components:components.length,bvhNodes:bvh.nodes.length}};
  scene.facets=buildFacets(scene,triangles);
  return scene;
}
export const prepareScene=preparePLTScene;

function buildBVH(triangles,leafSize) {
  const nodes=[];
  function build(ids) {
    const node={...combineBounds(ids.map(id=>triangles[id])),left:-1,right:-1,triangleIds:null},id=nodes.length;
    nodes.push(node);
    if(ids.length<=leafSize){node.triangleIds=ids;return id;}
    const cb=ids.reduce((b,i)=>expand(b,triangles[i].centroid),emptyBounds()),span=sub(cb.max,cb.min);
    const axis=span.indexOf(Math.max(...span));
    ids.sort((a,b)=>triangles[a].centroid[axis]-triangles[b].centroid[axis]||a-b);
    const half=ids.length>>1;node.left=build(ids.slice(0,half));node.right=build(ids.slice(half));return id;
  }
  const root=build(triangles.map(t=>t.id));return {root,nodes,leafSize};
}
function checkedInterval(options={}) {
  const minDistance=options.minDistance??options.near??1e-9,maxDistance=options.maxDistance??options.far??Infinity;
  if(!Number.isFinite(minDistance)||minDistance<0||(!Number.isFinite(maxDistance)&&maxDistance!==Infinity)||maxDistance<minDistance)
    throw new Error('Distance interval requires 0 <= minDistance <= maxDistance.');
  return {minDistance,maxDistance};
}
function checkedRay(origin,direction,options) {
  return {origin:vec(origin,'Ray origin'),direction:unit(direction,'Ray direction'),...checkedInterval(options),cullBackface:options?.cullBackface===true};
}
function boxRay(node,ray,far) {
  let lo=ray.minDistance,hi=Math.min(ray.maxDistance,far);
  const padding=epsilonAt(node.min,node.max,ray.origin);
  for(let i=0;i<3;i++) {
    const a=node.min[i]-padding,b=node.max[i]+padding,o=ray.origin[i],d=ray.direction[i];
    if(d===0){if(o<a||o>b)return null;continue;}
    let t0=(a-o)/d,t1=(b-o)/d;if(t0>t1)[t0,t1]=[t1,t0];
    lo=Math.max(lo,t0);hi=Math.min(hi,t1);if(lo>hi)return null;
  }
  return lo;
}
function triangleHit(t,r,far) {
  const e1=sub(t.b,t.a),e2=sub(t.c,t.a),p=cross(r.direction,e2),det=dot(e1,p);
  const threshold=Math.max(Number.MIN_VALUE,length(e1)*length(e2)*1e-13);
  if(r.cullBackface?det<=threshold:Math.abs(det)<=threshold)return null;
  const s=sub(r.origin,t.a),u=dot(s,p)/det;if(u< -1e-12||u>1+1e-12)return null;
  const q=cross(s,e1),v=dot(r.direction,q)/det;if(v< -1e-12||u+v>1+1e-12)return null;
  const d=dot(e2,q)/det;if(d<r.minDistance||d>Math.min(far,r.maxDistance))return null;
  const frontFace=dot(r.direction,t.normal)<0;
  return {distance:d,point:add(r.origin,scale(r.direction,d)),barycentric:[1-u-v,u,v],triangleId:t.id,
    triangle:t,meshId:t.meshId,materialIndex:t.materialIndex,geometricNormal:[...t.normal],
    normal:scale(t.normal,frontFace?1:-1),frontFace};
}
function rayQuery(scene,origin,direction,options,any) {
  const ray=checkedRay(origin,direction,options),stack=[scene.bvh.root];let best=null;
  while(stack.length) {
    const n=scene.bvh.nodes[stack.pop()];if(options?.stats)options.stats.nodes=(options.stats.nodes??0)+1;
    if(boxRay(n,ray,best?.distance??Infinity)===null)continue;
    if(n.triangleIds) {
      for(const id of n.triangleIds) {
        if(options?.stats)options.stats.triangles=(options.stats.triangles??0)+1;
        const hit=triangleHit(scene.triangles[id],ray,best?.distance??Infinity);
        if(hit&&(best===null||hit.distance<best.distance||(hit.distance===best.distance&&hit.triangleId<best.triangleId))) {
          if(any)return hit;best=hit;
        }
      }
    } else {
      const l=boxRay(scene.bvh.nodes[n.left],ray,best?.distance??Infinity),r=boxRay(scene.bvh.nodes[n.right],ray,best?.distance??Infinity);
      if(l!==null&&r!==null){if(l<r)stack.push(n.right,n.left);else stack.push(n.left,n.right);}
      else if(l!==null)stack.push(n.left);else if(r!==null)stack.push(n.right);
    }
  }
  return best;
}
export function rayNearest(scene,origin,direction,options={}) {return rayQuery(scene,origin,direction,options,false);}
export const intersectScene=rayNearest;
export function rayOccluded(scene,origin,direction,options={}) {return rayQuery(scene,origin,direction,options,true)!==null;}
/** Endpoint exclusion only; triangles incident on an endpoint remain occluders
 * everywhere else. A previous interaction's whole face is never skipped. */
export function segmentVisible(scene,source,destination,options={}) {
  const a=vec(source,'Visibility source'),b=vec(destination,'Visibility destination'),delta=sub(b,a),d=length(delta);
  const requested=options.endpointEpsilon??epsilonAt(a,b);
  if(!Number.isFinite(requested)||requested<0)throw new Error('endpointEpsilon must be finite and nonnegative.');
  if(d===0)return true;
  const e=Math.min(requested,d/4);
  return !rayOccluded(scene,a,delta,{minDistance:e,maxDistance:d-e});
}
/** Brute-force reference with exactly the same physical ray convention. */
export function rayNearestBrute(scene,origin,direction,options={}) {
  const ray=checkedRay(origin,direction,options);let best=null;
  for(const t of scene.triangles){const h=triangleHit(t,ray,best?.distance??Infinity);if(h&&(!best||h.distance<best.distance||h.distance===best.distance&&h.triangleId<best.triangleId))best=h;}
  return best;
}

function checkedCone(cone) {
  if(!cone)throw new Error('Cone is required.');
  const origin=vec(cone.origin,'Cone origin'),direction=unit(cone.direction,'Cone direction');
  const {minDistance:near,maxDistance:far}=checkedInterval({minDistance:cone.near??0,maxDistance:cone.far??Infinity});
  const radius0=cone.radius0??0,tanAngle=cone.tanAngle??0;
  for(const [key,v] of Object.entries({radius0,tanAngle}))if(!Number.isFinite(v)||v<0||v>1e9)throw new Error(`Cone ${key} must be finite in [0, 1e9].`);
  if(cone.sigma0!==undefined||cone.slope!==undefined)throw new Error('Use an explicit linear cone radius0/tanAngle for exact PLT contacts.');
  return {...cone,origin,direction,near,far,radius0,tanAngle};
}
/** Test an actual point against the finite linear cone. */
export function coneContains(cone,point,tolerance=1e-10) {
  const c=checkedCone(cone),p=vec(point,'Cone point');
  if(!Number.isFinite(tolerance)||tolerance<0)throw new Error('Cone tolerance must be finite and nonnegative.');
  const q=sub(p,c.origin),t=dot(q,c.direction);
  return t>=c.near-tolerance&&t<=c.far+tolerance&&length(sub(q,scale(c.direction,t)))<=c.radius0+c.tanAngle*t+tolerance;
}
function sphereCone(center,radius,cone) {
  const delta=sub(center,cone.origin),axial=dot(delta,cone.direction),e=epsilonAt(center,cone.origin);
  if(axial+radius<cone.near-e||axial-radius>cone.far+e)return false;
  const upper=Math.max(0,Math.min(cone.far,axial+radius+e));
  const radial=length(sub(delta,scale(cone.direction,axial)));
  return radial<=radius+cone.radius0+cone.tanAngle*upper+e;
}
function coneCandidates(scene,cone,options) {
  const result=[],stack=[scene.bvh.root];
  while(stack.length) {
    const n=scene.bvh.nodes[stack.pop()];if(options?.stats)options.stats.nodes=(options.stats.nodes??0)+1;
    if(!sphereCone(n.center,n.radius,cone))continue;
    if(n.triangleIds)for(const id of n.triangleIds){const t=scene.triangles[id];if(sphereCone(t.centroid,t.radius,cone))result.push(t);}
    else stack.push(n.right,n.left);
  }
  return result;
}
/** Conservative BVH broad phase; candidates are not physical contacts. */
export function gatherConeCandidates(scene,cone,options={}) {return coneCandidates(scene,checkedCone(cone),options);}
/** Every returned contact has passed the actual triangle/cone narrow phase. */
export function traceConeContacts(scene,cone,options={}) {
  const c=checkedCone(cone),contacts=[];
  for(const triangle of coneCandidates(scene,c,options)) {
    if(options.stats)options.stats.triangles=(options.stats.triangles??0)+1;
    const h=firstConeTriangleContact(triangle,c);
    if(h)contacts.push({...h,triangle,meshId:triangle.meshId,materialIndex:triangle.materialIndex});
  }
  return contacts.sort((a,b)=>a.distance-b.distance||a.triangleId-b.triangleId);
}
export function firstConeContact(scene,cone,options={}) {
  const c=checkedCone(cone),candidates=coneCandidates(scene,c,options).map(triangle=>({triangle,
    near:Math.max(c.near,Math.min(...[triangle.a,triangle.b,triangle.c].map(p=>dot(sub(p,c.origin),c.direction))))}));
  candidates.sort((a,b)=>a.near-b.near||a.triangle.id-b.triangle.id);
  let best=null;
  for(const {triangle,near} of candidates) {
    if(best&&near>best.distance+epsilonAt(triangle.a,c.origin))break;
    if(options.stats)options.stats.triangles=(options.stats.triangles??0)+1;
    const h=firstConeTriangleContact(triangle,{...c,far:Math.min(c.far,best?.distance??Infinity)});
    if(h&&(!best||h.distance<best.distance||h.distance===best.distance&&h.triangleId<best.triangleId))
      best={...h,triangle,meshId:triangle.meshId,materialIndex:triangle.materialIndex};
  }
  return best;
}

function checkedRegion(region) {
  if(!region)throw new Error('Interaction region is required.');
  const center=vec(region.center,'Region center'),direction=unit(region.direction,'Region direction'),radius=region.radius;
  const depth=region.depth??Infinity;
  if(!Number.isFinite(radius)||radius<=0||radius>1e9)throw new Error('Region radius must be finite and positive, <= 1e9.');
  if((!Number.isFinite(depth)&&depth!==Infinity)||depth<0)throw new Error('Region depth must be nonnegative.');
  return {...region,center,direction,radius,depth};
}
const metric=(p,r)=>{const q=sub(p,r.center),z=dot(q,r.direction);return scale(sub(q,scale(r.direction,.5*z)),1/r.radius);};
const unmetric=(p,r)=>add(r.center,scale(add(p,scale(r.direction,dot(p,r.direction))),r.radius));
export function regionContainsPoint(region,point,tolerance=1e-10) {
  const r=checkedRegion(region),p=vec(point,'Region point'),z=dot(sub(p,r.center),r.direction);
  return dot(metric(p,r),metric(p,r))<=1+tolerance&&(!Number.isFinite(r.depth)||z>=-tolerance&&z<=r.depth+tolerance);
}
function slabClip(polygon,r) {
  if(!Number.isFinite(r.depth))return polygon;
  let out=polygon;
  for(const [sign,bound] of [[1,0],[-1,-r.depth]]) {
    const source=out;out=[];
    if(!source.length)break;
    for(let i=0;i<source.length;i++) {
      const a=source[i],b=source[(i+1)%source.length],eps=epsilonAt(a,b,r.center);
      let fa=sign*dot(sub(a,r.center),r.direction)-bound,fb=sign*dot(sub(b,r.center),r.direction)-bound;
      if(Math.abs(fa)<=eps)fa=0;if(Math.abs(fb)<=eps)fb=0;
      if(fa>=0)out.push(a);
      if((fa<0&&fb>0)||(fa>0&&fb<0))out.push(add(a,scale(sub(b,a),fa/(fa-fb))));
    }
  }
  return out.filter((p,i)=>!out.slice(0,i).some(q=>distance(p,q)<=epsilonAt(p,q)));
}
function closestSegment(a,b) {
  const d=sub(b,a),ll=dot(d,d);return add(a,scale(d,ll?clamp(-dot(a,d)/ll,0,1):0));
}
function closestToOrigin(a,b,c) {
  // Projection onto a triangle, including its edge/vertex Voronoi regions.
  const ab=sub(b,a),ac=sub(c,a),area=cross(ab,ac);
  if(dot(area,area)<=1e-28*Math.max(dot(ab,ab)*dot(ac,ac),Number.MIN_VALUE)) {
    return [closestSegment(a,b),closestSegment(b,c),closestSegment(c,a)].sort((x,y)=>dot(x,x)-dot(y,y))[0];
  }
  const ap=scale(a,-1),d1=dot(ab,ap),d2=dot(ac,ap);
  if(d1<=0&&d2<=0)return a;
  const bp=scale(b,-1),d3=dot(ab,bp),d4=dot(ac,bp);if(d3>=0&&d4<=d3)return b;
  const vc=d1*d4-d3*d2;if(vc<=0&&d1>=0&&d3<=0)return add(a,scale(ab,d1/(d1-d3)));
  const cp=scale(c,-1),d5=dot(ab,cp),d6=dot(ac,cp);if(d6>=0&&d5<=d6)return c;
  const vb=d5*d2-d1*d6;if(vb<=0&&d2>=0&&d6<=0)return add(a,scale(ac,d2/(d2-d6)));
  const va=d3*d6-d5*d4;if(va<=0&&d4-d3>=0&&d5-d6>=0)return add(b,scale(sub(c,b),(d4-d3)/((d4-d3)+(d5-d6))));
  const denominator=va+vb+vc;
  if(Math.abs(denominator)<Number.MIN_VALUE)return a;
  return add(a,add(scale(ab,vb/denominator),scale(ac,vc/denominator)));
}
function circleEdgeParameters(a,b) {
  const d=sub(b,a),A=dot(d,d),B=2*dot(a,d),C=dot(a,a)-1;
  if(A===0)return [];
  let disc=B*B-4*A*C;if(disc<0&&disc> -1e-13*(B*B+Math.abs(4*A*C)))disc=0;
  if(disc<0)return [];
  const q=Math.sqrt(disc);return [(-B-q)/(2*A),(-B+q)/(2*A)].filter(t=>t>=0&&t<=1);
}
/** Exact intersection descriptor for a true triangle and the region ellipsoid.
 * The curved boundary is a unit-sphere disk clipped by the actual slab polygon
 * in metric coordinates. `polygon` is slab-clipped only, NOT a polygonal
 * approximation to the ellipsoid. `boundaryPoints` are exact edge contacts.
 */
export function clipTriangleToRegion(triangle,region) {
  const r=checkedRegion(region),polygon=slabClip([triangle.a,triangle.b,triangle.c],r);
  if(!polygon.length)return null;
  const points=polygon.map(p=>metric(p,r));
  let nearest=points.length===1?points[0]:closestSegment(points[0],points[1]),dist2=dot(nearest,nearest);
  for(let i=1;i+1<points.length;i++) {
    const q=closestToOrigin(points[0],points[i],points[i+1]),d2=dot(q,q);
    if(d2<dist2){nearest=q;dist2=d2;}
  }
  if(dist2>1+1e-10)return null;
  const normal=normalize(cross(sub(metric(triangle.b,r),metric(triangle.a,r)),sub(metric(triangle.c,r),metric(triangle.a,r))));
  const offset=dot(normal,points[0]),circleCenter=scale(normal,offset),circleRadius=Math.sqrt(Math.max(0,1-offset*offset));
  const boundaryPoints=[];
  const append=p=>{if(!boundaryPoints.some(q=>distance(q,p)<epsilonAt(p,q)))boundaryPoints.push(p);};
  for(let i=0;i<points.length;i++) {
    const a=points[i],b=points[(i+1)%points.length];
    if(dot(a,a)<=1+1e-12)append(polygon[i]);
    for(const t of circleEdgeParameters(a,b))append(unmetric(add(a,scale(sub(b,a),t)),r));
  }
  return {triangleId:triangle.id,triangle,polygon,metricPolygon:points,
    metricDisk:{center:circleCenter,normal,radius:circleRadius},
    boundaryPoints,closestPoint:unmetric(nearest,r),clipped:polygon.length!==3||points.some(p=>dot(p,p)>1+1e-12),
    contains:p=>regionContainsPoint(r,p)&&pointOnTriangle(triangle,p)};
}
function pointOnTriangle(t,p) {
  const eps=epsilonAt(t.a,t.b,t.c,p);
  if(Math.abs(dot(sub(p,t.a),t.normal))>eps)return false;
  return [t.a,t.b,t.c].every((a,i,points)=>dot(cross(sub(points[(i+1)%3],a),sub(p,a)),t.normal)>=-eps*Math.max(1,distance(points[(i+1)%3],a)));
}
// Prepared scene geometry is immutable. Cache only its eligible adjacency,
// never a region's selected subset, clipped surfaces or connected components.
// A WeakMap keeps this private and lets discarded scenes be collected.
const facetAdjacencyCache=new WeakMap();
function facetAdjacency(scene) {
  let adjacency=facetAdjacencyCache.get(scene);
  if(adjacency)return adjacency;
  adjacency=scene.triangles.map(()=>[]);
  for(const edge of scene.edges) {
    if(edge.kind!=='internal'||edge.valence!==2)continue;
    const [a,b]=edge.triangleIds,ta=scene.triangles[a],tb=scene.triangles[b];
    if(ta.materialIndex!==tb.materialIndex||dot(ta.normal,tb.normal)<1-1e-10)continue;
    const eps=epsilonAt(ta.a,ta.b,ta.c,tb.a,tb.b,tb.c);
    if([tb.a,tb.b,tb.c].some(p=>Math.abs(dot(sub(p,ta.a),ta.normal))>eps))continue;
    adjacency[a].push(b);adjacency[b].push(a);
  }
  facetAdjacencyCache.set(scene,adjacency);
  return adjacency;
}
function buildFacets(scene,triangles,clips=new Map()) {
  const selected=new Set(triangles.map(t=>t.id)),adjacency=facetAdjacency(scene),seen=new Set(),facets=[];
  for(const t of triangles) {
    if(seen.has(t.id))continue;
    const ids=[t.id];seen.add(t.id);
    for(let i=0;i<ids.length;i++)for(const other of adjacency[ids[i]])
      if(selected.has(other)&&!seen.has(other)){seen.add(other);ids.push(other);}
    facets.push({id:facets.length,triangleIds:ids,triangles:ids.map(id=>scene.triangles[id]),
      clippedTriangles:ids.map(id=>clips.get(id)).filter(Boolean),normal:[...t.normal],point:[...t.a],materialIndex:t.materialIndex,meshId:t.meshId});
  }
  return facets;
}
/** Discover every disconnected physical patch within one finite region. */
export function collectRegionGeometry(scene,region) {
  const r=checkedRegion(region),triangles=[],clippedTriangles=[],stack=[scene.bvh.root];
  while(stack.length) {
    const node=scene.bvh.nodes[stack.pop()];
    // A world-space sphere of radius 2r encloses the ellipsoid.
    if(distance(node.center,r.center)>node.radius+2*r.radius+epsilonAt(node.center,r.center))continue;
    if(node.triangleIds)for(const id of node.triangleIds) {
      const triangle=scene.triangles[id],clip=clipTriangleToRegion(triangle,r);
      if(clip){triangles.push(triangle);clippedTriangles.push(clip);}
    }
    else stack.push(node.right,node.left);
  }
  triangles.sort((a,b)=>a.id-b.id);clippedTriangles.sort((a,b)=>a.triangleId-b.triangleId);
  const clips=new Map(clippedTriangles.map(c=>[c.triangleId,c]));
  const wedges=scene.wedges.map(w=>clipEdgeToRegion(w,r)).filter(Boolean);
  const unsupportedEdges=scene.unsupportedEdges.filter(e=>clipEdgeToRegion({...e,l:distance(e.a,e.b)},r));
  return {region:r,triangles,triangleIds:triangles.map(t=>t.id),clippedTriangles,wedges,
    facets:buildFacets(scene,triangles,clips),unsupportedEdges,
    counts:{triangles:triangles.length,wedges:wedges.length,unsupportedEdges:unsupportedEdges.length}};
}
