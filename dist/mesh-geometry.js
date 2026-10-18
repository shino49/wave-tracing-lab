/**
 * Browser-safe, material-neutral geometry for finite triangle meshes.
 * No OBJ directive can cause a fetch, material load, or script execution.
 *
 * parseOBJ(text) -> { vertices, faces, warnings, source: 'obj' }.
 * prepareMesh(textOrParsed, options) -> a mesh with world-space triangle records.
 * Options: unitScale (authored units -> mm), rotationDeg [x,y,z] (Rz Ry Rx),
 * translation [x,y,z] (mm), weldTolerance (authored units, default exact),
 * creaseAngleDeg, rejectNonManifold. Authored positions are never recentered.
 * transformMesh(mesh, options) reapplies an absolute transform to authored data.
 *
 * Each triangle: {id,indices,weldedIndices,a,b,c,normal,area,centroid,radius}.
 * Each edge: {indices,a,b,length,triangleIds,valence,kind}. Indices reference
 * transformed vertices. kind is boundary, internal, crease, or nonmanifold.
 * Only boundary/crease edges are in diffractionEdges. Coplanar triangulation
 * edges are never diffraction edges. Quantized welding affects connectivity,
 * not authored/world positions. Normals follow OBJ winding; inconsistent winding
 * is reported. Open patches may be two-sided in the downstream physical model.
 *
 * gatherConeTriangles(mesh, cone) conservatively gathers triangle records for
 * an explicitly tail-truncated Gaussian cone, using enclosing spheres, not
 * triangle centroids. False positives are intentional; infinite Gaussian tails
 * are not claimed to have finite support. Cone origin/direction are world-space.
 * Cone radius: tail*hypot(sigma0,slope*t), or radius0+tanAngle*t. near >= 0;
 * far defaults to Infinity. Rays use physical distance along normalized direction.
 */

export const MESH_LIMITS = Object.freeze({
  maxBytes: 2 * 1024 * 1024,
  maxVertices: 8000,
  maxTriangles: 12000,
  maxAttributes: 48000,
  maxFaceVertices: 64,
  maxCoordinate: 1e6,
  maxWorldCoordinate: 1e9,
  minUnitScale: 1e-6,
  maxUnitScale: 1e6,
  maxTranslation: 1e6,
  maxRotationDeg: 36000,
});
export const add = (a,b) => [a[0]+b[0],a[1]+b[1],a[2]+b[2]];
export const sub = (a,b) => [a[0]-b[0],a[1]-b[1],a[2]-b[2]];
export const scale = (a,s) => [a[0]*s,a[1]*s,a[2]*s];
export const dot = (a,b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
export const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
export const length = a => Math.hypot(a[0],a[1],a[2]);
export const distance = (a,b) => length(sub(a,b));
export function normalize(a) {
  const n=length(a);
  if(!Number.isFinite(n)||n===0) throw new Error('Direction must be finite and nonzero.');
  return scale(a,1/n);
}
const fail = (message,line) => { throw new Error(`${line?`OBJ line ${line}: `:''}${message}`); };
const validVec = (v,label,max=Infinity) => {
  if(!Array.isArray(v)||v.length!==3||v.some(x=>!Number.isFinite(x)||Math.abs(x)>max))
    throw new Error(`${label} must contain three finite numbers within ±${max}.`);
  return [...v];
};
const numberToken = (s,line) => {
  // Reject JavaScript-specific numeric syntax, Infinity, NaN, and empty tokens.
  if(!/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(s)) fail(`Invalid numeric value '${s}'.`,line);
  const x=Number(s);
  if(!Number.isFinite(x)||Math.abs(x)>MESH_LIMITS.maxCoordinate) fail('Numeric value is outside the finite coordinate limit.',line);
  return x;
};
function objIndex(s,count,kind,line) {
  if(!/^[+-]?\d+$/.test(s)) fail(`Invalid ${kind} index.`,line);
  const n=Number(s), i=n>0?n-1:count+n;
  if(!Number.isSafeInteger(n)||n===0||i<0||i>=count) fail(`${kind} index ${s} is out of range. Forward references are unsupported.`,line);
  return i;
}
function triangleData(a,b,c) {
  const ab=sub(b,a), ac=sub(c,a), cr=cross(ab,ac), doubleArea=length(cr);
  const longest=Math.max(length(ab),length(ac),distance(b,c));
  if(!Number.isFinite(doubleArea)||longest===0||doubleArea<=longest*longest*1e-12)
    throw new Error('Degenerate or numerically unresolved triangle.');
  const centroid=scale(add(add(a,b),c),1/3);
  // Outward roundoff padding keeps the sphere conservative at triangle vertices.
  const radius=Math.max(distance(a,centroid),distance(b,centroid),distance(c,centroid))*(1+16*Number.EPSILON);
  return {a,b,c,area:doubleArea/2,normal:scale(cr,1/doubleArea),centroid,radius};
}
export function triangleArea(a,b,c) { return length(cross(sub(b,a),sub(c,a)))/2; }
export function triangleNormal(a,b,c) { return normalize(cross(sub(b,a),sub(c,a))); }

function triangulateFace(face,vertices,line) {
  const p=face.map(i=>vertices[i]);
  if(new Set(face).size!==face.length) fail('A face repeats a vertex index.',line);
  if(face.length===3) {
    try { triangleData(...p); } catch(error) { fail(error.message,line); }
    return [[...face]];
  }
  // Validate a simple, strictly convex, planar polygon before fan triangulation.
  // Concave, self-crossing, nonplanar, and collinear ambiguous n-gons are rejected.
  const n=[0,0,0];
  for(let i=0;i<p.length;i++) {
    const a=sub(p[i],p[0]),b=sub(p[(i+1)%p.length],p[0]),cr=cross(a,b);
    // Local offsets avoid cancellation when tiny patches have large authored
    // coordinates. A translation must not change n-gon acceptance.
    n[0]+=cr[0];n[1]+=cr[1];n[2]+=cr[2];
  }
  const span=Math.max(...p.map(v=>distance(v,p[0])));
  if(span===0||length(n)<=span*span*1e-12) fail('Ambiguous or degenerate polygon; supply triangles.',line);
  const normal=normalize(n), tolerance=span*1e-8;
  if(p.some(v=>Math.abs(dot(sub(v,p[0]),normal))>tolerance)) fail('Nonplanar polygon; supply explicit triangles.',line);
  const axis=Math.abs(normal[0])>Math.abs(normal[1])?(Math.abs(normal[0])>Math.abs(normal[2])?0:2):(Math.abs(normal[1])>Math.abs(normal[2])?1:2);
  const q=p.map(v=>v.filter((_,i)=>i!==axis));
  const turn=(a,b,c)=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
  const epsilon=span*span*1e-12;
  let winding=0;
  for(let i=0;i<q.length;i++) {
    const value=turn(q[i],q[(i+1)%q.length],q[(i+2)%q.length]);
    if(Math.abs(value)<=epsilon) fail('Collinear polygon corners are ambiguous; supply triangles.',line);
    if(!winding) winding=Math.sign(value);
    if(Math.sign(value)!==winding) fail('Concave polygon is unsupported; supply triangles.',line);
  }
  // Test all non-neighbor edges. This also rejects star polygons whose local
  // turns all have the same sign, and repeated non-adjacent positions.
  const overlaps=(a,b,c,d)=>Math.max(Math.min(a,b),Math.min(c,d))<=Math.min(Math.max(a,b),Math.max(c,d))+tolerance;
  for(let i=0;i<q.length;i++)for(let j=i+1;j<q.length;j++) {
    if(j===i+1||(i===0&&j===q.length-1))continue;
    const a=q[i],b=q[(i+1)%q.length],c=q[j],d=q[(j+1)%q.length];
    const t1=turn(a,b,c),t2=turn(a,b,d),t3=turn(c,d,a),t4=turn(c,d,b);
    if(overlaps(a[0],b[0],c[0],d[0])&&overlaps(a[1],b[1],c[1],d[1])&&
      t1*t2<=epsilon*epsilon&&t3*t4<=epsilon*epsilon)
      fail('Self-intersecting polygon is unsupported; supply simple triangles.',line);
  }
  const result=[];
  for(let i=1;i<face.length-1;i++) {
    const indices=[face[0],face[i],face[i+1]];
    try { triangleData(...indices.map(j=>vertices[j])); } catch(error) { fail(error.message,line); }
    result.push(indices);
  }
  return result;
}

export function parseOBJ(text) {
  if(typeof text!=='string') throw new Error('OBJ input must be text.');
  if(text.length>MESH_LIMITS.maxBytes||new TextEncoder().encode(text).byteLength>MESH_LIMITS.maxBytes)
    throw new Error('OBJ file exceeds the 2 MiB input limit.');
  if(text.includes('\0')) throw new Error('OBJ input contains a NUL byte.');
  const vertices=[],faces=[],warnings=[];
  let textureCount=0,normalCount=0;
  const ignored=new Set();
  const lines=text.replace(/^\uFEFF/,'').split(/\r\n|\n|\r/);
  for(let lineIndex=0;lineIndex<lines.length;lineIndex++) {
    const line=lines[lineIndex].split('#',1)[0].trim();
    if(!line)continue;
    const [directive,...args]=line.split(/\s+/), lineNo=lineIndex+1;
    if(directive==='v') {
      if(args.length!==3&&args.length!==4) fail('Vertices require x y z and optional w=1; vertex colors are unsupported.',lineNo);
      const values=args.map(s=>numberToken(s,lineNo));
      if(values.length===4&&values[3]!==1) fail('Homogeneous vertex w must be 1; convert to Cartesian coordinates first.',lineNo);
      vertices.push(values.slice(0,3));
      if(vertices.length>MESH_LIMITS.maxVertices) fail(`Vertex limit is ${MESH_LIMITS.maxVertices}.`,lineNo);
    } else if(directive==='vt') {
      if(args.length<1||args.length>3) fail('Texture coordinates require one to three values.',lineNo);
      args.forEach(s=>numberToken(s,lineNo));
      if(++textureCount>MESH_LIMITS.maxAttributes) fail('Texture-coordinate limit exceeded.',lineNo);
    } else if(directive==='vn') {
      if(args.length!==3) fail('Normals require three values.',lineNo);
      const n=args.map(s=>numberToken(s,lineNo));
      if(length(n)===0) fail('OBJ normal must be nonzero.',lineNo);
      if(++normalCount>MESH_LIMITS.maxAttributes) fail('Normal limit exceeded.',lineNo);
    } else if(directive==='f') {
      if(args.length<3||args.length>MESH_LIMITS.maxFaceVertices) fail(`Faces require 3–${MESH_LIMITS.maxFaceVertices} corners.`,lineNo);
      if(faces.length+args.length-2>MESH_LIMITS.maxTriangles) fail(`Triangle limit is ${MESH_LIMITS.maxTriangles}.`,lineNo);
      const face=args.map(s=>{
        const indices=s.split('/');
        if(indices.length>3||!indices[0]||(indices.length===2&&!indices[1])||(indices.length===3&&!indices[2]))
          fail('Malformed face corner; use v, v/vt, v//vn, or v/vt/vn.',lineNo);
        const vertex=objIndex(indices[0],vertices.length,'Vertex',lineNo);
        if(indices[1])objIndex(indices[1],textureCount,'Texture',lineNo);
        if(indices[2])objIndex(indices[2],normalCount,'Normal',lineNo);
        return vertex;
      });
      faces.push(...triangulateFace(face,vertices,lineNo));
    } else if(['o','g','s','usemtl','mtllib'].includes(directive)) {
      ignored.add(directive);
    } else {
      // In particular, call/csh/include/free-form surfaces are never evaluated.
      fail(`Unsupported directive '${directive}'. Only vertex/face geometry is accepted.`,lineNo);
    }
  }
  if(!vertices.length||!faces.length) throw new Error('OBJ must contain vertices and at least one nondegenerate face.');
  if(ignored.has('mtllib')||ignored.has('usemtl')) warnings.push('OBJ materials are ignored. No external files are requested; triangles are geometry only, with materials chosen downstream.');
  if(textureCount||normalCount)warnings.push('Texture/normal indices were validated; geometry normals are computed from triangle winding.');
  return {vertices,faces,warnings,source:'obj'};
}

function checkedSource(input) {
  const parsed=typeof input==='string'?parseOBJ(input):input;
  if(!parsed||!Array.isArray(parsed.vertices)||!Array.isArray(parsed.faces))throw new Error('Mesh requires vertices and triangle faces.');
  if(!parsed.vertices.length||parsed.vertices.length>MESH_LIMITS.maxVertices||!parsed.faces.length||parsed.faces.length>MESH_LIMITS.maxTriangles)
    throw new Error('Mesh vertex/triangle counts exceed supported limits or are empty.');
  const vertices=parsed.vertices.map(v=>validVec(v,'Authored vertex',MESH_LIMITS.maxCoordinate));
  const faces=parsed.faces.map(face=>{
    if(!Array.isArray(face)||face.length!==3||face.some(i=>!Number.isInteger(i)||i<0||i>=vertices.length))
      throw new Error('Prepared mesh faces must be valid triangle index triples.');
    triangleData(...face.map(i=>vertices[i]));
    return [...face];
  });
  return {vertices,faces,warnings:Array.isArray(parsed.warnings)?parsed.warnings.filter(v=>typeof v==='string'):[],source:parsed.source||'triangles'};
}
function checkedTransform(options={}) {
  const unitScale=options.unitScale??1;
  if(!Number.isFinite(unitScale)||unitScale<MESH_LIMITS.minUnitScale||unitScale>MESH_LIMITS.maxUnitScale)
    throw new Error(`unitScale must be in [${MESH_LIMITS.minUnitScale}, ${MESH_LIMITS.maxUnitScale}] mm per authored unit.`);
  const rotationDeg=validVec(options.rotationDeg??[0,0,0],'Rotation',MESH_LIMITS.maxRotationDeg);
  const translation=validVec(options.translation??[0,0,0],'Translation',MESH_LIMITS.maxTranslation);
  return {unitScale,rotationDeg,translation};
}
export function transformPoint(point,options={}) {
  const transform=checkedTransform(options);
  validVec(point,'Point',MESH_LIMITS.maxCoordinate);
  return applyTransform(point,transform);
}
function applyTransform(point,transform) {
  const [rx,ry,rz]=transform.rotationDeg.map(d=>d*Math.PI/180);
  const [cx,cy,cz]=[Math.cos(rx),Math.cos(ry),Math.cos(rz)], [sx,sy,sz]=[Math.sin(rx),Math.sin(ry),Math.sin(rz)];
  const [x,y,z]=scale(point,transform.unitScale);
  const y1=cx*y-sx*z,z1=sx*y+cx*z;
  const x2=cy*x+sy*z1,z2=-sy*x+cy*z1;
  const result=add([cz*x2-sz*y1,sz*x2+cz*y1,z2],transform.translation);
  return validVec(result,'Transformed point',MESH_LIMITS.maxWorldCoordinate);
}
function weldVertices(vertices,tolerance) {
  const ids=[],representatives=[],buckets=new Map();
  if(tolerance===0) {
    vertices.forEach((v,index)=>{
      const key=v.join(',');
      let id=buckets.get(key);
      if(id===undefined){id=representatives.length;representatives.push(index);buckets.set(key,id);}
      ids.push(id);
    });
  } else {
    vertices.forEach((v,index)=>{
      const cell=v.map(x=>Math.floor(x/tolerance));
      if(cell.some(x=>!Number.isSafeInteger(x))) throw new Error('Weld tolerance is too small for the coordinate magnitude; use exact welding (0).');
      let found=-1;
      for(let dx=-1;dx<=1;dx++)for(let dy=-1;dy<=1;dy++)for(let dz=-1;dz<=1;dz++) {
        const entries=buckets.get(`${cell[0]+dx},${cell[1]+dy},${cell[2]+dz}`)||[];
        for(const id of entries) if(distance(vertices[representatives[id]],v)<=tolerance&&(found===-1||id<found))found=id;
      }
      if(found===-1){found=representatives.length;representatives.push(index);const key=cell.join(',');if(!buckets.has(key))buckets.set(key,[]);buckets.get(key).push(found);}
      ids.push(found);
    });
  }
  return {ids,representatives};
}

export function prepareMesh(input,options={}) {
  const source=checkedSource(input), transform=checkedTransform(options);
  const weldTolerance=options.weldTolerance??0, creaseAngleDeg=options.creaseAngleDeg??0.01;
  if(!Number.isFinite(weldTolerance)||weldTolerance<0||weldTolerance>1)throw new Error('weldTolerance must be in [0, 1] authored units.');
  if(!Number.isFinite(creaseAngleDeg)||creaseAngleDeg<0||creaseAngleDeg>90)throw new Error('creaseAngleDeg must be in [0, 90].');
  const vertices=source.vertices.map(v=>applyTransform(v,transform));
  const welded=weldVertices(source.vertices,weldTolerance), warnings=[...source.warnings];
  const triangleKeys=new Set(), edgeMap=new Map();
  const triangles=source.faces.map((indices,id)=>{
    const weldedIndices=indices.map(i=>welded.ids[i]);
    if(new Set(weldedIndices).size!==3)throw new Error('Welding collapses a triangle; reduce weldTolerance.');
    const key=[...weldedIndices].sort((a,b)=>a-b).join(',');
    if(triangleKeys.has(key))throw new Error('Duplicate/coincident triangle faces are ambiguous and would double-count the surface.');
    triangleKeys.add(key);
    const t={id,indices:[...indices],weldedIndices,...triangleData(...indices.map(i=>vertices[i]))};
    for(let j=0;j<3;j++) {
      const ia=weldedIndices[j],ib=weldedIndices[(j+1)%3],pair=ia<ib?[ia,ib]:[ib,ia],edgeKey=pair.join(',');
      if(!edgeMap.has(edgeKey))edgeMap.set(edgeKey,{weldedIndices:pair,triangleIds:[],directions:[]});
      const e=edgeMap.get(edgeKey);e.triangleIds.push(id);e.directions.push(ia<ib?1:-1);
    }
    return t;
  });
  const boundaryEdges=[],internalEdges=[],creaseEdges=[],nonManifoldEdges=[],edges=[];
  let windingConflicts=0;
  for(const e of edgeMap.values()) {
    const indices=e.weldedIndices.map(i=>welded.representatives[i]);
    const a=vertices[indices[0]],b=vertices[indices[1]],valence=e.triangleIds.length;
    let kind;
    if(valence===1)kind='boundary';
    else if(valence===2) {
      const n0=triangles[e.triangleIds[0]].normal,n1=triangles[e.triangleIds[1]].normal;
      const cosine=Math.min(1,Math.abs(dot(n0,n1)));
      // A numerical tolerance avoids promoting same-plane tessellation to edges.
      kind=cosine+1e-12>=Math.cos(creaseAngleDeg*Math.PI/180)?'internal':'crease';
      if(e.directions[0]===e.directions[1])windingConflicts++;
    } else kind='nonmanifold';
    const edge={id:edges.length,indices,weldedIndices:[...e.weldedIndices],a,b,length:distance(a,b),triangleIds:[...e.triangleIds],valence,kind,orientationConsistent:valence!==2||e.directions[0]!==e.directions[1]};
    edges.push(edge);
    ({boundary:boundaryEdges,internal:internalEdges,crease:creaseEdges,nonmanifold:nonManifoldEdges})[kind].push(edge);
  }
  if(nonManifoldEdges.length) {
    if(options.rejectNonManifold!==false)throw new Error(`Mesh has ${nonManifoldEdges.length} non-manifold edges (more than two adjacent triangles).`);
    warnings.push(`${nonManifoldEdges.length} non-manifold edges are excluded from diffractionEdges; physical interpretation is ambiguous.`);
  }
  if(windingConflicts)warnings.push(`${windingConflicts} shared edges have inconsistent face winding; geometric normals retain authored orientation.`);
  const activeVertexIndices=[...new Set(source.faces.flat())],activeVertices=activeVertexIndices.map(i=>vertices[i]);
  const min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];
  for(const v of activeVertices)for(let i=0;i<3;i++){min[i]=Math.min(min[i],v[i]);max[i]=Math.max(max[i],v[i]);}
  const center=scale(add(min,max),.5),size=sub(max,min);
  return {
    source:source.source,authoredVertices:source.vertices.map(v=>[...v]),authoredFaces:source.faces.map(f=>[...f]),
    sourceWarnings:[...source.warnings],vertices,activeVertexIndices,triangles,edges,boundaryEdges,internalEdges,creaseEdges,nonManifoldEdges,
    diffractionEdges:[...boundaryEdges,...creaseEdges],weldedVertexCount:welded.representatives.length,
    bounds:{min,max,center,size,radius:Math.max(...activeVertices.map(v=>distance(v,center)))},
    totalArea:triangles.reduce((sum,t)=>sum+t.area,0),isClosed:boundaryEdges.length===0&&nonManifoldEdges.length===0,
    warnings,windingConflicts,transform,options:{weldTolerance,creaseAngleDeg,rejectNonManifold:options.rejectNonManifold!==false},
  };
}
export function transformMesh(mesh,options={}) {
  if(!mesh?.authoredVertices||!mesh?.authoredFaces)throw new Error('transformMesh requires a prepared mesh.');
  return prepareMesh({vertices:mesh.authoredVertices,faces:mesh.authoredFaces,warnings:mesh.sourceWarnings,source:mesh.source},
    {...mesh.options,...mesh.transform,...options});
}

function checkedCone(cone) {
  if(!cone)throw new Error('Cone specification is required.');
  const origin=validVec(cone.origin??[0,0,0],'Cone origin',MESH_LIMITS.maxWorldCoordinate);
  const direction=normalize(validVec(cone.direction??[0,0,1],'Cone direction'));
  const near=cone.near??0,far=cone.far??Infinity;
  if(!Number.isFinite(near)||near<0||(!Number.isFinite(far)&&far!==Infinity)||far<near)throw new Error('Cone requires 0 <= near <= far.');
  const linear=cone.radius0!==undefined||cone.tanAngle!==undefined;
  const sigma0=cone.sigma0??0,slope=cone.slope??0,tail=cone.tail??5,radius0=cone.radius0??0,tanAngle=cone.tanAngle??0;
  for(const [name,value] of Object.entries({sigma0,slope,tail,radius0,tanAngle}))
    if(!Number.isFinite(value)||value<0||value>1e9)throw new Error(`Cone ${name} must be finite in [0, 1e9].`);
  return {origin,direction,near,far,linear,sigma0,slope,tail,radius0,tanAngle};
}
const radiusAt=(t,c)=>c.linear?c.radius0+c.tanAngle*Math.max(0,t):c.tail*Math.hypot(c.sigma0,c.slope*Math.max(0,t));
export function coneRadius(t,cone) {
  if(!Number.isFinite(t)||t<0)throw new Error('Cone axial distance must be finite and nonnegative.');
  return radiusAt(t,checkedCone(cone));
}
function sphereCone(triangle,c) {
  const delta=sub(triangle.centroid,c.origin),t=dot(delta,c.direction),r=triangle.radius;
  // The projected sphere interval must intersect the finite axial interval.
  const tolerance=32*Number.EPSILON*Math.max(1,length(delta),r,c.near,Number.isFinite(c.far)?c.far:0);
  if(t+r<c.near-tolerance||t-r>c.far+tolerance)return false;
  const upper=Math.max(0,Math.min(c.far,t+r+tolerance));
  // radiusAt is monotone on t >= 0. Any point in the sphere is no farther
  // from the axis than centerRadial+r, and its cone radius is <= radiusAt(upper).
  const radial=length(sub(delta,scale(c.direction,t)));
  return radial<=r+radiusAt(upper,c)+tolerance;
}
export function triangleIntersectsCone(triangle,cone) { return sphereCone(triangle,checkedCone(cone)); }
export function gatherConeTriangles(mesh,cone) {
  if(!Array.isArray(mesh?.triangles))throw new Error('Cone gathering requires a prepared mesh.');
  const c=checkedCone(cone);
  return mesh.triangles.filter(t=>sphereCone(t,c));
}
function rayHit(origin,direction,triangle,minDistance,maxDistance,cullBackface) {
  const e1=sub(triangle.b,triangle.a),e2=sub(triangle.c,triangle.a),p=cross(direction,e2),det=dot(e1,p);
  const epsilon=Math.max(length(e1)*length(e2)*1e-13,Number.MIN_VALUE);
  if(cullBackface?det<=epsilon:Math.abs(det)<=epsilon)return null;
  const inv=1/det,s=sub(origin,triangle.a),u=dot(s,p)*inv;
  if(u< -1e-12||u>1+1e-12)return null;
  const q=cross(s,e1),v=dot(direction,q)*inv;
  if(v< -1e-12||u+v>1+1e-12)return null;
  const t=dot(e2,q)*inv;
  if(t<minDistance||t>maxDistance)return null;
  return {distance:t,barycentric:[1-u-v,u,v],point:add(origin,scale(direction,t)),triangleId:triangle.id};
}
function checkedRay(origin,direction,options={}) {
  origin=validVec(origin,'Ray origin',MESH_LIMITS.maxWorldCoordinate);
  direction=normalize(validVec(direction,'Ray direction'));
  const minDistance=options.minDistance??1e-9,maxDistance=options.maxDistance??Infinity;
  if(!Number.isFinite(minDistance)||minDistance<0||(!Number.isFinite(maxDistance)&&maxDistance!==Infinity)||maxDistance<minDistance)
    throw new Error('Ray distance interval must satisfy 0 <= minDistance <= maxDistance.');
  return {origin,direction,minDistance,maxDistance,cullBackface:options.cullBackface===true};
}
export function rayTriangleIntersect(origin,direction,triangle,options={}) {
  const r=checkedRay(origin,direction,options);
  return rayHit(r.origin,r.direction,triangle,r.minDistance,r.maxDistance,r.cullBackface);
}
export function rayOccluded(mesh,origin,direction,options={}) {
  const r=checkedRay(origin,direction,options),ignore=options.ignoreTriangle??options.ignoreIndex;
  const ignored=new Set(options.ignoreTriangleIds??[]);
  if(ignore!==undefined)ignored.add(ignore);
  return mesh.triangles.some(t=>!ignored.has(t.id)&&rayHit(r.origin,r.direction,t,r.minDistance,r.maxDistance,r.cullBackface)!==null);
}

function objFromIndexed(vertices,faces,title) {
  let text=`# ${title}\n# Millimetres; geometry only. Choose opaque/transmitting materials explicitly.\n`;
  text+=vertices.map(v=>`v ${v.join(' ')}`).join('\n')+'\n';
  for(const face of faces)for(let i=1;i<face.length-1;i++)text+=`f ${face[0]+1} ${face[i]+1} ${face[i+1]+1}\n`;
  return text;
}
function objFromPatches(patches,title) {
  let text=`# ${title}\n# Coordinates are millimetres. Faces are geometry only; choose materials downstream.\n`,offset=1;
  for(const patch of patches) {
    text+=patch.map(v=>`v ${v.join(' ')}`).join('\n')+'\n';
    // Explicit triangles avoid any dependence on browser OBJ triangulators.
    text+=`f ${offset} ${offset+1} ${offset+2}\nf ${offset} ${offset+2} ${offset+3}\n`;
    offset+=patch.length;
  }
  return text;
}
const rectangle=(x0,x1,y0,y1,z=0)=>[[x0,y0,z],[x1,y0,z],[x1,y1,z],[x0,y1,z]];
export function makePresetOBJ(name='double-slit',options={}) {
  const positive=(key,fallback,max=1000)=>{
    const value=options[key]??fallback;
    if(!Number.isFinite(value)||value<=0||value>max)throw new Error(`Preset ${key} must be positive and <= ${max}.`);
    return value;
  };
  if(['slit','slits','double-slit'].includes(name)) {
    const width=positive('width',.08),separation=positive('separation',.3),height=positive('height',.18);
    if(width>=separation)throw new Error('Double-slit width must be smaller than center separation.');
    return objFromPatches([-1,1].map(s=>rectangle(s*separation/2-width/2,s*separation/2+width/2,-height/2,height/2)),'Two finite slit apertures');
  }
  if(name==='opaque-double-slit') {
    const slitWidth=positive('slitWidth',options.width??.08),separation=positive('separation',.3),slitHeight=positive('slitHeight',options.height??.18);
    const wallWidth=positive('wallWidth',2),wallHeight=positive('wallHeight',2);
    if(slitWidth>=separation||separation+slitWidth>=wallWidth||slitHeight>=wallHeight)
      throw new Error('Slits must be separated and lie strictly inside the finite opaque wall.');
    const xs=[-wallWidth/2,-separation/2-slitWidth/2,-separation/2+slitWidth/2,separation/2-slitWidth/2,separation/2+slitWidth/2,wallWidth/2];
    const ys=[-wallHeight/2,-slitHeight/2,slitHeight/2,wallHeight/2];
    const vertices=ys.flatMap(y=>xs.map(x=>[x,y,0])),faces=[];
    for(let y=0;y<3;y++)for(let x=0;x<5;x++) {
      if(y===1&&(x===1||x===3))continue;
      const a=y*6+x;faces.push([a,a+1,a+7,a+6]);
    }
    return objFromIndexed(vertices,faces,'Finite opaque double-slit wall complement; two true open gaps');
  }
  if(name==='box') {
    const w=positive('width',2)/2,h=positive('height',2)/2,d=positive('depth',2)/2;
    return objFromIndexed([[-w,-h,-d],[w,-h,-d],[w,h,-d],[-w,h,-d],[-w,-h,d],[w,-h,d],[w,h,d],[-w,h,d]],
      [[0,3,2,1],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]],'Closed box; consistently outward winding');
  }
  if(name==='tetrahedron') {
    const w=positive('width',2)/2,h=positive('height',2)/2,d=positive('depth',2)/2;
    return objFromIndexed([[-w,-h,-d],[w,-h,-d],[-w,h,-d],[-w,-h,d]],[[0,2,1],[0,1,3],[0,3,2],[1,2,3]],'Closed tetrahedron; consistently outward winding');
  }
  if(['wedge','triangular-prism'].includes(name)) {
    const w=positive('width',2)/2,h=positive('height',2)/2,d=positive('depth',2)/2;
    return objFromIndexed([[-w,-h,-d],[w,-h,-d],[-w,-h,d],[-w,h,-d],[w,h,-d],[-w,h,d]],
      [[0,1,2],[3,5,4],[0,3,4,1],[1,4,5,2],[2,5,3,0]],'Closed right-triangular prism; consistently outward winding');
  }
  if(['square','finite-square','screen'].includes(name)) {
    const width=positive('width',name==='screen'?2:.3),height=positive('height',width);
    return objFromPatches([rectangle(-width/2,width/2,-height/2,height/2)],'Finite square/rectangle aperture');
  }
  if(['edge','edge-aperture'].includes(name)) {
    const width=positive('width',.5),height=positive('height',1);
    return objFromPatches([rectangle(-width,0,-height/2,height/2)],'Finite edge aperture patch; every outer edge is physical');
  }
  if(['folded','folded-surface','tilted'].includes(name)) {
    const width=positive('width',.36),height=positive('height',.24),depth=positive('depth',.07);
    // Two planar panels share a crease at x=0 and lie in different 3D planes.
    return objFromPatches([
      [[-width/2,-height/2,0],[0,-height/2,0],[0,height/2,0],[-width/2,height/2,0]],
      [[0,-height/2,0],[width/2,-height/2,depth],[width/2,height/2,depth],[0,height/2,0]],
    ],'Folded nonplanar surface');
  }
  throw new Error(`Unknown geometry preset '${name}'.`);
}
export const BUILTIN_OBJS=Object.freeze({
  'double-slit':makePresetOBJ('double-slit'),
  'finite-square':makePresetOBJ('finite-square'),
  'folded':makePresetOBJ('folded'),
  'edge':makePresetOBJ('edge'),
  'screen':makePresetOBJ('screen'),
  'box':makePresetOBJ('box'),
  'tetrahedron':makePresetOBJ('tetrahedron'),
  'triangular-prism':makePresetOBJ('triangular-prism'),
  'opaque-double-slit':makePresetOBJ('opaque-double-slit'),
});

/** Finite source-to-destination visibility, excluding only explicitly named faces. */
export function segmentVisible(mesh,source,destination,options={}) {
  const a=validVec(source,'Visibility source',MESH_LIMITS.maxWorldCoordinate);
  const b=validVec(destination,'Visibility destination',MESH_LIMITS.maxWorldCoordinate);
  const delta=sub(b,a),d=length(delta);
  if(d===0)return true;
  const machineEpsilon=64*Number.EPSILON*Math.max(1,...a.map(Math.abs),...b.map(Math.abs),d);
  const requested=options.endpointEpsilon??Math.max(1e-9,machineEpsilon);
  if(!Number.isFinite(requested)||requested<0)throw new Error('Visibility endpointEpsilon must be finite and nonnegative.');
  const epsilon=Math.min(requested,d/4);
  return !rayOccluded(mesh,a,delta,{ignoreTriangleIds:options.ignoreTriangleIds,minDistance:epsilon,maxDistance:d-epsilon});
}

/**
 * Stationary source-edge-destination path on a finite straight 3D edge.
 * Source/destination directions point away from the diffraction point.
 * Null means no interior/endpoint stationary point, or a degenerate on-edge
 * source/destination. No endpoint-clamped replacement of an outside stationary
 * point is made. Endpoint/corner asymptotics are the caller's responsibility.
 */
export function diffractionPoint(wedge,source,destination) {
  const src=validVec(source,'Diffraction source',MESH_LIMITS.maxWorldCoordinate);
  const dst=validVec(destination,'Diffraction destination',MESH_LIMITS.maxWorldCoordinate);
  const v=validVec(wedge.v??scale(add(wedge.a,wedge.b),.5),'Wedge midpoint',MESH_LIMITS.maxWorldCoordinate);
  const l=wedge.l??distance(wedge.a,wedge.b);
  if(!Number.isFinite(l)||l<=0)throw new Error('Wedge length must be positive and finite.');
  const e=normalize(wedge.e??sub(wedge.b,wedge.a));
  const s=sub(src,v),d=sub(dst,v),sa=dot(s,e),da=dot(d,e);
  const sr=length(sub(s,scale(e,sa))),dr=length(sub(d,scale(e,da)));
  const tolerance=64*Number.EPSILON*Math.max(1,l,length(s),length(d),length(v));
  if(sr+dr<=tolerance)return null;
  let axial=sa+(da-sa)*(sr/(sr+dr));
  if(axial< -l/2-tolerance||axial>l/2+tolerance)return null;
  axial=Math.max(-l/2,Math.min(l/2,axial));
  const point=add(v,scale(e,axial)),incident=sub(src,point),outgoing=sub(dst,point);
  const ri=length(incident),ro=length(outgoing);
  if(ri<=tolerance||ro<=tolerance)return null;
  const wi=scale(incident,1/ri),wo=scale(outgoing,1/ro);
  return {point,axial,ri,ro,wi,wo,e,pathLength:ri+ro,
    sinBeta:Math.sqrt(Math.max(0,1-dot(wi,e)**2)),
    atEndpoint:Math.abs(Math.abs(axial)-l/2)<=tolerance,
    fermatResidual:dot(wi,e)+dot(wo,e)};
}

/** Direct and two-leg visibility. Adjacent wedge triangles are excluded only
 * for legs terminating on that edge, never from the direct source-camera ray. */
export function diffractionVisibility(mesh,wedge,source,destination,point) {
  const p=point?.point??point??diffractionPoint(wedge,source,destination)?.point;
  const directVisible=segmentVisible(mesh,source,destination);
  if(!p)return {directVisible,sourceVisible:false,destinationVisible:false,visible:false};
  const options={ignoreTriangleIds:wedge.triangleIds??[]};
  const sourceVisible=segmentVisible(mesh,source,p,options),destinationVisible=segmentVisible(mesh,p,destination,options);
  return {directVisible,sourceVisible,destinationVisible,visible:sourceVisible&&destinationVisible};
}

/**
 * Extract actual opaque wedge frames, without choosing a material or changing
 * authored winding. surface must be 'closed-solid' (watertight, outward wound)
 * or 'open-plate' (each connected component is a planar zero-thickness plate).
 * Unsupported geometry disables the entire extraction instead of silently
 * dropping an ambiguous, concave, or nonplanar component.
 *
 * alpha is the solid INTERIOR angle in radians (plate=0); native UTD uses
 * n=2-alpha/PI. n1/n2 point outward, t1/t2 point along each face into the solid.
 * The front-oriented frame has e=cross(nff,tff), wi pointing edge -> source.
 * Source convention reference (original geometry implementation below):
 * https://wavetracer.dev/api/structwt_1_1utd_1_1wedge__edge__t.html
 */
export function extractWedges(mesh,options={}) {
  const surface=options.surface??'closed-solid';
  if(!['closed-solid','open-plate'].includes(surface))throw new Error('Wedge surface must be closed-solid or open-plate.');
  if(!Array.isArray(mesh?.triangles)||!Array.isArray(mesh?.edges))throw new Error('Wedge extraction requires a prepared mesh.');
  const unsupported=[],warnings=[],components=[],wedges=[];
  const problem=(code,message,extra={})=>unsupported.push({code,message,...extra});
  if(mesh.nonManifoldEdges.length)problem('nonmanifold','Opaque UTD geometry requires manifold edges.');
  if(mesh.windingConflicts)problem('inconsistent-winding','Opaque UTD geometry requires consistently wound adjacent faces.');
  const neighbors=mesh.triangles.map(()=>[]);
  for(const edge of mesh.edges)if(edge.valence===2) {
    const [a,b]=edge.triangleIds;neighbors[a].push(b);neighbors[b].push(a);
  }
  const componentOf=new Int32Array(mesh.triangles.length).fill(-1);
  for(let start=0;start<mesh.triangles.length;start++) {
    if(componentOf[start]>=0)continue;
    const id=components.length,triangleIds=[],queue=[start];componentOf[start]=id;
    for(let i=0;i<queue.length;i++) {
      const t=queue[i];triangleIds.push(t);
      for(const other of neighbors[t])if(componentOf[other]<0){componentOf[other]=id;queue.push(other);}
    }
    const triangles=triangleIds.map(i=>mesh.triangles[i]);
    const reference=triangles[0].a,normal=triangles[0].normal;
    let extent=0,signedVolume=0,volumeMagnitude=0;
    for(const t of triangles) {
      for(const p of [t.a,t.b,t.c])extent=Math.max(extent,distance(p,reference));
      const term=dot(sub(t.a,reference),cross(sub(t.b,reference),sub(t.c,reference)))/6;
      signedVolume+=term;volumeMagnitude+=Math.abs(term);
    }
    const boundary=mesh.boundaryEdges.filter(e=>componentOf[e.triangleIds[0]]===id);
    const closed=boundary.length===0;
    const component={id,triangleIds,closed,signedVolume,extent};components.push(component);
    if(surface==='closed-solid') {
      if(!closed)problem('open-solid','A solid component has open boundary edges; use an explicitly planar open plate or close the mesh.',{componentId:id});
      else {
        const tolerance=Math.max(extent**3*1e-12,volumeMagnitude*64*Number.EPSILON);
        if(Math.abs(signedVolume)<=tolerance)problem('zero-volume','A solid component has zero or numerically unresolved signed volume.',{componentId:id});
        else if(signedVolume<0)problem('inward-winding','A solid component is wound inward. Correct the authored winding; it is not flipped automatically.',{componentId:id});
      }
    } else {
      if(closed)problem('closed-plate','An open plate must have a boundary.',{componentId:id});
      const tolerance=Math.max(extent*1e-9,64*Number.EPSILON*length(reference));
      const nonplanar=triangles.some(t=>dot(t.normal,normal)<1-1e-9||[t.a,t.b,t.c].some(p=>Math.abs(dot(sub(p,reference),normal))>tolerance));
      if(nonplanar)problem('nonplanar-open-plate','A folded or curved open shell has no unambiguous solid wedge interior and is unsupported.',{componentId:id});
    }
  }
  // Gate winding/closedness before interpreting convex versus concave sides.
  if(unsupported.length)return {supported:false,surface,wedges:[],unsupported,warnings,components};
  for(const edge of mesh.diffractionEdges) {
    if(surface==='open-plate'&&edge.kind!=='boundary')continue;
    if(surface==='closed-solid'&&edge.kind!=='crease')continue;
    const t1face=mesh.triangles[edge.triangleIds[0]],t2face=edge.valence===2?mesh.triangles[edge.triangleIds[1]]:null;
    const v=scale(add(edge.a,edge.b),.5),direction=normalize(sub(edge.b,edge.a));
    const third=triangle=>mesh.vertices[triangle.indices.find((_,i)=>!edge.weldedIndices.includes(triangle.weldedIndices[i]))];
    const c1=third(t1face),c2=t2face?third(t2face):null,n1=[...t1face.normal],n2=t2face?[...t2face.normal]:scale(n1,-1);
    if(Math.abs(dot(n1,direction))>1e-9||Math.abs(dot(n2,direction))>1e-9) {
      problem('imprecise-weld','Welded edge endpoints do not lie in both adjacent face planes; use exact shared positions.',{edgeId:edge.id});continue;
    }
    let t1=normalize(cross(n1,direction));if(dot(t1,sub(c1,v))<0)t1=scale(t1,-1);
    let t2=t2face?normalize(cross(n2,direction)):[...t1];if(c2&&dot(t2,sub(c2,v))<0)t2=scale(t2,-1);
    let alpha=0;
    if(t2face) {
      const side1=dot(n1,sub(c2,v)),side2=dot(n2,sub(c1,v));
      const tolerance=1e-10*Math.max(edge.length,distance(c1,v),distance(c2,v));
      if(side1>tolerance&&side2>tolerance) {
        problem('concave-edge','Concave solid wedges are unsupported; their normals are not flipped into a fictitious convex wedge.',{edgeId:edge.id,triangleIds:[...edge.triangleIds]});continue;
      }
      if(side1>tolerance||side2>tolerance) {
        problem('ambiguous-wedge','Neighboring face halfspaces do not define a consistently oriented convex wedge.',{edgeId:edge.id});continue;
      }
      alpha=Math.PI-Math.acos(Math.max(-1,Math.min(1,dot(n1,n2))));
    }
    wedges.push({id:edge.id,edgeId:edge.id,segmentIds:[edge.id],vertexIds:[...edge.weldedIndices],a:[...edge.a],b:[...edge.b],v,l:edge.length,
      n1,t1,n2,t2,alpha,interiorAngle:alpha,exteriorAngle:2*Math.PI-alpha,
      triangleIds:[...edge.triangleIds],faceTriangleIds:[[edge.triangleIds[0]],[edge.triangleIds[1]??edge.triangleIds[0]]],componentId:componentOf[edge.triangleIds[0]],kind:edge.kind});
  }
  if(mesh.options.weldTolerance>0)warnings.push('Wedge connectivity uses the declared weld tolerance; source positions are preserved.');
  return {supported:unsupported.length===0,surface,wedges:unsupported.length?[]:(options.mergeCollinear===false?wedges:mergeCollinearWedges(wedges)),unsupported,warnings,components};
}

/** Select the native outward front face using edge-to-source direction wi. */
export function orientWedge(wedge,wi) {
  wi=normalize(validVec(wi,'Incident edge-to-source direction'));
  const first=dot(wi,wedge.n1)>0;
  const nff=first?wedge.n1:wedge.n2,tff=first?wedge.t1:wedge.t2,nbf=first?wedge.n2:wedge.n1;
  if(dot(wi,nff)<=0)return null;
  return {...wedge,nff:[...nff],tff:[...tff],nbf:[...nbf],e:normalize(cross(nff,tff)),frontTriangleId:wedge.faceTriangleIds?.[first?0:1]?.[0]??wedge.triangleIds[first?0:1]??wedge.triangleIds[0]};
}

/**
 * Stationary point for an incident plane wave and a finite detector point.
 * incidentDirection is propagation direction TOWARD the geometry, so wi=-di.
 * Phase-distance convention: di·point + |detector-point|. Native UTD supports
 * this plane-incidence distance limit; no finite-source curvature is invented.
 */
export function diffractionPointPlane(wedge,incidentDirection,destination) {
  const di=normalize(validVec(incidentDirection,'Plane-wave propagation direction'));
  const dst=validVec(destination,'Diffraction destination',MESH_LIMITS.maxWorldCoordinate);
  const v=validVec(wedge.v??scale(add(wedge.a,wedge.b),.5),'Wedge midpoint',MESH_LIMITS.maxWorldCoordinate);
  const l=wedge.l??distance(wedge.a,wedge.b),e=normalize(wedge.e??sub(wedge.b,wedge.a));
  if(!Number.isFinite(l)||l<=0)throw new Error('Wedge length must be positive and finite.');
  const c=dot(di,e),sinBeta=Math.sqrt(Math.max(0,1-c*c));
  if(sinBeta<1e-12)return null;
  const d=sub(dst,v),da=dot(d,e),dr=length(sub(d,scale(e,da)));
  let axial=da-dr*c/sinBeta;
  const tolerance=64*Number.EPSILON*Math.max(1,l,length(d),length(v));
  if(axial< -l/2-tolerance||axial>l/2+tolerance)return null;
  axial=Math.max(-l/2,Math.min(l/2,axial));
  const point=add(v,scale(e,axial)),outgoing=sub(dst,point),ro=length(outgoing);
  if(ro<=tolerance)return null;
  const wi=scale(di,-1),wo=scale(outgoing,1/ro),incidentPhaseDistance=dot(di,point);
  return {point,axial,ro,wi,wo,e,sinBeta,planeWave:true,incidentDirection:di,
    incidentPhaseDistance,opticalPath:incidentPhaseDistance+ro,
    atEndpoint:Math.abs(Math.abs(axial)-l/2)<=tolerance,
    fermatResidual:dot(wi,e)+dot(wo,e)};
}

/** Infinite source-side rays and finite edge-detector visibility. */
export function planeDiffractionVisibility(mesh,wedge,incidentDirection,destination,point) {
  const wi=scale(normalize(validVec(incidentDirection,'Plane-wave propagation direction')),-1);
  const dst=validVec(destination,'Visibility destination',MESH_LIMITS.maxWorldCoordinate);
  const p=point?.point??point??diffractionPointPlane(wedge,incidentDirection,dst)?.point;
  const epsilonAt=v=>Math.max(1e-9,64*Number.EPSILON*Math.max(1,...v.map(Math.abs)));
  const directVisible=!rayOccluded(mesh,dst,wi,{minDistance:epsilonAt(dst)});
  if(!p)return {directVisible,sourceVisible:false,destinationVisible:false,visible:false};
  validVec(p,'Diffraction point',MESH_LIMITS.maxWorldCoordinate);
  const sourceVisible=!rayOccluded(mesh,p,wi,{minDistance:epsilonAt(p),ignoreTriangleIds:wedge.triangleIds??[]});
  const destinationVisible=segmentVisible(mesh,p,dst,{ignoreTriangleIds:wedge.triangleIds??[]});
  return {directVisible,sourceVisible,destinationVisible,visible:sourceVisible&&destinationVisible};
}

/**
 * Remove tessellation-created endpoints on straight, same-frame physical edges.
 * The union of all adjacent triangles is retained for source/receiver visibility.
 * Junctions with more than two physical edge segments are never merged.
 */
export function mergeCollinearWedges(wedges) {
  const alive=new Map(),atVertex=new Map(),queue=[];
  const unique=a=>[...new Set(a)];
  const vertexKey=(w,i)=>w.vertexIds?.[i]??[w.a,w.b][i].join(',');
  const register=w=>{
    alive.set(w.id,w);
    for(let i=0;i<2;i++) {
      const key=vertexKey(w,i);
      if(!atVertex.has(key)){atVertex.set(key,new Set());queue.push(key);}
      atVertex.get(key).add(w.id);
    }
  };
  const unregister=w=>{
    alive.delete(w.id);
    for(let i=0;i<2;i++)atVertex.get(vertexKey(w,i))?.delete(w.id);
  };
  for(const w of wedges)register({...w,segmentIds:w.segmentIds??[w.id],faceTriangleIds:w.faceTriangleIds??[[w.triangleIds[0]],[w.triangleIds[1]??w.triangleIds[0]]]});
  for(let cursor=0;cursor<queue.length;cursor++) {
    const key=queue[cursor],ids=[...(atVertex.get(key)??[])];
    if(ids.length!==2)continue;
    const a=alive.get(ids[0]),b=alive.get(ids[1]);
    if(!a||!b||a.componentId!==b.componentId||a.kind!==b.kind||Math.abs(a.alpha-b.alpha)>1e-10)continue;
    const ai=vertexKey(a,0)===key?0:1,bi=vertexKey(b,0)===key?0:1;
    const shared=[a.a,a.b][ai],pa=[a.a,a.b][1-ai],pb=[b.a,b.b][1-bi];
    const da=normalize(sub(pa,shared)),db=normalize(sub(pb,shared));
    if(dot(da,db)>=0||length(cross(da,db))>1e-10)continue;
    const same=(u,v)=>distance(u,v)<=1e-9;
    const direct=same(a.n1,b.n1)&&same(a.t1,b.t1)&&same(a.n2,b.n2)&&same(a.t2,b.t2);
    const swapped=same(a.n1,b.n2)&&same(a.t1,b.t2)&&same(a.n2,b.n1)&&same(a.t2,b.t1);
    if(!direct&&!swapped)continue;
    const bfaces=direct?b.faceTriangleIds:[...b.faceTriangleIds].reverse();
    const merged={...a,id:Math.min(a.id,b.id),edgeId:Math.min(a.edgeId,b.edgeId),
      a:[...pa],b:[...pb],v:scale(add(pa,pb),.5),l:distance(pa,pb),
      vertexIds:[vertexKey(a,1-ai),vertexKey(b,1-bi)],segmentIds:unique([...a.segmentIds,...b.segmentIds]),
      triangleIds:unique([...a.triangleIds,...b.triangleIds]),
      faceTriangleIds:[unique([...a.faceTriangleIds[0],...bfaces[0]]),unique([...a.faceTriangleIds[1],...bfaces[1]])]};
    unregister(a);unregister(b);register(merged);
    queue.push(...merged.vertexIds);
  }
  return [...alive.values()].sort((a,b)=>a.id-b.id);
}
