/**
 * Bounded geometric admission for the first-order UTD solver, after extractWedges.
 * Local dihedral tests alone do not establish an embedded convex solid. This
 * module adds global supporting-halfspace tests, complete convex-polyhedron SAT
 * (face normals AND edge-cross-edge axes), and embedded planar triangulation
 * checks. No mesh, triangle, extraction record, or authored coordinate is changed.
 *
 * The caller must enforce its solver cap before extraction / other costly work.
 * This module independently refuses >128 triangles before doing pairwise work.
 * Predicates use local differences and Euclidean, scale-aware roundoff guards;
 * separated components must have a resolvable positive gap. Contact, nesting,
 * overlap, and unresolved near-contact are conservatively unsupported. Ordinary
 * triangulation edges/vertices in a single component remain legal, including
 * holes. This is floating-point geometric admission, not exact-arithmetic CSG.
 */
import {sub,add,scale,dot,cross,length,distance,normalize} from './mesh-geometry.js';

const MAX_TRIANGLES=128;
const ROUND=128*Number.EPSILON;
const positions=t=>[t.a,t.b,t.c];
const ids=t=>t.weldedIndices??t.indices;
const norm2=p=>Math.hypot(p[0],p[1]);
const sub2=(a,b)=>[a[0]-b[0],a[1]-b[1]];
const cross2=(a,b)=>a[0]*b[1]-a[1]*b[0];
const dot2=(a,b)=>a[0]*b[0]+a[1]*b[1];

function geometryScale(points) {
  const reference=points[0];
  let extent=0,magnitude=0;
  for(const p of points){extent=Math.max(extent,distance(p,reference));magnitude=Math.max(magnitude,length(p));}
  return {extent,magnitude};
}
function tolerance(...groups) {
  let extent=0,magnitude=0;
  for(const g of groups){extent=Math.max(extent,g.extent);magnitude=Math.max(magnitude,g.magnitude);}
  return Math.max(extent*1e-12,ROUND*magnitude,Number.MIN_VALUE);
}
function projectedRange(points,axis,reference) {
  let lo=Infinity,hi=-Infinity;
  for(const p of points){const x=dot(sub(p,reference),axis);lo=Math.min(lo,x);hi=Math.max(hi,x);}
  return [lo,hi];
}

/** Full SAT for bounded convex solids. Extra triangulation axes are harmless. */
function solidsContact(a,b,tol) {
  const reference=a.points[0];
  const separates=axis=>{
    const l=length(axis);if(l<=ROUND)return false;
    const n=scale(axis,1/l),ra=projectedRange(a.points,n,reference),rb=projectedRange(b.points,n,reference);
    return rb[0]-ra[1]>tol||ra[0]-rb[1]>tol;
  };
  for(const t of [...a.triangles,...b.triangles])if(separates(t.normal))return false;
  for(const u of a.edgeDirections)for(const v of b.edgeDirections)if(separates(cross(u,v)))return false;
  return true;
}

function planarCoordinates(a,b) {
  const origin=a.a,u=normalize(sub(a.b,a.a)),v=normalize(cross(a.normal,u));
  const project=p=>{const d=sub(p,origin);return [dot(d,u),dot(d,v)];};
  return {a:positions(a).map(project),b:positions(b).map(project),project};
}
function planarContact(a,b,tol) {
  let positiveArea=true;
  for(const tri of [a,b])for(let i=0;i<3;i++) {
    const edge=sub2(tri[(i+1)%3],tri[i]),l=norm2(edge),axis=[-edge[1]/l,edge[0]/l];
    const pa=a.map(p=>dot2(p,axis)),pb=b.map(p=>dot2(p,axis));
    const overlap=Math.min(Math.max(...pa),Math.max(...pb))-Math.max(Math.min(...pa),Math.min(...pb));
    if(overlap< -tol)return {contact:false,positiveArea:false};
    if(overlap<=tol)positiveArea=false;
  }
  return {contact:true,positiveArea};
}
function inside2(point,triangle,tol) {
  const orientation=Math.sign(cross2(sub2(triangle[1],triangle[0]),sub2(triangle[2],triangle[0])));
  return triangle.every((p,i)=>{
    const edge=sub2(triangle[(i+1)%3],p);
    return orientation*cross2(edge,sub2(point,p))>=-tol*norm2(edge);
  });
}
function planarIntersectionPoints(a,b,tol) {
  // Endpoints cover collinear contacts; proper crossings supply the remaining
  // vertices of a convex intersection polygon, including zero-area segments.
  const out=[...a.filter(p=>inside2(p,b,tol)),...b.filter(p=>inside2(p,a,tol))];
  for(let i=0;i<3;i++)for(let j=0;j<3;j++) {
    const p=a[i],q=b[j],r=sub2(a[(i+1)%3],p),s=sub2(b[(j+1)%3],q),den=cross2(r,s);
    if(Math.abs(den)<=ROUND*norm2(r)*norm2(s))continue;
    const d=sub2(q,p),u=cross2(d,s)/den,v=cross2(d,r)/den;
    if(u>=-tol/norm2(r)&&u<=1+tol/norm2(r)&&v>=-tol/norm2(s)&&v<=1+tol/norm2(s))
      out.push([p[0]+u*r[0],p[1]+u*r[1]]);
  }
  return out;
}
function distanceToSegment2(p,a,b) {
  const d=sub2(b,a),ll=dot2(d,d),t=ll?Math.max(0,Math.min(1,dot2(sub2(p,a),d)/ll)):0;
  return norm2([p[0]-a[0]-t*d[0],p[1]-a[1]-t*d[1]]);
}
function distanceToSegment3(p,a,b) {
  const d=sub(b,a),ll=dot(d,d),t=ll?Math.max(0,Math.min(1,dot(sub(p,a),d)/ll)):0;
  return distance(p,add(a,scale(d,t)));
}
function sharedPositions(a,b) {
  const bi=new Set(ids(b));
  return positions(a).filter((_,i)=>bi.has(ids(a)[i]));
}
function planeSection(triangle,plane,tol) {
  const p=positions(triangle),d=p.map(v=>dot(sub(v,plane.a),plane.normal)),out=[];
  if(d.every(x=>x>tol)||d.every(x=>x< -tol))return out;
  for(let i=0;i<3;i++) {
    const j=(i+1)%3;
    if(Math.abs(d[i])<=tol)out.push(p[i]);
    if((d[i]<0&&d[j]>0)||(d[i]>0&&d[j]<0))out.push(add(p[i],scale(sub(p[j],p[i]),d[i]/(d[i]-d[j]))));
  }
  return out;
}
function triangleContact(a,b,tol,sameComponent) {
  const shared=sameComponent?sharedPositions(a,b):[];
  const normalsCross=cross(a.normal,b.normal),sine=length(normalsCross);
  const coplanar=positions(b).every(p=>Math.abs(dot(sub(p,a.a),a.normal))<=tol)
    &&positions(a).every(p=>Math.abs(dot(sub(p,b.a),b.normal))<=tol);
  if(coplanar) {
    const projected=planarCoordinates(a,b),contact=planarContact(projected.a,projected.b,tol);
    if(!contact.contact)return null;
    if(contact.positiveArea)return 'overlap';
    if(!shared.length)return 'contact';
    const feature=shared.map(projected.project),points=planarIntersectionPoints(projected.a,projected.b,tol);
    // A legal intersection is exactly the shared topological vertex or edge.
    // Any other contact is a T junction, overlapping edge, or self-crossing.
    if(!points.length)return 'unresolved';
    if(feature.length===1)return points.every(p=>norm2(sub2(p,feature[0]))<=4*tol)?null:'contact';
    if(feature.length===2)return points.every(p=>distanceToSegment2(p,feature[0],feature[1])<=4*tol)?null:'contact';
    return 'overlap';
  }
  if(sine<=ROUND)return null; // Reliably distinct parallel planes.
  const sa=planeSection(a,b,tol),sb=planeSection(b,a,tol);
  if(!sa.length||!sb.length)return null;
  const axis=scale(normalsCross,1/sine),reference=a.a,ra=projectedRange(sa,axis,reference),rb=projectedRange(sb,axis,reference);
  const lo=Math.max(ra[0],rb[0]),hi=Math.min(ra[1],rb[1]);
  if(lo>hi+tol)return null;
  if(!shared.length)return 'crossing';
  // In a convex triangulated shell non-coplanar adjacent faces may intersect
  // along their shared edge, and non-adjacent faces at their shared vertex.
  const sample=sa[0],base=dot(sub(sample,reference),axis);
  const ends=[lo,hi].map(x=>add(sample,scale(axis,x-base)));
  if(shared.length===1)return ends.every(p=>distance(p,shared[0])<=4*tol)?null:'crossing';
  if(shared.length===2)return ends.every(p=>distanceToSegment3(p,shared[0],shared[1])<=4*tol)?null:'crossing';
  return 'crossing';
}

/** Return additional admission issues; existing extraction errors stay upstream. */
export function validateUTDGeometry(mesh,extracted) {
  const issues=[],warnings=[];
  const finish=()=>({supported:issues.length===0,issues,warnings});
  if(!Array.isArray(mesh?.triangles)||!Array.isArray(extracted?.components))
    throw new Error('UTD geometric admission requires a prepared mesh and wedge extraction.');
  if(mesh.triangles.length>MAX_TRIANGLES) {
    issues.push({code:'geometry-validation-limit',message:'UTD geometric admission supports at most 128 triangles.'});
    return finish();
  }
  if(extracted.supported===false)return finish();
  const triangleMap=new Map(mesh.triangles.map(t=>[t.id,t]));
  const componentOf=new Map();
  const components=extracted.components.map(c=>{
    const triangles=c.triangleIds.map(id=>triangleMap.get(id)),points=triangles.flatMap(positions);
    const geometry=geometryScale(points),tol=tolerance(geometry),vertexPositions=new Map(),edgeDirections=[];
    for(const t of triangles) {
      componentOf.set(t.id,c.id);
      positions(t).forEach((p,i)=>{
        const id=ids(t)[i],previous=vertexPositions.get(id);
        if(previous&&distance(p,previous)>tol&&!issues.some(x=>x.code==='imprecise-weld-geometry'&&x.componentId===c.id))
          issues.push({code:'imprecise-weld-geometry',message:'Welded topology joins distinct positions; exact shared positions are required for geometric UTD admission.',componentId:c.id});
        if(!previous)vertexPositions.set(id,p);
        edgeDirections.push(normalize(sub(positions(t)[(i+1)%3],p)));
      });
    }
    return {...c,...geometry,triangles,points:[...vertexPositions.values()],edgeDirections};
  });
  if(issues.length)return finish();
  if(extracted.surface==='closed-solid') {
    for(const c of components) {
      const tol=tolerance(c);
      outer:for(const t of c.triangles)for(const p of c.points)if(dot(sub(p,t.a),t.normal)>tol) {
        issues.push({code:'nonconvex-solid',message:'Every solid vertex must lie behind every outward face plane; this component is nonconvex or self-intersecting.',componentId:c.id,triangleIds:[t.id]});
        break outer;
      }
    }
    if(issues.length)return finish(); // SAT below requires convex components.
    for(let i=0;i<components.length;i++)for(let j=i+1;j<components.length;j++) {
      const a=components[i],b=components[j];
      if(solidsContact(a,b,tolerance(a,b)))issues.push({code:'intersecting-solid-components',message:'Convex solid components overlap, nest, touch, or have an unresolved gap; independent opaque UTD wedges require separated solids.',componentIds:[a.id,b.id]});
    }
    if(issues.length)return finish();
  }
  for(let i=0;i<mesh.triangles.length;i++)for(let j=i+1;j<mesh.triangles.length;j++) {
    const a=mesh.triangles[i],b=mesh.triangles[j],ca=componentOf.get(a.id),cb=componentOf.get(b.id),same=ca===cb;
    if(extracted.surface==='closed-solid'&&!same)continue; // Already covered by full SAT.
    const tol=tolerance(geometryScale(positions(a)),geometryScale(positions(b))),kind=triangleContact(a,b,tol,same);
    if(kind) {
      const plate=extracted.surface==='open-plate';
      issues.push({code:kind==='overlap'?'overlapping-surface-triangles':plate&&!same?'intersecting-plate-components':'self-intersecting-surface',
        message:kind==='overlap'?'Triangle interiors overlap in the same plane; duplicated opaque surface area is unsupported.':'Surface triangles meet outside their shared topological edge or vertex; crossing, touching, or unresolved geometry is unsupported.',
        triangleIds:[a.id,b.id],componentIds:same?[ca]:[ca,cb]});
      // One witness establishes rejection; avoid quadratic-size diagnostic output.
      return finish();
    }
  }
  return finish();
}
