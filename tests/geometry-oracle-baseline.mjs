/** Frozen v11 geometry oracle. Keep independent of the optimized query code.
 * These implementations are copied verbatim from the pre-optimization source
 * (except exporting the facet helper). Do not optimize this test-only module.
 */
import {add,sub,scale,dot,cross,length,distance} from '../dist/mesh-geometry.js';
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const epsilonAt=(...points)=>Math.max(1e-10,128*Number.EPSILON*Math.max(1,...points.flat().map(Math.abs)));

/** Convex cross-section of a triangle at a beam-axis depth, including flat faces. */
function sectionAt(triangle,cone,t) {
  const p=[triangle.a,triangle.b,triangle.c],z=p.map(v=>dot(sub(v,cone.origin),cone.direction)),out=[];
  const tol=epsilonAt(...p,cone.origin);
  for(let i=0;i<3;i++) {
    const j=(i+1)%3;
    if(Math.abs(z[i]-t)<=tol)out.push(p[i]);
    if((z[i]<t&&z[j]>t)||(z[i]>t&&z[j]<t))out.push(add(p[i],scale(sub(p[j],p[i]),(t-z[i])/(z[j]-z[i]))));
  }
  if(!out.length)return null;
  const axis=add(cone.origin,scale(cone.direction,t));
  if(out.length>=3&&Math.max(...z)-Math.min(...z)<=tol) {
    // For a coplanar section, an axis point within the triangle is its minimum.
    // Near a vertex, tolerance can duplicate section endpoints. That does not
    // make an oblique/parallel triangle coplanar with this cross-section.
    const n=triangle.normal;
    if(p.every((v,i)=>dot(cross(sub(p[(i+1)%3],v),sub(axis,v)),n)>=-tol))return {point:axis,radial:0};
  }
  let point=out[0],radial=distance(point,axis);
  for(let i=0;i<out.length;i++)for(let j=i+1;j<out.length;j++) {
    const delta=sub(out[j],out[i]),ll=dot(delta,delta);
    const q=add(out[i],scale(delta,ll?clamp(dot(sub(axis,out[i]),delta)/ll,0,1):0));
    const r=distance(q,axis);if(r<radial){radial=r;point=q;}
  }
  return {point,radial};
}

/** Earliest actual cone/triangle contact, after conservative broad-phase gather.
 * At depth t the nearest triangle-section point gives a convex distance-minus-
 * radius function. Golden-section minimization brackets feasibility, followed
 * by bisection of its first zero. This never substitutes the triangle centroid.
 */
export function firstConeTriangleContact(triangle,cone) {
  const depths=[triangle.a,triangle.b,triangle.c].map(p=>dot(sub(p,cone.origin),cone.direction));
  const lo=Math.max(cone.near,Math.min(...depths)),hi=Math.min(cone.far,Math.max(...depths));
  if(lo>hi)return null;
  const tol=epsilonAt(triangle.a,triangle.b,triangle.c,cone.origin);
  const test=t=>{const s=sectionAt(triangle,cone,t);return s?{...s,value:s.radial-cone.radius0-cone.tanAngle*t}:null;};
  const left=test(lo);if(left&&left.value<=tol)return {distance:lo,point:left.point,triangleId:triangle.id};
  if(hi-lo<=tol)return null;
  let a=lo,b=hi;
  const gr=(Math.sqrt(5)-1)/2;
  let c=b-gr*(b-a),d=a+gr*(b-a),fc=test(c),fd=test(d);
  for(let i=0;i<65;i++) {
    if((fc?.value??Infinity)<(fd?.value??Infinity)){b=d;d=c;fd=fc;c=b-gr*(b-a);fc=test(c);}
    else {a=c;c=d;fc=fd;d=a+gr*(b-a);fd=test(d);}
  }
  const candidates=[[(a+b)/2,test((a+b)/2)],[hi,test(hi)],[c,fc],[d,fd]].filter(v=>v[1]);
  candidates.sort((x,y)=>x[1].value-y[1].value);
  if(!candidates.length||candidates[0][1].value>tol)return null;
  let lower=lo,upper=candidates[0][0];
  for(let i=0;i<65;i++){const mid=(lower+upper)/2;if((test(mid)?.value??Infinity)<=tol)upper=mid;else lower=mid;}
  return {distance:upper,point:test(upper).point,triangleId:triangle.id};
}

export function buildFacets(scene,triangles,clips=new Map()) {
  const selected=new Set(triangles.map(t=>t.id)),adjacency=new Map(triangles.map(t=>[t.id,[]])),seen=new Set(),facets=[];
  for(const edge of scene.edges) {
    if(edge.kind!=='internal'||edge.valence!==2)continue;
    const [a,b]=edge.triangleIds;if(!selected.has(a)||!selected.has(b))continue;
    const ta=scene.triangles[a],tb=scene.triangles[b];
    if(ta.materialIndex!==tb.materialIndex||dot(ta.normal,tb.normal)<1-1e-10)continue;
    const eps=epsilonAt(ta.a,ta.b,ta.c,tb.a,tb.b,tb.c);
    if([tb.a,tb.b,tb.c].some(p=>Math.abs(dot(sub(p,ta.a),ta.normal))>eps))continue;
    adjacency.get(a).push(b);adjacency.get(b).push(a);
  }
  for(const t of triangles) {
    if(seen.has(t.id))continue;
    const ids=[t.id];seen.add(t.id);
    for(let i=0;i<ids.length;i++)for(const other of adjacency.get(ids[i]))if(!seen.has(other)){seen.add(other);ids.push(other);}
    facets.push({id:facets.length,triangleIds:ids,triangles:ids.map(id=>scene.triangles[id]),
      clippedTriangles:ids.map(id=>clips.get(id)).filter(Boolean),normal:[...t.normal],point:[...t.a],materialIndex:t.materialIndex,meshId:t.meshId});
  }
  return facets;
}
