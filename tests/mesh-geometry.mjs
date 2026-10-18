import assert from 'node:assert/strict';
import {
  MESH_LIMITS,parseOBJ,prepareMesh,transformMesh,transformPoint,triangleArea,triangleNormal,
  add,sub,scale,dot,cross,length,distance,normalize,
  coneRadius,triangleIntersectsCone,gatherConeTriangles,rayTriangleIntersect,rayOccluded,
  makePresetOBJ,BUILTIN_OBJS,segmentVisible,diffractionPoint,diffractionVisibility,
  extractWedges,orientWedge,diffractionPointPlane,planeDiffractionVisibility,
} from '../dist/mesh-geometry.js';

let checks=0;
const test=(name,fn)=>{try{fn();checks++;}catch(error){error.message=`${name}: ${error.message}`;throw error;}};
const near=(a,b,tolerance=1e-10)=>assert(Math.abs(a-b)<=tolerance*Math.max(1,Math.abs(a),Math.abs(b)),`${a} != ${b}`);
const vectorNear=(a,b,tolerance)=>a.forEach((x,i)=>near(x,b[i],tolerance));
const rejects=(text,pattern)=>assert.throws(()=>parseOBJ(text),pattern);
const basic='v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3';
const perimeter=mesh=>mesh.boundaryEdges.reduce((sum,e)=>sum+e.length,0);

for(const [name,obj] of Object.entries(BUILTIN_OBJS))test(`preset ${name}`,()=>{
  const mesh=prepareMesh(obj);
  assert(mesh.triangles.length>=2);
  assert(mesh.totalArea>0);
  assert(mesh.triangles.every(t=>t.area>0&&t.normal.every(Number.isFinite)));
  assert(mesh.boundaryEdges.length>0||mesh.isClosed);
});

test('vector operations',()=>{
  vectorNear(add([1,2,3],[2,3,4]),[3,5,7]);
  vectorNear(sub([1,2,3],[2,3,4]),[-1,-1,-1]);
  vectorNear(scale([1,2,3],2),[2,4,6]);
  assert.equal(dot([1,2,3],[4,5,6]),32);
  vectorNear(cross([1,0,0],[0,1,0]),[0,0,1]);
  near(length([3,4,0]),5);near(distance([1,0,0],[1,3,4]),5);
  vectorNear(normalize([0,0,5]),[0,0,1]);
  assert.throws(()=>normalize([0,0,0]),/nonzero/);
  near(triangleArea([0,0,0],[2,0,0],[0,3,0]),3);
  vectorNear(triangleNormal([0,0,0],[2,0,0],[0,3,0]),[0,0,1]);
});

test('OBJ comments, tabs, exponent and BOM',()=>{
  const mesh=parseOBJ('\ufeff# mesh\r\nv\t0 0 0 1\rv +1e0 0 0\nv 0 .1e1 0\nf 1 2 3 # face');
  assert.deepEqual(mesh.vertices,[[0,0,0],[1,0,0],[0,1,0]]);
  assert.deepEqual(mesh.faces,[[0,1,2]]);
});

test('OBJ all corner syntaxes and negative indices',()=>{
  const header='v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nvn 0 0 1\n';
  for(const face of ['f 1 2 3','f 1/1 2/2 3/3','f 1//1 2//1 3//1','f 1/1/1 2/2/1 3/3/1','f -3/-3/-1 -2/-2/-1 -1/-1/-1'])
    assert.deepEqual(parseOBJ(header+face).faces,[[0,1,2]]);
});

test('strict finite numeric and index validation',()=>{
  for(const bad of ['NaN','Infinity','-Infinity','1e309','0xFF','1e7','1_000','1foo'])
    rejects(`v ${bad} 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3`,/numeric|Numeric|limit/i);
  for(const f of ['0 2 3','4 2 3','-4 2 3','1.5 2 3','1e0 2 3','1/ 2 3','1// 2 3','1/1 2 3','1//1 2 3','1/1/1/1 2 3'])
    rejects(basic.replace('f 1 2 3',`f ${f}`),/index|Malformed|range/i);
  rejects('f 1 2 3\nv 0 0 0\nv 1 0 0\nv 0 1 0',/Forward references/);
  rejects(basic.replace('v 0 0 0','v 0 0 0 2'),/w must be 1/);
  rejects(basic+'\0',/NUL/);
  rejects('v 0 0 0',/at least/);
  rejects('v 0 0 0\nv 1 0 0\nv 2 0 0\nf 1 2 3',/Degenerate/);
  rejects(basic.replace('f 1 2 3','f 1 2 1'),/repeats/);
  assert.throws(()=>prepareMesh({vertices:[[0,0,0],[1,0,0],[0,1,0]],faces:[[0,1,100]]}),/index triples/);
});

test('bounded OBJ parsing',()=>{
  rejects('#'+'x'.repeat(MESH_LIMITS.maxBytes),/2 MiB/);
  // Multi-byte input must be checked by UTF-8 bytes, not JS character count.
  rejects('#'+'界'.repeat(Math.ceil(MESH_LIMITS.maxBytes/3)),/2 MiB/);
  rejects('v 0 0 0\n'.repeat(MESH_LIMITS.maxVertices+1),/Vertex limit/);
  rejects(basic.split('\nf ')[0]+'\n'+'f 1 2 3\n'.repeat(MESH_LIMITS.maxTriangles+1),/Triangle limit/);
  rejects(basic.split('\nf ')[0]+'\nf '+'1 '.repeat(65),/3–64/);
});

test('external-load and executable directives never run',()=>{
  const obj='mtllib https://example.invalid/material.mtl\nusemtl shader\n'+basic;
  const parsed=parseOBJ(obj);
  assert(parsed.warnings.some(w=>w.includes('No external files')));
  for(const directive of ['call other.obj','csh echo x','include ./a.obj','curv 0 1 1 2 3'])
    rejects(basic+'\n'+directive,/Unsupported directive/);
});

test('convex polygons triangulate without changing area',()=>{
  const square='v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4';
  for(const obj of [square,square.replace('f 1 2 3 4','f 4 3 2 1')]) {
    const mesh=prepareMesh(obj);
    assert.equal(mesh.triangles.length,2);near(mesh.totalArea,1);
    assert.equal(mesh.boundaryEdges.length,4);assert.equal(mesh.internalEdges.length,1);
    assert.equal(mesh.diffractionEdges.length,4);
  }
});

test('convex polygon acceptance preserves authored offset',()=>{
  const polygon=Array.from([[0,0,0],[.001,0,0],[.001,.001,0],[0,.001,0]],v=>'v '+v.map(x=>x+10000).join(' ')).join('\n')+'\nf 1 2 3 4';
  const mesh=prepareMesh(polygon);
  near(mesh.totalArea,1e-6,1e-12);
  assert.deepEqual(mesh.vertices[0],[10000,10000,10000]);
});

test('ambiguous polygons rejected',()=>{
  rejects('v 0 0 0\nv 1 0 0\nv .4 .4 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4 5',/Concave/);
  rejects('v 0 0 0\nv 1 0 0\nv 1 1 .2\nv 0 1 0\nf 1 2 3 4',/Nonplanar/);
  rejects('v 0 0 0\nv .5 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4 5',/Collinear/);
  rejects('v 0 0 0\nv 1 1 0\nv 0 1 0\nv 1 0 0\nf 1 2 3 4',/Ambiguous|Self-intersecting/);
  const star=Array.from({length:5},(_,i)=>`v ${Math.cos(2*Math.PI*i/5)} ${Math.sin(2*Math.PI*i/5)} 0`).join('\n')+'\nf 1 3 5 2 4';
  rejects(star,/Self-intersecting/);
});

test('exact seam welding and subdivision preserve boundary',()=>{
  const coarse=prepareMesh('v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3\nf 1 3 4');
  const seam=prepareMesh('v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3\nf 4 5 6');
  const subdivided=prepareMesh('v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv .5 .5 0\nf 1 2 5\nf 2 3 5\nf 3 4 5\nf 4 1 5');
  for(const mesh of [coarse,seam,subdivided]) {
    near(mesh.totalArea,1);near(perimeter(mesh),4);
    assert.equal(mesh.boundaryEdges.length,4);assert.equal(mesh.diffractionEdges.length,4);
    assert.equal(mesh.creaseEdges.length,0);assert.equal(mesh.nonManifoldEdges.length,0);
  }
  assert.equal(seam.weldedVertexCount,4);
  assert.equal(subdivided.internalEdges.length,4);
  // Collinear boundary subdivision changes segment count, never physical length.
  const split=prepareMesh({vertices:[[0,0,0],[.5,0,0],[1,0,0],[1,1,0],[0,1,0],[.5,.5,0]],faces:[[0,1,5],[1,2,5],[2,3,5],[3,4,5],[4,0,5]]});
  near(perimeter(split),perimeter(coarse));near(split.totalArea,coarse.totalArea);
  assert.equal(split.diffractionEdges.length,5);
});

test('quantized welding leaves authored coordinates untouched',()=>{
  const input={vertices:[[0,0,0],[1,0,0],[1,1,0],[1e-8,0,0],[1+1e-8,1,0],[0,1,0]],faces:[[0,1,2],[3,4,5]]};
  const mesh=prepareMesh(input,{weldTolerance:1e-7});
  assert.equal(mesh.boundaryEdges.length,4);assert.equal(mesh.internalEdges.length,1);
  assert.equal(mesh.weldedVertexCount,4);
  assert.deepEqual(mesh.authoredVertices,input.vertices);
  assert.deepEqual(mesh.vertices,input.vertices);
  assert.throws(()=>prepareMesh(basic,{weldTolerance:1}),/collapses/);
});

test('non-manifold and duplicate surface detection',()=>{
  const input={vertices:[[0,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]],faces:[[0,1,2],[1,0,3],[0,1,4]]};
  assert.throws(()=>prepareMesh(input),/non-manifold/);
  const warned=prepareMesh(input,{rejectNonManifold:false});
  assert.equal(warned.nonManifoldEdges.length,1);
  assert(warned.warnings.some(w=>w.includes('non-manifold')));
  assert(!warned.diffractionEdges.some(e=>e.kind==='nonmanifold'));
  assert.throws(()=>prepareMesh(basic+'\nf 3 2 1'),/Duplicate/);
});

test('nonplanar folds produce one real crease',()=>{
  const mesh=prepareMesh(BUILTIN_OBJS.folded);
  assert.equal(mesh.triangles.length,4);assert.equal(mesh.creaseEdges.length,1);
  assert.equal(mesh.internalEdges.length,2);assert.equal(mesh.boundaryEdges.length,6);
  assert.equal(mesh.diffractionEdges.length,7);
  assert(mesh.vertices.some(v=>v[2]!==0));
  assert(mesh.triangles.some(t=>Math.abs(t.normal[0])>.1));
});

test('closed tetrahedron and inconsistent-winding report',()=>{
  const mesh=prepareMesh({vertices:[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],faces:[[0,2,1],[0,1,3],[0,3,2],[1,2,3]]});
  assert(mesh.isClosed);assert.equal(mesh.boundaryEdges.length,0);assert.equal(mesh.creaseEdges.length,6);
  assert.equal(mesh.warnings.length,0);
  const bad=prepareMesh('v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3\nf 4 3 1');
  assert(bad.warnings.some(w=>w.includes('inconsistent')));
  assert.equal(bad.creaseEdges.length,0);
});

test('absolute units, authored location, rotations and translations',()=>{
  const parsed=parseOBJ('v 10 20 30\nv 11 20 30\nv 10 21 30\nf 1 2 3');
  const mesh=prepareMesh(parsed,{unitScale:2,rotationDeg:[0,0,90],translation:[1,2,3]});
  assert.deepEqual(mesh.authoredVertices,parsed.vertices);
  vectorNear(mesh.vertices[0],[-39,22,63]);
  near(mesh.totalArea,2);vectorNear(mesh.triangles[0].normal,[0,0,1]);
  const changed=transformMesh(mesh,{rotationDeg:[0,0,0],translation:[0,0,0]});
  vectorNear(changed.vertices[0],[20,40,60]); // Absolute transform; no compounding.
  vectorNear(transformPoint([1,0,0],{rotationDeg:[0,90,0]}),[0,0,-1]);
  for(const rotationDeg of [[23,47,-11],[-90,180,45],[360,720,-360]]) {
    const rotated=transformMesh(mesh,{rotationDeg});
    near(rotated.totalArea,mesh.totalArea);
    near(perimeter(rotated),perimeter(mesh));
    for(let i=0;i<mesh.vertices.length;i++)for(let j=0;j<mesh.vertices.length;j++)near(distance(rotated.vertices[i],rotated.vertices[j]),distance(mesh.vertices[i],mesh.vertices[j]));
  }
});

test('finite transform guards',()=>{
  for(const options of [{unitScale:0},{unitScale:Infinity},{unitScale:-1},{unitScale:1e7},{rotationDeg:[0,NaN,0]},{rotationDeg:[0,0,1e9]},{translation:[0,Infinity,0]},{translation:[0,0,1e8]},{weldTolerance:-1},{creaseAngleDeg:91}])
    assert.throws(()=>prepareMesh(basic,options));
  assert.throws(()=>prepareMesh('v 1000000 0 0\nv 1000000 1 0\nv 1000000 0 1\nf 1 2 3',{unitScale:1e6}),/Transformed point/);
});

test('cone intersects triangle away from its centroid',()=>{
  // A centroid-only radius test misses the bottom-left vertex of this triangle.
  const mesh=prepareMesh('v 0 0 1\nv 10 0 1\nv 0 10 1\nf 1 2 3');
  const cone={origin:[0,0,0],direction:[0,0,1],radius0:.01,tanAngle:0,near:0,far:2};
  assert(length(sub(mesh.triangles[0].centroid,[0,0,1]))>1);
  assert(triangleIntersectsCone(mesh.triangles[0],cone));
  assert.equal(gatherConeTriangles(mesh,cone).length,1);
  assert.equal(gatherConeTriangles(mesh,{...cone,origin:[30,30,0]}).length,0);
  assert.equal(gatherConeTriangles(mesh,{...cone,near:10,far:11}).length,0);
  near(coneRadius(3,{sigma0:4,slope:1,tail:2}),10);
  near(coneRadius(3,{radius0:4,tanAngle:2}),10);
});

test('finite cone slab boundaries',()=>{
  const mesh=prepareMesh('v -.01 -.01 2\nv .01 -.01 2\nv 0 .01 2\nf 1 2 3');
  const c={origin:[0,0,0],direction:[0,0,3],radius0:.1,near:2,far:2};
  assert.equal(gatherConeTriangles(mesh,c).length,1);
  assert.equal(gatherConeTriangles(mesh,{...c,near:2.1,far:3}).length,0);
  assert.throws(()=>gatherConeTriangles(mesh,{...c,direction:[0,0,0]}),/nonzero/);
  assert.throws(()=>gatherConeTriangles(mesh,{...c,sigma0:NaN}),/finite/);
  assert.throws(()=>gatherConeTriangles(mesh,{...c,near:3,far:1}),/near/);
});

let seed=1729;
const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
let testedSupportPoints=0;
test('random conservative gather contains every sampled in-cone surface point',()=>{
  // Independent direct point test, including oblique cones and triangles with
  // vertices outside the cone. It tests a property, not an implementation clone.
  for(let sample=0;sample<500;sample++) {
    const points=Array.from({length:3},()=>[random()*10-5,random()*10-5,random()*10]);
    const mesh=prepareMesh({vertices:points,faces:[[0,1,2]]});
    const cone={origin:[random()-.5,random()-.5,random()-.5],direction:normalize([random()-.5,random()-.5,1]),sigma0:.3+random(),slope:random()*.5,tail:2,near:0,far:8};
    const gathered=gatherConeTriangles(mesh,cone).length===1;
    for(let i=0;i<=10;i++)for(let j=0;j<=10-i;j++) {
      const u=i/10,v=j/10,point=add(add(scale(points[0],1-u-v),scale(points[1],u)),scale(points[2],v));
      const delta=sub(point,cone.origin),axial=dot(delta,cone.direction),radial=length(sub(delta,scale(cone.direction,axial)));
      if(axial>=cone.near&&axial<=cone.far&&radial<=cone.tail*Math.hypot(cone.sigma0,cone.slope*axial)) {
        assert(gathered,`Missed support point on trial ${sample}`);testedSupportPoints++;
      }
    }
  }
  assert(testedSupportPoints>1000);
});

test('cone gather is rigid-transform invariant',()=>{
  const mesh=prepareMesh(BUILTIN_OBJS.folded);
  const cone={origin:[.03,0,-1],direction:normalize([.03,0,1]),sigma0:.03,slope:.02,tail:2,near:0,far:3};
  const transform={rotationDeg:[42,-37,19],translation:[2,3,-7]};
  const rotated=transformMesh(mesh,transform);
  const rotatedCone={...cone,origin:transformPoint(cone.origin,transform),direction:transformPoint(cone.direction,{rotationDeg:transform.rotationDeg})};
  assert.deepEqual(gatherConeTriangles(mesh,cone).map(t=>t.id),gatherConeTriangles(rotated,rotatedCone).map(t=>t.id));
});

test('ray hit distances, edges, backface and finite segment occlusion',()=>{
  const mesh=prepareMesh(basic);
  const hit=rayTriangleIntersect([.2,.2,2],[0,0,-5],mesh.triangles[0]);
  near(hit.distance,2);vectorNear(hit.point,[.2,.2,0]);vectorNear(hit.barycentric,[.6,.2,.2]);
  assert.equal(rayTriangleIntersect([2,2,2],[0,0,-1],mesh.triangles[0]),null);
  assert(rayTriangleIntersect([0,0,2],[0,0,-1],mesh.triangles[0]));
  assert(rayOccluded(mesh,[.2,.2,2],[0,0,-1],{maxDistance:3}));
  assert(!rayOccluded(mesh,[.2,.2,2],[0,0,-1],{maxDistance:1}));
  assert(!rayOccluded(mesh,[.2,.2,2],[0,0,-1],{ignoreTriangle:0}));
  assert.equal(rayTriangleIntersect([.2,.2,-2],[0,0,1],mesh.triangles[0],{cullBackface:true}),null);
  assert(rayTriangleIntersect([.2,.2,-2],[0,0,1],mesh.triangles[0]));
});

test('preset dimensions and validation',()=>{
  const slits=prepareMesh(makePresetOBJ('double-slit',{width:.1,separation:.4,height:.2}));
  near(slits.totalArea,.04);near(slits.bounds.size[0],.5);near(slits.bounds.size[1],.2);
  assert.equal(slits.triangles.length,4);assert.equal(slits.boundaryEdges.length,8);
  assert.throws(()=>makePresetOBJ('double-slit',{width:.4,separation:.3}),/smaller/);
  assert.throws(()=>makePresetOBJ('square',{width:NaN}),/positive/);
  assert.throws(()=>makePresetOBJ('unknown'),/Unknown/);
});

test('finite 3D Fermat point has stationary optical length',()=>{
  const edge={a:[0,0,-2],b:[0,0,2],v:[0,0,0],l:4,e:[0,0,1],triangleIds:[]};
  const source=[3,4,-1],destination=[0,-2,1];
  const hit=diffractionPoint(edge,source,destination);
  assert(hit);vectorNear(hit.point,[0,0,3/7]);near(hit.fermatResidual,0,1e-13);
  near(dot(hit.wi,edge.e),-dot(hit.wo,edge.e),1e-13);
  for(const offset of [-.1,-.001,.001,.1]) {
    const p=add(hit.point,scale(edge.e,offset));
    assert(distance(source,p)+distance(p,destination)>hit.pathLength);
  }
  const swap=diffractionPoint(edge,destination,source);
  vectorNear(swap.point,hit.point);near(swap.pathLength,hit.pathLength);
  assert.equal(diffractionPoint(edge,[1,0,5],[-1,0,5]),null);
  assert.equal(diffractionPoint(edge,[0,0,-1],[0,0,1]),null);
  assert.equal(diffractionPoint(edge,[0,0,0],[1,0,0]),null);
  const endpoint=diffractionPoint(edge,[1,0,2],[-1,0,2]);
  assert(endpoint.atEndpoint);vectorNear(endpoint.point,[0,0,2]);
});

test('Fermat points and length are rigid-transform invariant',()=>{
  const a=[-1,2,3],b=[4,-2,6],src=[-2,1,-1],dst=[5,5,10];
  const edge={a,b},hit=diffractionPoint(edge,src,dst);
  assert(hit);
  const options={rotationDeg:[27,-19,37],translation:[15,-23,2]};
  const tr=p=>transformPoint(p,options);
  const transformed=diffractionPoint({a:tr(a),b:tr(b)},tr(src),tr(dst));
  vectorNear(transformed.point,tr(hit.point));near(transformed.pathLength,hit.pathLength);
  near(transformed.fermatResidual,0,1e-13);
});

test('visibility excludes adjacent faces only on edge legs',()=>{
  const mesh=prepareMesh('v -2 -2 0\nv 2 -2 0\nv 0 2 0\nf 1 2 3');
  assert(!segmentVisible(mesh,[0,0,-1],[0,0,1]));
  assert(segmentVisible(mesh,[0,0,-1],[0,0,1],{ignoreTriangleIds:[0]}));
  assert(segmentVisible(mesh,[3,0,-1],[3,0,1]));
  const visibility=diffractionVisibility(mesh,{triangleIds:[0]},[0,0,-1],[0,0,1],[0,0,0]);
  assert.deepEqual(visibility,{directVisible:false,sourceVisible:true,destinationVisible:true,visible:true});
  const blocker=prepareMesh('v -2 -2 -.5\nv 2 -2 -.5\nv 0 2 -.5\nf 1 2 3');
  const blocked=diffractionVisibility(blocker,{triangleIds:[]},[0,0,-1],[0,0,1],[0,0,0]);
  assert(!blocked.sourceVisible);assert(blocked.destinationVisible);assert(!blocked.visible);
});

function prism(polygon,topTriangles) {
  const n=polygon.length,vertices=[...polygon.map(p=>[...p,0]),...polygon.map(p=>[...p,1])];
  const faces=[];
  for(const f of topTriangles)faces.push([...f].reverse(),f.map(i=>i+n));
  for(let i=0;i<n;i++){const j=(i+1)%n;faces.push([i,j,j+n],[i,j+n,i+n]);}
  return {vertices,faces};
}
const boxInput=prism([[0,0],[1,0],[1,1],[0,1]],[[0,1,2],[0,2,3]]);
const tetraInput={vertices:[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],faces:[[0,2,1],[0,1,3],[0,3,2],[1,2,3]]};

test('closed box yields twelve actual right-angle wedges',()=>{
  const mesh=prepareMesh(boxInput),result=extractWedges(mesh,{surface:'closed-solid'});
  assert(result.supported,JSON.stringify(result.unsupported));
  assert.equal(result.wedges.length,12);assert.equal(result.unsupported.length,0);
  near(result.components[0].signedVolume,1);
  for(const w of result.wedges) {
    near(w.alpha,Math.PI/2);near(w.exteriorAngle,3*Math.PI/2);
    near(dot(w.n1,w.t1),0);near(dot(w.n2,w.t2),0);
    near(length(w.n1),1);near(length(w.t1),1);
    const oriented=orientWedge(w,w.n1);
    assert(oriented);near(dot(oriented.e,oriented.nff),0);near(dot(oriented.e,oriented.tff),0);
    vectorNear(oriented.e,cross(oriented.nff,oriented.tff));
    assert(dot(oriented.tff,sub(mesh.triangles[w.triangleIds[0]].centroid,w.v))>0);
    assert.equal(orientWedge(w,normalize(scale(add(w.n1,w.n2),-1))),null);
  }
});

test('closed tetrahedron wedges use actual face angles',()=>{
  const result=extractWedges(prepareMesh(tetraInput));
  assert(result.supported);assert.equal(result.wedges.length,6);
  near(result.components[0].signedVolume,1/6);
  const angles=result.wedges.map(w=>w.alpha).sort((a,b)=>a-b);
  for(let i=0;i<3;i++)near(angles[i],Math.acos(1/Math.sqrt(3)));
  for(let i=3;i<6;i++)near(angles[i],Math.PI/2);
});

test('opaque wedge gates reject open, inward, inconsistent and folded surfaces',()=>{
  const plate=prepareMesh(BUILTIN_OBJS['finite-square']);
  assert(extractWedges(plate).unsupported.some(e=>e.code==='open-solid'));
  const inverted=prepareMesh({...tetraInput,faces:tetraInput.faces.map(f=>[...f].reverse())});
  const inward=extractWedges(inverted);
  assert(!inward.supported);assert.equal(inward.wedges.length,0);
  assert(inward.unsupported.some(e=>e.code==='inward-winding'));
  const inconsistent=prepareMesh({...tetraInput,faces:tetraInput.faces.map((f,i)=>i===0?[...f].reverse():f)});
  assert(extractWedges(inconsistent).unsupported.some(e=>e.code==='inconsistent-winding'));
  const folded=extractWedges(prepareMesh(BUILTIN_OBJS.folded),{surface:'open-plate'});
  assert(!folded.supported);assert.equal(folded.wedges.length,0);
  assert(folded.unsupported.some(e=>e.code==='nonplanar-open-plate'));
  assert(!extractWedges(prepareMesh(tetraInput),{surface:'open-plate'}).supported);
});

test('each closed connected component needs outward winding',()=>{
  const input={vertices:[...tetraInput.vertices,...tetraInput.vertices.map(v=>add(v,[3,0,0]))],faces:[...tetraInput.faces,...tetraInput.faces.map(f=>[...f].reverse().map(i=>i+4))]};
  const result=extractWedges(prepareMesh(input));
  assert.equal(result.components.length,2);assert(!result.supported);
  assert(result.unsupported.some(e=>e.code==='inward-winding'&&e.componentId===1));
});

test('concave solid is explicitly unsupported, never reoriented as convex',()=>{
  const input=prism([[0,0],[2,0],[2,1],[1,1],[1,2],[0,2]],[[0,1,3],[1,2,3],[0,3,5],[3,4,5]]);
  const result=extractWedges(prepareMesh(input));
  near(result.components[0].signedVolume,3);
  assert(!result.supported);assert.equal(result.wedges.length,0);
  assert(result.unsupported.some(e=>e.code==='concave-edge'));
});

test('open planar plate frame has zero interior angle and two physical sides',()=>{
  const result=extractWedges(prepareMesh(BUILTIN_OBJS['double-slit']),{surface:'open-plate'});
  assert(result.supported);assert.equal(result.components.length,2);assert.equal(result.wedges.length,8);
  for(const w of result.wedges) {
    assert.equal(w.alpha,0);vectorNear(w.n2,scale(w.n1,-1));vectorNear(w.t2,w.t1);
    const front=orientWedge(w,w.n1),back=orientWedge(w,w.n2);
    assert(front&&back);vectorNear(front.e,scale(back.e,-1));
    vectorNear(front.nff,w.n1);vectorNear(back.nff,w.n2);
  }
});

test('wedge frames and interior angles are rigid-transform invariant',()=>{
  const mesh=prepareMesh(boxInput),before=extractWedges(mesh);
  const options={rotationDeg:[31,58,-43],translation:[5,-17,4]},after=extractWedges(transformMesh(mesh,options));
  assert(after.supported);assert.equal(after.wedges.length,before.wedges.length);
  for(let i=0;i<before.wedges.length;i++) {
    const a=before.wedges[i],b=after.wedges[i];
    near(a.alpha,b.alpha);near(a.l,b.l);
    vectorNear(b.v,transformPoint(a.v,options));
    vectorNear(b.n1,transformPoint(a.n1,{rotationDeg:options.rotationDeg}));
    vectorNear(b.t1,transformPoint(a.t1,{rotationDeg:options.rotationDeg}));
  }
  near(before.components[0].signedVolume,after.components[0].signedVolume);
});

test('plane-wave finite-edge stationarity and finite-source limit',()=>{
  const edge={a:[0,0,-4],b:[0,0,4]},di=normalize([1,0,.5]),dst=[3,2,1];
  const hit=diffractionPointPlane(edge,di,dst);
  assert(hit);near(hit.fermatResidual,0,1e-13);
  vectorNear(hit.wi,scale(di,-1));near(dot(hit.wo,hit.e),dot(di,hit.e));
  for(const offset of [-.1,-.001,.001,.1]) {
    const p=add(hit.point,scale(hit.e,offset));
    assert(dot(di,p)+distance(p,dst)>hit.opticalPath);
  }
  const far=diffractionPoint(edge,scale(di,-1e8),dst);
  vectorNear(far.point,hit.point,1e-7);
  assert.equal(diffractionPointPlane(edge,[0,0,1],dst),null);
  assert.equal(diffractionPointPlane(edge,[1,0,0],[3,0,5]),null);
  const endpoint=diffractionPointPlane(edge,[1,0,0],[3,0,4]);
  assert(endpoint.atEndpoint);vectorNear(endpoint.point,[0,0,4]);
});

test('plane-wave point and visibility rotate with complete geometry',()=>{
  const mesh=prepareMesh(boxInput),edge=extractWedges(mesh).wedges[0],di=normalize([.2,.1,-1]),dst=[3,3,-3];
  const hit=diffractionPointPlane(edge,di,dst);
  // Use a long oblique edge to guarantee an interior point for invariance.
  const line={a:[-2,1,-5],b:[2,-1,5]},point=diffractionPointPlane(line,di,dst);
  assert(point);
  const options={rotationDeg:[29,-41,13],translation:[2,-5,8]},tr=p=>transformPoint(p,options),rd=d=>transformPoint(d,{rotationDeg:options.rotationDeg});
  const rotated=diffractionPointPlane({a:tr(line.a),b:tr(line.b)},rd(di),tr(dst));
  vectorNear(rotated.point,tr(point.point));near(rotated.ro,point.ro);near(rotated.fermatResidual,0,1e-13);
  const first=planeDiffractionVisibility(mesh,edge,di,dst,hit);
  const changedMesh=transformMesh(mesh,options),changedEdge=extractWedges(changedMesh).wedges[0];
  const changedHit=hit?{point:tr(hit.point)}:undefined;
  const second=planeDiffractionVisibility(changedMesh,changedEdge,rd(di),tr(dst),changedHit);
  assert.deepEqual(first,second);
});

test('plane source visibility sees blockers toward infinity',()=>{
  const mesh=prepareMesh('v -2 -2 -.5\nv 2 -2 -.5\nv 0 2 -.5\nf 1 2 3');
  const result=planeDiffractionVisibility(mesh,{triangleIds:[]},[0,0,1],[0,0,1],[0,0,0]);
  assert.deepEqual(result,{directVisible:false,sourceVisible:false,destinationVisible:true,visible:false});
  const ignored=planeDiffractionVisibility(mesh,{triangleIds:[0]},[0,0,1],[0,0,1],[0,0,0]);
  assert.deepEqual(ignored,{directVisible:false,sourceVisible:true,destinationVisible:true,visible:true});
});

test('opaque 3D presets are closed and outward; aperture preset stays unchanged',()=>{
  for(const name of ['box','tetrahedron','triangular-prism','wedge']) {
    const mesh=prepareMesh(makePresetOBJ(name)),result=extractWedges(mesh);
    assert(mesh.isClosed,name);assert(result.supported,name+JSON.stringify(result.unsupported));
    assert(result.wedges.length>=6);assert(result.components[0].signedVolume>0);
    vectorNear(mesh.bounds.size,[2,2,2]);
  }
  const screen=prepareMesh(makePresetOBJ('screen'));
  vectorNear(screen.bounds.size,[2,2,0]);assert(extractWedges(screen,{surface:'open-plate'}).supported);
  const apertures=prepareMesh(makePresetOBJ('double-slit'));
  near(apertures.totalArea,2*.08*.18);
});

test('opaque double-slit complement has genuine gaps and manifold grid topology',()=>{
  const mesh=prepareMesh(makePresetOBJ('opaque-double-slit'));
  const result=extractWedges(mesh,{surface:'open-plate'});
  assert(result.supported,JSON.stringify(result.unsupported));
  assert.equal(result.components.length,1);assert.equal(result.wedges.length,12);
  near(mesh.totalArea,4-2*.08*.18);assert.equal(mesh.creaseEdges.length,0);
  for(const x of [-.15,.15])assert(segmentVisible(mesh,[x,0,-1],[x,0,1]));
  for(const x of [0,-.5,.5])assert(!segmentVisible(mesh,[x,0,-1],[x,0,1]));
  assert.throws(()=>makePresetOBJ('opaque-double-slit',{wallWidth:.2}),/inside/);
});

test('collinear wedge merging restores exact tessellation-independent physical edges',()=>{
  const n=8,vertices=[],faces=[];
  for(let y=0;y<=n;y++)for(let x=0;x<=n;x++)vertices.push([x/n,y/n,0]);
  for(let y=0;y<n;y++)for(let x=0;x<n;x++) {
    const a=y*(n+1)+x;faces.push([a,a+1,a+n+2],[a,a+n+2,a+n+1]);
  }
  const mesh=prepareMesh({vertices,faces});
  const raw=extractWedges(mesh,{surface:'open-plate',mergeCollinear:false});
  const merged=extractWedges(mesh,{surface:'open-plate'});
  assert.equal(raw.wedges.length,4*n);assert.equal(merged.wedges.length,4);
  const rawFaces=new Set(raw.wedges.flatMap(w=>w.triangleIds)),mergedFaces=new Set(merged.wedges.flatMap(w=>w.triangleIds));
  assert.deepEqual(mergedFaces,rawFaces);
  for(const w of merged.wedges){near(w.l,1);assert.equal(w.segmentIds.length,n);}
  // The point at y=1/2 used to fall on two collinear segment endpoints. It must
  // now occur once on that physical side, away from any true plate corner.
  const candidates=merged.wedges.map(w=>diffractionPointPlane(w,[0,0,1],[0,.5,1])).filter(Boolean);
  assert.equal(candidates.filter(p=>Math.abs(p.point[0])<1e-12&&Math.abs(p.point[1]-.5)<1e-12).length,1);
  const transformed=extractWedges(transformMesh(mesh,{rotationDeg:[13,-29,47],translation:[5,-2,1]}),{surface:'open-plate'});
  assert.equal(transformed.wedges.length,4);for(const w of transformed.wedges)near(w.l,1);
});

test('closed-solid collinear merge preserves both adjacent face groups',()=>{
  const base=prepareMesh(makePresetOBJ('box')),vertices=[],faces=[];
  for(const triangle of base.triangles) {
    const p=[triangle.a,triangle.b,triangle.c,scale(add(triangle.a,triangle.b),.5),scale(add(triangle.b,triangle.c),.5),scale(add(triangle.c,triangle.a),.5)];
    const offset=vertices.length;vertices.push(...p);
    faces.push(...[[0,3,5],[3,1,4],[5,4,2],[3,4,5]].map(f=>f.map(i=>i+offset)));
  }
  const mesh=prepareMesh({vertices,faces}),raw=extractWedges(mesh,{mergeCollinear:false}),merged=extractWedges(mesh);
  assert(merged.supported);assert.equal(raw.wedges.length,24);assert.equal(merged.wedges.length,12);
  for(const wedge of merged.wedges) {
    near(wedge.l,2);near(wedge.alpha,Math.PI/2);
    assert.equal(wedge.triangleIds.length,4);
    assert.equal(wedge.faceTriangleIds[0].length,2);assert.equal(wedge.faceTriangleIds[1].length,2);
    for(const direction of [wedge.n1,wedge.n2]) {
      const oriented=orientWedge(wedge,direction);
      vectorNear(mesh.triangles[oriented.frontTriangleId].normal,oriented.nff);
    }
  }
});

test('unused OBJ vertices do not move physical bounds or scene alignment',()=>{
  const base=prepareMesh(makePresetOBJ('screen'));
  const extra=prepareMesh({vertices:[...base.authoredVertices,[100000,100000,100000]],faces:base.authoredFaces});
  assert.deepEqual(extra.bounds,base.bounds);assert.equal(extra.vertices.length,base.vertices.length+1);
});
console.log(JSON.stringify({suite:'mesh-geometry',checks,testedSupportPoints,parserLimits:MESH_LIMITS,topologyInvariant:true,rigidTransforms:true,conservativeGather:true}));
