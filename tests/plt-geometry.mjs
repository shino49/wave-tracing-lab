import assert from 'node:assert/strict';
import {prepareMesh,makePresetOBJ,transformMesh,transformPoint,add,sub,scale,normalize,distance,dot,length,rayTriangleIntersect} from '../dist/mesh-geometry.js';
import {preparePLTScene,rayNearest,rayNearestBrute,rayOccluded,segmentVisible,gatherConeCandidates,traceConeContacts,firstConeContact,collectRegionGeometry,clipTriangleToRegion,regionContainsPoint,coneContains} from '../dist/plt-geometry.js';
let checks=0;
const test=(name,fn)=>{try{fn();checks++;}catch(error){error.message=`${name}: ${error.message}`;throw error;}};
const near=(a,b,eps=1e-9)=>assert(Math.abs(a-b)<=eps*Math.max(1,Math.abs(a),Math.abs(b)),`${a} != ${b}`);
const vn=(a,b,eps)=>{assert(a&&b);a.forEach((v,i)=>near(v,b[i],eps));};
const plate=(z=0,x=0,width=2)=>prepareMesh(makePresetOBJ('screen',{width,height:2}),{translation:[x,0,z]});
let seed=930821;
const random=()=>((seed=Math.imul(seed,1664525)+1013904223>>>0)/4294967296);
const brute=(s,o,d,options={})=>{
 let best=null;
 for(const t of s.triangles){const h=rayTriangleIntersect(o,d,t,options);if(h&&(!best||h.distance<best.distance||h.distance===best.distance&&h.triangleId<best.triangleId))best=h;}
 return best;
};

test('multiple disconnected objects retain true geometry/materials without convex gate',()=>{
 const scene=preparePLTScene([{mesh:plate(0,-3),materialIndex:2},{mesh:plate(3,3),materialIndex:5},{mesh:makePresetOBJ('box'),materialIndex:1}]);
 assert.equal(scene.triangles.length,16);assert.equal(scene.components.length,3);assert.equal(scene.wedges.length,20);
 assert.deepEqual(new Set(scene.triangles.map(t=>t.materialIndex)),new Set([2,5,1]));
 assert.equal(scene.unsupportedEdges.length,0);assert(scene.bounds.size[2]>=4);
 const a=rayNearest(scene,[-3,0,-4],[0,0,1]),b=rayNearest(scene,[3,0,-4],[0,0,1]);
 near(a.distance,4);near(b.distance,7);assert.equal(a.materialIndex,2);assert.equal(b.materialIndex,5);
 assert.equal(a.meshId,0);assert.equal(b.meshId,1);assert.equal(a.triangle.id,a.triangleId);
 assert.equal(a.frontFace,false);vn(a.geometricNormal,[0,0,1]);vn(a.normal,[0,0,-1]);
});

test('imported triangle material overrides and explicit material bounds',()=>{
 const m=plate();m.triangles[0].materialIndex=3;m.triangles[1].materialIndex=4;
 let s=preparePLTScene(m);assert.deepEqual(s.triangles.map(t=>t.materialIndex),[3,4]);
 s=preparePLTScene({mesh:m,materialIndices:[7,8]});assert.deepEqual(s.triangles.map(t=>t.materialIndex),[7,8]);
 assert.throws(()=>preparePLTScene({mesh:m,materialIndex:-1}),/nonnegative/);
 assert.throws(()=>preparePLTScene({mesh:m,materialIndices:[1]}),/one index/);
 assert.throws(()=>preparePLTScene(m,{materials:[{}]}),/outside/);
});

test('BVH nearest and occlusion equal independent brute force for 2500 3-D rays',()=>{
 const objects=[];
 for(let i=0;i<12;i++)objects.push({mesh:prepareMesh(makePresetOBJ(i%2?'tetrahedron':'box'),{
  rotationDeg:[random()*360,random()*360,random()*360],translation:[(random()-.5)*15,(random()-.5)*15,(random()-.5)*15]}),materialIndex:i});
 const s=preparePLTScene(objects,{leafSize:3});assert(s.bvh.nodes.length>20);
 for(let i=0;i<2500;i++){
  const origin=Array.from({length:3},()=>20*(random()-.5)),direction=Array.from({length:3},()=>random()-.5);
  const options={minDistance:i%3===0?.1:0,maxDistance:i%5===0?5:Infinity};
  const a=rayNearest(s,origin,direction,options),b=brute(s,origin,direction,options),c=rayNearestBrute(s,origin,direction,options);
  assert.equal(!!a,!!b);assert.equal(!!a,!!c);assert.equal(rayOccluded(s,origin,direction,options),!!b);
  if(a){near(a.distance,b.distance,1e-11);vn(a.point,b.point,1e-11);assert.equal(a.triangleId,b.triangleId);}
 }
 const stats={};rayNearest(s,[0,0,-20],[0,0,1],{stats});assert(stats.nodes>0);assert((stats.triangles??0)<s.triangles.length);
});

test('ray normalized distance, intervals, sidedness, signed zero and edge contact',()=>{
 const s=preparePLTScene([plate(0),plate(3)]),o=[0,0,-1];
 near(rayNearest(s,o,[0,0,7]).distance,1);near(rayNearest(s,o,[-0,0,1]).distance,1);
 near(rayNearest(s,o,[0,0,1],{minDistance:1.00001}).distance,4);
 assert.equal(rayNearest(s,o,[0,0,1],{maxDistance:.999}),null);
 near(rayNearest(s,o,[0,0,1],{near:1,far:1}).distance,1);
 assert.equal(rayNearest(s,o,[0,0,1],{cullBackface:true}),null);
 assert(rayNearest(s,[0,0,4],[0,0,-1],{cullBackface:true}));
 assert(rayOccluded(s,[1,1,-1],[0,0,1]));assert(!rayOccluded(s,[2,2,-1],[0,0,1]));
});

test('source visibility includes all opaque objects and never ignores an old face',()=>{
 const s=preparePLTScene([plate(0),plate(3)]);
 assert(!segmentVisible(s,[0,0,-3],[0,0,6]));
 assert(segmentVisible(s,[0,0,0],[0,0,2]));
 assert(!segmentVisible(s,[0,0,0],[0,0,6]));
 assert(!segmentVisible(s,[0,0,-3],[0,0,2],{ignoreTriangleIds:[0,1]}));
 assert(segmentVisible(s,[3,0,-3],[3,0,6]));
 assert(segmentVisible(s,[0,0,0],[0,0,0]));
 // Starting at a box surface must still hit its far wall after reflection/refraction.
 const box=preparePLTScene(makePresetOBJ('box'));
 near(rayNearest(box,[0,0,-1],[0,0,1],{minDistance:1e-7,ignoreTriangleIds:[0,1]}).distance,2);
});

test('conservative cone broad phase followed by actual corner contact',()=>{
 const m=prepareMesh({vertices:[[.9,0,2],[10,0,2],[.9,10,2]],faces:[[0,1,2]]}),s=preparePLTScene(m);
 const cone={origin:[0,0,0],direction:[0,0,3],radius0:1,tanAngle:0,near:0,far:5};
 assert.equal(gatherConeCandidates(s,cone).length,1);const h=firstConeContact(s,cone);assert(h);near(h.distance,2);
 assert(coneContains(cone,h.point));assert.equal(h.triangleId,0);
 assert.equal(firstConeContact(s,{...cone,radius0:.8}),null);
 assert.equal(firstConeContact(s,{...cone,far:1.9}),null);
 assert.equal(firstConeContact(s,{...cone,near:2.1}),null);
});

test('real cone contact rejects sphere false positives and finds disjoint screens',()=>{
 const off=prepareMesh({vertices:[[-5,-2,-11],[-5,2,-9],[-5,-2,-9]],faces:[[0,1,2]]});
 const offScene=preparePLTScene(off),cone={origin:[0,0,-20],direction:[0,0,1],radius0:3,tanAngle:.012,near:0,far:30};
 assert.equal(firstConeContact(offScene,cone),null);
 const s=preparePLTScene([plate(2),plate(5,2),plate(8,0)]),c={origin:[0,0,0],direction:[0,0,1],radius0:.2,tanAngle:.3,near:0,far:10};
 const hits=traceConeContacts(s,c);assert(hits.some(h=>h.meshId===0));assert(hits.some(h=>h.meshId===1));assert(hits.some(h=>h.meshId===2));
 near(hits[0].distance,2);assert(hits.every(h=>coneContains(c,h.point,1e-8)));
 const later=firstConeContact(s,{...c,near:2.01});near(later.distance,5);
});

test('cone BVH candidates never miss any exact narrow-phase contacts',()=>{
 const objects=[];for(let i=0;i<10;i++)objects.push(prepareMesh(makePresetOBJ('box'),{rotationDeg:[i*13,i*7,i*17],translation:[i-5,Math.sin(i)*3,i*1.5]}));
 const s=preparePLTScene(objects,{leafSize:2});
 for(let i=0;i<100;i++){
  const c={origin:[(random()-.5)*10,(random()-.5)*6,-4],direction:normalize([random()-.5,random()-.5,1]),radius0:random(),tanAngle:.1*random(),near:random(),far:30};
  const candidates=new Set(gatherConeCandidates(s,c).map(t=>t.id));
  // A ray-triangle hit on the axis is necessarily a cone contact.
  for(const t of s.triangles)if(rayTriangleIntersect(c.origin,c.direction,t,{minDistance:c.near,maxDistance:c.far}))assert(candidates.has(t.id));
  const contacts=traceConeContacts(s,c),first=firstConeContact(s,c);
  assert(contacts.every(h=>candidates.has(h.triangleId)&&coneContains(c,h.point,1e-7)));
  assert.equal(!!first,!!contacts.length);if(first){near(first.distance,contacts[0].distance,1e-9);assert.equal(first.triangleId,contacts[0].triangleId);}
 }
});

test('exact triangle-region test sees small corner and disk fully inside huge triangle',()=>{
 const r={center:[0,0,0],direction:[0,0,1],radius:1,depth:2};
 const s=preparePLTScene({vertices:[[.9,0,0],[100,0,0],[.9,100,0],[-100,-100,.5],[100,-100,.5],[0,100,.5]],faces:[[0,1,2],[3,4,5]]});
 const g=collectRegionGeometry(s,r);assert.deepEqual(g.triangleIds,[0,1]);
 const corner=g.clippedTriangles[0],disk=g.clippedTriangles[1];
 assert(corner.boundaryPoints.length>=3);assert(corner.clipped);
 assert.equal(disk.boundaryPoints.length,0);assert(disk.metricDisk.radius>0);assert(disk.contains([0,0,.5]));assert(!disk.contains([2,0,.5]));
 for(const clip of g.clippedTriangles){assert(regionContainsPoint(r,clip.closestPoint));for(const p of clip.boundaryPoints)assert(regionContainsPoint(r,p,1e-7));}
});

test('region slab selects actual triangle intersections and clips oblique physical edges',()=>{
 const r={center:[0,0,0],direction:[0,0,1],radius:2,depth:3};
 const s=preparePLTScene({vertices:[[-8,1,-6],[8,1,6],[8,2,6]],faces:[[0,1,2]]});
 const g=collectRegionGeometry(s,r);assert.equal(g.triangles.length,1);assert(g.wedges.length>=1);
 for(const w of g.wedges){assert(w.originalA&&w.originalB);for(const p of [w.a,w.b])assert(regionContainsPoint(r,p,1e-7));near(distance(w.a,w.b),w.l);}
 const behind=preparePLTScene(plate(-1));assert.equal(collectRegionGeometry(behind,r).triangles.length,0);
 const all=collectRegionGeometry(behind,{...r,depth:Infinity});assert.equal(all.triangles.length,2);
});

test('zero-depth region and tangent triangle contacts are finite and rigid-invariant',()=>{
 const s=preparePLTScene({vertices:[[-3,0,-1],[3,0,1],[-3,2,-1]],faces:[[0,1,2]]});
 const r={center:[0,0,0],direction:[0,0,1],radius:1,depth:0};
 const g=collectRegionGeometry(s,r);assert.equal(g.triangles.length,1);assert(g.clippedTriangles[0].closestPoint.every(Number.isFinite));
 const tr={rotationDeg:[33,-18,71],translation:[30,-50,10]},p=v=>transformPoint(v,tr),d=v=>transformPoint(v,{rotationDeg:tr.rotationDeg});
 const transformed=preparePLTScene({vertices:s.meshes[0].mesh.vertices.map(p),faces:[[0,1,2]]});
 const other=collectRegionGeometry(transformed,{...r,center:p(r.center),direction:d(r.direction)});
 assert.equal(other.triangles.length,1);vn(p(g.clippedTriangles[0].closestPoint),other.clippedTriangles[0].closestPoint,1e-8);
 const touch=preparePLTScene({vertices:[[1,0,0],[2,0,0],[1,1,0]],faces:[[0,1,2]]});
 const clip=clipTriangleToRegion(touch.triangles[0],r);assert(clip);vn(clip.closestPoint,[1,0,0]);
});

test('region includes disjoint patches, excludes misses and never joins separate facets',()=>{
 const s=preparePLTScene([plate(0,-1.2,1),plate(.4,1.2,1),plate(5,0,1),plate(.5,10,1)]);
 const g=collectRegionGeometry(s,{center:[0,0,0],direction:[0,0,1],radius:2,depth:3});
 assert.deepEqual(new Set(g.triangles.map(t=>t.meshId)),new Set([0,1]));assert.equal(g.facets.length,2);
 assert(g.facets.every(f=>new Set(f.triangles.map(t=>t.meshId)).size===1));
 assert(g.wedges.every(w=>w.meshId<2));
});

test('boundary and solid wedge extraction retain endpoints, normals and native n',()=>{
 const s=preparePLTScene([plate(),{mesh:prepareMesh(makePresetOBJ('box'),{translation:[4,0,0]}),surface:'closed-solid'}]);
 assert.equal(s.wedges.filter(w=>w.kind==='boundary').length,4);assert.equal(s.wedges.filter(w=>w.kind==='crease').length,12);
 for(const w of s.wedges){near(w.n,2-w.alpha/Math.PI);near(w.l,distance(w.a,w.b));near(dot(w.n1,w.e),0);near(dot(w.n2,w.e),0);assert(w.triangleIds.every(id=>s.triangles[id]));}
 assert(s.wedges.filter(w=>w.kind==='boundary').every(w=>w.n===2));
 assert(s.wedges.filter(w=>w.kind==='crease').every(w=>Math.abs(w.n-1.5)<1e-10));
});

function concavePrism() {
 const outline=[[0,0],[2,0],[2,1],[1,1],[1,2],[0,2]],vertices=[];
 for(const z of [-1,1])for(const p of outline)vertices.push([...p,z]);
 const top=[[0,1,3],[1,2,3],[0,3,5],[3,4,5]],faces=[];
 for(const f of top){faces.push([...f].reverse());faces.push(f.map(i=>i+6));}
 for(let i=0;i<6;i++){const j=(i+1)%6;faces.push([i,j,j+6],[i,j+6,i+6]);}
 return {vertices,faces};
}

test('nonconvex solid keeps opaque triangles and local convex wedges',()=>{
 const s=preparePLTScene(concavePrism());assert.equal(s.triangles.length,20);
 assert(s.unsupportedEdges.some(e=>e.code==='concave-edge'));assert(s.wedges.length>=10);
 assert(rayOccluded(s,[.5,.5,-3],[0,0,1]));assert(!rayOccluded(s,[1.5,1.5,-3],[0,0,1]));
 assert(s.wedges.every(w=>w.n>1&&w.n<=2));
 const other=preparePLTScene([concavePrism(),plate(4,5)]);assert.equal(other.wedges.filter(w=>w.meshId===1).length,4);
});

test('malformed topology remains opaque and only invalid local wedges are excluded',()=>{
 const source={vertices:[[0,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]],faces:[[0,1,2],[1,0,3],[0,1,4]]};
 const s=preparePLTScene([source,plate(4,5)]);assert(s.unsupportedEdges.some(e=>e.code==='nonmanifold'));
 assert.equal(s.wedges.filter(w=>w.meshId===1).length,4);
 assert(rayOccluded(s,[.2,.2,-1],[0,0,1]));assert(s.wedges.some(w=>w.meshId===0&&w.kind==='boundary'));
 const folded=preparePLTScene(makePresetOBJ('folded'));assert(folded.wedges.length>0);assert(folded.unsupportedEdges.some(e=>e.code==='ambiguous-open-crease'));
 const inward=prepareMesh(makePresetOBJ('box'));inward.authoredFaces=inward.authoredFaces.map(f=>[...f].reverse());
 const bad=preparePLTScene(inward);assert.equal(bad.wedges.length,0);assert(bad.unsupportedEdges.some(e=>e.code==='inward-winding'));assert(rayOccluded(bad,[0,0,-3],[0,0,1]));
});

test('rigid transform of entire scene/rays/cones/regions preserves queries',()=>{
 const meshes=[prepareMesh(makePresetOBJ('box')),
  prepareMesh(makePresetOBJ('screen',{width:2,height:3}),{translation:[4,0,3]})];
 const transform={rotationDeg:[23,41,-17],translation:[21,-34,59]},p=v=>transformPoint(v,transform),d=v=>transformPoint(v,{rotationDeg:transform.rotationDeg});
 const transformed=meshes.map(m=>prepareMesh({vertices:m.vertices.map(p),faces:m.triangles.map(t=>t.indices)}));
 const a=preparePLTScene(meshes),b=preparePLTScene(transformed);
 assert.equal(a.wedges.length,b.wedges.length);assert.equal(a.unsupportedEdges.length,b.unsupportedEdges.length);
 for(let i=0;i<200;i++){
  const origin=[(random()-.5)*10,(random()-.5)*6,-5],direction=normalize([random()-.5,random()-.5,1]);
  const h=rayNearest(a,origin,direction),j=rayNearest(b,p(origin),d(direction));assert.equal(!!h,!!j);
  if(h){near(h.distance,j.distance,1e-9);vn(p(h.point),j.point,1e-9);vn(d(h.normal),j.normal,1e-9);}
 }
 const cone={origin:[0,0,-5],direction:[0,0,1],radius0:.2,tanAngle:.1,near:0,far:20};
 const h=firstConeContact(a,cone),j=firstConeContact(b,{...cone,origin:p(cone.origin),direction:d(cone.direction)});
 near(h.distance,j.distance,1e-8);vn(p(h.point),j.point,1e-8);
 const region={center:[0,0,-1],direction:[0,0,1],radius:1.5,depth:3};
 const ga=collectRegionGeometry(a,region),gb=collectRegionGeometry(b,{...region,center:p(region.center),direction:d(region.direction)});
 assert.deepEqual(ga.triangleIds,gb.triangleIds);assert.equal(ga.wedges.length,gb.wedges.length);
 for(const w of ga.wedges){const v=gb.wedges.find(v=>v.id===w.id);assert(v);near(w.l,v.l,1e-8);vn(p(w.a),v.a,1e-8);vn(p(w.b),v.b,1e-8);vn(d(w.n1),v.n1,1e-8);}
 for(const [source,destination] of [[[0,0,-5],[0,0,5]],[[4,0,-5],[4,0,5]],[[8,0,-5],[8,0,5]]])assert.equal(segmentVisible(a,source,destination),segmentVisible(b,p(source),p(destination)));
});

test('invalid inputs have explicit errors rather than NaN geometry',()=>{
 assert.throws(()=>preparePLTScene([]),/1–256/);
 assert.throws(()=>preparePLTScene('v 0 0 0\nf 1 2 3'),/out of range/);
 assert.throws(()=>preparePLTScene('call external.obj'),/Unsupported/);
 assert.throws(()=>preparePLTScene([plate(),plate()]),/duplicate/);
 assert.throws(()=>preparePLTScene({vertices:[[0,0,0],[1,0,0],[2,0,0]],faces:[[0,1,2]]}),/Degenerate/);
 const s=preparePLTScene(plate());
 for(const direction of [[0,0,0],[NaN,0,1]])assert.throws(()=>rayNearest(s,[0,0,0],direction),/finite|nonzero/);
 assert.throws(()=>rayNearest(s,[0,0,0],[0,0,1],{minDistance:2,maxDistance:1}),/interval/);
 assert.throws(()=>firstConeContact(s,{origin:[0,0,0],direction:[0,0,1],radius0:-1}),/radius0/);
 assert.throws(()=>firstConeContact(s,{origin:[0,0,0],direction:[0,0,1],sigma0:1}),/explicit linear/);
 assert.throws(()=>collectRegionGeometry(s,{center:[0,0,0],direction:[0,0,1],radius:0}),/radius/);
 assert.throws(()=>collectRegionGeometry(s,{center:[0,0,0],direction:[0,0,1],radius:1,depth:-1}),/depth/);
 assert.throws(()=>segmentVisible(s,[0,0,0],[0,0,2],{endpointEpsilon:-1}),/endpointEpsilon/);
});

console.log(`PLT general 3-D geometry: ${checks} groups passed.`);
