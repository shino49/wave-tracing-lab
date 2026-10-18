import assert from 'node:assert/strict';
import {prepareMesh,transformMesh,transformPoint,makePresetOBJ,add,sub,scale,dot,length,distance,normalize} from '../dist/mesh-geometry.js';
import {prepareUTDScene,builtInSceneOBJ,evaluatePoint,computeReceiverPoint,clipEdgeToRegion,outgoingConeForDirection,stablePlaneExcess,firstConeTriangleContact,UTD_LIMITS} from '../dist/utd-scene.js';
let checks=0;
const test=(name,f)=>{try{f();checks++;}catch(e){e.message=`${name}: ${e.message}`;throw e;}};
const near=(a,b,tol=1e-9)=>assert(Math.abs(a-b)<=tol*Math.max(1,Math.abs(a),Math.abs(b)),`${a} != ${b}`);
const vn=(a,b,tol)=>a.forEach((x,i)=>near(x,b[i],tol));
const scene=name=>prepareUTDScene(builtInSceneOBJ(name));

function gridPlate(nx=4,ny=8) {
 const vertices=[],faces=[];
 for(let y=0;y<=ny;y++)for(let x=0;x<=nx;x++)vertices.push([-1+2*x/nx,-10+20*y/ny,0]);
 for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){const a=y*(nx+1)+x,b=a+1,c=b+nx+1,d=c-1;faces.push([a,b,c],[a,c,d]);}
 return prepareMesh({vertices,faces});
}
function subdivideFaces(mesh) {
 const vertices=mesh.vertices.map(v=>[...v]),faces=[];
 for(const t of mesh.triangles){const i=vertices.length;vertices.push(scale(add(add(t.a,t.b),t.c),1/3));const[a,b,c]=t.indices;faces.push([a,b,i],[b,c,i],[c,a,i]);}
 return prepareMesh({vertices,faces});
}

test('mm presets are real 3D outward geometry and finite tall plate',()=>{
 const screen=scene('screen'),box=scene('box'),wedge=scene('wedge');
 assert(screen.supported&&box.supported&&wedge.supported);
 near(screen.mesh.bounds.size[0],2);near(screen.mesh.bounds.size[1],20);
 assert(box.mesh.isClosed&&wedge.mesh.isClosed);assert(box.mesh.bounds.size[2]>0&&wedge.mesh.bounds.size[2]>0);
 assert.equal(screen.counts.clippedEdges,2);assert.equal(box.counts.physicalEdges,12);assert.equal(wedge.counts.physicalEdges,9);
 for(const s of [screen,box,wedge])assert(s.warnings.some(t=>t.includes('not irradiance')));
});

test('Gaussian cone conventions and finite region depth',()=>{
 const s=scene('screen'),p=s.params;
 near(s.incomingCone.radius0,3*p.sigma0Mm);near(s.incomingCone.tanAngle,3/(2*p.k*p.sigma0Mm));
 near(s.region.distance,5);near(s.region.radius,3+s.incomingCone.tanAngle*5);
 near(s.region.depth,2*s.region.radius);vn(s.region.axes,[s.region.radius,s.region.radius,2*s.region.radius]);
 const out=outgoingConeForDirection(s,normalize([.5,.4,1]));
 near(out.tanAngle,s.incomingCone.tanAngle);near(out.radius0,2*s.region.radius);assert(out.conservative);
 for(let a=0;a<20;a++)for(let b=0;b<20;b++){
   const theta=Math.PI*a/19,phi=2*Math.PI*b/19,point=[s.region.radius*Math.sin(theta)*Math.cos(phi),s.region.radius*Math.sin(theta)*Math.sin(phi),2*s.region.radius*Math.cos(theta)];
   const projected=sub(point,scale(out.direction,dot(point,out.direction)));assert(length(projected)<=out.radius0+1e-10);
 }
});

test('cone contact sees a corner even when axis misses and centroid lies outside',()=>{
 const m=prepareMesh({vertices:[[.9,0,2],[10,0,2],[.9,10,2]],faces:[[0,1,2]]});
 const cone={origin:[0,0,0],direction:[0,0,1],radius0:1,tanAngle:0,near:0,far:10};
 const h=firstConeTriangleContact(m.triangles[0],cone);assert(h);near(h.distance,2);assert(h.point[0]>=.9);assert(distance(h.point,[0,0,2])<=1+1e-9);
 assert.equal(firstConeTriangleContact(m.triangles[0],{...cone,radius0:.8}),null);
 // An almost-endpoint section can contain duplicate vertices; it must not be
 // mistaken for a coplanar triangle containing an axis that is 5 mm away.
 const offAxis=prepareMesh({vertices:[[-5,-2,-11],[-5,2,-9],[-5,-2,-9]],faces:[[0,1,2]]});
 assert.equal(firstConeTriangleContact(offAxis.triangles[0],{origin:[0,0,-20],direction:[0,0,1],radius0:3,tanAngle:.012,near:0,far:30}),null);
});

test('actual oblique 3D segment clips to ellipsoid and forward slab',()=>{
 const w={a:[-8,1,-6],b:[8,1,6],l:20};
 const r={center:[0,0,0],direction:[0,0,1],radius:2,depth:4};
 const clipped=clipEdgeToRegion(w,r);assert(clipped);assert(clipped.clippedStart&&clipped.clippedEnd);
 for(const p of [clipped.a,clipped.b]){assert(p[2]>=-1e-9&&p[2]<=4+1e-9);assert((p[0]**2+p[1]**2)/4+p[2]**2/16<=1+1e-9);near(p[2],.75*p[0]);}
 near(clipped.a[2],0);assert(clipped.b[2]>0);
 assert.equal(clipEdgeToRegion({a:[5,5,0],b:[5,5,2],l:2},r),null);
});

test('triangle subdivisions preserve field and one GO reflection',()=>{
 const coarse=scene('screen'),fine=prepareUTDScene(gridPlate());
 assert.equal(fine.counts.physicalEdges,4);assert.equal(fine.wedges.length,coarse.wedges.length);
 for(const dst of [[0,0,20],[.6,0,20],[1,0,20],[1.4,.2,20],[0,0,-20],[.4,.2,-20],[1,0,-20]]){
   const a=evaluatePoint(coarse,dst),b=evaluatePoint(fine,dst);vn(a.soft,b.soft,2e-9);vn(a.hard,b.hard,2e-9);near(a.f,b.f,2e-9);assert.equal(a.directWeight,b.directWeight);assert.equal(a.reflectionCount,b.reflectionCount);
 }
 assert.equal(evaluatePoint(fine,[0,0,-20]).reflectionCount,1);
 // A shared triangulation diagonal must block the full direct field, not half.
 assert.equal(evaluatePoint(coarse,[0,0,20]).directWeight,0);
 const box=scene('box'),boxFine=prepareUTDScene(subdivideFaces(box.mesh));
 assert.equal(boxFine.counts.physicalEdges,12);
 for(const dst of [[1.5,0,20],[-1.5,.2,20],[0,0,-20]])near(evaluatePoint(box,dst).f,evaluatePoint(boxFine,dst).f,2e-9);
});
test('shallow real creases cannot masquerade as one planar reflection facet',()=>{
 const original=prepareMesh(builtInSceneOBJ('box')),vertices=original.vertices.map(v=>[...v]);vertices[0][2]=-1.0001;
 const bent=prepareUTDScene(prepareMesh({vertices,faces:original.triangles.map(t=>t.indices)}));
 assert(!bent.supported);assert(bent.unsupported.some(v=>v.code==='nonplanar-reflection-facet'));
 const rotated=prepareUTDScene(transformMesh(original,{rotationDeg:[17,-23,31],translation:[80,-40,13]}),{sourceOrigin:[80,-40,-7],detectorCenter:[80,-40,33]});
 assert(!rotated.unsupported.some(v=>v.code==='nonplanar-reflection-facet'));
});

test('whole 3D configuration rigid rotation and translation preserve fields',()=>{
 const tr={rotationDeg:[23,31,17],translation:[2,-3,4]},point=p=>transformPoint(p,tr),direction=p=>transformPoint(p,{rotationDeg:tr.rotationDeg});
 for(const name of ['screen','box','wedge']){
   const first=scene(name),p=first.params;
   const changed=prepareUTDScene(transformMesh(first.mesh,tr),{...p,sourceOrigin:point(p.sourceOrigin),incidentDirection:direction(p.incidentDirection),detectorCenter:point(p.detectorCenter),detectorU:direction(p.detectorU),detectorV:direction(p.detectorV)});
   near(first.region.distance,changed.region.distance,1e-8);near(first.region.radius,changed.region.radius,1e-8);
   for(const dst of [[0,0,20],[.3,.2,20],[1.5,.1,20],[-1.4,-.2,20],[.2,.1,-20]]){
     const a=evaluatePoint(first,dst),b=evaluatePoint(changed,point(dst));vn(a.soft,b.soft,1e-8);vn(a.hard,b.hard,1e-8);near(a.f,b.f,1e-8);assert.equal(a.valid,b.valid);
   }
 }
});

test('depth and actual wedge orientation affect coherent field',()=>{
 const a=scene('wedge'),b=prepareUTDScene(makePresetOBJ('triangular-prism',{width:2,height:20,depth:1}));
 assert(Math.abs(evaluatePoint(a,[.4,0,20]).f-evaluatePoint(b,[.4,0,20]).f)>1e-5);
 const yaw=prepareUTDScene(a.mesh,{incidentDirection:normalize([.25,0,1])});
 assert(yaw.supported);assert(Math.abs(evaluatePoint(a,[.4,0,20]).f-evaluatePoint(yaw,[.4,0,20]).f)>1e-4);
 assert(yaw.wedges.some(w=>Math.abs(w.a[2]-w.b[2])>1e-5)||yaw.wedges.some(w=>Math.abs(w.a[2])>.1));
});

test('direct and reflection boundaries use symmetric half with regular UTD terms',()=>{
 const s=scene('screen');
 const mid=evaluatePoint(s,[1,0,20],{details:true});near(mid.directWeight,.5);assert(mid.valid);assert.equal(mid.edgeCount,2);
 const l=evaluatePoint(s,[1-1e-8,0,20]),r=evaluatePoint(s,[1+1e-8,0,20]);
 vn(mid.soft,scale(add(l.soft,r.soft),.5),1e-7);vn(mid.hard,scale(add(l.hard,r.hard),.5),1e-7);
 assert(mid.contributions.some(c=>Math.hypot(...c.soft)>1e-5));
 const reflected=evaluatePoint(s,[1,0,-20]);assert.equal(reflected.reflectionCount,1);near(reflected.geometricPower,.25);
 const reflLeft=evaluatePoint(s,[1-1e-8,0,-20]),reflRight=evaluatePoint(s,[1+1e-8,0,-20]);
 vn(reflected.soft,scale(add(reflLeft.soft,reflRight.soft),.5),1e-7);vn(reflected.hard,scale(add(reflLeft.hard,reflRight.hard),.5),1e-7);
 near(evaluatePoint(s,[1+1e-7,0,20],{boundaryEpsilon:2e-6}).directWeight,.5);
});

test('endpoint and unmodeled higher-order shadows are masked, not false zeros',()=>{
 const short=prepareUTDScene(makePresetOBJ('screen',{width:2,height:2}));
 const corner=evaluatePoint(short,[1,1,20]);assert(!corner.valid);assert(corner.invalidReasons.some(v=>v.includes('corner')||v.includes('endpoint')));assert(Number.isNaN(corner.power));
 const deep=evaluatePoint(scene('box'),[0,0,20]);assert(!deep.valid);assert(deep.invalidReasons.includes('no-first-order-path'));
 const tall=scene('screen');assert(evaluatePoint(tall,[1,0,20]).valid);
 const clipped=evaluatePoint(tall,[1,2.8,20]);assert(!clipped.valid);assert(clipped.invalidReasons.includes('region-truncated-stationary-neighborhood'));
});

test('outgoing native conditional f has no Gaussian amplitude or 1/pdf multiplier',()=>{
 const s=scene('screen'),wide=prepareUTDScene(s.mesh,{sigma0Mm:2});
 for(const dst of [[0,0,20],[.5,0,20],[1,0,20],[2,0,20]]){
   const a=evaluatePoint(s,dst),b=evaluatePoint(wide,dst);assert(a.valid&&b.valid);near(a.f,b.f);near(a.power,a.f);
   near(a.f,(a.soft[0]**2+a.soft[1]**2+a.hard[0]**2+a.hard[1]**2)/2);
   near(evaluatePoint(s,dst,{boundary:'soft'}).f,a.soft[0]**2+a.soft[1]**2);
 }
});

test('plane approximation, disconnected depth and unsupported folded-shell gates',()=>{
 const narrow=prepareUTDScene(builtInSceneOBJ('screen'),{sigma0Mm:.05});assert(!narrow.supported);assert(narrow.unsupported.some(x=>x.code==='plane-curvature-guard'));
 const folded=prepareUTDScene(makePresetOBJ('folded'));assert(!folded.supported);assert(folded.unsupported.some(x=>x.code==='nonplanar-open-plate'));
 const m=prepareMesh(builtInSceneOBJ('screen')),vertices=[...m.vertices,...m.vertices.map(p=>add(p,[0,0,20]))],faces=[...m.authoredFaces,...m.authoredFaces.map(f=>f.map(i=>i+m.vertices.length))];
 const depth=prepareUTDScene({vertices,faces});assert(!depth.supported);assert(depth.unsupported.some(x=>x.code==='multiple-depth-regions'));
});

test('stable phase residual and GPU precision bound',()=>{
 near(stablePlaneExcess([0,0,1],[0,0,0],[1,0,1e12]),5e-13,1e-20);
 near(stablePlaneExcess([0,0,1],[0,0,0],[0,0,-3]),6);
 assert(scene('screen').gpuSafety.safe);
 const far=prepareUTDScene(builtInSceneOBJ('screen'),{detectorCenter:[0,0,1e6]});assert(!far.gpuSafety.safe);assert(far.gpuSafety.maxPhaseArgumentBound>1e5);
});

test('detector pixel center mapping and packed record contracts',()=>{
 const s=scene('screen');vn(computeReceiverPoint(s,0,0,1,1),s.params.detectorCenter);
 vn(computeReceiverPoint(s,0,0,2,2),[-1.5,1.5,20]);
 assert.equal(s.packedTriangles.length,s.mesh.triangles.length*12);assert.equal(s.packedEdges.length,s.wedges.length*24);assert.equal(s.packedEdgeAdjacency.length,s.wedges.length*4);
 for(const w of s.wedges)for(const endpoint of [w.a,w.b])assert(endpoint[2]>=s.region.center[2]-1e-9);
 assert.equal(UTD_LIMITS.maxTriangles,128);assert.equal(UTD_LIMITS.maxEdges,256);
});

console.log(JSON.stringify({suite:'utd-scene',checks,quantity:'conditional directional power transfer f',real3D:true,coherentGO:true,subdivisionInvariant:true,endpointMasks:true}));
