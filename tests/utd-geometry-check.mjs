import assert from 'node:assert/strict';
import {prepareMesh,extractWedges,makePresetOBJ,transformMesh,add,sub,scale,dot,normalize} from '../dist/mesh-geometry.js';
import {validateUTDGeometry} from '../dist/utd-geometry-check.js';

let checks=0;
const test=(name,f)=>{try{f();checks++;}catch(e){e.message=`${name}: ${e.message}`;throw e;}};
const mesh=(name,options={})=>prepareMesh(makePresetOBJ(name,options));
const combine=(...parts)=>{
  const vertices=[],faces=[];
  for(const part of parts){const offset=vertices.length;vertices.push(...part.vertices);faces.push(...part.triangles.map(t=>t.indices.map(i=>i+offset)));}
  return prepareMesh({vertices,faces});
};
const check=(m,surface=m.isClosed?'closed-solid':'open-plate')=>{
  const extracted=extractWedges(m,{surface});
  assert(extracted.supported,`Fixture must pass local extraction: ${JSON.stringify(extracted.unsupported)}`);
  return validateUTDGeometry(m,extracted);
};
const accepted=m=>assert(check(m).supported,JSON.stringify(check(m).issues));
const rejected=(m,code)=>{const result=check(m);assert(!result.supported);if(code)assert(result.issues.some(x=>x.code===code),JSON.stringify(result.issues));};
function subdivide(m) {
  const vertices=m.vertices.map(p=>[...p]),faces=[];
  for(const t of m.triangles){const i=vertices.length;vertices.push(scale(add(add(t.a,t.b),t.c),1/3));const[a,b,c]=t.indices;faces.push([a,b,i],[b,c,i],[c,a,i]);}
  return prepareMesh({vertices,faces});
}
function grid(nx=5,ny=7) {
  const vertices=[],faces=[];
  for(let y=0;y<=ny;y++)for(let x=0;x<=nx;x++)vertices.push([x/nx,y/ny,0]);
  for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){const a=y*(nx+1)+x,b=a+1,c=b+nx+1,d=c-1;faces.push([a,b,c],[a,c,d]);}
  return prepareMesh({vertices,faces});
}

test('built-in convex solids, screen, and opaque plate with two holes pass',()=>{
  for(const name of ['box','wedge','tetrahedron','screen','opaque-double-slit'])accepted(mesh(name));
});
test('ordinary triangulation edges and shared vertices are admitted',()=>{
  for(const m of [grid(),subdivide(mesh('box')),subdivide(mesh('opaque-double-slit')),subdivide(mesh('screen'))])accepted(m);
});
test('well-separated boxes pass; overlap, nested solids, face/edge/vertex touches fail',()=>{
  const a=mesh('box');
  accepted(combine(a,transformMesh(a,{translation:[3,0,0]})));
  rejected(combine(a,transformMesh(a,{translation:[1.5,.2,.1]})),'intersecting-solid-components');
  rejected(combine(a,mesh('box',{width:.5,height:.5,depth:.5})),'intersecting-solid-components');
  rejected(combine(a,transformMesh(mesh('box',{height:1,depth:1}),{translation:[2,0,0]})),'intersecting-solid-components');
  rejected(combine(a,transformMesh(mesh('box',{depth:1}),{translation:[2,2,.1]})),'intersecting-solid-components');
  rejected(combine(a,transformMesh(a,{translation:[2,2,2]})),'intersecting-solid-components');
});
test('edge-cross-edge SAT separates skew boxes missed by every face normal',()=>{
  const a=mesh('box',{width:3,height:.4,depth:.5});
  const b=transformMesh(a,{rotationDeg:[102.88302859291434,135.66085585858673,28.590249186381698],translation:[.37820438016206026,-.3073791414499283,1.4427819766569883]});
  for(const t of [...a.triangles,...b.triangles]) {
    const pa=a.vertices.map(p=>dot(sub(p,a.vertices[0]),t.normal)),pb=b.vertices.map(p=>dot(sub(p,a.vertices[0]),t.normal));
    assert(Math.min(...pa)<Math.max(...pb)&&Math.min(...pb)<Math.max(...pa),'No face-normal axis separates this fixture.');
  }
  const m=combine(a,b);accepted(m);
  for(const rotationDeg of [[23,31,17],[91,27,133]])accepted(transformMesh(m,{rotationDeg,translation:[2,-3,4]}));
});
test('positive-area coplanar overlaps fail even with opposite authored normals',()=>{
  const a=mesh('screen');
  rejected(combine(a,transformMesh(a,{translation:[.5,.25,0]})),'overlapping-surface-triangles');
  const reverse=prepareMesh({vertices:a.vertices.map(p=>add(p,[.25,.5,0])),faces:a.triangles.map(t=>[...t.indices].reverse())});
  rejected(combine(a,reverse),'overlapping-surface-triangles');
});
test('bow-tie point contacts and coplanar T-junctions between components fail',()=>{
  rejected(prepareMesh({vertices:[[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]],faces:[[0,1,2],[0,3,4]]}),'intersecting-plate-components');
  rejected(combine(mesh('screen'),transformMesh(mesh('screen',{width:1,height:1}),{translation:[0,1.5,0]})),'intersecting-plate-components');
});
test('orthogonal and oblique crossing plates fail; nonintersecting plates pass',()=>{
  const a=mesh('screen');
  rejected(combine(a,transformMesh(a,{rotationDeg:[90,0,0]})),'intersecting-plate-components');
  rejected(combine(a,transformMesh(a,{rotationDeg:[31,23,0],translation:[.1,0,0]})),'intersecting-plate-components');
  accepted(combine(a,transformMesh(a,{rotationDeg:[90,0,0],translation:[4,0,0]})));
  accepted(combine(a,transformMesh(a,{translation:[0,0,.001]})));
  accepted(combine(a,transformMesh(a,{translation:[3,0,0]})));
});
test('a connected plate cannot wrap over its existing surface',()=>{
  // A triangle fan winds over itself while every pair of adjacent triangles has
  // consistent winding and every individual normal remains +z.
  const vertices=[[0,0,0]];
  for(const angle of [0,80,160,240,320,400])vertices.push([Math.cos(angle*Math.PI/180),Math.sin(angle*Math.PI/180),0]);
  const faces=[];for(let i=1;i<6;i++)faces.push([0,i,i+1]);
  rejected(prepareMesh({vertices,faces}),'overlapping-surface-triangles');
});
test('global support planes independently reject concavity and self-crossed solids',()=>{
  const a=mesh('box'),vertices=a.vertices.map(p=>[...p]);vertices[6]=[0,0,0];
  const dent=prepareMesh({vertices,faces:a.triangles.map(t=>t.indices)}),extracted=extractWedges(dent,{surface:'closed-solid'});
  // Exercise the global guard independently from the earlier local guard.
  const result=validateUTDGeometry(dent,{...extracted,supported:true});
  assert(result.issues.some(x=>x.code==='nonconvex-solid'));
  const crossed=a.vertices.map(p=>[...p]);[crossed[6],crossed[0]]=[crossed[0],crossed[6]];
  const folded=prepareMesh({vertices:crossed,faces:a.triangles.map(t=>t.indices)}),base=extractWedges(folded,{surface:'closed-solid'});
  assert(validateUTDGeometry(folded,{...base,supported:true}).issues.some(x=>x.code==='nonconvex-solid'));
});
test('rigid rotation and translation preserve admission and rejection',()=>{
  const a=mesh('box'),plate=mesh('screen'),fixtures=[
    [mesh('opaque-double-slit'),true],[subdivide(a),true],
    [combine(a,transformMesh(a,{translation:[3,0,0]})),true],
    [combine(a,transformMesh(a,{translation:[1.5,.2,.1]})),false],
    [combine(a,mesh('box',{width:.5,height:.5,depth:.5})),false],
    [combine(a,transformMesh(a,{translation:[2,2,2]})),false],
    [combine(a,transformMesh(mesh('box',{height:1,depth:1}),{translation:[2,0,0]})),false],
    [combine(plate,transformMesh(plate,{translation:[.5,.2,0]})),false],
    [combine(plate,transformMesh(plate,{rotationDeg:[90,0,0]})),false],
  ];
  for(const rotationDeg of [[23,31,17],[90,90,90],[.001,147.4,-62.3]])for(const [m,supported] of fixtures)
    assert.equal(check(transformMesh(m,{rotationDeg,translation:[2,-3,4]})).supported,supported);
});
test('scale-aware checks admit small/large geometry and translated subdivisions',()=>{
  for(const unitScale of [1e-5,1,1e5])for(const name of ['box','opaque-double-slit'])
    accepted(transformMesh(subdivide(mesh(name)),{unitScale,rotationDeg:[23,31,17]}));
  accepted(transformMesh(subdivide(mesh('box')),{rotationDeg:[23,31,17],translation:[1e6,-1e6,1e6]}));
});
test('input records and author positions are not mutated',()=>{
  const m=mesh('opaque-double-slit'),extracted=extractWedges(m,{surface:'open-plate'}),before=JSON.stringify({m,extracted});
  assert(validateUTDGeometry(m,extracted).supported);assert.equal(JSON.stringify({m,extracted}),before);
});
test('the pairwise validator refuses the solver cap before any geometry work',()=>{
  const result=validateUTDGeometry({triangles:Array(129).fill(null)},{components:[]});
  assert(!result.supported);assert.equal(result.issues[0].code,'geometry-validation-limit');
});

console.log(JSON.stringify({suite:'utd-geometry-admission',checks,globalConvexity:true,completeSolidSAT:true,embeddedPlanarSurfaces:true,rigidTransformInvariant:true}));
