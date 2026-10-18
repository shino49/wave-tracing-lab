/** Exact old/new regression and bounded cone-query benchmark.
 * Run: node tests/geometry-oracle.mjs
 * JSON report: node tests/geometry-oracle.mjs --report research/geometry-oracle-validation.json
 * Optional facet-only cache timing: measured on region results below.
 */
import assert from 'node:assert/strict';
import {performance} from 'node:perf_hooks';
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {firstConeTriangleContact} from '../dist/utd-scene.js';
import {preparePLTScene,collectRegionGeometry} from '../dist/plt-geometry.js';
import {makePresetOBJ,prepareMesh,sub,cross,dot,normalize,transformPoint} from '../dist/mesh-geometry.js';
import {firstConeTriangleContact as originalContact,buildFacets as originalFacets} from './geometry-oracle-baseline.mjs';

const INITIAL_SEED=0x793de027;
let seed=INITIAL_SEED;
const random=()=>((seed=Math.imul(seed,1664525)+1013904223>>>0)/4294967296);
const signed=()=>2*random()-1;
let nextId=0;
const makeTriangle=points=>({id:nextId++,a:points[0],b:points[1],c:points[2],normal:normalize(cross(sub(points[1],points[0]),sub(points[2],points[0])))});
const cone=(overrides={})=>({origin:[0,0,0],direction:[0,0,1],radius0:1,tanAngle:0,near:0,far:10,...overrides});
const cases=[];
const push=(group,points,c)=>cases.push({group,triangle:makeTriangle(points),cone:c});

// Flat sections: interior, vertex/edge contact, outside, and tolerance sides.
for(const x of [0,1,1-1e-10,1+1e-10,1+1.01e-10,1-1e-12,1+1e-12,2]) {
  for(const dz of [0,1e-12,9.9e-11,1.01e-10,1e-8]) {
    push('coplanar-nearly-flat',[[x,-1,2],[x+2,-1,2+dz],[x,1,2-dz]],cone());
    push('coplanar-axis-inside',[[-2,-1,2],[2,-1,2+dz],[0,2,2-dz]],cone({radius0:0}));
  }
}
// Grazing a cylinder's side, a widening cone, and the zero-radius axis.
for(const x of [0,1e-12,1,1-1e-10,1+1e-10,1+1.01e-10]) {
  for(const radius0 of [0,.1,1])for(const tanAngle of [0,.1,1]) {
    push('grazing-and-parallel',[[x,-2,1],[x,2,3],[x,-1,5]],cone({radius0,tanAngle}));
  }
}
// Thin and tiny triangles retain all original endpoint duplication behavior.
for(const width of [1e-13,1e-11,1e-9,1e-7,1e-3])for(const radius0 of [0,.1,1]) {
  push('thin',[[.5,-width,1],[1.5,width,5],[1,2*width,2]],cone({radius0,tanAngle:.1}));
  push('tiny',[[.5,-width,2],[.5+width,width,2+width],[.5-width,width,2-width]],cone({radius0}));
}
// Exact and adjacent depth endpoints, including a zero-width interval.
for(const near of [0,1,1-1e-12,1+1e-12,2,3])for(const far of [1-1e-12,1,1+1e-12,2,3,10]) {
  if(near>far)continue;
  push('near-far-endpoints',[[.7,0,1],[1.8,1,2],[.8,-1,3]],cone({near,far,tanAngle:.2}));
  push('near-far-endpoints',[[1,0,1],[2,1,2],[2,-1,3]],cone({near,far}));
}
// Rigid transforms and coordinate scales also probe the scale-aware epsilon.
const canonical=cases.slice();
for(const factor of [1e-9,1e-6,1e-3,1,1e3,1e6]) {
  for(const rotationDeg of [[0,0,0],[31,-47,83],[-89.999999,179.9,270]]) {
    const translation=factor===1e6?[1e8,-2e8,3e8]:[7*factor,-11*factor,3*factor];
    const point=p=>transformPoint(p,{rotationDeg}).map((x,k)=>x*factor+translation[k]);
    const direction=d=>normalize(transformPoint(d,{rotationDeg}));
    for(let i=0;i<canonical.length;i+=5) {
      const base=canonical[i],c=base.cone,t=base.triangle;
      push('scales-and-rotations',[point(t.a),point(t.b),point(t.c)],{
        ...c,origin:point(c.origin),direction:direction(c.direction),radius0:c.radius0*factor,near:c.near*factor,far:c.far*factor});
    }
  }
}
// Large translations test epsilon growth without changing a tiny local shape.
for(const translation of [[1e9-50,-1e9+50,1e9-50],[1e6,-1e6,1e6]]) {
  const shift=p=>p.map((x,k)=>x+translation[k]);
  for(const x of [1,1-1e-5,1+1e-5,1+1e-4]) {
    push('large-translations',[shift([x,-1,1]),shift([x,1,3]),shift([x,2,4.5])],cone({origin:translation}));
  }
}
// Seeded randomized intersections and misses, with periodic flat, thin,
// zero-radius, near/far-clipped, and far=Infinity cases.
for(let i=0;i<12000;i++) {
  const factor=10**(-6+12*random()),origin=Array.from({length:3},()=>signed()*factor*3);
  const direction=normalize([signed(),signed(),signed()]);
  const points=Array.from({length:3},()=>Array.from({length:3},(_,k)=>origin[k]+signed()*factor*5));
  if(i%13===0)for(const p of points)p[2]=origin[2]+factor*2;
  if(i%17===0)points[2]=points[0].map((v,k)=>v*.5+points[1][k]*.5+signed()*factor*1e-9);
  const near=i%7===0?random()*factor*2:0;
  push('randomized',points,{origin,direction,radius0:i%11===0?0:factor*random()*2,
    tanAngle:random()*.7,near,far:i%5===0?Infinity:near+random()*factor*8});
}

// Load only the private scalar helpers from these known local modules. This
// tests the production implementation without adding a public test export.
const sourcePaths=['dist/utd-scene.js','dist/plt-geometry.js','tests/geometry-oracle-baseline.mjs'];
const sourceTexts=Object.fromEntries(sourcePaths.map(path=>[path,readFileSync(new URL('../'+path,import.meta.url),'utf8')]));
const sourceHashes=Object.fromEntries(sourcePaths.map(path=>[path,createHash('sha256').update(sourceTexts[path]).digest('hex')]));
const epsilonFunctions=['dist/utd-scene.js','dist/plt-geometry.js'].map(path=>{
  const source=sourceTexts[path],start=source.indexOf('const pointMagnitude='),end=source.indexOf('\n};',source.indexOf('const epsilonAt=',start))+3;
  assert(start>=0&&end>start,'Scalar epsilon helper was not found');
  return new Function(source.slice(start,end)+'\nreturn epsilonAt;')();
});
const originalEpsilon=(...points)=>Math.max(1e-10,128*Number.EPSILON*Math.max(1,...points.flat().map(Math.abs)));
let epsilonCases=0;
// Independent generator keeps the cone and region fixtures unchanged.
let epsilonSeed=0x05fda190;
const epsilonRandom=()=>((epsilonSeed=Math.imul(epsilonSeed,1664525)+1013904223>>>0)/4294967296);
for(let count=0;count<=6;count++)for(let i=0;i<1500;i++) {
  const points=Array.from({length:count},()=>Array.from({length:3},()=>{
    const sign=epsilonRandom()<.5?-1:1;
    return sign*10**(-320+628*epsilonRandom());
  }));
  const expected=originalEpsilon(...points);
  for(const fn of epsilonFunctions)assert(Object.is(fn(...points),expected));
  epsilonCases++;
}
for(const value of [0,-0,Number.MIN_VALUE,Number.MAX_VALUE,Infinity,-Infinity,NaN]) {
  for(const fn of epsilonFunctions)assert(Object.is(fn([value,value,value]),originalEpsilon([value,value,value])));
  epsilonCases++;
}

const counts={},hits={};
const benchmarkCases=[];
for(const [i,item] of cases.entries()) {
  const before=originalContact(item.triangle,item.cone),after=firstConeTriangleContact(item.triangle,item.cone);
  try {assert.deepEqual(after,before);}catch(error){error.message=`Exact cone oracle case ${i} (${item.group}): ${error.message}`;throw error;}
  if(before) {
    // Returning a triangle vertex keeps its original reference too.
    const ref=[item.triangle.a,item.triangle.b,item.triangle.c].find(p=>p===before.point);
    if(ref)assert.equal(after.point,ref);
  }
  counts[item.group]=(counts[item.group]??0)+1;hits[item.group]=(hits[item.group]??0)+(before?1:0);
  if(item.group==='randomized'&&benchmarkCases.length<1200)benchmarkCases.push(item);
}

// A subdivided plate plus a box and disconnected plate test material barriers,
// disconnected subsets, repeated region changes and scene-local cache lifetime.
const vertices=[],faces=[],materialIndices=[],N=12;
for(let y=0;y<=N;y++)for(let x=0;x<=N;x++)vertices.push([x-N/2,y-N/2,0]);
for(let y=0;y<N;y++)for(let x=0;x<N;x++) {
  const a=y*(N+1)+x,b=a+1,d=a+N+1,c=d+1;
  faces.push([a,b,c],[a,c,d]);materialIndices.push(x%4===0?1:0,x%4===0?1:0);
}
const scene=preparePLTScene([{mesh:{vertices,faces},materialIndices},
  {mesh:prepareMesh(makePresetOBJ('box'),{translation:[0,0,3]}),materialIndex:2},
  {mesh:prepareMesh(makePresetOBJ('screen'),{translation:[8,0,1]}),materialIndex:3}]);
assert.deepEqual(scene.facets,originalFacets(scene,scene.triangles));
let regionCount=0,regionFacetCount=0;
for(let i=0;i<300;i++) {
  const region={center:[signed()*9,signed()*8,signed()*3],direction:normalize([signed(),signed(),signed()]),radius:.03+random()*5,depth:i%6===0?Infinity:i%7===0?0:random()*4};
  const g=collectRegionGeometry(scene,region),clips=new Map(g.clippedTriangles.map(c=>[c.triangleId,c]));
  assert.deepEqual(g.facets,originalFacets(scene,g.triangles,clips));
  regionCount++;regionFacetCount+=g.facets.length;
}
// A separately prepared scene cannot reuse another scene's facet/material map.
const separate=preparePLTScene({mesh:{vertices,faces},materialIndices:materialIndices.map(x=>x+5)});
assert.deepEqual(separate.facets,originalFacets(separate,separate.triangles));
const separateRegion=collectRegionGeometry(separate,{center:[0,0,0],direction:[0,0,1],radius:2,depth:1});
assert.deepEqual(separateRegion.facets,originalFacets(separate,separateRegion.triangles,new Map(separateRegion.clippedTriangles.map(c=>[c.triangleId,c]))));

// Alternate benchmark order to avoid consistently favoring a warmer JIT/CPU.
// The same deterministic 1200 mixed cases are used by both implementations.
const measure=fn=>{const start=performance.now();let checksum=0;for(const item of benchmarkCases){const h=fn(item.triangle,item.cone);if(h)checksum+=h.distance+h.point[0];}return {ms:performance.now()-start,checksum};};
measure(originalContact);measure(firstConeTriangleContact);
const oldRuns=[],newRuns=[];
for(let round=0;round<5;round++) {
  const a=round%2?measure(firstConeTriangleContact):measure(originalContact);
  const b=round%2?measure(originalContact):measure(firstConeTriangleContact);
  assert.equal(a.checksum,b.checksum);
  oldRuns.push(round%2?b.ms:a.ms);newRuns.push(round%2?a.ms:b.ms);
}
const median=values=>[...values].sort((a,b)=>a-b)[values.length>>1];
const report={suite:'geometry-exact-oracle',seed:INITIAL_SEED,comparison:'Exact deep equality, including contact coordinates, distance, null results and vertex references',
  sourceHashes,epsilonCases,epsilonImplementations:2,coneCases:cases.length,counts,hits,facetRegions:regionCount+1,facetCount:regionFacetCount,
  result:'passed',iterations:{golden:65,bisection:65},tolerance:'unchanged',
  benchmark:{environment:`Node ${process.version} ${process.platform} ${process.arch}`,caseCount:benchmarkCases.length,rounds:5,
    originalMs:oldRuns,optimizedMs:newRuns,originalMedianMs:median(oldRuns),optimizedMedianMs:median(newRuns),speedup:median(oldRuns)/median(newRuns),
    scope:'CPU firstConeTriangleContact only, same mixed inputs; this is not an end-to-end rendering speedup'},
  limitations:['Equivalence preserves baseline geometry behavior; it does not prove the baseline algorithm mathematically exact.',
    'The facet adjacency cache relies on the documented immutability of prepared scene geometry. Rebuild a scene to change its mesh or materials.']};
const reportFlag=process.argv.indexOf('--report');
if(reportFlag!==-1){assert(process.argv[reportFlag+1],'--report requires a path');writeFileSync(process.argv[reportFlag+1],JSON.stringify(report,null,2)+'\n');}
console.log(JSON.stringify(report,null,2));
