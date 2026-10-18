/** Independent v25 rectangle-plane oracle. No production RNG, camera, mesh,
 * BVH, triangle-intersection or path-tracing code is used to form expectations.
 * Production helpers are used only to build the separately compared CPU result.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {createPLTRandomStates} from '../dist/plt-core.js';
import {buildResidentReferenceBatch} from '../dist/plt-resident-reference.js';
import {createResidentHardwareTestFixture} from '../dist/plt-resident-test-presets.js';

// Coordinate ranges are [x,y,z]. Exactly seven rectangles / fourteen triangles.
const rectangle=(name,axis,position,ranges,albedo,emission)=>({name,axis,position,ranges,albedo,emission});
export const rectangles=Object.freeze([
 rectangle('x-mirror',0,2,[[2,2],[-8,8],[1,2]],.5,0),
 rectangle('z-mirror',2,3,[[-8,8],[-8,8],[3,3]],.75,0),
 rectangle('negative-bright',2,-2,[[-6,-4],[0,2],[-2,-2]],0,8),
 rectangle('negative-dim',2,-2,[[-6,-4],[2,4],[-2,-2]],0,2),
 rectangle('positive-bright',2,-2,[[6,8],[0,2],[-2,-2]],0,8),
 rectangle('positive-dim',2,-2,[[6,8],[2,4],[-2,-2]],0,2),
 rectangle('black-baffle',2,1,[[.9375,1.0625],[.21875,.28125],[1,1]],0,0),
]);
const unsigned=value=>Number(BigInt.asUintN(32,BigInt(value)));
const multiply=(a,b)=>Number(BigInt.asUintN(32,BigInt(a)*BigInt(b)));
function independentRNG(state){
 return {get state(){return state;},word(){state=unsigned(state+1831565813);let t=multiply(state^(state>>>15),state|1);t=unsigned(t^(unsigned(t+multiply(t^(t>>>7),t|61))));return unsigned(t^(t>>>14));}};
}
const normalized=v=>{const d=Math.sqrt(v.reduce((sum,x)=>sum+x*x,0));return v.map(x=>x/d);};
function cameraDirection(pixel,size,words){
 // Analytic orthonormal frame of camera (0,0,0) -> (4,1,4), up (0,1,0).
 const forward=[4/Math.sqrt(33),1/Math.sqrt(33),4/Math.sqrt(33)],right=[-1/Math.sqrt(2),0,1/Math.sqrt(2)],up=[-1/Math.sqrt(66),8/Math.sqrt(66),-1/Math.sqrt(66)];
 const half=Math.tan(Math.PI/36),u=(2*((pixel%size)+words[0]/4294967296)/size-1)*half,v=(1-2*(Math.floor(pixel/size)+words[1]/4294967296)/size)*half;
 return normalized(forward.map((x,i)=>x+u*right[i]+v*up[i]));
}
function traceRectanglePath(direction,surfaces=rectangles){
 let origin=[0,0,0],weight=1;const hits=[];
 for(let depth=0;depth<6;depth++){
  let nearest=null;
  for(const surface of surfaces){
   const {axis,position,ranges}=surface;if(direction[axis]===0)continue;
   const distance=(position-origin[axis])/direction[axis];if(distance<1e-7||nearest&&distance>=nearest.distance)continue;
   const point=origin.map((x,i)=>x+distance*direction[i]);
   if(!ranges.every(([lo,hi],i)=>i===axis||point[i]>=lo&&point[i]<=hi))continue;
   nearest={surface,distance,point,direction:[...direction]};
  }
  if(!nearest)return {power:0,mirrors:hits.filter(x=>x.surface.albedo>0).length,hits,termination:'escaped'};
  hits.push(nearest);const {surface}=nearest;
  if(surface.albedo===0)return {power:weight*surface.emission,mirrors:hits.length-1,hits,termination:'absorbed'};
  weight*=surface.albedo;origin=nearest.point;direction=direction.map((x,i)=>i===surface.axis?-x:x);
 }
 throw new Error('Independent rectangle oracle exceeded frozen depth');
}
const histogram=items=>items.reduce((out,x)=>{out[x]=(out[x]??0)+1;return out;},{});
export function structuredOracle(size=32){
 assert([8,16,32].includes(size));const paths=[],image=[],initial=[],final=[];
 for(let pixel=0;pixel<size*size;pixel++){
  const state=unsigned(7391^multiply(pixel+1,2654435761)),rng=independentRNG(state);initial.push(state);let mean=0;
  for(let sample=0;sample<2;sample++){
   const start=rng.state,words=[rng.word(),rng.word()],direction=cameraDirection(pixel,size,words),path=traceRectanglePath(direction);
   paths.push({pixel,sample,...path,direction,start,final:rng.state,words});mean+=path.power/2;
  }
  image.push(mean);final.push(rng.state);
 }
 return {size,paths,image,initial,final,powers:histogram(paths.map(p=>p.power)),mirrors:histogram(paths.map(p=>p.mirrors)),pixels:histogram(image)};
}
function matchSurface(triangle){
 const vertices=[triangle.a,triangle.b,triangle.c];
 const candidates=rectangles.filter(s=>vertices.every(p=>p[s.axis]===s.position&&s.ranges.every(([lo,hi],i)=>p[i]>=lo&&p[i]<=hi)));
 assert.equal(candidates.length,1,'Each production triangle must belong to exactly one frozen rectangle');return candidates[0];
}
function verifyGeometry(fixture){
 assert.deepEqual(fixture.camera,{position:[0,0,0],target:[4,1,4],up:[0,1,0],fov:10});assert.equal(fixture.scene.triangles.length,14);
 const byId=new Map(),groups=new Map();
 for(const t of fixture.scene.triangles){const surface=matchSurface(t),material=fixture.scene.materials[t.materialIndex];assert.equal(material.albedo,surface.albedo);assert.equal(material.emission,surface.emission);assert.equal(material.type,surface.albedo?'mirror':'diffuse');if(surface.emission)assert.equal(material.twoSidedEmission,true);byId.set(t.id,surface);const group=groups.get(surface.name)??[];group.push(t);groups.set(surface.name,group);}
 for(const s of rectangles){const group=groups.get(s.name);assert.equal(group.length,2);const vertices=new Set(group.flatMap(t=>[t.a,t.b,t.c].map(p=>JSON.stringify(p))));assert.equal(vertices.size,4);const varying=s.ranges.filter((_,i)=>i!==s.axis);assert.equal(group.reduce((sum,t)=>sum+t.area,0),(varying[0][1]-varying[0][0])*(varying[1][1]-varying[1][0]));}
 assert.equal(fixture.scene.emitters.length,8);assert.equal(fixture.scene.emitterArea,16);return byId;
}
export async function compareStructuredCPU(size){
 const expected=structuredOracle(size),fixture=createResidentHardwareTestFixture('structured-mirror-panels'),byId=verifyGeometry(fixture);
 const options={width:size,height:size,samplesPerPixel:2,maxDepth:6,seed:7391,wave:false,nee:false},rng=createPLTRandomStates(options);assert.deepEqual([...rng],expected.initial);
 const reference=await buildResidentReferenceBatch(fixture.scene,fixture.camera,options,{tile:{x:0,y:0,width:size,height:size},samplesPerPixel:2,sampleStart:0,rngStates:rng});
 assert.deepEqual([...rng],expected.final);assert.deepEqual([...reference.audit.reference],expected.image);assert.equal(reference.audit.paths.length,expected.paths.length);
 for(let index=0;index<expected.paths.length;index++){
  const want=expected.paths[index],actual=reference.audit.paths[index],label=`${size}/${want.pixel}/${want.sample}`;
  assert.deepEqual(actual.key,{wave:false,pixel:want.pixel,sample:want.sample},label);assert.equal(actual.value,want.power,label);assert.equal(actual.scatters.length,want.mirrors,label);assert.equal(actual.vertices.length,want.hits.length,label);assert.equal(actual.termination.reason,want.termination,label);
  assert.deepEqual(actual.rng.words,want.words,label);assert.equal(actual.rng.start,want.start);assert.equal(actual.rng.final,want.final);assert.equal(actual.rng.count,2);assert.equal(actual.localEvents.length,0);assert.equal(actual.contributions.length,want.power>0?1:0);
  for(let k=0;k<want.hits.length;k++){
   const hit=want.hits[k],got=actual.vertices[k];assert.equal(byId.get(got.triangleId)?.name,hit.surface.name,label);assert(Math.abs(got.distance-hit.distance)<1e-10,label);
   for(let axis=0;axis<3;axis++){assert(Math.abs(got.center[axis]-hit.point[axis])<1e-10,label);assert(Math.abs(got.direction[axis]-hit.direction[axis])<1e-13,label);}
  }
 }
 assert(Object.keys(expected.pixels).length>5,'Structured scene must retain mixed-sample boundary pixels');assert(expected.paths.some(p=>p.hits[0].surface.name==='black-baffle'));assert(expected.paths.some(p=>p.mirrors===1));assert(expected.paths.some(p=>p.mirrors===2));
 return {size,paths:expected.paths.length,pathPowerHistogram:expected.powers,mirrorCountHistogram:expected.mirrors,pixelPowerHistogram:expected.pixels,imageSHA256:crypto.createHash('sha256').update(JSON.stringify(expected.image)).digest('hex'),everyPathTopologyAndRngCompared:true,independentRectanglePlaneOracle:true};
}
export async function main(){
 const results=[];for(const size of [8,16,32])results.push(await compareStructuredCPU(size));
 const frozen=results.find(x=>x.size===32);assert.deepEqual(frozen.pathPowerHistogram,{'0':181,'0.75':511,'1.5':424,'3':423,'6':509});
 assert.deepEqual(frozen.mirrorCountHistogram,{'0':181,'1':933,'2':934});
 const report={allPass:true,actualGPUExecuted:false,browserValidated:false,physicalGPUValidated:false,oracle:'Independent BigInt RNG, analytic fixed camera frame, seven rectangle-plane intersections, axis reflection and dyadic transport; no production triangle/BVH/camera/RNG used for expectations',results};
 fs.writeFileSync(new URL('../research/plt-resident-structured-oracle-validation.json',import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
}
if(process.argv[1]&&pathToFileURL(process.argv[1]).href===import.meta.url)await main();
