// CPU fixtures exercise the gate's sampling policy, not physical GPU execution.
import assert from 'node:assert/strict';
import {prepareUTDScene,builtInSceneOBJ} from '../dist/utd-scene.js';
import {makeCPUReference,parityGate} from '../dist/utd-engine.js';
let checks=0;
const tolerance={absolute:2e-5,relative:.005,boundaryEpsilon:2e-6};
const copy=r=>({...r,data:new Float32Array(r.data)});
const caught=f=>{let error;try{f();}catch(e){error=e;}assert.ok(error,'Expected gate rejection');return error;};
const originalCoordinates=(w,h)=>[0,Math.floor((h-1)/4),Math.floor((h-1)/2),Math.floor(3*(h-1)/4),h-1].flatMap(y=>[0,Math.floor((w-1)/4),Math.floor((w-1)/2),Math.floor(3*(w-1)/4),w-1].map(x=>[x,y]));
const slit=prepareUTDScene(builtInSceneOBJ('opaque-double-slit'),{boundary:'soft',sourceOrigin:[0,0,-5]});
let defaultFixture,defaultGate;
for(const [size,expectedValid] of [[33,84],[65,248],[97,512]]){
 const engine=makeCPUReference(slit,size,size),fixture=await engine.run();engine.destroy();
 assert.equal(fixture.valid,expectedValid);
 const report=parityGate(slit,fixture);
 assert.equal(report.passed,true);assert.equal(report.points,81);assert.equal(report.validPoints,4);
 assert.deepEqual(report.tolerance,tolerance);assert.equal(report.coverage.strategy,'refined-9x9');
 assert.equal(report.coverage.basePoints,25);assert.equal(report.coverage.supplementalPoints,56);assert.equal(report.coverage.uniquePoints,81);assert.equal(report.coverage.exhaustive,false);
 assert.deepEqual(report.samples.slice(0,25).map(s=>[s.x,s.y]),originalCoordinates(size,size));
 assert.ok(report.samples.slice(0,25).every(s=>s.status==='both-invalid'));
 const a=(size-1)/8,b=7*(size-1)/8;
 assert.deepEqual(report.samples.filter(s=>s.status==='pass').map(s=>[s.x,s.y]),[[a,a],[b,a],[a,b],[b,b]]);
 assert.equal(new Set(report.samples.map(s=>`${s.x},${s.y}`)).size,81);
 if(size===33){defaultFixture=fixture;defaultGate=report;}
 checks++;
}
// Rectangular grids refine each axis independently, preserving the 25 base checks.
{
 const engine=makeCPUReference(slit,33,65),fixture=await engine.run();engine.destroy();
 const report=parityGate(slit,fixture);assert.equal(report.points,81);assert.equal(report.validPoints,4);
 assert.deepEqual(report.samples.slice(0,25).map(s=>[s.x,s.y]),originalCoordinates(33,65));
 assert.deepEqual(report.samples.filter(s=>s.status==='pass').map(s=>[s.x,s.y]),[[4,8],[28,8],[4,56],[28,56]]);checks++;
}
const selected=[4,4],index=(selected[1]*33+selected[0])*4;
assert.ok(!originalCoordinates(33,33).some(([x,y])=>x===selected[0]&&y===selected[1]));
for(const [label,modify,kind] of [
 ['power corruption',data=>data[index]+=1,'numericDisagreements'],
 ['geometric power corruption',data=>data[index+1]+=1,'numericDisagreements'],
 ['false-negative GPU mask',data=>data[index+2]=0,'maskDisagreements'],
 ['false-positive GPU mask',data=>data[(4*33+12)*4+2]=1,'maskDisagreements'],
 ['all GPU masks false',data=>{for(let i=2;i<data.length;i+=4)data[i]=0;},'maskDisagreements'],
]){
 const bad=copy(defaultFixture);modify(bad.data);const error=caught(()=>parityGate(slit,bad));
 assert.ok(error.parity[kind]>0,label);assert.equal(error.parity.points,81,label);assert.equal(error.parity.passed,false,label);
 assert.deepEqual(error.parity.tolerance,tolerance);
 assert.deepEqual(error.parity.samples.map(s=>[s.x,s.y]),defaultGate.samples.map(s=>[s.x,s.y]),'GPU output must not steer supplemental coordinates');
 checks++;
}
// Existing coarse-grid mismatches reject without buying extra opportunities to pass.
{
 const bad=copy(defaultFixture);bad.data[2]=1;const error=caught(()=>parityGate(slit,bad));
 assert.equal(error.parity.points,25);assert.equal(error.parity.coverage.strategy,'base-5x5');assert.ok(error.parity.maskDisagreements>0);checks++;
}
// The common screen keeps its original 25-check cost and behavior.
{
 const scene=prepareUTDScene(builtInSceneOBJ('screen'),{boundary:'soft'}),engine=makeCPUReference(scene,33,33),r=await engine.run();engine.destroy();
 const report=parityGate(scene,r);assert.equal(report.points,25);assert.equal(report.coverage.strategy,'base-5x5');assert.equal(report.coverage.supplementalPoints,0);checks++;
}
// Truly all-invalid scenes are never accepted merely because both masks agree.
{
 const scene=prepareUTDScene(builtInSceneOBJ('screen'),{detectorCenter:[0,0,2]}),engine=makeCPUReference(scene,33,33),r=await engine.run();engine.destroy();assert.equal(r.valid,0);
 const error=caught(()=>parityGate(scene,r));assert.equal(error.parity.points,81);assert.equal(error.parity.validPoints,0);assert.match(error.message,/没有有效/);checks++;
}
// Bounded refinement is still sampling: a narrow valid pixel between both grids
// must cause explicit rejection, not an exhaustive CPU search or false pass.
{
 const scene=prepareUTDScene(builtInSceneOBJ('screen'),{sourceOrigin:[40+6/33,0,-5],detectorCenter:[40,0,20],sigma0Mm:.001,wavelengthMm:.000001}),engine=makeCPUReference(scene,33,33),r=await engine.run();engine.destroy();assert.equal(r.valid,1);
 const error=caught(()=>parityGate(scene,r));assert.equal(error.parity.points,81);assert.equal(error.parity.validPoints,0);assert.equal(error.parity.coverage.exhaustive,false);checks++;
}
// Tiny grids retain the legacy 25 sample entries and do not add duplicate probes.
{
 const scene=prepareUTDScene(builtInSceneOBJ('screen'),{detectorCenter:[0,0,2]}),engine=makeCPUReference(scene,2,2),r=await engine.run();engine.destroy();
 const error=caught(()=>parityGate(scene,r));assert.equal(error.parity.points,25);assert.equal(error.parity.coverage.uniquePoints,4);assert.equal(error.parity.coverage.supplementalPoints,0);assert.equal(error.parity.coverage.exhaustive,true);checks++;
}
console.log(`Parity coverage: ${checks} groups passed; unchanged tolerances, 25 base checks, at most 81 total checks.`);
