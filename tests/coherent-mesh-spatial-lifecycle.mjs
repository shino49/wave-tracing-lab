import assert from 'node:assert/strict';
import {executeCoherentMeshTask,prepareCoherentMeshTask} from '../dist/coherent-mesh-runtime.js';
import {validateCoherentSceneSource} from '../dist/coherent-mesh-scene.js';
const base={resolution:16,pixelQuadrature:1,backend:'cpu'},source={...prepareCoherentMeshTask(base).source,spatial:{type:'disk',radiusMm:.03,normal:[0,0,1],samples:3,seed:4}};
const channels=['coherent','incoherent','geometric','crossTerm','coherentXYZ','incoherentXYZ','geometricXYZ'];
for(const normal of [[0,0,0],[NaN,0,1],Array(3),[0,Infinity,1]])assert.throws(()=>validateCoherentSceneSource({...source,spatial:{...source.spatial,normal}}));
assert.throws(()=>prepareCoherentMeshTask({...base,source:{...source,position:[0,3.05,50]}}),/intersect|contain/);
const progress=[],previews=[];const result=await executeCoherentMeshTask({...base,source},{progress:p=>progress.push(p.fraction),preview:p=>previews.push(p)});
assert.equal(result.spatialSamples,3);assert.equal(result.spatialComplete,true);assert.equal(result.diagnostics.geometryReuseAcrossSources,false);
assert(progress.every((p,i)=>!i||p>=progress[i-1]));assert.equal(progress.at(-1),1);assert(previews.some(p=>!p.spatialComplete));assert(previews.every(p=>p.width===16&&p.height===16&&!p.complete));
const controller=new AbortController();let count=0;
await assert.rejects(executeCoherentMeshTask({...base,source},{signal:controller.signal,preview:p=>{if(p.spatialSamplesIntegrated===2){count++;controller.abort();}}}),{name:'AbortError'});assert.equal(count,1);
const restart=await executeCoherentMeshTask({...base,source});for(const k of channels)assert.deepEqual(restart[k],result[k]);
let calls=0,resets=0,destroyed=0;const failing=await executeCoherentMeshTask({...base,backend:'auto',source},{gpuFactory:async()=>({backend:{kind:'test'},destroy(){destroyed++},async evaluate({sampleCount}){calls++;if(calls===3)throw Error('deliberate failure on later source pass');return {coherent:new Float64Array(sampleCount).fill(777),incoherent:new Float64Array(sampleCount).fill(888)};}}),reset:()=>resets++});
assert.equal(resets,1);assert.equal(destroyed,1);assert.equal(failing.diagnostics.attempt,1);for(const k of channels)assert.deepEqual(failing[k],result[k]);assert.equal(failing.diagnostics.productionCPUShadow,false);
console.log('PASS disk validation; whole-support detector exclusion; monotone native-grid spatial progress; cancellation during second source; deterministic restart; GPU failure clears all scalar/XYZ spatial accumulation before CPU replay');
// Source-dependent reflection and UTD data may not be reused from disk center.
const {prepareCoherentSpatialSources}=await import('../dist/coherent-mesh-spatial.js');
for(const [preset,reciprocal] of [['two-mirror',false],['double-slit',false],['reflected-edge',false],['reflected-edge',true]]){
 const options={...base,preset,maxReflections:2,diffraction:preset!=='two-mirror',mixedDiffraction:preset==='reflected-edge',spectrum:{mode:'lines',lines:[{wavelengthNm:460,powerFraction:1},{wavelengthNm:650,powerFraction:2}]}},prepared=prepareCoherentMeshTask(options),point=reciprocal?{...prepared.source,position:prepared.detector.center}:prepared.source,disk={...point,spatial:{type:'disk',radiusMm:.005,normal:[0,0,1],samples:2,seed:7391}},nodes=prepareCoherentSpatialSources(disk);
 if(reciprocal)options.detector={...prepared.detector,center:prepared.source.position,width:.5,height:.5};
 const mixed=await executeCoherentMeshTask({...options,source:disk}),refs=[];for(const node of nodes)refs.push(await executeCoherentMeshTask({...options,source:node.source}));
 console.log('Source-route coverage',preset,reciprocal?'reciprocal':'forward',JSON.stringify({D:mixed.diagnostics.diffractedRoutes,RD:mixed.diagnostics.reflectionDiffractionRoutes,DR:mixed.diagnostics.diffractionReflectionRoutes,R:mixed.diagnostics.reflectedRoutes}));
 if(preset!=='two-mirror')assert(mixed.diagnostics.diffractedRoutes>0,`${preset} must exercise diffraction`);
 if(preset==='reflected-edge')assert((reciprocal?mixed.diagnostics.diffractionReflectionRoutes:mixed.diagnostics.reflectionDiffractionRoutes)>0,reciprocal?'must exercise DR':'must exercise RD');
 for(const key of channels)for(let i=0;i<mixed[key].length;i++){const expected=(refs[0][key][i]+refs[1][key][i])/2;assert(Math.abs(mixed[key][i]-expected)<=1e-20+Math.abs(expected)*1e-12,`${preset}/${key}/${i}`);}
}
console.log('PASS disk source recomputation matches independent source renders for RR, D, RD and DR with exact two-line XYZ');
