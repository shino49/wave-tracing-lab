import assert from 'node:assert/strict';
import fs from 'node:fs';
import {besselJ0,besselJ1,makeFresnelEvaluator,fresnelPsfDensity,geometricPsfRadius,
  geometricPsfDensity,spectralQuadrature,generateKernels,sampleRadialXYZ,
  diskRectangleArea,integratePixelXYZ,buildPaddedKernel,renderPointDetector} from '../dist/pinhole-optics.js';
const hooks={yield:()=>Promise.resolve()};
function near(actual,expected,relative=1e-10,absolute=1e-13) {
  assert.ok(Math.abs(actual-expected)<=absolute+relative*Math.abs(expected),`${actual} != ${expected}`);
}
const oracle=JSON.parse(fs.readFileSync(new URL('../research/pinhole-fresnel-math-reference.json',import.meta.url)));
let maxRelativeOracleError=0,maxAbsoluteOracleError=0;
for(const row of oracle.bessel) {near(besselJ0(row.x),row.j0,2e-11,2e-14);near(besselJ1(row.x),row.j1,2e-11,2e-14);}
for(const row of oracle.rows) {
  const actual=fresnelPsfDensity(row.radiusMm,row),error=Math.abs(actual-row.density);
  maxRelativeOracleError=Math.max(maxRelativeOracleError,error/Math.max(1e-14,row.density));maxAbsoluteOracleError=Math.max(maxAbsoluteOracleError,error);
  near(actual,row.density,2e-8,2e-10);
}
// Nonzero finite-distance defocus: exact on-axis sin(beta/2)/(beta/2),
// including a strong Fresnel minimum; a Fraunhofer-only model fails this test.
for(const d of [.04,.16,.5])for(const s of [200,300])for(const nm of [360,830]){
  const e=makeFresnelEvaluator({diameterMm:d,depthMm:s,wavelengthNm:nm}),u=e.beta/2;
  near(e.density(0),e.areaMm2/(nm*1e-6*50)**2*(Math.sin(u)/u)**2,1e-11);
  for(const r of [0,.01,.1,1,11.31])assert.ok(Number.isFinite(e.density(r))&&e.density(r)>=0);
}
// A finite radial aperture in the detector converges toward, but is never
// divided by, the analytic infinite-plane mass one.
const evaluator=makeFresnelEvaluator({diameterMm:.04,depthMm:200,wavelengthNm:550});
function radialMass(radius) {
  const count=Math.ceil(radius/.01/2)*2,dr=radius/count;let sum=0;
  for(let i=0;i<=count;i++)sum+=(i===0||i===count?1:i%2?4:2)*2*Math.PI*i*dr*evaluator.density(i*dr);
  return sum*dr/3;
}
const mass8=radialMass(8),mass64=radialMass(64);
assert.ok(mass8>.97&&mass8<.999&&mass64>mass8&&mass64<1);
near(geometricPsfRadius({diameterMm:.5,depthMm:200}),.3125);
near(geometricPsfDensity(0,{diameterMm:.5,depthMm:200}),1/(Math.PI*.3125**2));
assert.equal(geometricPsfDensity(.4,{diameterMm:.5,depthMm:200}),0);
for(const r of [.024,.1,.3125]){
  near(diskRectangleArea(r,-1,1,-1,1),Math.PI*r*r);
  near(diskRectangleArea(r,0,1,0,1),Math.PI*r*r/4);
  assert.equal(diskRectangleArea(r,2,3,2,3),0);
}
const spectral=spectralQuadrature({spectralStepNm:1}),colors=Array.from({length:4},(_,b)=>[0,1,2].map(c=>spectral.reduce((s,q)=>s+q.basisXYZ[b][c],0)));
near(colors[0][1],.9889001165954525);
for(let b=1;b<4;b++)for(let c=0;c<3;c++)assert.ok(colors[b][c]>0&&colors[b][c]<=colors[0][c]);
const narrow=spectralQuadrature({mode:'narrowband',wavelengthNm:550});assert.equal(narrow.length,1);assert.equal(narrow[0].weightNm,1);
assert.throws(()=>spectralQuadrature({mode:'narrowband',wavelengthNm:850}),RangeError);
assert.throws(()=>makeFresnelEvaluator({diameterMm:1,depthMm:200,wavelengthNm:360}),RangeError);
assert.throws(()=>geometricPsfRadius({diameterMm:0,depthMm:200}),RangeError);
const kernel=await generateKernels({resolution:32,diameterMm:.16,depths:[200,300],spectralStepNm:5},hooks);
assert.equal(kernel.layers.length,2);assert.equal(kernel.layers[0].basis.length,4);assert.equal(kernel.spectralSamples,95);
near(kernel.throughput,Math.PI*.08**2/50**2);assert.ok(kernel.maxRadiusMm>Math.SQRT2*8);
assert.throws(()=>sampleRadialXYZ(kernel,0,0,kernel.maxRadiusMm+1),RangeError);
const geometric=await renderPointDetector(kernel,{kind:'geometric'},hooks);
for(let c=0;c<3;c++)near(geometric.powerXYZ[c],kernel.throughput*kernel.layers[0].basis[0].colorXYZ[c],2e-12);
const geometricEdge=await renderPointDetector(kernel,{kind:'geometric',centerMm:[4,0]},hooks);
for(let c=0;c<3;c++)near(geometricEdge.powerXYZ[c],geometric.powerXYZ[c]/2,2e-12);
const wave=await renderPointDetector(kernel,{},hooks);
for(let c=0;c<3;c++)assert.ok(wave.powerXYZ[c]>geometric.powerXYZ[c]*.98&&wave.powerXYZ[c]<geometric.powerXYZ[c]);
const padded=await buildPaddedKernel(kernel,0,0,{},hooks);assert.equal(padded.size,64);assert.equal(padded.channels.length,3);
for(const data of padded.channels){
  assert.ok(data.every(v=>Number.isFinite(v)&&v>=0));
  for(let i=0;i<padded.size;i++){assert.equal(data[32*padded.size+i],0);assert.equal(data[i*padded.size+32],0);}
  for(const [x,y]of[[1,2],[31,1],[7,8]]){near(data[y*64+x],data[(64-y)*64+(64-x)]);near(data[y*64+x],data[x*64+y]);}
}
assert.throws(()=>integratePixelXYZ(kernel,0,0,0,0,{pixelQuadrature:3}),RangeError);
await assert.rejects(generateKernels({resolution:32,diameterMm:.16,depths:[250]},{shouldAbort:()=>true}),{name:'AbortError'});
let yields=0;
await assert.rejects(generateKernels({resolution:32,diameterMm:.16,depths:[250]},{yield:async()=>{yields++},shouldAbort:()=>yields>1}),{name:'AbortError'});
await assert.rejects(buildPaddedKernel(kernel,0,0,{}, {signal:{aborted:true}}),{name:'AbortError'});
await assert.rejects(buildPaddedKernel(kernel,0,0,{paddedSize:32},hooks),RangeError);
// Same incident spectrum: increasing diameter doubles area quadratically.
const small=await generateKernels({resolution:32,diameterMm:.08,depths:[200],spectralStepNm:5},hooks);
near(kernel.throughput/small.throughput,4);
const smallGeometric=await renderPointDetector(small,{kind:'geometric'},hooks);
for(let c=0;c<3;c++)near(geometric.powerXYZ[c]/smallGeometric.powerXYZ[c],4,1e-11);
console.log(JSON.stringify({test:'pinhole-optics',passed:true,independentOracleCases:oracle.rows.length,maxRelativeOracleError,maxAbsoluteOracleError,finiteRadialMass:{radius8Mm:mass8,radius64Mm:mass64},spectralSamples:kernel.spectralSamples,geometricAreaScaling:4},null,2));
