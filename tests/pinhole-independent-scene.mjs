/** Independent source-raster / BVH and physical-resolution admission checks. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createPinholeScene,buildPinholeSourceLayers,tracePinholeRay} from '../dist/pinhole-scene.js';
const scene=createPinholeScene(),hooks={yield:()=>{}},sources=[];
// Two layers reduce audit memory only: raw geometry, illumination, and density
// maps are identical to the runtime's eight-layer mode; PSF depth error has its
// own eight-layer test in pinhole-independent.mjs.
for(const resolution of [512,1024])sources.push(await buildPinholeSourceLayers(scene,{resolution,diameterMm:.5,depthCount:2},hooks));
let rayCount=0,maxDepthError=0,maxCos4Error=0,maxLayerError=0;let seed=0x51524731;
const next=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed;};
for(const source of sources){
 const n=source.resolution,pitch=source.pixelPitchMm;
 for(let k=0;k<2048;k++){
  const x=(next()>>>12)%n,y=(next()>>>12)%n,i=y*n+x,qx=(x+.5-n/2)*pitch,qy=(n/2-y-.5)*pitch;
  const hit=tracePinholeRay(scene,[0,0,0],[-qx/50,-qy/50,-1]);rayCount++;
  assert.equal(source.objectIndex[i],hit?.objectIndex??-1,'Center-ray raster / BVH first object mismatch');
  if(hit){const error=Math.abs(source.depthMm[i]+hit.positionMm[2]);maxDepthError=Math.max(maxDepthError,error);assert.ok(error<3e-5,'Center-ray depth mismatch');}
 }
 for(let i=0;i<n*n;i++){
  const x=i%n,y=Math.floor(i/n),qx=(x+.5-n/2)*pitch,qy=(n/2-y-.5)*pitch,expected=source.radiance[i]/(1+(qx*qx+qy*qy)/2500)**2;
  maxCos4Error=Math.max(maxCos4Error,Math.abs(source.density[i]-expected));
  let layer=0;for(const d of source.layers)for(const b of d.basis)layer+=b[i];
  maxLayerError=Math.max(maxLayerError,Math.abs(layer-source.density[i]));
  if(source.objectIndex[i]<0)assert.equal(source.density[i],0,'Exact black source exterior');
 }
 assert.ok(source.metadata.maxProjectedExtentMm<3,'All nonblack mesh geometry must lie in central six millimetres');
}
assert.ok(maxCos4Error<1e-7&&maxLayerError<1e-7,'Source density unit/layer conservation mismatch');
const masses=sources.map(s=>s.metadata.projectedSourceMass),relativeMassChange=Math.abs(masses[1]-masses[0])/masses[1];
assert.ok(relativeMassChange<.005,'Source-raster total mass is not converging within 0.5%');
const report={passed:true,sceneId:scene.id,rayCount,maxDepthErrorMm:maxDepthError,maxCos4Error,maxLayerConservationError:maxLayerError,
 sourceMassByResolution:{512:masses[0],1024:masses[1]},relativeMassChange,
 scope:'Fixed-scene center-ray source-raster mass convergence, not a certified boundary-error or optical-image bound',
 sourceDepthRangeMm:scene.metadata.sourceDepthRangeMm,fullProjectedExtentMm:sources[0].metadata.maxProjectedExtentMm,
 visibilityAtD05:sources.map(s=>({resolution:s.resolution,...s.metadata.pupilVisibility}))};
fs.writeFileSync(new URL('../research/pinhole-independent-scene-validation.json',import.meta.url),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({passed:true,rayCount,maxDepthErrorMm:maxDepthError,maxCos4Error,maxLayerError,masses,relativeMassChange,visibilityAtD05:report.visibilityAtD05.map(v=>({resolution:v.resolution,affectedVisiblePixelShare:v.affectedVisiblePixelShare,affectedSourceSignalShare:v.affectedSourceSignalShare,newlyRevealedSourceSignalShareByRim:v.newlyRevealedSourceSignalShareByRim}))},null,2));
