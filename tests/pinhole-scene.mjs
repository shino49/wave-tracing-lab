import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {deflateSync} from 'node:zlib';
import {PINHOLE_BASIS_NAMES,createPinholeScene,buildPinholeSourceLayers,projectPinholePoint,createPinholePointCalibration,pinholeDepthLayers,pinholeDepthWeights,tracePinholeRay,pinholeBasisLinearRGB} from '../dist/pinhole-scene.js';

const scene=createPinholeScene(),noWait={yield:()=>{}};
assert.equal(scene.meshes.length,5);
assert.equal(scene.metadata.triangleCount,1032);
assert(scene.metadata.sourceDepthRangeMm[0]>=200&&scene.metadata.sourceDepthRangeMm[1]<=300);
assert.deepEqual(PINHOLE_BASIS_NAMES,['neutral','red','green','blue']);
assert(scene.meshes.every(m=>m.vertices.length>0&&m.faces.length>0));
assert.equal(scene.metadata.nearPupilOccluders,false);
assert.deepEqual(projectPinholePoint([8,-4,-200],50),{xMm:-2,yMm:1,depthMm:200,magnification:-.25});
const point=createPinholePointCalibration({positionMm:[1.234,-.567,-237]});
assert.equal(point.kind,'exact-point');assert.equal(point.projectedMass,1);
assert.equal(point.xMm,-50*1.234/237);assert.equal(point.yMm,50*.567/237);
assert(!('layers' in point));assert(!('raster' in point));
assert.throws(()=>projectPinholePoint([0,0,0]),RangeError);
assert.throws(()=>createPinholePointCalibration({basis:'magenta'}),RangeError);
assert.throws(()=>createPinholeScene({meshes:[]}),RangeError);
assert.throws(()=>pinholeDepthLayers(1),RangeError);
for(const count of [4,8,16]){
 const depths=pinholeDepthLayers(count);
 for(const s of [200,201.15,214.6,225,250,275.55,299.9,300]){
  const {indices,weights}=pinholeDepthWeights(s,depths);
  assert(Math.abs(weights[0]+weights[1]-1)<1e-14);
  assert(Math.abs(weights.reduce((v,w,k)=>v+w/depths[indices[k]],0)-1/s)<1e-16);
 }
}
const progress=[],small=await buildPinholeSourceLayers(scene,{resolution:128,depthCount:4},{...noWait,onProgress:p=>progress.push(p.fraction)});
assert(progress.length>10);assert.equal(progress.at(-1),1);assert(progress.every((v,i)=>i===0||v>=progress[i-1]));
assert(small.diagnostics===small.metadata&&small.overview===small.geometry);
assert.equal(small.layers.length,4);assert.equal(small.previewRGBA.length,128*128*4);
assert.equal(small.geometry.aperture.centerMm[2],0);assert.equal(small.geometry.detector.cornersMm[0][2],50);
assert(small.metadata.maxProjectedExtentMm<3);
assert(small.metadata.sourceGuardMm>1);
assert.equal(small.metadata.sourceBrightnessApplied,false);
assert.equal(small.metadata.throughputFactorApplied,false);
assert.equal(small.metadata.pixelAreaApplied,false);
let foreground=0;
for(let i=0;i<small.density.length;i++){
 let sum=0,weightedInvDepth=0;for(const layer of small.layers)for(const basis of layer.basis){assert(Number.isFinite(basis[i])&&basis[i]>=0);sum+=basis[i];weightedInvDepth+=basis[i]/layer.depthMm;}
 assert(Math.abs(sum-small.density[i])<8e-8);
 if(small.objectIndex[i]<0){assert.equal(sum,0);assert.equal(small.radiance[i],0);assert.deepEqual(Array.from(small.previewRGBA.slice(i*4,i*4+3)),[0,0,0]);}
 else{foreground++;if(sum>0)assert(Math.abs(weightedInvDepth/sum-1/small.depthMm[i])<3e-10);}
}
assert(foreground>1000);
for(let object=0;object<scene.meshes.length;object++)assert(small.objectIndex.includes(object));
for(let basis=0;basis<4;basis++)assert(small.basisIndex.includes(basis));
// Independent BVH ray intersections agree with triangle-raster depth and object.
let checked=0;
for(let y=3;y<128;y+=3)for(let x=2;x<128;x+=3){
 const i=y*128+x,qx=(x+.5)*8/128-4,qy=4-(y+.5)*8/128,hit=tracePinholeRay(scene,[0,0,0],[-qx/50,-qy/50,-1]);
 assert.equal(hit?.objectIndex??-1,small.objectIndex[i]);if(hit){assert(Math.abs(-hit.positionMm[2]-small.depthMm[i])<2e-5);checked++;}
}
assert(checked>200);
// Aperture diameter affects only visibility diagnostics. Throughput is applied
// later in optics, not in source density or the fixed physical scene framing.
const wide=await buildPinholeSourceLayers(scene,{resolution:128,depthCount:4,diameterMm:.6},noWait);
for(let k=0;k<4;k++)for(let b=0;b<4;b++)assert.deepEqual(wide.layers[k].basis[b],small.layers[k].basis[b]);
assert.deepEqual(wide.radiance,small.radiance);
assert(wide.metadata.pupilVisibility.affectedPixelCount>0);
assert(wide.metadata.pupilVisibility.newlyRevealedByRim.some(v=>v>0));
assert(wide.metadata.pupilVisibility.angularViewDisagreementByRim.some(v=>v>0));
assert(Math.abs(wide.metadata.pupilVisibility.parallaxBoundPixels-.4)<1e-14);
// Genuine first-surface occlusion in an independently specified geometry.
const quad=(z,x0,x1,y0,y1,basis,id)=>({id,name:id,basis,albedo:1,vertices:[[x0,y0,z],[x1,y0,z],[x1,y1,z],[x0,y1,z]],faces:[[0,1,2],[0,2,3]]});
const occlusionScene=createPinholeScene({meshes:[quad(-280,-10,10,-8,8,'blue','rear'),quad(-210,-3,3,-7,7,'red','front')],lights:[{direction:[0,0,1],strength:1}]});
const occluded=await buildPinholeSourceLayers(occlusionScene,{resolution:128,depthCount:4},noWait),mid=64*128+64;
assert.equal(occluded.objectIndex[mid],1);assert.equal(occluded.basisIndex[mid],1);assert(Math.abs(occluded.depthMm[mid]-210)<1e-5);
assert.equal(occluded.layers.reduce((s,l)=>s+l.basis[3][mid],0),0);
const colors=pinholeBasisLinearRGB();assert(colors[1][0]>colors[1][1]&&colors[1][0]>colors[1][2]);assert(colors[2][1]>colors[2][0]&&colors[2][1]>colors[2][2]);assert(colors[3][2]>colors[3][0]&&colors[3][2]>colors[3][1]);
// Validation and cancellation take place before expensive scene allocations.
for(const options of [{resolution:0},{resolution:NaN},{depthCount:0},{sensorWidthMm:0},{sensorWidthMm:2},{detectorDistanceMm:0},{diameterMm:Infinity},{diameterMm:.61},{depthMaxMm:250},{resolution:1024,depthCount:64}])await assert.rejects(buildPinholeSourceLayers(scene,options,noWait),RangeError);
const abort=new AbortController();abort.abort();await assert.rejects(buildPinholeSourceLayers(scene,{}, {signal:abort.signal}),{name:'AbortError'});
let yields=0;await assert.rejects(buildPinholeSourceLayers(scene,{resolution:128},{yield:()=>{yields++;},isCancelled:()=>yields>=2}),{name:'AbortError'});assert.equal(yields,2);
const medium=await buildPinholeSourceLayers(scene,{resolution:256,depthCount:8},noWait),large=await buildPinholeSourceLayers(scene,{resolution:512,depthCount:8},noWait);
assert.equal(large.resolution,512);assert.equal(large.layers[0].basis[0].length,512**2);assert.equal(large.pixelPitchMm,8/512);
assert.equal(small.metadata.maxProjectedExtentMm,large.metadata.maxProjectedExtentMm);
assert(Math.abs(medium.metadata.projectedSourceMass/large.metadata.projectedSourceMass-1)<.01);
assert(Math.abs(small.metadata.projectedSourceMass/large.metadata.projectedSourceMass-1)<.02);
const worst=await buildPinholeSourceLayers(scene,{resolution:512,depthCount:8,diameterMm:.6},noWait);
assert(worst.metadata.pupilVisibility.affectedSourceSignalShare<.03);
assert(worst.metadata.pupilVisibility.newlyRevealedSourceSignalShareByRim.every(v=>v<.03));

const report={status:'pass',scene:scene.metadata,tests:['Actual 1032-triangle geometry and all five objects visible','Inverted projection and exact off-grid point coordinates','Reciprocal-depth moment preservation at 4/8/16 layers','Radiance/depth/basis maps match independent BVH intersections','Layer-mass conservation and exact zero-radiance background','First-surface occlusion removes rear-object radiance','Source maps invariant under aperture diameter; no duplicated throughput','Center and full-rim source visibility, including newly revealed patches','128/256/512 fixed physical framing and source-mass convergence','Finite validation, allocation budget and cancellation'],independentRayChecks:checked,sourceMassByResolution:{128:small.metadata.projectedSourceMass,256:medium.metadata.projectedSourceMass,512:large.metadata.projectedSourceMass},visibilityDefault:large.metadata.pupilVisibility,visibilityMaximumDiameter:worst.metadata.pupilVisibility,timings512:large.timings,limitations:['One centre sample per source pixel; subpixel boundary integration is not claimed','Circular-pupil kernels retain centre-pupil visibility; partial masks are measured but not integrated','Directional diffuse lighting and synthetic spectra, not measured scene radiometry','Visibility diagnostic samples do not prove a worst-case wave-image error bound']};
if(process.argv.includes('--artifacts')){
 const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../research/pinhole-scene');await fs.mkdir(root,{recursive:true});
 const crc32=bytes=>{let crc=-1;for(const byte of bytes){crc^=byte;for(let j=0;j<8;j++)crc=(crc>>>1)^((crc&1)?0xedb88320:0);}return (crc^-1)>>>0;};
 const chunk=(name,data)=>{const n=Buffer.from(name),len=Buffer.alloc(4),crc=Buffer.alloc(4);len.writeUInt32BE(data.length);crc.writeUInt32BE(crc32(Buffer.concat([n,data])));return Buffer.concat([len,n,data,crc]);};
 const png=(n,rgba)=>{const header=Buffer.alloc(13);header.writeUInt32BE(n,0);header.writeUInt32BE(n,4);header[8]=8;header[9]=6;const raw=Buffer.alloc(n*(n*4+1));for(let y=0;y<n;y++)raw.set(rgba.subarray(y*n*4,(y+1)*n*4),y*(n*4+1)+1);return Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),chunk('IHDR',header),chunk('IDAT',deflateSync(raw)),chunk('IEND',Buffer.alloc(0))]);};
 await fs.writeFile(path.join(root,'ideal-inverted-source-512.png'),png(512,large.previewRGBA));
 const upright=new Uint8ClampedArray(large.previewRGBA.length);for(let i=0;i<512**2;i++)upright.set(large.previewRGBA.subarray(i*4,i*4+4),(512**2-1-i)*4);
 await fs.writeFile(path.join(root,'source-objects-upright-512.png'),png(512,upright));
 const visibility=new Uint8ClampedArray(512**2*4);for(let i=0;i<512**2;i++){const v=worst.visibilityMask[i];visibility.set(v?[255,80,25,255]:large.objectIndex[i]>=0?[40,55,75,255]:[0,0,0,255],4*i);}await fs.writeFile(path.join(root,'pupil-visibility-affected-d060-512.png'),png(512,visibility));
 await fs.writeFile(path.join(root,'geometry.json'),JSON.stringify(large.overview));
 await fs.writeFile(path.join(root,'validation.json'),JSON.stringify(report,null,2)+'\n');
}
console.log(JSON.stringify(report,null,2));
