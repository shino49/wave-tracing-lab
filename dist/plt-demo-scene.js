/** A finite 3-D aperture bench for native backward PLT.
 * The complete image is traced through actual triangles. No screen-space wave
 * pattern is added. Local UTD changes aperture/edge response; this is not a
 * global coherent-field or calibrated-radiance demonstration.
 */
import {preparePLTScene} from './plt-geometry.js';
import {prepareMesh,makePresetOBJ} from './mesh-geometry.js';

export const PLT_DEMO_SCENE_ID='slitbench-demo';
export const PLT_DEMO_DEFAULTS=Object.freeze({
 scene:Object.freeze({preset:PLT_DEMO_SCENE_ID,emission:8,albedo:.65}),
 camera:Object.freeze({yaw:0,pitch:0,distanceScale:1}),
 options:Object.freeze({width:64,height:48,samplesPerPixel:128,maxDepth:2,wavelengthMm:.2,spectralMode:'monochrome',wave:true,nee:true,boundary:'unpolarized',seed:7391}),
 progressive:Object.freeze({tileSize:32,batchSamples:4,maxRuntimeMs:600000,previewIntervalMs:100}),
 exposure:0,
 title:'有限孔径 · 三维光学台',
 description:'有限面光源、厚遮挡板、不透明盒体与棱柱、双色地面共同参与实际光传输。局部 UTD 改变孔径与边缘响应；相干性限于单个交互区域。',
 scaleNote:'单位 mm · 孔径 1.2 × 2.9 · 挡板厚 0.4 · 默认 λ 0.2 mm，尺度化红外演示',
 display:Object.freeze({kind:'fixed-log',gain:4,whitePoint:8,gamma:2.2}),
 limits:Object.freeze({wavelengthMinMm:.025,wavelengthMaxMm:.2,validatedWidth:64,validatedHeight:48,maxDepth:2}),
});

const quad=(vertices,materialIndex)=>({mesh:{vertices,faces:[[0,1,2],[0,2,3]]},materialIndex,surface:'open-plate'});
const rectangle=(x0,x1,y0,y1,z,m)=>quad([[x0,y0,z],[x1,y0,z],[x1,y1,z],[x0,y1,z]],m);
const solid=(width,height,depth,translation,materialIndex,rotationDeg=[0,0,0],shape='box')=>({mesh:prepareMesh(makePresetOBJ(shape,{width,height,depth}),{translation,rotationDeg}),materialIndex,surface:'auto'});

export function createPLTDemoScene({emission=8,albedo=.65}={}){
 if(!Number.isFinite(emission)||emission<0||emission>100)throw new RangeError('Demo emitter radiance must be within 0…100');
 if(!Number.isFinite(albedo)||albedo<0||albedo>1)throw new RangeError('Demo floor albedo must be within 0…1');
 const materials=[
  {type:'diffuse',albedo,emission:0},
  {type:'diffuse',albedo:.025,emission:0},
  {type:'diffuse',albedo:0,emission,twoSidedEmission:true},
  {type:'diffuse',albedo:.8,emission:0},
  {type:'diffuse',albedo:0,emission:.075*emission,twoSidedEmission:true},
  {type:'diffuse',albedo:.15,emission:0},
 ];
 // Shared floor topology makes tile boundaries material changes, not fictitious
 // independent open-plate diffraction edges.
 const nx=8,nz=8,vertices=[],faces=[],materialIndices=[];
 for(let z=0;z<=nz;z++)for(let x=0;x<=nx;x++)vertices.push([-5+x*1.25,-1.5,-6.2+z*1.4]);
 for(let z=0;z<nz;z++)for(let x=0;x<nx;x++){
  const a=z*(nx+1)+x,b=a+1,c=a+nx+2,d=a+nx+1,m=(x+z)%2===0?0:5;
  faces.push([a,c,b],[a,d,c]);materialIndices.push(m,m);
 }
 const entries=[
  {mesh:{vertices,faces},materialIndices,surface:'open-plate'},
  rectangle(-5,5,-1.5,4.5,-6.2,0),
  quad([[-5,-1.5,5],[-5,-1.5,-6.2],[-5,4.5,-6.2],[-5,4.5,5]],0),
  rectangle(-2.8,2.8,-1.3,3.3,-5.8,2),
  solid(2.5,4.8,.4,[-1.85,.9,-3],1),
  solid(2.5,4.8,.4,[1.85,.9,-3],1),
  solid(1.2,1.1,.4,[0,-.95,-3],1),
  solid(1.2,.8,.4,[0,2.9,-3],1),
  solid(1.1,1.4,1.1,[1.6,-.8,.6],3,[0,25,0]),
  solid(1.3,1.3,1.3,[-1.8,-.85,.5],3,[0,-18,0],'triangular-prism'),
  quad([[-3,4,0],[-3,4,3],[3,4,3],[3,4,0]],4),
 ];
 const scene=preparePLTScene(entries,{materials});
 scene.emitters=scene.triangles.filter(t=>materials[t.materialIndex].emission>0);
 scene.emitterArea=scene.emitters.reduce((sum,t)=>sum+t.area,0);
 scene.camera={position:[5.5,3.7,11],target:[0,.3,-2],up:[0,1,0],fov:42};
 scene.preset=PLT_DEMO_SCENE_ID;
 scene.demoMetadata={name:PLT_DEMO_DEFAULTS.title,lengthUnit:'mm',apertureWidth:1.2,apertureHeight:2.9,baffleThickness:.4,roomWidth:10,sourceRadiance:emission,fillRadiance:.075*emission,
  sourceSize:[5.6,4.6],sourcePosition:[0,1,-5.8],floorTiles:[nx,nz],triangles:scene.triangles.length,wedges:scene.wedges.length,
  scope:'finite-depth native positive path ensemble; local UTD aperture/edge response; relative camera units; no global coherence',
  validation:'Compare the same wave path tape with UTD coefficients disabled to separate edge response from finite-cone footprint and sampling. Resolution changes the native camera footprint.'};
 return scene;
}
