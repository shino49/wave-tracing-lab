// Executes the actual WGSL through wgsl_reflect's CPU interpreter. This is a
// stronger check than reimplementing equations in JS, but is NOT GPU execution.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
import {prepareUTDScene,builtInSceneOBJ,evaluatePoint,computeReceiverPoint} from '../dist/utd-scene.js';
import {prepareMesh} from '../dist/mesh-geometry.js';
import {packUTDSceneGPU,UTD_GPU_UNIFORM_BYTES} from '../dist/utd-gpu-pack.js';
let reflection;
if(process.env.WGSL_REFLECT_MODULE) reflection=await import(pathToFileURL(process.env.WGSL_REFLECT_MODULE));
else reflection=await import('wgsl_reflect/wgsl_reflect.module.js');
const {WgslReflect,WgslParser,WgslExec}=reflection;
const source=fs.readFileSync(new URL('../dist/utd-functions.wgsl',import.meta.url),'utf8')+'\n'+
  fs.readFileSync(new URL('../dist/utd-compute.wgsl',import.meta.url),'utf8');
const reflected=new WgslReflect(source),ast=new WgslParser().parse(source);
assert.equal(reflected.uniforms[0].size,UTD_GPU_UNIFORM_BYTES);
assert.deepEqual(reflected.storage.map(s=>s.stride),[48,96,16,16,32]);
const variants=[
  ['plate transmission','screen',{}],
  ['convex box','box',{}],
  ['convex prism','wedge',{}],
  ['opaque slit wall','opaque-double-slit',{}],
  ['plate reflection','screen',{detectorCenter:[0,0,-20]}],
  ['box reflection','box',{detectorCenter:[0,0,-20]}],
  ['prism reflection','wedge',{detectorCenter:[0,0,-20]}],
  ['horizontal yaw','screen',{incidentDirection:[.1,0,Math.sqrt(.99)]}],
  ['vertical yaw','screen',{incidentDirection:[0,.1,Math.sqrt(.99)]}],
  ['combined yaw','box',{incidentDirection:[.1,.12,Math.sqrt(1-.01-.0144)]}],
  ['higher frequency','screen',{wavelengthMm:.01}],
  ['very high safe frequency','screen',{wavelengthMm:.005}],
  ['curvature rejected','screen',{sourceOrigin:[0,0,-20],sigma0Mm:.1}],
  ['receiver endpoints','screen',{detectorHeightMm:30}],
  ['soft boundary','screen',{boundary:'soft'}],
  ['hard boundary','screen',{boundary:'hard'}],
  ['shadow seam exact','screen',{detectorWidthMm:2.5}],
  ['shadow seam right','screen',{detectorWidthMm:2.5,detectorCenter:[1e-5,0,20]}],
  ['shadow seam left','screen',{detectorWidthMm:2.5,detectorCenter:[-1e-5,0,20]}],
  ['low frequency masked','screen',{wavelengthMm:100,phaseCurvatureLimit:Math.PI}],
  ['receiver inside region','screen',{detectorCenter:[0,0,2]}],
  ['no incoming contact','screen',{sourceOrigin:[40,0,-5],detectorCenter:[40,0,20]}],
];
const shift=[100000,-200000,300000];
function translatedOBJ(obj,offset) {
  return obj.split('\n').map(line=>line.startsWith('v ')?'v '+line.trim().split(/\s+/).slice(1).map((s,i)=>Number(s)+offset[i]).join(' '):line).join('\n');
}
const shifted=prepareMesh(translatedOBJ(builtInSceneOBJ('screen'),shift));
variants.push(['large translation',shifted,{sourceOrigin:shift.map((v,i)=>v+[0,0,-5][i]),detectorCenter:shift.map((v,i)=>v+[0,0,20][i])}]);
function rotateOBJ(obj,a,b) {
  return obj.split('\n').map(line=>{
    if(!line.startsWith('v '))return line;
    const [x,y,z]=line.trim().split(/\s+/).slice(1).map(Number);
    const yy=y*Math.cos(a)-z*Math.sin(a),zz=y*Math.sin(a)+z*Math.cos(a);
    return 'v '+[x*Math.cos(b)+zz*Math.sin(b),yy,-x*Math.sin(b)+zz*Math.cos(b)].join(' ');
  }).join('\n');
}
variants.push(['tilted 3D plate',prepareMesh(rotateOBJ(builtInSceneOBJ('screen'),.2,.35)),{}]);
variants.push(['tilted 3D box',prepareMesh(rotateOBJ(builtInSceneOBJ('box'),.15,-.25)),{}]);
variants.push(['tilted 3D prism',prepareMesh(rotateOBJ(builtInSceneOBJ('wedge'),-.1,.2)),{}]);
const report={execution:'CPU interpretation of actual WGSL via wgsl_reflect, not physical GPU',uniformBytes:160,storageStrides:[48,96,16,16,32],scenes:[],maxPowerError:0,maxGeometricError:0,validityMismatches:0,edgeCountMismatches:0};
for(const [label,input,state]of variants) {
  const scene=prepareUTDScene(typeof input==='string'?builtInSceneOBJ(input):input,state);
  const width=5,height=5,packed=packUTDSceneGPU(scene,width,height),out=new Float32Array(width*height*4).fill(-123);
  assert(packed.triangles.byteLength>=48&&packed.edges.byteLength>=96&&packed.facets.byteLength>=32);
  const bind={0:{0:packed.uniforms,1:packed.triangles,2:packed.edges,3:packed.adjacency,4:out,5:packed.facets}};
  // Three row batches exercise baseRow and prevent repeated writes to row zero.
  for(let base=0;base<height;base+=2){bind[0][0]=packed.createUniforms(base,Math.min(2,height-base));new WgslExec(ast).dispatchWorkgroups('main',[1,1,1],bind);}
  let maxPower=0,maxGeo=0,valid=0,invalid=0;
  for(let y=0;y<height;y++)for(let x=0;x<width;x++) {
    const cpu=evaluatePoint(scene,computeReceiverPoint(scene,x,y,width,height),{boundaryEpsilon:2e-6}),i=(y*width+x)*4;
    const got=[...out.slice(i,i+4)];assert(got.every(Number.isFinite),`${label}: finite GPU output`);
    assert.equal(cpu.valid,got[2]>.5,`${label} (${x},${y}) validity; ${cpu.invalidReasons}`);
    assert.equal(cpu.edgeCount,got[3],`${label} (${x},${y}) edge count`);
    if(cpu.valid)valid++;else invalid++;
    if(!scene.supported){assert.equal(got[0],0);assert.equal(got[1],0);continue;}
    const pe=Math.abs(cpu.f-got[0]),ge=Math.abs(cpu.geometricPower-got[1]);
    maxPower=Math.max(maxPower,pe);maxGeo=Math.max(maxGeo,ge);
    assert(pe<2e-5+.005*Math.abs(cpu.f),`${label} (${x},${y}) power: CPU=${cpu.f}, WGSL=${got[0]}, error=${pe}`);
    assert(ge<2e-5+.005*Math.abs(cpu.geometricPower),`${label} (${x},${y}) geometric power: ${ge}`);
  }
  report.maxPowerError=Math.max(report.maxPowerError,maxPower);report.maxGeometricError=Math.max(report.maxGeometricError,maxGeo);
  report.scenes.push({label,supported:scene.supported,points:25,valid,invalid,maxPowerError:maxPower,maxGeometricError:maxGeo});
}
const normal=prepareUTDScene(builtInSceneOBJ('screen'));
assert.throws(()=>packUTDSceneGPU(normal,98,5),/resolution/);
assert.throws(()=>packUTDSceneGPU(normal,5,5,1e-14),/seam epsilon/);
assert.throws(()=>packUTDSceneGPU({...normal,gpuSafety:{safe:false,reasons:['test guard']}},5,5),/numerical guard/);
const highK=prepareUTDScene(builtInSceneOBJ('screen'),{wavelengthMm:.0005});
assert.equal(highK.gpuSafety.safe,false);
assert.throws(()=>packUTDSceneGPU(highK,5,5),/numerical guard/);
report.points=report.scenes.reduce((s,c)=>s+c.points,0);
console.log(JSON.stringify(report,null,2));
fs.writeFileSync(new URL('../research/utd-wgsl-validation.json',import.meta.url),JSON.stringify(report,null,2)+'\n');
