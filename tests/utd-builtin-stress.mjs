// Synthetic builtin-error stress, NOT a reproduction of the user's GPU/driver.
// Runs the actual browser compute-engine module through native Dawn with file-
// backed fetch. The negative control changes shader text only in memory.
// Use an executable WEBGPU_EXECUTION_BACKEND and optional WEBGPU_MODULE.
// Driver selection (e.g. VK_ICD_FILENAMES) is controlled externally.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const root=path.resolve(process.argv[2]||process.cwd());
const backend=process.env.WEBGPU_EXECUTION_BACKEND;
assert.ok(['vulkan','metal','d3d11','d3d12','opengl','opengles'].includes(backend),
  'Set WEBGPU_EXECUTION_BACKEND to an executable native backend; null is refused.');
const load=file=>import(pathToFileURL(path.join(root,'dist',file)).href);
const [{prepareUTDScene,builtInSceneOBJ,evaluatePoint,computeReceiverPoint},{prepareMesh},{makeGPUEngine}]=await Promise.all([
  load('utd-scene.js'),load('mesh-geometry.js'),load('utd-gpu-engine.js'),
]);
const {create,globals}=process.env.WEBGPU_MODULE
  ? await import(pathToFileURL(path.resolve(process.env.WEBGPU_MODULE)).href)
  : await import('webgpu');
Object.assign(globalThis,globals);
const read=name=>fs.readFileSync(path.join(root,'dist',name),'utf8');
const hardened=read('utd-functions.wgsl'),compute=read('utd-compute.wgsl');
assert.match(hardened,/fn\s+utd_cos_small\s*\(/,'Current shader must contain small-angle hardening');
assert.match(hardened,/abs\(t\*utd_sinc\(t\)\)/,'Current shader must use stable t*sinc(t)');
function replaceFunction(source,name,replacement){
  const start=source.indexOf(`fn ${name}(`);assert.ok(start>=0,`Missing function ${name}`);
  const open=source.indexOf('{',start);let depth=1,end=open+1;
  for(;end<source.length&&depth;end++){if(source[end]==='{')depth++;else if(source[end]==='}')depth--;}
  assert.equal(depth,0,`Unbalanced function ${name}`);
  return source.slice(0,start)+replacement+source.slice(end);
}
let vulnerable=replaceFunction(hardened,'utd_sinc',`fn utd_sinc(x:f32)->f32 {
  if(abs(x)<0.001) { let q=x*x; return 1.0-q/6.0+q*q/120.0; }
  return sin(x)/x;
}`);
assert.equal((vulnerable.match(/utd_cos_small\(h\)/g)||[]).length,1);
assert.equal((vulnerable.match(/abs\(t\*utd_sinc\(t\)\)/g)||[]).length,1);
vulnerable=vulnerable.replace('utd_cos_small(h)','cos(h)').replace('abs(t*utd_sinc(t))','abs(sin(t))');
function stressed(source,amplitude){
  return source.replace(/\bsin\s*\(/g,'utd_synthetic_sin(')+`\n
// Deliberate synthetic error. This does not identify any real driver's behavior.
fn utd_synthetic_sin(x:f32)->f32 {
  return sin(x)+select(0.0,sign(x)*${amplitude},abs(x)>0.000001&&abs(x)<0.1);
}\n`;
}
const mesh=prepareMesh(builtInSceneOBJ('screen')),c=mesh.bounds.center;
const scene=prepareUTDScene(mesh,{wavelengthMm:.05,sigma0Mm:1,incidentDirection:[0,0,1],
  sourceOrigin:[c[0],c[1],c[2]-Math.max(5,Math.hypot(...mesh.bounds.size))],
  detectorCenter:[c[0],c[1],c[2]+20],detectorU:[1,0,0],detectorV:[0,1,0],
  detectorWidthMm:6,detectorHeightMm:6,surface:'open-plate',boundary:'soft'});
let valid=0,peak=0;
for(let y=0;y<33;y++)for(let x=0;x<33;x++){
  const p=evaluatePoint(scene,computeReceiverPoint(scene,x,y,33,33));
  if(p.valid){valid++;peak=Math.max(peak,Math.fround(p.f));}
}
assert.equal(valid,825);assert.equal(peak.toPrecision(5),'1.4794');
let gpu=create([`backend=${backend}`]);
const adapter=await gpu.requestAdapter();assert.ok(adapter,'No executable adapter');
const provenance=Object.fromEntries(['vendor','architecture','device','description','isFallbackAdapter'].map(k=>[k,adapter.info[k]]));
const report={type:'Synthetic sin-error stress; not the user’s actual adapter failure',adapter:provenance,
  requestedBackend:backend,perturbation:'sin(x) ± sign(x)*0.0001 only where 0.000001 < abs(x) < 0.1',
  defaultScene:{width:33,height:33,valid,peak},cases:[]};
const previousNavigator=Object.getOwnPropertyDescriptor(globalThis,'navigator'),previousFetch=globalThis.fetch;
Object.defineProperty(globalThis,'navigator',{configurable:true,value:{gpu,userAgent:`Node ${process.version}; synthetic WGSL stress`}});
let activeSource=hardened;
globalThis.fetch=async input=>{
  assert.ok(['./utd-functions.wgsl','./utd-compute.wgsl'].includes(String(input)),`Unexpected fetch ${input}`);
  return {ok:true,status:200,text:async()=>String(input).endsWith('utd-functions.wgsl')?activeSource:compute};
};
try{
  for(const amplitude of [0.0001,-0.0001]){
    for(const [label,source,shouldPass] of [['hardened',hardened,true],['vulnerable negative control',vulnerable,false]]){
      activeSource=stressed(source,amplitude);
      const engine=await makeGPUEngine(scene,33,33);let output,error;
      try{output=await engine.run();}catch(e){error=e;}finally{engine.destroy();}
      if(shouldPass){
        assert.equal(error,undefined,`${label} with ${amplitude}: ${error?.message}`);
        assert.equal(output.parity.passed,true);assert.equal(output.valid,825);
        assert.equal(output.parity.samples.length,25,'Successful gate retains all samples');
        report.cases.push({label,amplitude,expected:'pass',passed:true,parity:output.parity});
        console.log(`Synthetic ${amplitude} sin stress: hardened shader PASS, ratio ${output.parity.normalizedToleranceRatio}`);
      }else{
        assert.ok(error,'Vulnerable negative control must fail unchanged parity gate');
        assert.ok(error.diagnostic,'Actual engine must retain failure diagnostics');
        assert.equal(error.diagnostic.stage,'runtime-parity');
        assert.equal(error.diagnostic.parity.passed,false);
        assert.equal(error.diagnostic.parity.samples.length,25,'Failure diagnostics must retain all 25 samples');
        assert.ok(error.diagnostic.parity.numericDisagreements>0,'Must reproduce numerical mismatch, not another failure');
        assert.equal(error.diagnostic.parity.tolerance.absolute,2e-5);
        assert.equal(error.diagnostic.parity.tolerance.relative,.005);
        report.cases.push({label,amplitude,expected:'reject',passed:true,diagnostic:error.diagnostic});
        console.log(`Synthetic ${amplitude} sin stress: vulnerable shader correctly REJECTED, ratio ${error.diagnostic.parity.normalizedToleranceRatio}; 25 samples retained`);
      }
    }
  }
  report.passed=true;
  if(process.env.WEBGPU_STRESS_REPORT)fs.writeFileSync(process.env.WEBGPU_STRESS_REPORT,JSON.stringify(report,null,2)+'\n');
  console.log('Synthetic stress adapter:',JSON.stringify(provenance));
}finally{
  globalThis.fetch=previousFetch;
  if(previousNavigator)Object.defineProperty(globalThis,'navigator',previousNavigator);else delete globalThis.navigator;
  gpu=null;
}
