// Executes the unchanged WGSL with native Dawn. Requires an explicitly selected
// executable backend. A software adapter is real WGSL execution, not physical GPU
// evidence; the returned adapter provenance must accompany every saved report.
// Run from the repo root, or pass that root as the first argument.
// WEBGPU_MODULE can point at an external installation's webgpu/index.js.
// WEBGPU_EXECUTION_REPORT optionally names an output JSON report.
// Driver discovery (e.g. VK_ICD_FILENAMES) is entirely caller-controlled.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
const root=path.resolve(process.argv[2]||process.cwd());
const backend=process.env.WEBGPU_EXECUTION_BACKEND;
assert.ok(['vulkan','metal','d3d11','d3d12','opengl','opengles'].includes(backend),
  'Set WEBGPU_EXECUTION_BACKEND to an executable native backend; null is refused.');
const load=file=>import(pathToFileURL(path.join(root,'dist',file)).href);
const [{prepareUTDScene,builtInSceneOBJ,evaluatePoint,computeReceiverPoint},{prepareMesh},{packUTDSceneGPU},{parityGate}]=await Promise.all([
  load('utd-scene.js'),load('mesh-geometry.js'),load('utd-gpu-pack.js'),load('utd-engine.js'),
]);
const {create,globals}=process.env.WEBGPU_MODULE
  ? await import(pathToFileURL(path.resolve(process.env.WEBGPU_MODULE)).href)
  : await import('webgpu');
const webgpuPackagePath=process.env.WEBGPU_MODULE?path.join(path.dirname(path.resolve(process.env.WEBGPU_MODULE)),'package.json'):createRequire(import.meta.url).resolve('webgpu/package.json');
const webgpuVersion=JSON.parse(fs.readFileSync(webgpuPackagePath,'utf8')).version;
Object.assign(globalThis,globals);
let gpu=create([`backend=${backend}`]);
const adapter=await gpu.requestAdapter();assert.ok(adapter,`No executable ${backend} adapter available`);
const adapterInfo=Object.fromEntries(['vendor','architecture','device','description','isFallbackAdapter'].map(k=>[k,adapter.info[k]]));
const device=await adapter.requestDevice();
const sourceFiles=['utd-functions.wgsl','utd-compute.wgsl'];
const source=sourceFiles.map(name=>fs.readFileSync(path.join(root,'dist',name),'utf8')).join('\n');
const sourceHashes=Object.fromEntries(sourceFiles.map(name=>[name,createHash('sha256').update(fs.readFileSync(path.join(root,'dist',name))).digest('hex')]));
const report={execution:'Native Dawn WGSL dispatch and buffer readback; software adapters are not physical GPU evidence',
  requestedBackend:backend,adapter:adapterInfo,sourceHashes,nodeVersion:process.version,webgpuVersion,
  tolerance:{absolute:2e-5,relative:.005},scenes:[]};
const failures=[];
const previousNavigator=Object.getOwnPropertyDescriptor(globalThis,'navigator');
const previousFetch=globalThis.fetch;
Object.defineProperty(globalThis,'navigator',{configurable:true,value:{gpu,userAgent:`Node ${process.version}; native Dawn ${webgpuVersion}`}});
globalThis.fetch=async input=>{
  const name=String(input).replace(/^\.\//,'');
  assert.ok(sourceFiles.includes(name),`Unexpected test fetch: ${input}`);
  return {ok:true,status:200,text:async()=>fs.readFileSync(path.join(root,'dist',name),'utf8')};
};
try {
  const module=device.createShaderModule({label:'UTD execution regression',code:source});
  const info=await module.getCompilationInfo();
  assert.equal(info.messages.filter(m=>m.type==='error').length,0,'WGSL compiler errors');
  const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});
  const cases=[33,65,97].map(n=>({label:`Exact UI defaults ${n}x${n}`,preset:'screen',size:n,ui:true,state:{}}));
  cases.push(...[33,65,97].map(n=>({label:`Exact opaque-slit UI defaults ${n}x${n}`,preset:'opaque-double-slit',size:n,ui:true,state:{}})));
  cases.push(...['box','wedge'].map(preset=>({label:`Exact ${preset} UI defaults 33x33`,preset,size:33,ui:true,state:{}})));
  for(const [label,preset,state] of [
    ['Plate transmission','screen',{}],['Convex box','box',{}],['Convex prism','wedge',{}],
    ['Plate reflection','screen',{detectorCenter:[0,0,-20]}],['Hard boundary','screen',{boundary:'hard'}],
    ['Horizontal yaw','screen',{incidentDirection:[.1,0,Math.sqrt(.99)]}],
    ['Higher frequency','screen',{wavelengthMm:.01}],
  ])cases.push({label,preset,state,size:5});
  for(const item of cases){
    const mesh=prepareMesh(builtInSceneOBJ(item.preset)),c=mesh.bounds.center;
    const state=item.ui?{wavelengthMm:.05,sigma0Mm:1,incidentDirection:[0,0,1],
      sourceOrigin:[c[0],c[1],c[2]-Math.max(5,Math.hypot(...mesh.bounds.size))],
      detectorCenter:[c[0],c[1],c[2]+20],detectorU:[1,0,0],detectorV:[0,1,0],
      detectorWidthMm:6,detectorHeightMm:6,surface:['box','wedge'].includes(item.preset)?'closed-solid':'open-plate',boundary:'soft'}:item.state;
    const scene=prepareUTDScene(mesh,state),w=item.size,h=item.size,packed=packUTDSceneGPU(scene,w,h);
    const resources=[];
    try {
      const make=(size,usage,data)=>{const b=device.createBuffer({size:Math.max(16,Math.ceil(size/16)*16),usage});resources.push(b);if(data)device.queue.writeBuffer(b,0,data);return b;};
      const uniforms=packed.createUniforms(0,8),uniform=make(uniforms.byteLength,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,uniforms);
      const storage=a=>make(a.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,a);
      const tri=storage(packed.triangles),edges=storage(packed.edges),adjacency=storage(packed.adjacency),facets=storage(packed.facets);
      const bytes=w*h*16,out=make(bytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC),read=make(bytes,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
      const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[uniform,tri,edges,adjacency,out,facets].map((buffer,binding)=>({binding,resource:{buffer}}))});
      device.pushErrorScope('validation');
      for(let base=0;base<h;base+=8){
        device.queue.writeBuffer(uniform,0,packed.createUniforms(base,Math.min(8,h-base)));
        const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
        pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(w/8),1);pass.end();
        device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
      }
      const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(out,0,read,0,bytes);device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);const data=new Float32Array(read.getMappedRange(0,bytes).slice(0));read.unmap();
      const error=await device.popErrorScope();assert.equal(error,null,error?.message);
      const r={label:item.label,preset:item.preset,width:w,height:h,parameters:scene.params,
        valid:0,total:w*h,cpuPeak:0,gpuPeak:0,maskMismatches:0,edgeMismatches:0,numericalFailures:0,
        maxToleranceRatio:0,maxAbsolutePowerError:0,gate:null,gatePoints:[],failures:[]};
      try{r.gate=parityGate(scene,{width:w,height:h,data});}catch(error){r.gate={passed:false,error:error.message};}
      const coords=[0,Math.floor((w-1)/4),Math.floor((w-1)/2),Math.floor(3*(w-1)/4),w-1];
      for(let y=0;y<h;y++)for(let x=0;x<w;x++){
        const position=computeReceiverPoint(scene,x,y,w,h),expected=evaluatePoint(scene,position,{boundaryEpsilon:2e-6});
        const i=(y*w+x)*4,actual=[...data.slice(i,i+4)],gpuValid=actual[2]>.5;
        const ratio=Math.abs(actual[0]-expected.f)/(2e-5+.005*Math.abs(expected.f));
        const geometricRatio=Math.abs(actual[1]-expected.geometricPower)/(2e-5+.005*Math.abs(expected.geometricPower));
        const maskMismatch=gpuValid!==expected.valid,edgeMismatch=actual[3]!==expected.edgeCount;
        if(maskMismatch)r.maskMismatches++;if(edgeMismatch)r.edgeMismatches++;
        const numericFailure=expected.valid&&(!actual.every(Number.isFinite)||actual[0]<0||actual[1]<0||ratio>1||geometricRatio>1);
        if(numericFailure)r.numericalFailures++;
        if(expected.valid){r.valid++;r.cpuPeak=Math.max(r.cpuPeak,Math.fround(expected.f));r.gpuPeak=Math.max(r.gpuPeak,actual[0]);r.maxToleranceRatio=Math.max(r.maxToleranceRatio,ratio,geometricRatio);r.maxAbsolutePowerError=Math.max(r.maxAbsolutePowerError,Math.abs(actual[0]-expected.f));}
        const point={x,y,position,actual:{power:actual[0],geometricPower:actual[1],valid:gpuValid,edgeCount:actual[3]},
          expected:{power:expected.f,geometricPower:expected.geometricPower,valid:expected.valid,edgeCount:expected.edgeCount,invalidReasons:expected.invalidReasons},
          toleranceRatio:ratio,geometricToleranceRatio:geometricRatio};
        if(r.gate.samples?.some(s=>s.x===x&&s.y===y)||(coords.includes(x)&&coords.includes(y)))r.gatePoints.push(point);
        if(maskMismatch||edgeMismatch||numericFailure)r.failures.push(point);
      }
      if(item.ui){
        const {makeGPUEngine}=await load('utd-gpu-engine.js');
        const engine=await makeGPUEngine(scene,w,h),progress=[];
        try{
          const actualEngine=await engine.run({onProgress:p=>progress.push(p)});
          assert.equal(actualEngine.parity.passed,true,'Actual GPU engine parity');
          assert.equal(actualEngine.valid,r.valid,'Actual GPU engine validity count');
          assert.equal(progress.length,Math.ceil(h/8),'Actual GPU engine executes every row batch');
          assert.equal(progress.at(-1),1,'Actual GPU engine finishes all row batches');
          assert.deepEqual(actualEngine.data,data,'Actual engine output must match independently dispatched WGSL');
          r.actualEngine={passed:true,backend:actualEngine.backend,valid:actualEngine.valid,
            batches:progress.length,bitwiseEqualToIndependentDispatch:true,parity:actualEngine.parity};
        }finally{engine.destroy();}
      }
      report.scenes.push(r);
      if(!r.gate.passed||r.maskMismatches||r.edgeMismatches||r.numericalFailures)failures.push(item.label);
      console.log(`${item.label}: ${r.gate.passed?'gate PASS':'gate FAIL'}; ${r.valid}/${r.total} valid; max tolerance ratio ${r.maxToleranceRatio}; masks ${r.maskMismatches}; edges ${r.edgeMismatches}`);
    }finally{for(const b of resources)b.destroy();}
  }
  report.passed=failures.length===0;
  if(process.env.WEBGPU_EXECUTION_REPORT)fs.writeFileSync(process.env.WEBGPU_EXECUTION_REPORT,JSON.stringify(report,null,2)+'\n');
  console.log('Adapter provenance:',JSON.stringify(adapterInfo));
  assert.equal(failures.length,0,`Native WGSL execution failures: ${failures.join(', ')}`);
}finally{
  globalThis.fetch=previousFetch;
  if(previousNavigator)Object.defineProperty(globalThis,'navigator',previousNavigator);
  else delete globalThis.navigator;
  device.destroy();gpu=null;
}
