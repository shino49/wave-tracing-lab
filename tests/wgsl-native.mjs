// Native Dawn/Tint WGSL validation. The null backend validates WGSL modules and
// pipeline descriptors without hardware execution. It is not a rendering test.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const {create,globals}=process.env.WEBGPU_MODULE
  ? await import(pathToFileURL(path.resolve(process.env.WEBGPU_MODULE)).href)
  : await import('webgpu');
Object.assign(globalThis,globals);
const root=path.resolve(process.argv[2]||process.cwd());
let gpu=create(['backend=null']);
const adapter=await gpu.requestAdapter();
assert.ok(adapter,'Native Dawn null adapter unavailable');
const device=await adapter.requestDevice();
try {
  const read=name=>fs.readFileSync(path.join(root,'dist',name),'utf8');
  const functions=read('utd-functions.wgsl');
  const combined=functions+'\n'+read('utd-compute.wgsl');
  const cases=[
    ['Gaussian kernel',read('kernel.wgsl'),true],
    ['UTD function library',functions,false],
    ['UTD compute shader',combined,true],
  ];
  for(const [label,code,pipeline] of cases){
    device.pushErrorScope('validation');
    const module=device.createShaderModule({label,code});
    const info=await module.getCompilationInfo();
    const errors=info.messages.filter(m=>m.type==='error');
    const validation=await device.popErrorScope();
    assert.equal(errors.length,0,`${label}: ${errors.map(m=>`${m.lineNum}:${m.linePos} ${m.message}`).join('\n')}`);
    assert.equal(validation,null,validation?.message);
    if(pipeline) await device.createComputePipelineAsync({label,layout:'auto',compute:{module,entryPoint:'main'}});
    console.log(`${label}: native WGSL validation PASS${pipeline?' (pipeline creation PASS)':''}`);
  }
  // Guard the validator itself: wgsl_reflect 1.4.0 accepts this reserved name.
  device.pushErrorScope('validation');
  const bad=device.createShaderModule({label:'reserved keyword negative control',code:'struct Bad { meta: vec4f };'});
  const info=await bad.getCompilationInfo();
  assert.ok(info.messages.some(m=>m.type==='error'&&m.message.includes('reserved keyword')),'Native compiler must reject reserved meta identifier');
  assert.ok(await device.popErrorScope(),'Reserved word must produce validation error');
  console.log('Reserved-keyword negative control: PASS');
} finally {
  device.destroy();
  gpu=null;
}
