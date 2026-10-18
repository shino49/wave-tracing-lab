import fs from 'node:fs';
import assert from 'node:assert/strict';
import {create,globals} from 'webgpu';
import {executeCoherentMeshTask} from '../dist/coherent-mesh-runtime.js';
import {createCoherentMeshSpectralGPURunner} from '../dist/coherent-mesh-spectral-gpu.js';
Object.assign(globalThis,globals);assert.equal(process.env.WEBGPU_EXECUTION_BACKEND,'vulkan');
const gpu=create(['backend=vulkan']),shaderSource=fs.readFileSync('dist/utd-functions.wgsl','utf8')+'\n'+fs.readFileSync('dist/coherent-mesh-spectral-compute.wgsl','utf8'),scene=JSON.parse(fs.readFileSync('dist/examples/coherent-lloyd-d65-disk.json','utf8')),controller=new AbortController(),start=performance.now();let record;
await assert.rejects(executeCoherentMeshTask({scene,resolution:512,pixelQuadrature:1,spectralSamples:16,backend:'auto'},{signal:controller.signal,spectralGpuFactory:options=>createCoherentMeshSpectralGPURunner({...options,gpu,shaderSource}),preview:p=>{record={firstPreviewHostMs:performance.now()-start,width:p.width,height:p.height,rowStart:p.rowStart,rowCount:p.rowCount,spatialSamples:p.spatialSamples,spatialSamplesIntegrated:p.spatialSamplesIntegrated,spatialComplete:p.spatialComplete,spectralComplete:p.spectralComplete,backend:p.backend,scope:'Native SwiftShader first complete spectral tile, source1/16; aborted immediately afterward; not browser delivery time'};controller.abort();}}),{name:'AbortError'});
assert(record&&!record.spatialComplete&&record.spectralComplete);fs.writeFileSync('research/spatial-first-preview.json',JSON.stringify(record,null,2)+'\n');console.log(record.firstPreviewHostMs);
