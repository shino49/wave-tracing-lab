import {UTD_PROBE_SHADER} from './utd-probe-shader.js';
import {makeUTDProbeReferences,decodeUTDProbe,assessUTDProbe,UTD_PROBE_ROWS,UTD_PROBE_STRIDE_VEC4,makeUTDProbeShader} from './utd-probe-reference.js';
function withTimeout(promise,ms,label){let id;return Promise.race([promise,new Promise((_,reject)=>{id=setTimeout(()=>reject(new Error(label)),ms);})]).finally(()=>clearTimeout(id));}
export function selectUTDProbePixels(parity,width,height){
  const samples=parity?.samples??[],bad=samples.filter(s=>s.status==='numeric-mismatch'||s.status==='validity-mismatch');
  const score=s=>Math.max(s.normalizedToleranceRatio?.f??0,s.normalizedToleranceRatio?.geometricPower??0);
  const worst=[...bad].sort((a,b)=>score(b)-score(a))[0],pass=samples.find(s=>s.status==='pass');
  const selected=[];
  for(const s of [bad[0],worst,pass,...samples]){if(!s)continue;const p=[s.x,s.y];if(p.some((v,i)=>!Number.isInteger(v)||v<0||v>=[width,height][i]))continue;if(!selected.some(a=>a[0]===p[0]&&a[1]===p[1]))selected.push(p);if(selected.length===3)break;}
  return selected;
}
export async function collectUTDProbe({device,source,scene,packed,buffers,result,parity,signal,uniformBytes=packed.uniforms}){
  const pixels=selectUTDProbePixels(parity,result.width,result.height),edgeCount=Math.min(2,scene.wedges.length);
  if(pixels.length!==3||!edgeCount)return {available:false,reason:'No selected pixels or clipped edges to trace'};
  if(signal?.aborted)throw new Error('Trace cancelled');
  const code=makeUTDProbeShader(source,UTD_PROBE_SHADER,pixels,result.width,result.height),resources=[];
  let mapped=false,read=null,scope=false,traceData;
  try{
    device.pushErrorScope('validation');scope=true;
    const module=device.createShaderModule({label:'UTD failure trace, separate from main computation',code});
    const info=await withTimeout(module.getCompilationInfo(),20000,'Diagnostic module compilation timeout'),errors=info.messages.filter(m=>m.type==='error');
    if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}:${m.linePos} ${m.message}`).join('; '));
    const pipeline=await withTimeout(device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'utd_diagnostic_probe'}}),20000,'Diagnostic pipeline compilation timeout');
    if(signal?.aborted)throw new Error('Trace cancelled');
    const bytes=pixels.length*edgeCount*UTD_PROBE_STRIDE_VEC4*16;
    const out=device.createBuffer({size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});resources.push(out);
    read=device.createBuffer({size:bytes,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});resources.push(read);
    const traceBuffers=[...buffers];traceBuffers[4]=out;
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:traceBuffers.map((buffer,binding)=>({binding,resource:{buffer}}))});
    const enc=device.createCommandEncoder(),pass=enc.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(pixels.length*edgeCount);pass.end();enc.copyBufferToBuffer(out,0,read,0,bytes);device.queue.submit([enc.finish()]);
    await withTimeout(read.mapAsync(GPUMapMode.READ),15000,'Diagnostic readback timeout');mapped=true;
    if(signal?.aborted)throw new Error('Trace cancelled');
    traceData=new Float32Array(read.getMappedRange(0,bytes).slice(0));read.unmap();mapped=false;
    const error=await device.popErrorScope();scope=false;if(error)throw new Error(error.message);
  }finally{
    if(mapped){try{read.unmap();}catch{}}
    for(const buffer of resources){try{buffer.destroy();}catch{}}
    if(scope){try{await device.popErrorScope();}catch{}}
  }
  const gpu=decodeUTDProbe(traceData,pixels,edgeCount),cpu=makeUTDProbeReferences(scene,pixels,result.width,result.height),cpuAtGPUInputs=assessUTDProbe(gpu);
  const traceSamples=gpu.map((g,i)=>{
    const originalGPU=Array.from(result.data.slice((g.pixel[1]*result.width+g.pixel[0])*4,(g.pixel[1]*result.width+g.pixel[0])*4+4));
    const reference=cpu.find(c=>c.pixel?.[0]===g.pixel[0]&&c.pixel?.[1]===g.pixel[1]&&c.edgeIndex===g.edgeIndex)??cpu.find(c=>c.pixel?.[0]===g.pixel[0]&&c.pixel?.[1]===g.pixel[1]);
    const reconstructed=g.rows[29].reconstructedPower;
    const traceToMainRatio=Math.abs(reconstructed-originalGPU[0])/(2e-5+.005*Math.abs(originalGPU[0]));
    return {pixel:g.pixel,edgeIndex:g.edgeIndex,edgeId:reference?.edgeId,originalGPU,
      traceMatchesMainPower:Number.isFinite(traceToMainRatio)&&traceToMainRatio<=1,traceToMainToleranceRatio:traceToMainRatio,
      cpuModel:reference,cpuAtGPUInputs:cpuAtGPUInputs[i],gpu:g};
  });
  return {schema:'utd-intermediate-trace-v1',available:true,entryPoint:'utd_diagnostic_probe',selectedPixels:pixels,
    capturedEdges:edgeCount,totalClippedEdges:scene.wedges.length,allEdgesCaptured:edgeCount===scene.wedges.length,
    note:'Separate diagnostic entrypoint reuses the same functions and input buffers. The main GPU output remains authoritative; differences from reconstructed power are diagnostic evidence, never silently replaced.',
    inputSnapshot:{uniformsU32:Array.from(new Uint32Array(uniformBytes.buffer,uniformBytes.byteOffset,8)),uniformsF32:Array.from(new Float32Array(uniformBytes.buffer,uniformBytes.byteOffset+32,32)),clippedEdgesF32:Array.from(packed.edges.slice(packed.counts.physicalEdges*24,(packed.counts.physicalEdges+edgeCount)*24))},
    rowSchema:UTD_PROBE_ROWS,samples:traceSamples};
}
