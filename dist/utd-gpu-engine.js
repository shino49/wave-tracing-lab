import {packUTDSceneGPU} from './utd-gpu-pack.js';
import {parityGate,CancelledError} from './utd-engine.js';

export const UTD_GPU_BUILD_ID='webgpu-diagnostics-2026-10-12.1';
function adapterSummary(adapter){
  const info={};
  for(const key of ['vendor','architecture','device','description','isFallbackAdapter']){
    try{const value=adapter.info?.[key];if(typeof value==='string'||typeof value==='boolean')info[key]=value;}catch{}
  }
  return info;
}
function sourceFingerprint(source){let hash=2166136261;for(let i=0;i<source.length;i++){hash^=source.charCodeAt(i);hash=Math.imul(hash,16777619);}return (hash>>>0).toString(16).padStart(8,'0');}
function timeout(promise,ms,label){let id;return Promise.race([promise,new Promise((_,reject)=>{id=setTimeout(()=>reject(new Error(label)),ms);})]).finally(()=>clearTimeout(id));}
export async function makeGPUEngine(scene,width,height,{onLost=()=>{}}={}){
  if(!navigator.gpu)throw new Error('此浏览器未提供 WebGPU');
  const packed=packUTDSceneGPU(scene,width,height);
  const adapter=await navigator.gpu.requestAdapter({powerPreference:'low-power'});
  if(!adapter)throw new Error('没有可用的 WebGPU 适配器');
  const device=await adapter.requestDevice();let destroyed=false,lostReason=null;const resources=[];
  const destroy=()=>{if(destroyed)return;destroyed=true;for(const b of resources){try{b.destroy();}catch{}}device.destroy();};
  device.lost.then(info=>{if(!destroyed){lostReason=info.message||'WebGPU 设备连接中断';onLost(lostReason);}});
  device.addEventListener('uncapturederror',e=>{if(!destroyed){lostReason=e.error?.message||'WebGPU 验证错误';onLost(lostReason);}});
  try{
    const paths=['./utd-functions.wgsl','./utd-compute.wgsl'];
    const source=(await Promise.all(paths.map(async path=>{const r=await fetch(path);if(!r.ok)throw new Error(`无法读取着色器（${r.status}）`);return r.text();}))).join('\n');
    const diagnosticContext={buildId:UTD_GPU_BUILD_ID,shaderFingerprintFNV1a:sourceFingerprint(source),adapter:adapterSummary(adapter),
      browser:navigator.userAgent??null,dimensions:{width,height},params:scene.params,
      ...(scene.mirrorTransport?{mirrorTransport:{order:scene.mirrorTransport.order,mirror:scene.mirrorTransport.mirror,physical:scene.mirrorTransport.physical,phaseGauge:scene.mirrorTransport.phaseGauge,admission:scene.mirrorTransport.admission}}:{}),
      geometry:{...packed.counts,bounds:scene.mesh.bounds},gpuSafety:scene.gpuSafety};
    const module=device.createShaderModule({code:source});
    const diagnostics=await module.getCompilationInfo();
    const errors=diagnostics.messages.filter(m=>m.type==='error');
    if(errors.length)throw new Error(errors.map(m=>`WGSL ${m.lineNum}:${m.linePos} ${m.message}`).join('; '));
    const pipeline=await timeout(device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}}),20000,'WebGPU 管线编译超时');
    const make=(size,usage,data)=>{
      if(size>device.limits.maxBufferSize||size>device.limits.maxStorageBufferBindingSize)throw new Error('场景超过当前 GPU 缓冲区上限');
      const b=device.createBuffer({size:Math.max(16,Math.ceil(size/16)*16),usage});resources.push(b);if(data)device.queue.writeBuffer(b,0,data);return b;
    };
    const uniformData=packed.createUniforms(0,8);let lastUniformData=uniformData;
    const uniform=make(uniformData.byteLength,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,uniformData);
    const tri=make(packed.triangles.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,packed.triangles);
    const edges=make(packed.edges.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,packed.edges);
    const adjacency=make(packed.adjacency.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,packed.adjacency);
    const facets=make(packed.facets.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,packed.facets);
    const bytes=width*height*16,out=make(bytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC),read=make(bytes,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[uniform,tri,edges,adjacency,out,facets].map((buffer,binding)=>({binding,resource:{buffer}}))});
    return {
      type:'WebGPU',width,height,destroy,
      async run({signal,onProgress=()=>{}}={}){
        const check=()=>{if(signal?.aborted||destroyed)throw new CancelledError();if(lostReason)throw new Error(lostReason);};
        const safely=async promise=>{try{return await promise;}catch(error){check();throw error;}};
        check();
        for(let base=0;base<height;base+=8){
          check();const count=Math.min(8,height-base);lastUniformData=packed.createUniforms(base,count);device.queue.writeBuffer(uniform,0,lastUniformData);
          device.pushErrorScope('validation');
          let failure=null,validation=null;
          try{
            const enc=device.createCommandEncoder(),pass=enc.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(width/8),Math.ceil(count/8));pass.end();device.queue.submit([enc.finish()]);
            await safely(timeout(device.queue.onSubmittedWorkDone(),15000,'GPU 单批计算超时，已停止该计算'));
          }catch(error){failure=error;}
          finally{try{validation=await device.popErrorScope();}catch(error){failure??=error;}}
          check();if(failure)throw failure;if(validation)throw new Error(validation.message);onProgress((base+count)/height);
          await new Promise(r=>setTimeout(r,0));
        }
        check();const enc=device.createCommandEncoder();enc.copyBufferToBuffer(out,0,read,0,bytes);device.queue.submit([enc.finish()]);
        await safely(timeout(read.mapAsync(GPUMapMode.READ),15000,'GPU 结果读回超时'));
        let data;try{check();data=new Float32Array(read.getMappedRange(0,bytes).slice(0));}finally{try{read.unmap();}catch{}}
        check();let valid=0;for(let i=0;i<width*height;i++){
          const f=data[4*i],g=data[4*i+1],mask=data[4*i+2],edges=data[4*i+3];
          if(!Number.isFinite(f)||f<0||!Number.isFinite(g)||g<0||(mask!==0&&mask!==1)||!Number.isInteger(edges)||edges<0||edges>scene.wedges.length)throw new Error('GPU 结果包含无效数值或有效性标记');
          if(mask===1)valid++;
        }
        const result={data,width,height,valid,invalid:width*height-valid,reasons:{},backend:'WebGPU'};
        try{result.parity=parityGate(scene,result);}
        catch(error){
          error.diagnostic={...diagnosticContext,stage:'runtime-parity',error:error.message,parity:error.parity};
          try{
            const {collectUTDProbe}=await import('./utd-gpu-diagnostics.js');
            error.diagnostic.intermediateTrace=await collectUTDProbe({device,source,scene,packed,buffers:[uniform,tri,edges,adjacency,out,facets],result,parity:error.parity,signal,uniformBytes:lastUniformData});
          }catch(traceError){error.diagnostic.intermediateTrace={available:false,error:traceError.message||String(traceError)};}
          if(signal?.aborted||destroyed)throw new CancelledError();
          throw error;
        }
        check();return result;
      },
    };
  }catch(error){destroy();throw error;}
}
