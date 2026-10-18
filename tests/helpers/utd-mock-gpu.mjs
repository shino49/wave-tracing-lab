/** Host-side WebGPU lifecycle mock. Does not compile or execute WGSL. */
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {prepareUTDScene,builtInSceneOBJ,evaluatePoint,computeReceiverPoint} from '../../dist/utd-scene.js';
const scene=prepareUTDScene(builtInSceneOBJ('screen'));
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};
export function cpuFixture(s=scene,width=9,height=9){
  const data=new Float32Array(width*height*4);let valid=0;
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const r=evaluatePoint(s,computeReceiverPoint(s,x,y,width,height),{boundaryEpsilon:2e-6}),i=(y*width+x)*4;
    data[i]=r.valid?r.f:0;data[i+1]=r.valid?r.geometricPower:0;data[i+2]=r.valid?1:0;data[i+3]=r.edgeCount||0;valid+=Number(r.valid);
  }
  return {data,width,height,valid,invalid:width*height-valid};
}

// Minimal mock implementing only the WebGPU lifecycle exercised by this adapter.
const USAGE={UNIFORM:1,COPY_DST:2,STORAGE:4,COPY_SRC:8,MAP_READ:16};
export function mockGPU({fixture=cpuFixture(),...options}={}){
  const state={buffers:[],fetches:[],submits:0,dispatches:[],destroyCalls:0,mappedReads:0,unmaps:0,scopes:0,writeCalls:0,shaderCalls:0,pipelineCalls:0};
  const lost=deferred(),events=new Map();
  let workCalls=0;
  const device={
    limits:{maxBufferSize:options.maxBufferSize??1e7,maxStorageBufferBindingSize:1e7},lost:lost.promise,
    addEventListener:(name,fn)=>events.set(name,fn),
    destroy(){state.destroyCalls++;},
    createShaderModule(){state.shaderCalls++;const trace=state.shaderCalls>1;return {getCompilationInfo:async()=>({messages:(trace?options.traceCompileMessages:options.compileMessages)??[]})};},
    createComputePipelineAsync:async()=>{state.pipelineCalls++;if(state.pipelineCalls>1){options.onTrace?.();if(options.tracePipelineWait)await options.tracePipelineWait;}if(options.pipelineError)throw new Error(options.pipelineError);if(options.pipelineWait)await options.pipelineWait;return {getBindGroupLayout:()=>({})};},
    createBuffer(desc){
      const b={...desc,storage:new ArrayBuffer(desc.size),destroyCalls:0,mapped:false,destroyed:false,
        destroy(){this.destroyCalls++;this.destroyed=true;this.mapped=false;},
        async mapAsync(){options.onMap?.();if(options.mapWait)await options.mapWait;if(options.mapError)throw new Error(options.mapError);this.mapped=true;},
        getMappedRange(offset=0,size=this.size){state.mappedReads++;if(options.rangeError)throw new Error(options.rangeError);if(this.destroyed)throw new Error('mock buffer already destroyed');assert(this.mapped,'readback must be mapped');return this.storage.slice(offset,offset+size);},
        unmap(){state.unmaps++;this.mapped=false;},
      };
      if(desc.usage===(USAGE.STORAGE|USAGE.COPY_SRC))new Float32Array(b.storage).set(fixture.data);
      state.buffers.push(b);return b;
    },
    createBindGroup:()=>({}),
    pushErrorScope:()=>state.scopes++,
    popErrorScope:async()=>{state.scopes--;return options.validationError?{message:options.validationError}:null;},
    createCommandEncoder:()=>({
      beginComputePass:()=>({setPipeline(){},setBindGroup(){},dispatchWorkgroups:(...args)=>state.dispatches.push(args),end(){}}),
      copyBufferToBuffer:(src,so,dst,doff,size)=>{new Uint8Array(dst.storage,doff,size).set(new Uint8Array(src.storage,so,size));},finish:()=>({}),
    }),
    queue:{
      writeBuffer(buffer,offset,data){state.writeCalls++;const bytes=ArrayBuffer.isView(data)?new Uint8Array(data.buffer,data.byteOffset,data.byteLength):new Uint8Array(data);new Uint8Array(buffer.storage,offset,bytes.byteLength).set(bytes);},
      submit(){state.submits++;if(options.submitError)throw new Error(options.submitError);},
      async onSubmittedWorkDone(){workCalls++;options.onWork?.(workCalls);if(options.workWait)await options.workWait;if(options.workError)throw new Error(options.workError);},
    },
  };
  state.lose=message=>lost.resolve({message});
  state.uncaptured=message=>events.get('uncapturederror')?.({error:{message}});
  state.assertDestroyed=()=>{assert.equal(state.destroyCalls,1);assert(state.buffers.every(b=>b.destroyCalls===1),'every allocated buffer must be destroyed exactly once');};
  Object.defineProperty(globalThis,'navigator',{value:{gpu:{requestAdapter:async()=>options.noAdapter?null:{info:options.adapterInfo??{},requestDevice:async()=>device}}},configurable:true});
  globalThis.GPUBufferUsage=USAGE;globalThis.GPUMapMode={READ:1};
  globalThis.fetch=async path=>{
    state.fetches.push(path);assert(['./utd-functions.wgsl','./utd-compute.wgsl'].includes(path),'only bundled shader fetches allowed');
    if(options.fetchError)return {ok:false,status:503};
    return {ok:true,text:()=>fs.readFile(new URL('../../dist/'+path.slice(2),import.meta.url),'utf8')};
  };
  return state;
}
