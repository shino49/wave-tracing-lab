// Experimental numerical operator only. Geometry/admission are CPU-owned.
// This module does not register a UI mode or publish an unverified GPU buffer.
import {coefficient, abs2} from './utd-math.js';

export const DD_GPU_SCOPE = 'Selected native-centre-connected two-region local-plane conditional power; CPU geometry/admission, GPU coefficient and conditional-power stages; no cross-path phase or calibrated irradiance';
export const DD_RECORD_FLOATS = 12;
export const DD_OUTPUT_FLOATS = 4;
export const DD_MAX_RECORDS = 65536;
// Independent DD gate. The pre-existing single-region 25/81-point gate and
// its tolerance are untouched. Check every lane of every output record.
export const DD_PARITY_TOLERANCE = Object.freeze({absolute:1e-24, relative:.005, boundaryEpsilon:2e-6});

function fail(message) { throw new RangeError(`DD operator: ${message}`); }
function checkRecords(records) {
  if (!Array.isArray(records) || records.length < 1 || records.length > DD_MAX_RECORDS)
    fail(`expected 1..${DD_MAX_RECORDS} point records`);
  for (const [i, record] of records.entries()) {
    if (!record || typeof record.valid !== 'boolean') fail(`record ${i} requires an explicit Boolean CPU admission mask`);
    // Explicit invalid masks carry no meaningful geometry. Canonical zeros are
    // used for them on both backends, with no coefficient evaluation.
    if (!record.valid) continue;
    if (!['soft','hard'].includes(record.boundary)) fail(`record ${i} requires soft or hard boundary`);
    if (!Array.isArray(record.angles) || record.angles.length !== 4 ||
        !record.angles.every(x => Number.isFinite(x) && x >= 0 && x <= 2*Math.PI))
      fail(`record ${i} angles must be four finite exterior-sector angles in [0,2*pi]`);
    for (const key of ['k','ro1','ro2']) {
      if (!(Number.isFinite(record[key]) && record[key] >= 1e-6 && record[key] <= 1e6))
        fail(`record ${i} ${key} is outside the bounded numeric domain [1e-6,1e6]`);
    }
    for (const range of [record.ro1,record.ro2]) {
      if (!(record.k*range >= 1e-6 && record.k*range <= 100000))
        fail(`record ${i} k*range must lie in [1e-6,100000]`);
    }
  }
}
function evaluateChecked(record) {
  if (!record.valid) return {power:0,stage1Power:0,stage2Power:0,valid:false};
  const {k,angles,ro1,ro2,boundary} = record;
  const stage1Power = abs2(coefficient(k,2,angles[0],angles[1],ro1,1,boundary,DD_PARITY_TOLERANCE.boundaryEpsilon))/ro1;
  const stage2Power = abs2(coefficient(k,2,angles[2],angles[3],ro2,1,boundary,DD_PARITY_TOLERANCE.boundaryEpsilon))/ro2;
  const power = stage1Power*stage2Power;
  if (![power,stage1Power,stage2Power].every(x=>Number.isFinite(x)&&x>=0))
    fail('CPU reference returned a nonfinite or negative conditional power');
  return {power,stage1Power,stage2Power,valid:true};
}
export function evaluateDDRecord(record) {
  checkRecords([record]);
  return evaluateChecked(record);
}
export function packDDRecords(records) {
  checkRecords(records);
  const data = new Float32Array(records.length*DD_RECORD_FLOATS);
  for (const [i,record] of records.entries()) {
    if (!record.valid) continue;
    data.set([...record.angles,record.k,record.ro1,record.ro2,record.boundary==='soft'?-1:1,1,0,0,0],i*DD_RECORD_FLOATS);
  }
  return {data,count:records.length};
}

/** Exhaustive numerical gate. It certifies backend agreement, not physics. */
export function ddParityGate(records,data) {
  checkRecords(records);
  if (!(data instanceof Float32Array) || data.length !== records.length*DD_OUTPUT_FLOATS)
    fail('readback must be an exact stride-4 Float32Array');
  const gate = {passed:true,coverage:'full-array',points:records.length,valid:0,invalid:0,
    tolerance:{...DD_PARITY_TOLERANCE},nonfinite:0,negative:0,maskMismatches:0,invalidNonzero:0,
    numericalFailures:0,maxToleranceRatio:0,maxAbsolutePowerError:0,maxRelativePowerError:0,
    positivePowerRange:{min:null,max:null},failures:[]};
  for (let i=0;i<records.length;i++) {
    const reference=evaluateChecked(records[i]);
    const expected=[reference.power,reference.stage1Power,reference.stage2Power,Number(reference.valid)];
    const actual=Array.from(data.subarray(4*i,4*i+4));
    let failed=false;
    if (!actual.every(Number.isFinite)) { gate.nonfinite++;failed=true; }
    if (actual.some(x=>x<0)) { gate.negative++;failed=true; }
    if (actual[3]!==expected[3]) { gate.maskMismatches++;failed=true; }
    if (!reference.valid && actual.some(x=>x!==0)) { gate.invalidNonzero++;failed=true; }
    const ratios=[];
    if (reference.valid) {
      gate.valid++;
      for (let lane=0;lane<3;lane++) {
        const difference=Math.abs(actual[lane]-expected[lane]);
        const ratio=difference/(DD_PARITY_TOLERANCE.absolute+DD_PARITY_TOLERANCE.relative*Math.abs(expected[lane]));
        ratios.push(Number.isFinite(ratio)?ratio:null);
        if (Number.isFinite(ratio)) gate.maxToleranceRatio=Math.max(gate.maxToleranceRatio,ratio);
        if (!Number.isFinite(ratio) || ratio>1) { gate.numericalFailures++;failed=true; }
      }
      if (Number.isFinite(actual[0])) {
        const difference=Math.abs(actual[0]-reference.power);
        gate.maxAbsolutePowerError=Math.max(gate.maxAbsolutePowerError,difference);
        if (reference.power>0) gate.maxRelativePowerError=Math.max(gate.maxRelativePowerError,difference/reference.power);
      }
      if (reference.power>0) {
        gate.positivePowerRange.min=Math.min(gate.positivePowerRange.min??Infinity,reference.power);
        gate.positivePowerRange.max=Math.max(gate.positivePowerRange.max??0,reference.power);
      }
    } else gate.invalid++;
    if (failed) {
      gate.passed=false;
      // Bound diagnostics, never coverage: every record is still checked.
      if (gate.failures.length<32) gate.failures.push({index:i,actual,expected,toleranceRatios:ratios});
    }
  }
  if (!gate.passed) {
    const error=new Error('DD full-array GPU/CPU parity failed; unverified output withheld');
    error.name='DDParityError';error.parity=gate;
    throw error;
  }
  return gate;
}
function cancelled() { const error=new Error('DD GPU execution cancelled or destroyed');error.name='AbortError';return error; }
function timeout(promise,ms,label) {
  let timer;
  return Promise.race([promise,new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error(label)),ms);})])
    .finally(()=>clearTimeout(timer));
}

/** Owns its device and buffers; no data is returned before the exhaustive gate. */
export async function makeDDGPUEngine(inputRecords,{gpu=globalThis.navigator?.gpu}={}) {
  checkRecords(inputRecords);
  // Snapshot before asynchronous work. Caller mutations cannot race gate inputs.
  const records=inputRecords.map(record=>record.valid?{...record,angles:[...record.angles]}:{valid:false});
  const packed=packDDRecords(records);
  if (!gpu) throw new Error('WebGPU unavailable for the experimental DD operator');
  const adapter=await gpu.requestAdapter({powerPreference:'low-power'});
  if (!adapter) throw new Error('No WebGPU adapter for the experimental DD operator');
  const device=await adapter.requestDevice();
  const resources=[];
  let destroyed=false,running=false,lostReason=null;
  const onUncaptured=event=>{if(!destroyed)lostReason=event.error?.message||'DD uncaptured WebGPU error';};
  device.addEventListener('uncapturederror',onUncaptured);
  device.lost.then(info=>{if(!destroyed)lostReason=info.message||'DD device lost';});
  const destroy=()=>{
    if (destroyed) return;
    destroyed=true;
    for (const buffer of resources) { try {buffer.destroy();} catch {} }
    device.removeEventListener?.('uncapturederror',onUncaptured);
    device.destroy();
  };
  const check=signal=>{
    if(destroyed||signal?.aborted)throw cancelled();
    if(lostReason)throw new Error(lostReason);
  };
  try {
    const paths=['./utd-functions.wgsl','./utd-dd-compute.wgsl'];
    const sources=await Promise.all(paths.map(async url=>{
      const response=await fetch(url);
      if(!response.ok)throw new Error(`Cannot load DD shader ${url}: ${response.status}`);
      return response.text();
    }));
    check();
    const module=device.createShaderModule({label:'Experimental two-region conditional operator',code:sources.join('\n')});
    const diagnostics=await module.getCompilationInfo();
    const errors=diagnostics.messages.filter(item=>item.type==='error');
    if(errors.length)throw new Error(errors.map(item=>`WGSL ${item.lineNum}:${item.linePos} ${item.message}`).join('; '));
    const pipeline=await timeout(device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'dd_main'}}),20000,'DD pipeline compilation timed out');
    check();
    const make=(size,usage,data)=>{
      if(size>device.limits.maxBufferSize || size>device.limits.maxStorageBufferBindingSize)
        throw new Error('DD records exceed current WebGPU buffer limits');
      const buffer=device.createBuffer({size,usage});resources.push(buffer);
      if(data)device.queue.writeBuffer(buffer,0,data);
      return buffer;
    };
    const input=make(packed.data.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,packed.data);
    const bytes=records.length*16;
    const output=make(bytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
    const readback=make(bytes,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[input,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
    const adapterInfo={};
    for(const key of ['vendor','architecture','device','description','isFallbackAdapter'])
      if(adapter.info?.[key]!==undefined)adapterInfo[key]=adapter.info[key];
    return {
      type:'WebGPU DD conditional operator',count:records.length,destroy,
      async run({signal}={}) {
        check(signal);
        if(running)throw new Error('A DD GPU run is already in progress');
        running=true;let mapped=false,scope=false;
        try {
          device.pushErrorScope('validation');scope=true;
          const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
          pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(records.length/64));pass.end();
          encoder.copyBufferToBuffer(output,0,readback,0,bytes);
          device.queue.submit([encoder.finish()]);
          await timeout(device.queue.onSubmittedWorkDone(),15000,'DD dispatch timed out');
          check(signal);
          await timeout(readback.mapAsync(GPUMapMode.READ),15000,'DD readback timed out');mapped=true;
          check(signal);
          const data=new Float32Array(readback.getMappedRange(0,bytes).slice(0));
          readback.unmap();mapped=false;
          const validation=await device.popErrorScope();scope=false;
          if(validation)throw new Error(validation.message);
          check(signal);
          const parity=ddParityGate(records,data);
          check(signal);
          return {data,count:records.length,valid:parity.valid,invalid:parity.invalid,parity,
            backend:'WebGPU',adapter:adapterInfo,scope:DD_GPU_SCOPE};
        } finally {
          if(mapped) {try{readback.unmap();}catch{}}
          if(scope) {try{await device.popErrorScope();}catch{}}
          running=false;
        }
      },
    };
  } catch(error) {destroy();throw error;}
}
