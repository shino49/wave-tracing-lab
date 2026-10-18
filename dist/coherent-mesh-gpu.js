/** Native WebGPU summation of deterministic, sample-major coherent routes.
 * CPU: geometry (caller), robust reduced phase (caller), finite/domain checks.
 * GPU: complex route sum and coherent/incoherent intensities, at native sample
 * count. Bounded storage is reused in chunks; no CPU shadow/oracle is run.
 */
export const COHERENT_MESH_GPU_BUILD_ID = 'coherent-mesh-route-sum-2026-10-13.1';
export const COHERENT_MESH_GPU_LIMITS = Object.freeze({
  maxSamples: 262144, maxRoutes: 65, chunkSamples: 8192,
  maxIntensityBound: 1e30, phaseBound: Math.fround(Math.PI),
});
const now = () => globalThis.performance?.now?.() ?? Date.now();
const abortError = () => Object.assign(new Error('Coherent mesh GPU calculation cancelled or destroyed'), {name: 'AbortError'});
const fail = message => { throw new RangeError(`Coherent mesh GPU: ${message}`); };
const TIMING_SCOPE = 'Host elapsed validation/upload/encoding/submission/queue wait/readback; not GPU timestamp-query timings';
function adapterInfo(adapter) {
  const info = {};
  for (const key of ['vendor', 'architecture', 'device', 'description', 'isFallbackAdapter']) {
    try { const value = adapter.info?.[key]; if (typeof value === 'string' || typeof value === 'boolean') info[key] = value; } catch {}
  }
  return info;
}
export function coherentMeshGPUCapacity({maxSamples = COHERENT_MESH_GPU_LIMITS.maxSamples} = {}) {
  if (!Number.isSafeInteger(maxSamples) || maxSamples < 1 || maxSamples > COHERENT_MESH_GPU_LIMITS.maxSamples) fail('maxSamples must be an integer in 1..262144');
  const chunkSamples = Math.min(maxSamples, COHERENT_MESH_GPU_LIMITS.chunkSamples);
  const recordBytes = chunkSamples * COHERENT_MESH_GPU_LIMITS.maxRoutes * 16;
  const outputBytes = chunkSamples * 16;
  return {maxSamples, maxRoutes: COHERENT_MESH_GPU_LIMITS.maxRoutes, chunkSamples, recordBytes, outputBytes,
    uniformBytes: 16, safetyBytes: 16, residentBytes: recordBytes + 2 * outputBytes + 48,
    maxSingleBufferBytes: recordBytes};
}
/** Owns one device. Calls must be sequential. Any failed/cancelled evaluation
 * permanently destroys the runner; no partial result escapes. Records are
 * borrowed until evaluate resolves, with each chunk snapshotted and checked
 * immediately before upload. SharedArrayBuffer input is deliberately refused.
 */
export async function createCoherentMeshGPURunner(options = {}) {
  const capacity = coherentMeshGPUCapacity(options);
  const timeoutMs = options.timeoutMs ?? 600000;
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0 || timeoutMs > 2147483647) fail('timeoutMs must be positive and at most 2147483647');
  const gpu = options.gpu ?? globalThis.navigator?.gpu;
  if (!gpu) throw new Error('WebGPU unavailable; no CPU replacement was run');
  let device, dead = false, busy = false, reason = null, fatalReject, onUncaptured;
  const buffers = [], fetchController = new AbortController();
  const fatal = new Promise((_, reject) => { fatalReject = reject; });
  fatal.catch(() => {});
  const onAbort = () => destroy(abortError());
  function destroy(error = abortError()) {
    if (dead) return;
    dead = true; reason = error; fatalReject(error); fetchController.abort();
    options.signal?.removeEventListener('abort', onAbort);
    if (onUncaptured) device?.removeEventListener?.('uncapturederror', onUncaptured);
    for (const buffer of buffers) { try { buffer.destroy(); } catch {} }
    try { device?.destroy(); } catch {}
  }
  function check(signal, deadline) {
    if (dead) throw reason ?? abortError();
    if (options.signal?.aborted || signal?.aborted) { destroy(); throw reason; }
    if (deadline !== undefined && now() >= deadline) throw new Error('Coherent mesh GPU operation timed out');
  }
  async function wait(promise, label, deadline, signal) {
    // The operation is already started by the caller. Observe its rejection
    // even if a synchronous deadline/abort check wins before Promise.race.
    Promise.resolve(promise).catch(() => {});
    check(signal, deadline);
    let timer;
    const timeout = new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`Coherent mesh GPU ${label} timed out`)), Math.max(1, deadline - now())); });
    try { const result = await Promise.race([promise, fatal, timeout]); check(signal, deadline); return result; }
    finally { clearTimeout(timer); }
  }
  options.signal?.addEventListener('abort', onAbort, {once: true});
  const U = globalThis.GPUBufferUsage ?? {MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128};
  const MAP_READ = globalThis.GPUMapMode?.READ ?? 1;
  const start = now(), initDeadline = start + timeoutMs;
  let backend, pipeline, uniform, recordsBuffer, output, readback, flags, flagReadback, bindGroup;
  try {
    check(undefined, initDeadline);
    const adapter = await wait(gpu.requestAdapter({powerPreference: 'high-performance'}), 'adapter request', initDeadline);
    if (!adapter) throw new Error('No WebGPU adapter available');
    const info = adapterInfo(adapter);
    const software = /swiftshader|llvmpipe|lavapipe|software/i.test(Object.values(info).join(' ')) || info.isFallbackAdapter === true;
    backend = {kind: 'webgpu', label: `WebGPU${software ? ' software adapter' : ''}: ${info.description || info.device || info.architecture || info.vendor || 'adapter identity unavailable'}`,
      adapter: info, softwareAdapter: software, physicalGPUValidated: false, browserValidated: false, cpuShadow: false};
    const limits = adapter.limits ?? {};
    const requirements = {maxBufferSize: capacity.maxSingleBufferBytes, maxStorageBufferBindingSize: capacity.maxSingleBufferBytes,
      maxComputeWorkgroupsPerDimension: Math.ceil(capacity.chunkSamples / 64), maxComputeWorkgroupSizeX: 64, maxComputeInvocationsPerWorkgroup: 64, maxStorageBuffersPerShaderStage: 3};
    for (const [key, required] of Object.entries(requirements)) {
      if (limits[key] !== undefined && Number(limits[key]) < required) throw new Error(`WebGPU capacity insufficient: ${key}=${limits[key]}, requires ${required}`);
    }
    // A late requestDevice result still belongs to us after abort/timeout.
    const requested = Promise.resolve(adapter.requestDevice()).then(d => {
      if (dead) { try { d.destroy(); } catch {} } else device = d;
      return d;
    });
    await wait(requested, 'device request', initDeadline);
    check(undefined, initDeadline);
    device.lost?.then(info => { if (!dead) destroy(new Error(`Coherent mesh WebGPU device lost: ${info.message || info.reason || 'unknown'}`)); });
    onUncaptured = event => destroy(new Error(`Coherent mesh WebGPU uncaptured error: ${event.error?.message || 'unknown'}`));
    device.addEventListener?.('uncapturederror', onUncaptured);
    let code = options.shaderSource;
    if (code === undefined) {
      const response = await wait(fetch(new URL('./coherent-mesh-compute.wgsl', import.meta.url), {signal: fetchController.signal}), 'shader fetch', initDeadline);
      if (!response.ok) throw new Error(`Cannot load coherent mesh shader (${response.status})`);
      code = await wait(response.text(), 'shader text', initDeadline);
    }
    if (typeof code !== 'string' || !code.trim()) fail('shaderSource must be a nonempty WGSL string');
    device.pushErrorScope('validation');
    const module = device.createShaderModule({label: 'Coherent mesh bounded route sum', code});
    const compilation = await wait(module.getCompilationInfo(), 'shader compilation', initDeadline);
    const compilationError = await wait(device.popErrorScope(), 'shader validation', initDeadline);
    const errors = compilation.messages.filter(message => message.type === 'error');
    if (errors.length || compilationError) throw new Error(`Coherent mesh WGSL compilation failed: ${errors.map(error => error.message).join('; ')} ${compilationError?.message ?? ''}`);
    pipeline = await wait(device.createComputePipelineAsync({label: 'Coherent mesh complex field accumulation', layout: 'auto', compute: {module, entryPoint: 'accumulateFields'}}), 'pipeline creation', initDeadline);
    device.pushErrorScope('out-of-memory'); device.pushErrorScope('validation');
    const allocate = (label, size, usage) => { const buffer = device.createBuffer({label, size, usage}); buffers.push(buffer); return buffer; };
    uniform = allocate('Coherent mesh chunk parameters', 16, U.UNIFORM | U.COPY_DST);
    recordsBuffer = allocate('Coherent mesh bounded route records', capacity.recordBytes, U.STORAGE | U.COPY_DST);
    output = allocate('Coherent mesh chunk fields', capacity.outputBytes, U.STORAGE | U.COPY_SRC);
    readback = allocate('Coherent mesh chunk readback', capacity.outputBytes, U.MAP_READ | U.COPY_DST);
    flags = allocate('Coherent mesh finite inspection', 16, U.STORAGE | U.COPY_SRC | U.COPY_DST);
    flagReadback = allocate('Coherent mesh inspection readback', 16, U.MAP_READ | U.COPY_DST);
    bindGroup = device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: [
      {binding: 0, resource: {buffer: uniform}}, {binding: 1, resource: {buffer: recordsBuffer}},
      {binding: 2, resource: {buffer: output}}, {binding: 3, resource: {buffer: flags}},
    ]});
    const allocationError = await wait(device.popErrorScope(), 'buffer validation', initDeadline);
    const allocationOOM = await wait(device.popErrorScope(), 'buffer allocation', initDeadline);
    if (allocationError || allocationOOM) throw new Error(`Coherent mesh buffer creation failed: ${(allocationError || allocationOOM).message}`);
    check(undefined, initDeadline);
  } catch (error) { destroy(error); throw error; }
  const initializationMs = now() - start;
  async function mapped(buffer, size, deadline, signal) {
    let isMapped = false;
    try {
      await wait(buffer.mapAsync(MAP_READ, 0, size), 'readback map', deadline, signal);
      isMapped = true;
      return buffer.getMappedRange(0, size).slice(0);
    } finally { if (isMapped) buffer.unmap(); }
  }
  return {
    backend, capacity,
    async evaluate({records, sampleCount, routeCount, signal} = {}) {
      check(signal);
      if (busy) throw new Error('Coherent mesh GPU runner methods must be sequential');
      busy = true;
      const localAbort = () => destroy(abortError());
      signal?.addEventListener('abort', localAbort, {once: true});
      const begin = now(), deadline = begin + timeoutMs;
      const timing = {initializationMs, validationMs: 0, uploadEncodeMs: 0, queueWaitMs: 0, readbackMs: 0, totalMs: 0,
        chunkCount: 0, sampleCount, routeCount, timingScope: TIMING_SCOPE, cpuShadow: false};
      try {
        check(signal, deadline);
        if (!Number.isSafeInteger(sampleCount) || sampleCount < 1 || sampleCount > capacity.maxSamples) fail(`sampleCount must be an integer in 1..${capacity.maxSamples}`);
        if (!Number.isSafeInteger(routeCount) || routeCount < 0 || routeCount > capacity.maxRoutes) fail('routeCount must be an integer in 0..65');
        const length = sampleCount * routeCount * 4;
        if (!Number.isSafeInteger(length) || !(records instanceof Float32Array) || records.length !== length) fail(`records must be Float32Array(${length})`);
        if (typeof SharedArrayBuffer !== 'undefined' && records.buffer instanceof SharedArrayBuffer) fail('shared route records are not supported');
        const coherent = new Float32Array(sampleCount), incoherent = new Float32Array(sampleCount);
        const real = new Float32Array(sampleCount), imag = new Float32Array(sampleCount);
        for (let base = 0; base < sampleCount; base += capacity.chunkSamples) {
          check(signal, deadline);
          let time = now();
          const count = Math.min(capacity.chunkSamples, sampleCount - base);
          const chunk = records.slice(base * routeCount * 4, (base + count) * routeCount * 4);
          for (let sample = 0; sample < count; sample++) {
            let amplitudeBound = 0;
            for (let route = 0; route < routeCount; route++) {
              const index = 4 * (sample * routeCount + route), amplitude = chunk[index], phase = chunk[index + 1];
              if (!Number.isFinite(amplitude) || amplitude < 0) fail(`record ${base * routeCount + sample * routeCount + route} amplitude must be finite and nonnegative`);
              if (!Number.isFinite(phase) || Math.abs(phase) > COHERENT_MESH_GPU_LIMITS.phaseBound) fail(`record ${base * routeCount + sample * routeCount + route} phase must be reduced to [-pi, pi] before f32 conversion`);
              if (chunk[index + 2] !== 0 || chunk[index + 3] !== 0) fail('reserved record fields must be zero');
              amplitudeBound += amplitude;
            }
            if (amplitudeBound * amplitudeBound > COHERENT_MESH_GPU_LIMITS.maxIntensityBound) fail(`sample ${base + sample} intensity bound exceeds the safe f32 domain`);
          }
          timing.validationMs += now() - time;
          check(signal, deadline);
          time = now();
          device.pushErrorScope('validation');
          if (chunk.length) device.queue.writeBuffer(recordsBuffer, 0, chunk);
          device.queue.writeBuffer(uniform, 0, new Uint32Array([count, routeCount, 0, 0]));
          device.queue.writeBuffer(flags, 0, new Uint32Array(4));
          const encoder = device.createCommandEncoder({label: 'Coherent mesh native chunk'});
          const pass = encoder.beginComputePass();
          pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup);
          pass.dispatchWorkgroups(Math.ceil(count / 64)); pass.end();
          encoder.copyBufferToBuffer(output, 0, readback, 0, count * 16);
          encoder.copyBufferToBuffer(flags, 0, flagReadback, 0, 16);
          device.queue.submit([encoder.finish()]); timing.chunkCount++;
          timing.uploadEncodeMs += now() - time;
          time = now();
          await wait(device.queue.onSubmittedWorkDone(), 'queue completion', deadline, signal);
          const validationError = await wait(device.popErrorScope(), 'command validation', deadline, signal);
          if (validationError) throw new Error(`Coherent mesh command validation failed: ${validationError.message}`);
          timing.queueWaitMs += now() - time;
          time = now();
          const safety = new Uint32Array(await mapped(flagReadback, 16, deadline, signal));
          if (safety[0] || safety[1] || safety[2] !== count) throw Object.assign(new Error('Coherent mesh GPU finite/domain inspection failed; output withheld'), {
            name: 'CoherentMeshGPUValidationError', validation: {nonFinite: safety[0], outsideDomain: safety[1], inspected: safety[2], expected: count, cpuParityChecked: false},
          });
          const values = new Float32Array(await mapped(readback, count * 16, deadline, signal));
          for (let sample = 0; sample < count; sample++) {
            const index = sample * 4;
            for (let component = 0; component < 4; component++) {
              const value = values[index + component];
              if (!Number.isFinite(value) || Math.abs(value) > 1.0001e30 || (component < 2 && value < 0)) throw new Error(`Coherent mesh GPU invalid output at sample ${base + sample}; output withheld`);
            }
            coherent[base + sample] = values[index]; incoherent[base + sample] = values[index + 1];
            real[base + sample] = values[index + 2]; imag[base + sample] = values[index + 3];
          }
          timing.readbackMs += now() - time;
          // Yield a task, not only microtasks, so UI cancellation stays responsive.
          if (base + count < sampleCount) await wait(new Promise(resolve => setTimeout(resolve, 0)), 'chunk yield', deadline, signal);
        }
        check(signal, deadline); timing.totalMs = now() - begin;
        return {coherent, incoherent, real, imag, backend, timing};
      } catch (error) { destroy(error); throw error; }
      finally { signal?.removeEventListener('abort', localAbort); busy = false; }
    },
    destroy() { destroy(); },
  };
}
