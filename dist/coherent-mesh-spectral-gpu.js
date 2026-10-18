/** Native bounded spectral complete-route scalar transport.
 * CPU supplies wavelength-independent geometry and sampled wavelength records.
 * GPU evaluates UTD(k), robust split optical phases, coherent route sums,
 * per-wavelength intensities, then incoherent spectral scalar/XYZ integration.
 * No production CPU coefficient evaluation or parity/shadow calculation.
 * Route record, 16 f32: amplitude factor, material phase [-pi,pi], excess hi/lo;
 * kind (0 empty,1 GO,2 D,3 slab transmission), n/glassLength, phiI, phiO; L,sinBeta,boundaryR,reserved 0;
 * four stable cot deltas (difference +/-, sum +/-), snapped only at true poles.
 * D amplitude includes spherical spreading and any external mirror factor.
 * Spectral record, 8 f32: inverse-wavelength hi/lo (mm^-1), k, power weight,
 * CIE xbar/ybar/zbar, reserved 0. Records may be shared or sample-major.
 */
export const COHERENT_MESH_SPECTRAL_GPU_BUILD_ID = 'coherent-mesh-spectral-complete-routes-2026-10-14.1';
export const COHERENT_MESH_SPECTRAL_GPU_LIMITS = Object.freeze({
  maxSamples: 262144, maxRoutes: 65, maxWavelengths: 128, chunkSamples: 2048,
  maxRouteWavelengthEvaluations: 262144, maxIntensityBound: 1e30,
  maxOpticalExcessMm: 1e5, minSinBeta: 1e-3, maxReducedDistanceMm: 1e5,
  phaseBound: Math.fround(Math.PI), routeStride: 16, spectralStride: 8,
});
const now = () => globalThis.performance?.now?.() ?? Date.now();
const abortError = () => Object.assign(new Error('Coherent mesh spectral GPU calculation cancelled or destroyed'), {name: 'AbortError'});
const fail = message => { throw new RangeError(`Coherent mesh spectral GPU: ${message}`); };
const TIMING_SCOPE = 'Host elapsed validation/upload/encoding/submission/queue wait/readback; not GPU timestamp-query timings';
function adapterInfo(adapter) {
  const info = {};
  for (const key of ['vendor', 'architecture', 'device', 'description', 'isFallbackAdapter']) {
    try { const value = adapter.info?.[key]; if (typeof value === 'string' || typeof value === 'boolean') info[key] = value; } catch {}
  }
  return info;
}
export function coherentMeshSpectralGPUCapacity({maxSamples = COHERENT_MESH_SPECTRAL_GPU_LIMITS.maxSamples} = {}) {
  if (!Number.isSafeInteger(maxSamples) || maxSamples < 1 || maxSamples > COHERENT_MESH_SPECTRAL_GPU_LIMITS.maxSamples) fail('maxSamples must be an integer in 1..262144');
  const chunkSamples = Math.min(maxSamples, COHERENT_MESH_SPECTRAL_GPU_LIMITS.chunkSamples);
  const recordBytes = chunkSamples * COHERENT_MESH_SPECTRAL_GPU_LIMITS.maxRoutes * 64;
  const spectralBytes = chunkSamples * COHERENT_MESH_SPECTRAL_GPU_LIMITS.maxWavelengths * 32;
  const fieldBytes = chunkSamples * COHERENT_MESH_SPECTRAL_GPU_LIMITS.maxWavelengths * 16;
  const outputBytes = chunkSamples * 48;
  return {maxSamples, maxRoutes: 65, maxWavelengths: 128, chunkSamples,
    recordBytes, spectralBytes, fieldBytes, outputBytes, uniformBytes: 16, safetyBytes: 16,
    residentBytes: recordBytes + spectralBytes + fieldBytes + 2 * outputBytes + 48,
    maxSingleBufferBytes: Math.max(recordBytes, spectralBytes, fieldBytes)};
}
/** Owns one device. Calls must be sequential. Any failed/cancelled evaluation
 * permanently destroys the runner; no partial result escapes. Records are
 * borrowed until evaluate resolves, with each chunk snapshotted and checked
 * immediately before upload. SharedArrayBuffer input is deliberately refused.
 */
export async function createCoherentMeshSpectralGPURunner(options = {}) {
  const capacity = coherentMeshSpectralGPUCapacity(options);
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
    if (deadline !== undefined && now() >= deadline) throw new Error('Coherent mesh spectral GPU operation timed out');
  }
  async function wait(promise, label, deadline, signal) {
    // The operation is already started by the caller. Observe its rejection
    // even if a synchronous deadline/abort check wins before Promise.race.
    Promise.resolve(promise).catch(() => {});
    check(signal, deadline);
    let timer;
    const timeout = new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`Coherent mesh spectral GPU ${label} timed out`)), Math.max(1, deadline - now())); });
    try { const result = await Promise.race([promise, fatal, timeout]); check(signal, deadline); return result; }
    finally { clearTimeout(timer); }
  }
  options.signal?.addEventListener('abort', onAbort, {once: true});
  const U = globalThis.GPUBufferUsage ?? {MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128};
  const MAP_READ = globalThis.GPUMapMode?.READ ?? 1;
  const start = now(), initDeadline = start + timeoutMs;
  let backend, pipeline, integrationPipeline, uniform, recordsBuffer, spectralBuffer, fields, output, readback, flags, flagReadback, bindGroup, integrationBindGroup;
  try {
    check(undefined, initDeadline);
    const adapter = await wait(gpu.requestAdapter({powerPreference: 'high-performance'}), 'adapter request', initDeadline);
    if (!adapter) throw new Error('No WebGPU adapter available');
    const info = adapterInfo(adapter);
    const software = /swiftshader|llvmpipe|lavapipe|software/i.test(Object.values(info).join(' ')) || info.isFallbackAdapter === true;
    backend = {kind: 'webgpu', label: `WebGPU${software ? ' software adapter' : ''}: ${info.description || info.device || info.architecture || info.vendor || 'adapter identity unavailable'}`,
      adapter: info, softwareAdapter: software, physicalGPUValidated: false, browserValidated: false, cpuShadow: false,
      description: 'WebGPU wavelength-dependent UTD coefficients and robust optical phase, coherent route sum and square, then incoherent spectral scalar/XYZ integration'};
    const limits = adapter.limits ?? {};
    const requirements = {maxBufferSize: capacity.maxSingleBufferBytes, maxStorageBufferBindingSize: capacity.maxSingleBufferBytes,
      maxComputeWorkgroupsPerDimension: Math.max(capacity.maxWavelengths, Math.ceil(capacity.chunkSamples / 64)), maxComputeWorkgroupSizeX: 64, maxComputeInvocationsPerWorkgroup: 64, maxStorageBuffersPerShaderStage: 4};
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
    device.lost?.then(info => { if (!dead) destroy(new Error(`Coherent mesh spectral WebGPU device lost: ${info.message || info.reason || 'unknown'}`)); });
    onUncaptured = event => destroy(new Error(`Coherent mesh spectral WebGPU uncaptured error: ${event.error?.message || 'unknown'}`));
    device.addEventListener?.('uncapturederror', onUncaptured);
    let code = options.shaderSource;
    if (code === undefined) {
      const includes = [];
      for (const filename of ['./utd-functions.wgsl', './coherent-mesh-spectral-compute.wgsl']) {
        const response = await wait(fetch(new URL(filename, import.meta.url), {signal: fetchController.signal}), 'shader fetch', initDeadline);
        if (!response.ok) throw new Error(`Cannot load coherent mesh spectral shader (${response.status})`);
        includes.push(await wait(response.text(), 'shader text', initDeadline));
      }
      code = includes.join('\n');
    }
    if (typeof code !== 'string' || !code.trim()) fail('shaderSource must be a nonempty WGSL string');
    device.pushErrorScope('validation');
    const module = device.createShaderModule({label: 'Coherent mesh spectral bounded route sum', code});
    const compilation = await wait(module.getCompilationInfo(), 'shader compilation', initDeadline);
    const compilationError = await wait(device.popErrorScope(), 'shader validation', initDeadline);
    const errors = compilation.messages.filter(message => message.type === 'error');
    if (errors.length || compilationError) throw new Error(`Coherent mesh spectral WGSL compilation failed: ${errors.map(error => error.message).join('; ')} ${compilationError?.message ?? ''}`);
    pipeline = await wait(device.createComputePipelineAsync({label: 'Coherent mesh spectral complex field accumulation', layout: 'auto', compute: {module, entryPoint: 'accumulateSpectralFields'}}), 'pipeline creation', initDeadline);
    integrationPipeline = await wait(device.createComputePipelineAsync({label: 'Incoherent spectral scalar and XYZ integration', layout: 'auto', compute: {module, entryPoint: 'integrateSpectralIntensities'}}), 'integration pipeline creation', initDeadline);
    device.pushErrorScope('out-of-memory'); device.pushErrorScope('validation');
    const allocate = (label, size, usage) => { const buffer = device.createBuffer({label, size, usage}); buffers.push(buffer); return buffer; };
    uniform = allocate('Coherent mesh spectral chunk parameters', 16, U.UNIFORM | U.COPY_DST);
    recordsBuffer = allocate('Coherent mesh spectral bounded route records', capacity.recordBytes, U.STORAGE | U.COPY_DST);
    spectralBuffer = allocate('Bounded sampled spectral records', capacity.spectralBytes, U.STORAGE | U.COPY_DST);
    fields = allocate('Bounded per-wavelength intensities', capacity.fieldBytes, U.STORAGE);
    output = allocate('Coherent mesh spectral chunk fields', capacity.outputBytes, U.STORAGE | U.COPY_SRC);
    readback = allocate('Coherent mesh spectral chunk readback', capacity.outputBytes, U.MAP_READ | U.COPY_DST);
    flags = allocate('Coherent mesh spectral finite inspection', 16, U.STORAGE | U.COPY_SRC | U.COPY_DST);
    flagReadback = allocate('Coherent mesh spectral inspection readback', 16, U.MAP_READ | U.COPY_DST);
    bindGroup = device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: [
      {binding: 0, resource: {buffer: uniform}}, {binding: 1, resource: {buffer: recordsBuffer}},
      {binding: 2, resource: {buffer: spectralBuffer}}, {binding: 3, resource: {buffer: fields}},
      {binding: 5, resource: {buffer: flags}},
    ]});
    integrationBindGroup = device.createBindGroup({layout: integrationPipeline.getBindGroupLayout(0), entries: [
      {binding: 0, resource: {buffer: uniform}}, {binding: 2, resource: {buffer: spectralBuffer}},
      {binding: 3, resource: {buffer: fields}}, {binding: 4, resource: {buffer: output}},
      {binding: 5, resource: {buffer: flags}},
    ]});
    const allocationError = await wait(device.popErrorScope(), 'buffer validation', initDeadline);
    const allocationOOM = await wait(device.popErrorScope(), 'buffer allocation', initDeadline);
    if (allocationError || allocationOOM) throw new Error(`Coherent mesh spectral buffer creation failed: ${(allocationError || allocationOOM).message}`);
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
    async evaluate({records, spectralRecords, sampleCount, routeCount, wavelengthCount, signal} = {}) {
      check(signal);
      if (busy) throw new Error('Coherent mesh spectral GPU runner methods must be sequential');
      busy = true;
      const localAbort = () => destroy(abortError());
      signal?.addEventListener('abort', localAbort, {once: true});
      const begin = now(), deadline = begin + timeoutMs;
      const timing = {initializationMs, validationMs: 0, uploadEncodeMs: 0, queueWaitMs: 0, readbackMs: 0, totalMs: 0,
        chunkCount: 0, sampleCount, routeCount, wavelengthCount, timingScope: TIMING_SCOPE, cpuShadow: false};
      try {
        check(signal, deadline);
        if (!Number.isSafeInteger(sampleCount) || sampleCount < 1 || sampleCount > capacity.maxSamples) fail(`sampleCount must be an integer in 1..${capacity.maxSamples}`);
        if (!Number.isSafeInteger(routeCount) || routeCount < 0 || routeCount > capacity.maxRoutes) fail('routeCount must be an integer in 0..65');
        if (!Number.isSafeInteger(wavelengthCount) || wavelengthCount < 1 || wavelengthCount > capacity.maxWavelengths) fail('wavelengthCount must be an integer in 1..128');
        if (!(records instanceof Float32Array) || records.length !== sampleCount * routeCount * 16) fail('records must be a sample-major Float32Array of 16-float complete routes');
        const sharedSpectrum = spectralRecords?.length === wavelengthCount * 8;
        if (!(spectralRecords instanceof Float32Array) || (!sharedSpectrum && spectralRecords.length !== sampleCount * wavelengthCount * 8)) fail('spectralRecords must be Float32Array of 8-float shared or sample-major nodes');
        if (typeof SharedArrayBuffer !== 'undefined' && [records, spectralRecords].some(a => a.buffer instanceof SharedArrayBuffer)) fail('shared-memory inputs are not supported');
        const sharedNodes = sharedSpectrum ? spectralRecords.slice() : null;
        const result = Object.fromEntries(['coherent', 'incoherent', 'geometric', 'coherentXYZ', 'incoherentXYZ', 'geometricXYZ'].map(name => [name, new Float32Array(sampleCount * (name.endsWith('XYZ') ? 3 : 1))]));
        // Bound one submitted dispatch by route×wavelength work as well as
        // memory. Poll/yield between chunks so cancellation remains responsive.
        const chunkSamples = Math.min(capacity.chunkSamples, Math.max(1, Math.floor(COHERENT_MESH_SPECTRAL_GPU_LIMITS.maxRouteWavelengthEvaluations / (Math.max(routeCount, 1) * wavelengthCount))));
        timing.chunkSamples = chunkSamples;
        for (let base = 0; base < sampleCount; base += chunkSamples) {
          check(signal, deadline);
          let time = now();
          const count = Math.min(chunkSamples, sampleCount - base);
          const chunk = records.slice(base * routeCount * 16, (base + count) * routeCount * 16);
          const nodes = sharedNodes ?? spectralRecords.slice(base * wavelengthCount * 8, (base + count) * wavelengthCount * 8);
          const spectralBounds = new Float64Array(sharedSpectrum ? 1 : count);
          for (let s = 0; s < spectralBounds.length; s++) {
            let weightBound = 0;
            for (let w = 0; w < wavelengthCount; w++) {
              const at = (s * wavelengthCount + w) * 8;
              for (let c = 0; c < 8; c++) if (!Number.isFinite(nodes[at + c])) fail('spectral record contains a nonfinite value');
              const hi = nodes[at], lo = nodes[at + 1], inverse = hi + lo, k = nodes[at + 2], weight = nodes[at + 3];
              if (inverse < 1e6 / 830 - 1e-9 || inverse > 1e6 / 360 + 1e-9 || hi <= 0 || Math.abs(lo) > Math.abs(hi) * 6e-8) fail('inverse wavelength must be a split visible-band value (360..830 nm)');
              if (Math.abs(k - 2 * Math.PI * inverse) > 2e-7 * k) fail('spectral k does not match the split inverse wavelength');
              if (weight < 0 || weight > 1e6 || nodes[at + 4] < 0 || nodes[at + 5] < 0 || nodes[at + 6] < 0 || Math.max(nodes[at + 4], nodes[at + 5], nodes[at + 6]) > 4 || nodes[at + 7] < 0 || nodes[at + 7] > 1e6) fail('spectral weights/CIE values are outside the supported domain');
              weightBound += weight * Math.max(1, nodes[at + 4], nodes[at + 5], nodes[at + 6]);
            }
            spectralBounds[s] = Math.max(1, weightBound);
          }
          for (let sample = 0; sample < count; sample++) {
            let amplitudeBound = 0;
            for (let route = 0; route < routeCount; route++) {
              const at = 16 * (sample * routeCount + route), kind = chunk[at + 4];
              for (let c = 0; c < 16; c++) if (!Number.isFinite(chunk[at + c])) fail('route record contains a nonfinite value');
              if (kind === 0) { for (let c = 0; c < 16; c++) if (chunk[at + c] !== 0) fail('empty route record must be all zero'); continue; }
              if (kind !== 1 && kind !== 2 && kind !== 3) fail('route kind must be 0, 1 (GO), 2 (D), or 3 (slab transmission)');
              const amplitude = chunk[at], phase = chunk[at + 1], hi = chunk[at + 2], lo = chunk[at + 3];
              if (amplitude < 0 || Math.abs(phase) > COHERENT_MESH_SPECTRAL_GPU_LIMITS.phaseBound) fail('route amplitude or material phase outside its domain');
              if (Math.abs(hi + lo) > COHERENT_MESH_SPECTRAL_GPU_LIMITS.maxOpticalExcessMm || Math.abs(lo) > Math.max(Math.abs(hi) * 6e-8, 1e-37)) fail('optical excess must be a canonical split within +/-100000 mm');
              if (kind === 1) { for (let c = 5; c < 16; c++) if (chunk[at + c] !== 0) fail('GO diffraction fields must be zero'); amplitudeBound += amplitude; }
              else if (kind === 3) { if(chunk[at+5]<0||chunk[at+5]>1e5)fail('Slab glass length outside supported domain'); for(let c=6;c<16;c++)if(chunk[at+c]!==0)fail('Slab reserved fields must be zero'); amplitudeBound+=amplitude; }
              else {
                const n = chunk[at + 5], phiI = chunk[at + 6], phiO = chunk[at + 7], L = chunk[at + 8], sinBeta = chunk[at + 9], boundary = chunk[at + 10];
                if (n < 1 || n > 2 || phiI < 0 || phiO < 0 || Math.max(phiI, phiO) > n * Math.PI + 2e-6 || !(L >= 1e-20 && L <= 1e5) || sinBeta < Math.fround(1e-3) || sinBeta > 1 || (boundary !== -1 && boundary !== 1) || chunk[at + 11] !== 0) fail('UTD geometry outside the supported domain');
                for (let c = 12; c < 16; c++) if (Math.abs(chunk[at + c]) > n * Math.PI + 2e-6) fail('stable cot delta outside the principal wedge interval');
                // Conservative coefficient bound, not a CPU UTD evaluation.
                amplitudeBound += amplitude * (16 * Math.sqrt(L) / sinBeta);
              }
            }
            if (amplitudeBound * amplitudeBound * spectralBounds[sharedSpectrum ? 0 : sample] > COHERENT_MESH_SPECTRAL_GPU_LIMITS.maxIntensityBound) fail(`sample ${base + sample} exceeds the safe intensity domain`);
          }
          timing.validationMs += now() - time;
          check(signal, deadline); time = now();
          device.pushErrorScope('validation');
          if (chunk.length) device.queue.writeBuffer(recordsBuffer, 0, chunk);
          device.queue.writeBuffer(spectralBuffer, 0, nodes);
          device.queue.writeBuffer(uniform, 0, new Uint32Array([count, routeCount, wavelengthCount, sharedSpectrum ? 1 : 0]));
          device.queue.writeBuffer(flags, 0, new Uint32Array(4));
          const encoder = device.createCommandEncoder({label: 'Complete spectral route chunk'});
          const pass = encoder.beginComputePass();
          pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup);
          pass.dispatchWorkgroups(Math.ceil(count / 64), wavelengthCount); pass.end();
          const integration = encoder.beginComputePass();
          integration.setPipeline(integrationPipeline); integration.setBindGroup(0, integrationBindGroup);
          integration.dispatchWorkgroups(Math.ceil(count / 64)); integration.end();
          encoder.copyBufferToBuffer(output, 0, readback, 0, count * 48);
          encoder.copyBufferToBuffer(flags, 0, flagReadback, 0, 16);
          device.queue.submit([encoder.finish()]); timing.chunkCount++;
          timing.uploadEncodeMs += now() - time; time = now();
          await wait(device.queue.onSubmittedWorkDone(), 'queue completion', deadline, signal);
          const validationError = await wait(device.popErrorScope(), 'command validation', deadline, signal);
          if (validationError) throw new Error(`Coherent mesh spectral command validation failed: ${validationError.message}`);
          timing.queueWaitMs += now() - time; time = now();
          const safety = new Uint32Array(await mapped(flagReadback, 16, deadline, signal));
          if (safety[0] || safety[1] || safety[2] !== count * wavelengthCount || safety[3] !== count) throw Object.assign(new Error('Coherent mesh spectral GPU finite/domain inspection failed; output withheld'), {
            name: 'CoherentMeshSpectralGPUValidationError', validation: {nonFinite: safety[0], outsideDomain: safety[1], inspectedWavelengths: safety[2], expectedWavelengths: count * wavelengthCount, inspectedSamples: safety[3], expectedSamples: count, cpuParityChecked: false},
          });
          const values = new Float32Array(await mapped(readback, count * 48, deadline, signal));
          for (let sample = 0; sample < count; sample++) {
            for (let kind = 0; kind < 3; kind++) {
              const name = ['coherent', 'incoherent', 'geometric'][kind], at = sample * 12 + kind * 4;
              for (let c = 0; c < 4; c++) if (!Number.isFinite(values[at + c]) || values[at + c] < 0 || values[at + c] > 1.0001e30) throw new Error(`Coherent mesh spectral GPU invalid output at sample ${base + sample}; output withheld`);
              result[name][base + sample] = values[at + 3];
              result[name + 'XYZ'].set(values.subarray(at, at + 3), (base + sample) * 3);
            }
          }
          timing.readbackMs += now() - time;
          if (base + count < sampleCount) await wait(new Promise(resolve => setTimeout(resolve, 0)), 'chunk yield', deadline, signal);
        }
        check(signal, deadline); timing.totalMs = now() - begin;
        return {...result, backend, timing};
      } catch (error) { destroy(error); throw error; }
      finally { signal?.removeEventListener('abort', localAbort); busy = false; }
    },
    destroy() { destroy(); },
  };
}
/** Geometric angle reduction, never wavelength-dependent coefficient work. */
export function coherentSpectralCotDeltas(n, phiI, phiO, epsilon = 1e-12) {
  if (![n, phiI, phiO, epsilon].every(Number.isFinite) || n < 1 || n > 2 || epsilon < 0) fail('invalid cot-delta geometry');
  return [phiO - phiI, phiO + phiI].flatMap(beta => [1, -1].map(side => {
    const nearest = Math.floor((beta + side * Math.PI) / (2 * Math.PI * n) + .5);
    const delta = beta - (2 * Math.PI * n * nearest - side * Math.PI);
    return Math.abs(delta) <= epsilon ? 0 : delta;
  }));
}
export const packCoherentSpectralCotDeltas = coherentSpectralCotDeltas;
