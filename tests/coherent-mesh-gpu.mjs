/** Offline native Dawn/Tint compilation and executable WGSL tests.
 * Execute with:
 * VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json WEBGPU_EXECUTION_BACKEND=vulkan node tests/coherent-mesh-gpu.mjs --execute
 * SwiftShader is software execution, never browser or physical-GPU validation.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {create, globals} from 'webgpu';
import {createCoherentMeshGPURunner, coherentMeshGPUCapacity, COHERENT_MESH_GPU_LIMITS} from '../dist/coherent-mesh-gpu.js';
Object.assign(globalThis, globals);
const shader = fs.readFileSync(new URL('../dist/coherent-mesh-compute.wgsl', import.meta.url), 'utf8');
const execute = process.argv.includes('--execute');
const report = {suite: 'coherent-mesh-gpu', schemaVersion: 1, executionRequested: execute, browserValidated: false,
  physicalGPUValidated: false, cpuShadow: false, oracleTolerance: {normalized: 8e-6, denseSingleRoute: 2e-7, fieldScale: 'maximum per-sample sum of route amplitudes', intensityScale: 'square of maximum per-sample sum of route amplitudes'}, tests: [], failures: [], nativeCases: [], negativeControls: [], hashes: {}};
for (const file of ['dist/coherent-mesh-gpu.js', 'dist/coherent-mesh-compute.wgsl', 'tests/coherent-mesh-gpu.mjs']) {
  report.hashes[file] = createHash('sha256').update(fs.readFileSync(new URL(`../${file}`, import.meta.url))).digest('hex');
}
async function test(name, run) {
  try { await run(); report.tests.push(name); console.log(`PASS ${name}`); }
  catch (error) { report.failures.push({name, error: error.stack, validation: error.validation}); console.error(`FAIL ${name}: ${error.stack}`); }
}
const deferred = () => { let resolve, reject; const promise = new Promise((r, j) => { resolve = r; reject = j; }); return {promise, resolve, reject}; };
const tick = () => new Promise(resolve => setTimeout(resolve, 0));
function mock({queueWait, mapWait, scopeWait, scopeAt = 1, onScope, limits, compileError, validationError, mapError, invalidOutput, invalidSafety} = {}) {
  const lost = deferred(), buffers = [], events = {};
  let destroyed = 0, submits = 0, unmaps = 0, scope = 0, dispatched = 0;
  const device = {
    lost: lost.promise, limits: {maxBufferSize: 1e9, maxStorageBufferBindingSize: 1e9},
    addEventListener: (name, fn) => { events[name] = fn; }, removeEventListener: name => { delete events[name]; },
    destroy() { destroyed++; }, pushErrorScope() {},
    async popErrorScope() { if (++scope === scopeAt) { onScope?.(); if (scopeWait) await scopeWait; if (validationError) return {message: validationError}; } return null; },
    createShaderModule: () => ({getCompilationInfo: async () => ({messages: compileError ? [{type: 'error', message: compileError}] : []})}),
    createComputePipelineAsync: async () => ({getBindGroupLayout: () => ({})}), createBindGroup: () => ({}),
    createBuffer(options) {
      const buffer = {...options, bytes: new ArrayBuffer(options.size), dead: false, destroy() { this.dead = true; },
        async mapAsync() { if (mapWait) await mapWait; if (mapError) throw new Error(mapError); },
        unmap() { unmaps++; }, getMappedRange(offset = 0, size = options.size) { return this.bytes.slice(offset, offset + size); }};
      buffers.push(buffer); return buffer;
    },
    createCommandEncoder() {
      const copies = [];
      return {beginComputePass: () => ({setPipeline() {}, setBindGroup() {}, dispatchWorkgroups(n) { dispatched = n; }, end() {}}),
        copyBufferToBuffer(source, so, destination, dest, size) { copies.push(() => new Uint8Array(destination.bytes, dest, size).set(new Uint8Array(source.bytes, so, size))); },
        finish: () => ({copies})};
    },
    queue: {
      writeBuffer(buffer, offset, input) { new Uint8Array(buffer.bytes, offset, input.byteLength).set(new Uint8Array(input.buffer, input.byteOffset, input.byteLength)); },
      submit(commands) {
        submits++;
        const [uniform, , output, , flags] = buffers;
        new Uint32Array(flags.bytes)[2] = invalidSafety ? 0 : new Uint32Array(uniform.bytes)[0];
        if (invalidOutput !== undefined) new Float32Array(output.bytes)[0] = invalidOutput;
        for (const command of commands) for (const copy of command.copies) copy();
      },
      async onSubmittedWorkDone() { if (queueWait) await queueWait; },
    },
  };
  return {device, buffers, lost, events, gpu: {requestAdapter: async () => ({info: {description: 'mock lifecycle only'}, limits: limits ?? device.limits, requestDevice: async () => device})},
    get destroyed() { return destroyed; }, get submits() { return submits; }, get unmaps() { return unmaps; }, get dispatched() { return dispatched; }};
}
const zeroInput = (sampleCount = 2, routeCount = 1) => ({records: new Float32Array(sampleCount * routeCount * 4), sampleCount, routeCount});
const makeMock = async (options = {}, runnerOptions = {}) => { const x = mock(options), runner = await createCoherentMeshGPURunner({gpu: x.gpu, shaderSource: shader, maxSamples: 4, ...runnerOptions}); return {x, runner}; };
function recordsFrom(samples) {
  const routeCount = samples[0].length, records = new Float32Array(samples.length * routeCount * 4);
  for (let s = 0; s < samples.length; s++) for (let r = 0; r < routeCount; r++) records.set(samples[s][r], 4 * (s * routeCount + r));
  return {records, sampleCount: samples.length, routeCount};
}
// Independent array oracle: only used by this offline test, never by runner.
function oracle({records, sampleCount, routeCount}) {
  const result = Object.fromEntries(['coherent', 'incoherent', 'real', 'imag'].map(k => [k, new Float64Array(sampleCount)]));
  let maxAmplitudeSum = 0;
  for (let s = 0; s < sampleCount; s++) {
    let re = 0, im = 0, power = 0, amplitudeSum = 0;
    for (let r = 0; r < routeCount; r++) {
      const i = 4 * (s * routeCount + r), a = records[i], p = records[i + 1];
      re += a * Math.cos(p); im += a * Math.sin(p); power += a * a; amplitudeSum += a;
    }
    result.real[s] = re; result.imag[s] = im; result.coherent[s] = re * re + im * im; result.incoherent[s] = power;
    maxAmplitudeSum = Math.max(maxAmplitudeSum, amplitudeSum);
  }
  result.maxAmplitudeSum = maxAmplitudeSum;
  return result;
}
function compare(actual, expected, tolerance = 8e-6) {
  const errors = {};
  for (const key of ['coherent', 'incoherent', 'real', 'imag']) {
    assert.equal(actual[key].length, expected[key].length);
    let maxAbsoluteError = 0, referencePeak = 0;
    for (let i = 0; i < actual[key].length; i++) { assert(Number.isFinite(actual[key][i])); maxAbsoluteError = Math.max(maxAbsoluteError, Math.abs(actual[key][i] - expected[key][i])); referencePeak = Math.max(referencePeak, Math.abs(expected[key][i])); }
    const scale = key === 'real' || key === 'imag' ? expected.maxAmplitudeSum : expected.maxAmplitudeSum ** 2;
    errors[key] = {maxAbsoluteError, normalizedError: maxAbsoluteError / Math.max(scale, 1e-30), normalizationScale: scale, referencePeak, relativeToReferencePeak: maxAbsoluteError / Math.max(referencePeak, 1e-30)};
    assert(maxAbsoluteError <= tolerance * Math.max(scale, 1e-30), `${key}: ${JSON.stringify(errors[key])}`);
  }
  return errors;
}
function randomInput(sampleCount, routeCount, seed = 9121) {
  const records = new Float32Array(sampleCount * routeCount * 4);
  const next = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; };
  for (let i = 0; i < records.length; i += 4) { records[i] = next(); records[i + 1] = (2 * next() - 1) * Math.PI; }
  return {records, sampleCount, routeCount};
}
await test('native capacity supports 262144 samples/65 routes within 9 MiB GPU residency', () => {
  const c = coherentMeshGPUCapacity(); assert.equal(c.maxSamples, 262144); assert.equal(c.maxRoutes, 65); assert.equal(c.chunkSamples, 8192);
  assert(c.residentBytes < 9 * 1024 ** 2); assert.equal(coherentMeshGPUCapacity({maxSamples: 1}).chunkSamples, 1);
  for (const maxSamples of [0, -1, 1.1, 262145, Infinity, Number.MAX_SAFE_INTEGER]) assert.throws(() => coherentMeshGPUCapacity({maxSamples}));
  report.capacity = c;
});
await test('adapter/capacity/timeout option errors fail without a CPU substitute', async () => {
  await assert.rejects(createCoherentMeshGPURunner({gpu: {requestAdapter: async () => null}}), /adapter/);
  for (const timeoutMs of [0, -1, Infinity, 2147483648]) await assert.rejects(createCoherentMeshGPURunner({timeoutMs}), /timeoutMs/);
  const x = mock({limits: {maxBufferSize: 10, maxStorageBufferBindingSize: 10}});
  await assert.rejects(createCoherentMeshGPURunner({gpu: x.gpu, shaderSource: shader}), /capacity/); assert.equal(x.buffers.length, 0);
});
await test('exact count/type/phase/amplitude/reserved/overflow guards reject all invalid records', async () => {
  const inputs = [
    {...zeroInput(), records: [0, 0, 0, 0]}, {...zeroInput(), records: new Float32Array(4)},
    {...zeroInput(), sampleCount: 0}, {...zeroInput(), sampleCount: 5}, {...zeroInput(), sampleCount: 1.5},
    {...zeroInput(), routeCount: 66}, {...zeroInput(), routeCount: -1}, {...zeroInput(), routeCount: Infinity},
    recordsFrom([[[NaN, 0]]]), recordsFrom([[[Infinity, 0]]]), recordsFrom([[[-1, 0]]]),
    recordsFrom([[[1, NaN]]]), recordsFrom([[[1, 4]]]), recordsFrom([[[1, -4]]]),
    recordsFrom([[[1, 0, 1]]]), recordsFrom([[[1, 0, 0, NaN]]]), recordsFrom([[[1e16, 0]]]),
  ];
  if (typeof SharedArrayBuffer !== 'undefined') inputs.push({records: new Float32Array(new SharedArrayBuffer(16)), sampleCount: 1, routeCount: 1});
  for (const input of inputs) {
    const {x, runner} = await makeMock(); await assert.rejects(runner.evaluate(input), RangeError);
    assert.equal(x.submits, 0); assert.equal(x.destroyed, 1); assert(x.buffers.every(b => b.dead)); await assert.rejects(runner.evaluate(zeroInput()));
  }
});
await test('zero-route/repeated native-sized evaluations reuse buffers; no stale accumulators', async () => {
  const {x, runner} = await makeMock();
  try {
    for (const input of [zeroInput(4, 0), zeroInput(1, 1), zeroInput(3, 1)]) {
      const result = await runner.evaluate(input); assert(result.coherent.every(v => v === 0)); assert.equal(result.real.length, input.sampleCount);
      assert.equal(result.timing.cpuShadow, false); assert.equal(result.timing.chunkCount, 1);
    }
    assert.equal(x.buffers.length, 6); assert.equal(x.submits, 3); assert.equal(x.unmaps, 6);
  } finally { runner.destroy(); runner.destroy(); }
  assert.equal(x.destroyed, 1);
});
await test('pre-abort and late device acquisition cancellation/timeout do not leak ownership', async () => {
  const cancelled = new AbortController(); cancelled.abort(); const initial = mock();
  await assert.rejects(createCoherentMeshGPURunner({gpu: initial.gpu, shaderSource: shader, signal: cancelled.signal}), e => e.name === 'AbortError'); assert.equal(initial.buffers.length, 0);
  for (const mode of ['abort', 'timeout']) {
    const pending = deferred(), c = new AbortController(); let destroyed = 0;
    const p = createCoherentMeshGPURunner({gpu: {requestAdapter: async () => ({requestDevice: () => pending.promise})}, shaderSource: shader, signal: c.signal, timeoutMs: mode === 'timeout' ? 10 : 10000});
    const rejection = assert.rejects(p); await tick(); if (mode === 'abort') c.abort(); await rejection;
    pending.resolve({destroy() { destroyed++; }}); await tick(); assert.equal(destroyed, 1);
  }
});
await test('device-adoption microtask races clean up exactly once', async () => {
  for (let depth = 0; depth < 6; depth++) {
    const c = new AbortController(), x = mock();
    const gpu = {requestAdapter: async () => ({requestDevice() { let later = Promise.resolve(); for (let i = 0; i < depth; i++) later = later.then(() => {}); later.then(() => c.abort()); return Promise.resolve(x.device); }})};
    await assert.rejects(createCoherentMeshGPURunner({gpu, shaderSource: shader, signal: c.signal}), e => e.name === 'AbortError');
    await tick(); assert.equal(x.destroyed, 1);
  }
});
await test('queue/map waits, concurrent calls, abort/destroy/loss/uncaptured/timeout are bounded', async () => {
  for (const stage of ['queue', 'map']) for (const mode of ['abort', 'destroy', 'device-loss', 'uncaptured', 'timeout']) {
    const hold = deferred(), c = new AbortController();
    const {x, runner} = await makeMock({[stage === 'queue' ? 'queueWait' : 'mapWait']: hold.promise}, {timeoutMs: mode === 'timeout' ? 15 : 10000});
    const p = runner.evaluate({...zeroInput(), signal: c.signal}); const rejection = assert.rejects(p);
    await tick(); await assert.rejects(runner.evaluate(zeroInput()), /sequential/); assert.equal(x.destroyed, 0);
    if (mode === 'abort') c.abort(); if (mode === 'destroy') runner.destroy();
    if (mode === 'device-loss') x.lost.resolve({message: 'test loss'});
    if (mode === 'uncaptured') x.events.uncapturederror({error: {message: 'test uncaptured'}});
    await rejection; hold.resolve(); assert.equal(x.destroyed, 1); assert(x.buffers.every(b => b.dead)); await assert.rejects(runner.evaluate(zeroInput()));
  }
});
await test('uncooperative shader/command error scopes terminate on abort/loss/timeout', async () => {
  for (const stage of ['shader', 'command']) for (const mode of ['abort', 'device-loss', 'timeout']) {
    const hold = deferred(), entered = deferred(), c = new AbortController();
    const x = mock({scopeWait: hold.promise, scopeAt: stage === 'shader' ? 1 : 4, onScope: entered.resolve});
    const creating = createCoherentMeshGPURunner({gpu: x.gpu, maxSamples: 4, shaderSource: shader, signal: c.signal, timeoutMs: mode === 'timeout' ? 20 : 10000});
    const p = stage === 'shader' ? creating : (await creating).evaluate(zeroInput());
    const rejection = assert.rejects(p); await entered.promise;
    if (mode === 'abort') c.abort(); if (mode === 'device-loss') x.lost.resolve({message: 'scope loss'});
    await rejection; hold.resolve(); assert.equal(x.destroyed, 1); assert(x.buffers.every(b => b.dead));
  }
});
await test('compile/allocation/map/safety/readback faults withhold output and release buffers', async () => {
  for (const options of [{compileError: 'bad WGSL'}, {validationError: 'invalid shader'}, {validationError: 'allocation failed', scopeAt: 3}]) {
    const x = mock(options); await assert.rejects(createCoherentMeshGPURunner({gpu: x.gpu, shaderSource: shader})); assert.equal(x.destroyed, 1); assert(x.buffers.every(b => b.dead));
  }
  for (const options of [{mapError: 'map failure'}, {invalidOutput: NaN}, {invalidOutput: -1}, {invalidOutput: Infinity}, {invalidSafety: true}]) {
    const {x, runner} = await makeMock(options); await assert.rejects(runner.evaluate(zeroInput())); assert.equal(x.destroyed, 1); assert(x.buffers.every(b => b.dead));
  }
});
await test('native Dawn/Tint compiles actual WGSL and rejects malformed shader', async () => {
  const gpu = create(['backend=null']);
  const runner = await createCoherentMeshGPURunner({gpu, shaderSource: shader, maxSamples: 4}); runner.destroy();
  await assert.rejects(createCoherentMeshGPURunner({gpu, shaderSource: shader + '\nINVALID WGSL', maxSamples: 4}), /WGSL/);
});
if (execute) {
  const backend = process.env.WEBGPU_EXECUTION_BACKEND;
  assert(['vulkan', 'metal', 'd3d11', 'd3d12', 'opengl', 'opengles'].includes(backend), 'Explicit executable backend required; null backend refused');
  const gpu = create([`backend=${backend}`]); report.requestedBackend = backend;
  report.driverSelection = {VK_ICD_FILENAMES: process.env.VK_ICD_FILENAMES ?? null};
  const make = options => createCoherentMeshGPURunner({gpu, shaderSource: shader, ...options});
  await test('actual native sums match independent CPU oracle, phase flips, dark and bright ports', async () => {
    const input = recordsFrom([[[1, 0], [1, 0]], [[1, 0], [1, Math.PI]], [[1, 0], [1, -Math.PI]], [[2, Math.PI / 2], [1, -Math.PI / 2]], [[0, Math.PI], [0, 0]]]);
    const runner = await make({maxSamples: 5});
    try {
      const got = await runner.evaluate(input), errors = compare(got, oracle(input));
      assert.equal(got.coherent[0], 4); assert.equal(got.incoherent[0], 2);
      assert(got.coherent[1] < 1e-12 && got.coherent[2] < 1e-12); assert.equal(got.incoherent[1], 2);
      assert.equal(got.coherent[4], 0); assert.equal(got.incoherent[4], 0);
      report.adapter = got.backend; report.nativeCases.push({name: 'phase flips and bright/dark ports', errors, timing: got.timing, values: [...got.coherent]});
    } finally { runner.destroy(); }
  });
  await test('actual single-route phase invariance, zero routes and amplitude-squared power scaling', async () => {
    const input = randomInput(131, 65), runner = await make({maxSamples: 131});
    try {
      const original = await runner.evaluate(input), scaled = {...input, records: input.records.slice()};
      for (let i = 0; i < scaled.records.length; i += 4) scaled.records[i] *= 2;
      const doubled = await runner.evaluate(scaled); compare(doubled, oracle(scaled));
      for (const key of ['coherent', 'incoherent', 'real', 'imag']) {
        const scale = key === 'real' || key === 'imag' ? 2 : 4;
        for (let i = 0; i < original[key].length; i++) assert.equal(doubled[key][i], scale * original[key][i]);
      }
      const single = recordsFrom(Array.from({length: 131}, (_, i) => [[2, -Math.PI + 2 * Math.PI * i / 130]]));
      const field = await runner.evaluate(single); for (const value of field.coherent) assert(Math.abs(value - 4) < 2e-6);
      const empty = await runner.evaluate(zeroInput(131, 0)); assert(empty.coherent.every(v => v === 0) && empty.incoherent.every(v => v === 0));
      report.nativeCases.push({name: '65-route power scaling, phase invariance, zero routes', errors: compare(original, oracle(input))});
    } finally { runner.destroy(); }
  });
  await test('actual dense reduced-phase sweep resolves quadrant boundaries and f32 endpoint rounding', async () => {
    const count = 16385, runner = await make({maxSamples: count});
    try {
      const input = recordsFrom(Array.from({length: count}, (_, i) => [[1, -Math.PI + 2 * Math.PI * i / (count - 1)]]));
      const got = await runner.evaluate(input), errors = compare(got, oracle(input), 2e-7);
      assert.equal(input.records[1], -COHERENT_MESH_GPU_LIMITS.phaseBound);
      assert.equal(input.records[input.records.length - 3], COHERENT_MESH_GPU_LIMITS.phaseBound);
      report.nativeCases.push({name: 'dense reduced phase including +/- pi and quadrant boundaries', errors});
      for (const amplitude of [1e-14, 1e14]) {
        const ranged = recordsFrom([[[amplitude, 0], [amplitude, 0]], [[amplitude, 0], [amplitude, Math.PI]]]);
        const fields = await runner.evaluate(ranged); compare(fields, oracle(ranged));
      }
    } finally { runner.destroy(); }
  });
  await test('actual chunk boundary and repeated sizes preserve sample-major route orientation', async () => {
    const runner = await make({maxSamples: 8209});
    try {
      for (const count of [8209, 1, 67, 8192]) {
        const input = randomInput(count, 65, count + 9121), got = await runner.evaluate(input);
        report.nativeCases.push({name: 'sample-major 65-route stream', sampleCount: count, errors: compare(got, oracle(input)), timing: got.timing});
        assert.equal(got.timing.chunkCount, Math.ceil(count / 8192));
      }
    } finally { runner.destroy(); }
  });
  if (!process.argv.includes('--skip-large')) await test('actual native 262144-sample/65-route field, bounded GPU storage, all samples checked', async () => {
    const input = randomInput(262144, 65), runner = await make();
    try {
      const got = await runner.evaluate(input), errors = compare(got, oracle(input));
      assert.equal(got.coherent.length, 262144); assert.equal(got.timing.chunkCount, 32);
      report.nativeCases.push({name: 'native 262144 samples x 65 routes', errors, timing: got.timing, capacity: runner.capacity});
    } finally { runner.destroy(); }
  });
  await test('actual finite/domain gate rejects negative and overflowing mutated outputs', async () => {
    for (const [name, replacement] of [['negative', 'vec4f(-1.0, incoherent, field.x, field.y)'], ['overflow', 'vec4f(1e35, incoherent, field.x, field.y)'], ['nan', 'vec4f(bitcast<f32>(0x7fc00000u + p.reserved0), incoherent, field.x, field.y)']]) {
      const mutated = shader.replace('vec4f(coherentIntensity, incoherent, field.x, field.y)', replacement); assert.notEqual(mutated, shader);
      const runner = await make({maxSamples: 2, shaderSource: mutated});
      try {
        await assert.rejects(runner.evaluate(recordsFrom([[[1, 0]], [[1, 0]]])), error => {
          assert.equal(error.name, 'CoherentMeshGPUValidationError'); report.negativeControls.push({name, validation: error.validation}); return true;
        });
      } finally { runner.destroy(); }
    }
  });
  await test('finite wrong shader passes finite gate but independent offline oracle detects it', async () => {
    const mutated = shader.replace('reduced_cis(phase)', 'reduced_cis(-phase)'); assert.notEqual(mutated, shader);
    const runner = await make({maxSamples: 67, shaderSource: mutated});
    try { const input = randomInput(67, 3), got = await runner.evaluate(input); assert.throws(() => compare(got, oracle(input))); report.negativeControls.push({name: 'finite incorrect phase sign caught only by independent oracle'}); }
    finally { runner.destroy(); }
  });
  await test('native active cancellation permanently withholds partial multi-chunk results', async () => {
    const runner = await make({maxSamples: 8209}), input = randomInput(8209, 65), c = new AbortController();
    try {
      const pending = runner.evaluate({...input, signal: c.signal}); setTimeout(() => c.abort(), 0);
      await assert.rejects(pending, error => error.name === 'AbortError'); await assert.rejects(runner.evaluate(input));
    } finally { runner.destroy(); }
  });
}
report.passed = report.failures.length === 0; report.completedAt = new Date().toISOString();
if (process.env.COHERENT_MESH_GPU_REPORT) fs.writeFileSync(process.env.COHERENT_MESH_GPU_REPORT, JSON.stringify(report, null, 2) + '\n');
console.log(`${report.tests.length} passed, ${report.failures.length} failed`); if (report.failures.length) process.exitCode = 1;
