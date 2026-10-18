/** Native Dawn/Tint spectral complete-route tests. Software execution is not
 * browser or physical-GPU validation. Run with explicit native backend:
 * VK_ICD_FILENAMES=/usr/lib/chromium/vk_swiftshader_icd.json WEBGPU_EXECUTION_BACKEND=vulkan node tests/coherent-mesh-spectral-gpu.mjs --execute
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {create, globals} from 'webgpu';
import {createCoherentMeshSpectralGPURunner, coherentMeshSpectralGPUCapacity, packCoherentSpectralCotDeltas, COHERENT_MESH_SPECTRAL_GPU_LIMITS} from '../dist/coherent-mesh-spectral-gpu.js';
import {createCoherentMeshGPURunner} from '../dist/coherent-mesh-gpu.js';
import {erfcxScaled} from '../dist/utd-math.js';
Object.assign(globalThis, globals);
const include = fs.readFileSync(new URL('../dist/utd-functions.wgsl', import.meta.url), 'utf8');
const body = fs.readFileSync(new URL('../dist/coherent-mesh-spectral-compute.wgsl', import.meta.url), 'utf8');
const shader = include + '\n' + body;
const execute = process.argv.includes('--execute');
const report = {suite: 'coherent-mesh-spectral-gpu', schemaVersion: 1, executionRequested: execute, browserValidated: false,
  physicalGPUValidated: false, cpuShadow: false, tests: [], failures: [], nativeCases: [], negativeControls: [], hashes: {}};
for (const file of ['dist/coherent-mesh-spectral-gpu.js', 'dist/coherent-mesh-spectral-compute.wgsl', 'dist/utd-functions.wgsl', 'tests/coherent-mesh-spectral-gpu.mjs']) report.hashes[file] = createHash('sha256').update(fs.readFileSync(new URL(`../${file}`, import.meta.url))).digest('hex');
async function test(name, run) {
  try { await run(); report.tests.push(name); console.log(`PASS ${name}`); }
  catch (error) { report.failures.push({name, error: error.stack, validation: error.validation}); console.error(`FAIL ${name}: ${error.stack}`); }
}
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return {promise, resolve}; };
const tick = () => new Promise(resolve => setTimeout(resolve, 0));
function mock({queueWait, mapWait, scopeWait, scopeAt = 1, onScope, limits, compileError, validationError, mapError, invalidOutput, invalidSafety} = {}) {
  const lost = deferred(), buffers = [], events = {}; let destroyed = 0, submits = 0, unmaps = 0, scope = 0;
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
      return {beginComputePass: () => ({setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {}}),
        copyBufferToBuffer(source, so, destination, dest, size) { copies.push(() => new Uint8Array(destination.bytes, dest, size).set(new Uint8Array(source.bytes, so, size))); }, finish: () => ({copies})};
    },
    queue: {
      writeBuffer(buffer, offset, input) { new Uint8Array(buffer.bytes, offset, input.byteLength).set(new Uint8Array(input.buffer, input.byteOffset, input.byteLength)); },
      submit(commands) {
        submits++; const [uniform, , , , output, , flags] = buffers; const p = new Uint32Array(uniform.bytes);
        new Uint32Array(flags.bytes).set([0, 0, invalidSafety ? 0 : p[0] * p[2], p[0]]);
        if (invalidOutput !== undefined) new Float32Array(output.bytes)[0] = invalidOutput;
        for (const command of commands) for (const copy of command.copies) copy();
      },
      async onSubmittedWorkDone() { if (queueWait) await queueWait; },
    },
  };
  return {device, buffers, lost, events, gpu: {requestAdapter: async () => ({info: {description: 'mock lifecycle only'}, limits: limits ?? device.limits, requestDevice: async () => device})},
    get destroyed() { return destroyed; }, get submits() { return submits; }, get unmaps() { return unmaps; }};
}
const split = value => [Math.fround(value), Math.fround(value - Math.fround(value))];
const node = (nm = 550, weight = 1, cie = [.4, .7, .2]) => [...split(1e6 / nm), 2 * Math.PI * 1e6 / nm, weight, ...cie, 0];
const go = (amplitude = 1, phase = 0, excess = 0) => [amplitude, phase, ...split(excess), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const diffracted = (amplitude = 30, phase = .3, excess = 26.56, n = 1.5, phiI = .4, phiO = 2.8, L = 12, sinBeta = .8, boundary = -1) => [amplitude, phase, ...split(excess), 2, n, phiI, phiO, L, sinBeta, boundary, 0, ...packCoherentSpectralCotDeltas(n, phiI, phiO)];
function inputFrom(samples, nodes = [node()]) {
  const routeCount = samples[0].length, records = new Float32Array(samples.length * routeCount * 16);
  samples.forEach((sample, s) => sample.forEach((route, r) => records.set(route, (s * routeCount + r) * 16)));
  return {records, spectralRecords: new Float32Array(nodes.flat()), sampleCount: samples.length, routeCount, wavelengthCount: nodes.length};
}
const zeroInput = (sampleCount = 2, routeCount = 1) => ({records: new Float32Array(sampleCount * routeCount * 16), spectralRecords: new Float32Array(node()), sampleCount, routeCount, wavelengthCount: 1});
const makeMock = async (options = {}, runnerOptions = {}) => { const x = mock(options), runner = await createCoherentMeshSpectralGPURunner({gpu: x.gpu, shaderSource: shader, maxSamples: 4, ...runnerOptions}); return {x, runner}; };
const cmul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
// Independent f64 scalar UTD arithmetic using packed geometric deltas. This
// test oracle never runs in production. The erfcx f64 reference is independent
// of native scalar WGSL recurrence/lowering and validated separately.
function coefficientFromRecord(records, at, k) {
  const n = records[at + 5], L = records[at + 8], sinBeta = records[at + 9], R = records[at + 10];
  if (n === 1) return [0, 0];
  const terms = [];
  for (let j = 0; j < 4; j++) {
    const d = records[at + 12 + j];
    if (d === 0) { terms.push([0, 0]); continue; }
    const t = d / 2, h = d / (2 * n), sinc = x => Math.abs(x) < 1e-4 ? 1 - x * x / 6 + x ** 4 / 120 : Math.sin(x) / x;
    const u = Math.sqrt(2 * k * L) * Math.abs(Math.sin(t));
    const scale = (j % 2 === 0 ? 1 : -1) * Math.sqrt(2 * Math.PI * k * L) / (1 + u) * Math.sign(d) * n * sinc(t) * Math.cos(h) / sinc(h);
    terms.push(cmul([Math.SQRT1_2, Math.SQRT1_2], erfcxScaled(u / (1 + u))).map(x => x * scale));
  }
  const sum = [0, 1].map(c => terms[0][c] + terms[1][c] + R * (terms[2][c] + terms[3][c]));
  return cmul([Math.SQRT1_2, -Math.SQRT1_2], sum).map(x => -x / (2 * n * Math.sqrt(2 * Math.PI * k) * sinBeta));
}
function oracle(input) {
  const {records, spectralRecords: nodes, sampleCount, routeCount, wavelengthCount} = input;
  const shared = nodes.length === wavelengthCount * 8;
  const result = Object.fromEntries(['coherent', 'incoherent', 'geometric', 'coherentXYZ', 'incoherentXYZ', 'geometricXYZ'].map(name => [name, new Float64Array(sampleCount * (name.endsWith('XYZ') ? 3 : 1))]));
  for (let s = 0; s < sampleCount; s++) for (let w = 0; w < wavelengthCount; w++) {
    const wn = ((shared ? 0 : s) * wavelengthCount + w) * 8, inverse = nodes[wn] + nodes[wn + 1];
    let re = 0, im = 0, inc = 0, geo = 0;
    for (let r = 0; r < routeCount; r++) {
      const at = (s * routeCount + r) * 16, a = records[at]; if (!records[at + 4]) continue;
      const cycles = (records[at + 2] + records[at + 3]) * inverse;
      const p = records[at + 1] - 2 * Math.PI * (cycles - Math.round(cycles));
      let field = [a * Math.cos(p), a * Math.sin(p)];
      if (records[at + 4] === 2) field = cmul(field, coefficientFromRecord(records, at, nodes[wn + 2]));
      const power = field[0] ** 2 + field[1] ** 2; re += field[0]; im += field[1]; inc += power;
      if (records[at + 4] === 1) geo += power;
    }
    const values = {coherent: re * re + im * im, incoherent: inc, geometric: geo};
    for (const name of Object.keys(values)) {
      const weighted = values[name] * nodes[wn + 3]; result[name][s] += weighted;
      for (let c = 0; c < 3; c++) result[name + 'XYZ'][s * 3 + c] += weighted * nodes[wn + 4 + c];
    }
  }
  return result;
}
function compare(actual, expected, tolerance = 8e-5) {
  const errors = {};
  for (const name of Object.keys(expected)) {
    assert.equal(actual[name].length, expected[name].length);
    let maxAbsoluteError = 0, peak = 0;
    for (let i = 0; i < expected[name].length; i++) { assert(Number.isFinite(actual[name][i])); maxAbsoluteError = Math.max(maxAbsoluteError, Math.abs(actual[name][i] - expected[name][i])); peak = Math.max(peak, Math.abs(expected[name][i])); }
    errors[name] = {maxAbsoluteError, peak, relativeToPeak: maxAbsoluteError / Math.max(peak, 1e-20)};
    assert(maxAbsoluteError <= tolerance * Math.max(peak, 1e-20), `${name}: ${JSON.stringify(errors[name])}`);
  }
  return errors;
}
await test('bounded capacity: 262144 samples, 65 routes, 128 wavelengths, residency below21 MiB', () => {
  const c = coherentMeshSpectralGPUCapacity(); report.capacity = c;
  assert.equal(c.maxSamples, 262144); assert.equal(c.maxRoutes, 65); assert.equal(c.maxWavelengths, 128); assert(c.residentBytes < 21 * 1024 ** 2);
  assert.equal(coherentMeshSpectralGPUCapacity({maxSamples: 1}).chunkSamples, 1);
  for (const maxSamples of [0, -1, 1.1, 262145, Infinity]) assert.throws(() => coherentMeshSpectralGPUCapacity({maxSamples}));
});
await test('adapter and initialization capacity/timeout guards fail without CPU replacement', async () => {
  await assert.rejects(createCoherentMeshSpectralGPURunner({gpu: {requestAdapter: async () => null}}), /adapter/);
  for (const timeoutMs of [0, -1, Infinity, 2147483648]) await assert.rejects(createCoherentMeshSpectralGPURunner({timeoutMs}), /timeoutMs/);
  for (const limits of [{maxBufferSize: 10}, {maxStorageBufferBindingSize: 10}, {maxStorageBuffersPerShaderStage: 3}, {maxComputeWorkgroupsPerDimension: 127}]) {
    const x = mock({limits}); await assert.rejects(createCoherentMeshSpectralGPURunner({gpu: x.gpu, shaderSource: shader}), /capacity/); assert.equal(x.buffers.length, 0);
  }
});
await test('strict record/node dimensions, split-phase, UTD, reserved and intensity guards reject invalid data', async () => {
  const inputs = [ {...zeroInput(), records: []}, {...zeroInput(), records: new Float32Array(3)}, {...zeroInput(), sampleCount: 0}, {...zeroInput(), sampleCount: 5}, {...zeroInput(), routeCount: 66}, {...zeroInput(), wavelengthCount: 0}, {...zeroInput(), wavelengthCount: 129}, {...zeroInput(), spectralRecords: new Float32Array(7)} ];
  for (const [at, value] of [[0, NaN], [0, -1], [0, 1e16], [1, 4], [2, 100001], [3, 1], [4, 4], [5, 1]]) { const input = inputFrom([[go()]]); input.records[at] = value; inputs.push(input); }
  for (const [at, value] of [[5, .9], [5, 2.1], [6, -1], [7, 7], [8, 0], [8, 1e6], [9, 0], [10, 0], [11, 1], [12, 10], [15, NaN]]) { const input = inputFrom([[diffracted()]]); input.records[at] = value; inputs.push(input); }
  for (const [at, value] of [[0, NaN], [0, 1], [1, 1], [2, 1], [3, -1], [3, 1e7], [4, -1], [5, 5], [6, Infinity], [7, -1], [7, 1000001]]) { const input = zeroInput(); input.spectralRecords[at] = value; inputs.push(input); }
  for(const [at,value] of [[5,-1],[5,100001],[6,1]]){const input=inputFrom([[go()]]);input.records[4]=3;input.records[at]=value;inputs.push(input);}
  const badEmpty = zeroInput(); badEmpty.records[0] = 1; inputs.push(badEmpty);
  if (typeof SharedArrayBuffer !== 'undefined') { inputs.push({...zeroInput(), records: new Float32Array(new SharedArrayBuffer(128))}); inputs.push({...zeroInput(), spectralRecords: new Float32Array(new SharedArrayBuffer(32))}); }
  for (const input of inputs) { const {x, runner} = await makeMock(); await assert.rejects(runner.evaluate(input), RangeError); assert.equal(x.submits, 0); assert.equal(x.destroyed, 1); assert(x.buffers.every(b => b.dead)); }
});
await test('empty-route, repeated-size evaluations reuse buffers and inspect both passes', async () => {
  const {x, runner} = await makeMock();
  try { for (const n of [4, 1, 3]) { const got = await runner.evaluate(zeroInput(n, 0)); assert(got.coherent.every(x => x === 0)); assert.equal(got.coherentXYZ.length, n * 3); assert.equal(got.timing.cpuShadow, false); } assert.equal(x.buffers.length, 8); assert.equal(x.submits, 3); assert.equal(x.unmaps, 6); }
  finally { runner.destroy(); runner.destroy(); } assert.equal(x.destroyed, 1);
});
await test('initial abort and late device adoption timeout/abort do not leak device ownership', async () => {
  const c = new AbortController(); c.abort(); const x = mock(); await assert.rejects(createCoherentMeshSpectralGPURunner({gpu: x.gpu, shaderSource: shader, signal: c.signal}), e => e.name === 'AbortError');
  for (const mode of ['abort', 'timeout']) {
    const pending = deferred(), c = new AbortController(); let destroyed = 0;
    const p = createCoherentMeshSpectralGPURunner({gpu: {requestAdapter: async () => ({requestDevice: () => pending.promise})}, shaderSource: shader, signal: c.signal, timeoutMs: mode === 'timeout' ? 10 : 10000});
    const rejected = assert.rejects(p); await tick(); if (mode === 'abort') c.abort(); await rejected; pending.resolve({destroy() { destroyed++; }}); await tick(); assert.equal(destroyed, 1);
  }
});
await test('queue/map waits, concurrency, active abort, destruction, loss, uncaptured errors and timeout are bounded', async () => {
  for (const stage of ['queue', 'map']) for (const mode of ['abort', 'destroy', 'loss', 'uncaptured', 'timeout']) {
    const hold = deferred(), c = new AbortController(), {x, runner} = await makeMock({[stage === 'queue' ? 'queueWait' : 'mapWait']: hold.promise}, {timeoutMs: mode === 'timeout' ? 15 : 10000});
    const rejected = assert.rejects(runner.evaluate({...zeroInput(), signal: c.signal})); await tick(); await assert.rejects(runner.evaluate(zeroInput()), /sequential/);
    if (mode === 'abort') c.abort(); if (mode === 'destroy') runner.destroy(); if (mode === 'loss') x.lost.resolve({message: 'test loss'}); if (mode === 'uncaptured') x.events.uncapturederror({error: {message: 'test uncaptured'}});
    await rejected; hold.resolve(); assert.equal(x.destroyed, 1); assert(x.buffers.every(b => b.dead)); await assert.rejects(runner.evaluate(zeroInput()));
  }
});
await test('uncooperative compilation/command error scopes terminate and release owned resources', async () => {
  for (const stage of ['shader', 'command']) for (const mode of ['abort', 'loss', 'timeout']) {
    const hold = deferred(), entered = deferred(), c = new AbortController(), x = mock({scopeWait: hold.promise, scopeAt: stage === 'shader' ? 1 : 4, onScope: entered.resolve});
    const creating = createCoherentMeshSpectralGPURunner({gpu: x.gpu, maxSamples: 4, shaderSource: shader, signal: c.signal, timeoutMs: mode === 'timeout' ? 20 : 10000});
    const rejected = assert.rejects(stage === 'shader' ? creating : (await creating).evaluate(zeroInput())); await entered.promise;
    if (mode === 'abort') c.abort(); if (mode === 'loss') x.lost.resolve({message: 'test loss'}); await rejected; hold.resolve(); assert.equal(x.destroyed, 1);
  }
});
await test('shader, allocation, mapping, finite/readback and inspection-count faults withhold all output', async () => {
  for (const opts of [{compileError: 'bad WGSL'}, {validationError: 'bad module'}, {validationError: 'allocation failed', scopeAt: 3}]) { const x = mock(opts); await assert.rejects(createCoherentMeshSpectralGPURunner({gpu: x.gpu, shaderSource: shader})); assert.equal(x.destroyed, 1); }
  for (const opts of [{mapError: 'map failure'}, {invalidOutput: NaN}, {invalidOutput: -1}, {invalidOutput: Infinity}, {invalidSafety: true}]) { const {x, runner} = await makeMock(opts); await assert.rejects(runner.evaluate(zeroInput())); assert.equal(x.destroyed, 1); assert(x.buffers.every(b => b.dead)); }
});
await test('native Dawn/Tint compiles both actual spectral WGSL entrypoints and rejects malformed code', async () => {
  const gpu = create(['backend=null']); const runner = await createCoherentMeshSpectralGPURunner({gpu, shaderSource: shader, maxSamples: 4}); runner.destroy();
  await assert.rejects(createCoherentMeshSpectralGPURunner({gpu, shaderSource: shader + '\nINVALID WGSL', maxSamples: 4}), /WGSL/);
});
if (execute) {
  const backend = process.env.WEBGPU_EXECUTION_BACKEND; assert(['vulkan', 'metal', 'd3d11', 'd3d12', 'opengl', 'opengles'].includes(backend), 'Explicit executable backend required; null refused');
  const gpu = create([`backend=${backend}`]); report.requestedBackend = backend; report.driverSelection = {VK_ICD_FILENAMES: process.env.VK_ICD_FILENAMES ?? null};
  const make = options => createCoherentMeshSpectralGPURunner({gpu, shaderSource: shader, ...options});
  await test('native monochromatic GO+D/RD/DR template evaluation matches f64 and established mono GPU sum', async () => {
    const input = inputFrom([[go(), diffracted()], [go(.6, -.2, 1.34), diffracted(20, -.5, 26.57)], [go(0), diffracted(40, .8, 2.26, 2, .5, 2.1, 3, 1, 1)]]);
    const runner = await make({maxSamples: 3}); let mono;
    try {
      const got = await runner.evaluate(input); report.adapter = got.backend;
      report.nativeCases.push({name: 'K=1 UTD plus GO', errors: compare(got, oracle(input)), timing: got.timing});
      const packed = new Float32Array(3 * 2 * 4);
      for (let s = 0; s < 3; s++) for (let r = 0; r < 2; r++) {
        const at = (s * 2 + r) * 16, out = (s * 2 + r) * 4, d = input.records[at + 4] === 2 ? coefficientFromRecord(input.records, at, input.spectralRecords[2]) : [1, 0];
        const cycles = (input.records[at + 2] + input.records[at + 3]) * (input.spectralRecords[0] + input.spectralRecords[1]);
        let phase = input.records[at + 1] + Math.atan2(d[1], d[0]) - 2 * Math.PI * (cycles - Math.round(cycles)); phase = ((phase + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
        packed[out] = input.records[at] * Math.hypot(...d); packed[out + 1] = phase;
      }
      mono = await createCoherentMeshGPURunner({gpu, shaderSource: fs.readFileSync(new URL('../dist/coherent-mesh-compute.wgsl', import.meta.url), 'utf8'), maxSamples: 3});
      const monoResult = await mono.evaluate({records: packed, sampleCount: 3, routeCount: 2});
      for (let i = 0; i < 3; i++) assert(Math.abs(got.coherent[i] - monoResult.coherent[i]) < 8e-5 * Math.max(monoResult.coherent[i], 1));
    } finally { runner.destroy(); mono?.destroy(); }
  });
  await test('native split optical phase spans +/-100000mm without large-argument f32 phase loss', async () => {
    const samples = Array.from({length: 1025}, (_, i) => [go(), go(.7, .3, -1e5 + 2e5 * i / 1024)]);
    const input = inputFrom(samples, [node(360), node(547.78), node(830)]), runner = await make({maxSamples: samples.length});
    try { const got = await runner.evaluate(input); report.nativeCases.push({name: 'bounded large optical excess', errors: compare(got, oracle(input), 1.5e-6), timing: got.timing}); }
    finally { runner.destroy(); }
  });
  await test('native exact and near-pole stable cot deltas preserve zero/+/- seams smaller than f32 angle resolution', async () => {
    const shifts = [-1e-5, -1e-7, -1e-10, 0, 1e-10, 1e-7, 1e-5];
    const samples = shifts.map(d => [go(d < 0 ? 1 : d === 0 ? .5 : 0), diffracted(.2, 0, .003, 2, .7, .7 + Math.PI + d, 5, 1, -1)]);
    const input = inputFrom(samples, [node(360, .25), node(550, .5), node(830, .25)]), runner = await make({maxSamples: samples.length});
    assert.equal(input.records[3 * 32 + 16 + 13], 0); assert.notEqual(input.records[2 * 32 + 16 + 13], 0); assert.notEqual(input.records[4 * 32 + 16 + 13], 0);
    try { report.nativeCases.push({name: 'exact and near cot poles', errors: compare(await runner.evaluate(input), oracle(input))}); }
    finally { runner.destroy(); }
  });
  await test('native two distinct spectral lines sum intensities and XYZ, never cross-wavelength fields', async () => {
    const samples = [[go(), go(.8, .37, .00117)], [go(.5, -.9), diffracted(20, .2, 2.26)]];
    const a = node(450, .3, [.34, .038, 1.77]), b = node(650, .7, [.28, .107, 0]);
    const multi = inputFrom(samples, [a, b]), runner = await make({maxSamples: 2});
    try {
      const got = await runner.evaluate(multi), one = await runner.evaluate(inputFrom(samples, [a])), two = await runner.evaluate(inputFrom(samples, [b]));
      compare(got, oracle(multi));
      for (const name of ['coherent', 'incoherent', 'geometric', 'coherentXYZ', 'incoherentXYZ', 'geometricXYZ']) for (let i = 0; i < got[name].length; i++) assert(Math.abs(got[name][i] - one[name][i] - two[name][i]) < 2e-6 * Math.max(1, got[name][i]));
      report.nativeCases.push({name: 'two independent lines; weighted scalar and XYZ', errors: compare(got, oracle(multi))});
    } finally { runner.destroy(); }
  });
  await test('native sample-major varying wavelengths, 128 nodes, all65 routes and bounded work chunks', async () => {
    const sampleCount = 73, wavelengthCount = 128, samples = Array.from({length: sampleCount}, (_, s) => Array.from({length: 65}, (_, r) => r % 4 ? go(.03 + r / 1000, (r % 6 - 3) * .5, r * .007 + s * .000113) : diffracted(.3, .2, 2.26 + s * .011 + r * .017, 1.5 + (r % 3) * .2, .7, 2.8, 4 + r, .8, r % 2 ? -1 : 1)));
    const input = inputFrom(samples); input.wavelengthCount = wavelengthCount; input.spectralRecords = new Float32Array(sampleCount * wavelengthCount * 8);
    for (let s = 0; s < sampleCount; s++) for (let w = 0; w < wavelengthCount; w++) input.spectralRecords.set(node(360 + 470 * ((w + .5 + s * .337) % wavelengthCount) / wavelengthCount, 1 / wavelengthCount, [.2 + w / 200, .8 - w / 200, .1 + s / 100]), (s * wavelengthCount + w) * 8);
    const runner = await make({maxSamples: sampleCount});
    try { const got = await runner.evaluate(input); assert.equal(got.timing.chunkCount, 3); assert.equal(got.timing.chunkSamples, 31); report.nativeCases.push({name: '73 samples x65 routes x128 varying nodes', errors: compare(got, oracle(input)), timing: got.timing}); }
    finally { runner.destroy(); }
  });
  await test('native empty routes and repeated chunk dimensions do not retain prior field or color values', async () => {
    const runner = await make({maxSamples: 2061});
    try { for (const count of [2061, 1, 67, 2048]) { const input = inputFrom(Array.from({length: count}, (_, i) => [go(.5, .1, i * .00017), go(.6, .9, i * .00021)])); const got = await runner.evaluate(input); compare(got, oracle(input), 2e-6); assert.equal(got.timing.chunkCount, Math.ceil(count / 2048)); const zero = await runner.evaluate(zeroInput(count, 0)); assert(zero.coherent.every(x => x === 0)); assert(zero.coherentXYZ.every(x => x === 0)); } }
    finally { runner.destroy(); }
  });
  if (!process.argv.includes('--skip-large')) await test('native full262144-sample spectral output checks every sample with bounded GPU residency', async () => {
    const sampleCount = 262144, records = new Float32Array(sampleCount * 32);
    for (let s = 0; s < sampleCount; s++) { records.set(go(.2 + (s % 13) / 100, .31, s * .00011), s * 32); records.set(go(.7, -.42, s * .00017), s * 32 + 16); }
    const input = {records, spectralRecords: new Float32Array(node()), sampleCount, routeCount: 2, wavelengthCount: 1}, runner = await make();
    try { const got = await runner.evaluate(input); assert.equal(got.coherent.length, sampleCount); assert.equal(got.timing.chunkCount, 128); report.nativeCases.push({name: 'native262144 spectral samples', errors: compare(got, oracle(input), 3e-6), timing: got.timing, capacity: runner.capacity}); }
    finally { runner.destroy(); }
  });
  await test('native finite/domain guards catch mutated negative, overflow and nonfinite spectral intensities', async () => {
    for (const [name, replacement] of [['negative', 'vec4f(-1.0, incoherent, geometric, 0.0)'], ['overflow', 'vec4f(1e35, incoherent, geometric, 0.0)'], ['nan', 'vec4f(bitcast<f32>(0x7fc00000u + p.sharedSpectrum), incoherent, geometric, 0.0)']]) {
      const mutated = shader.replace('vec4f(coherentIntensity, incoherent, geometric, 0.0)', replacement); assert.notEqual(mutated, shader); const runner = await make({maxSamples: 2, shaderSource: mutated});
      try { await assert.rejects(runner.evaluate(inputFrom([[go()], [go()]])), error => { assert.equal(error.name, 'CoherentMeshSpectralGPUValidationError'); report.negativeControls.push({name, validation: error.validation}); return true; }); }
      finally { runner.destroy(); }
    }
  });
  await test('finite wrong phase, UTD coefficient and spectral integration pass finite gate but fail independent oracle', async () => {
    const input = inputFrom([[go(), go(.7, .8, .00133), diffracted(20, .4, 26.56)], [go(.6, -.3), go(.9, -.7, .00619), diffracted(30, -.2, 2.26)]], [node(450, .4, [.34, .038, 1.77]), node(650, .6, [.28, .107, 0])]);
    for (const [name, from, to] of [
      ['phase-sign', 'record.field.y - 6.283185307179586', 'record.field.y + 6.283185307179586'],
      ['wrong-finite-UTD', 'routeField, spectral_utd(node.wave.z, record)', 'routeField, 1.7 * spectral_utd(node.wave.z, record)'],
      ['cross-wavelength-field-square', 'coherentValue += intensity.x * weight;', 'coherentValue = pow(sqrt(coherentValue) + sqrt(intensity.x * weight), vec4f(2.0));'],
      ['spectral-weight', 'node.wave.w * vec4f(node.cie.xyz, 1.0)', '1.2 * node.wave.w * vec4f(node.cie.xyz, 1.0)'],
    ]) {
      const mutated = shader.replace(from, to); assert.notEqual(mutated, shader); const runner = await make({maxSamples: 2, shaderSource: mutated});
      try { const got = await runner.evaluate(input); assert.throws(() => compare(got, oracle(input))); report.negativeControls.push({name, finiteGatePassed: true, independentOracleRejected: true}); }
      finally { runner.destroy(); }
    }
  });
  await test('native in-flight cancellation withholds every partial multi-chunk result and kills runner', async () => {
    const input = inputFrom(Array.from({length: 2061}, (_, i) => [go(), diffracted(30, .4, i * .0011)]), Array.from({length: 128}, (_, w) => node(360 + 470 * (w + .5) / 128, 1 / 128)));
    const runner = await make({maxSamples: input.sampleCount}), c = new AbortController();
    try { const pending = runner.evaluate({...input, signal: c.signal}); setTimeout(() => c.abort(), 0); await assert.rejects(pending, e => e.name === 'AbortError'); await assert.rejects(runner.evaluate(input)); }
    finally { runner.destroy(); }
  });
}
report.passed = report.failures.length === 0; report.completedAt = new Date().toISOString();
if (process.env.COHERENT_MESH_SPECTRAL_GPU_REPORT) fs.writeFileSync(process.env.COHERENT_MESH_SPECTRAL_GPU_REPORT, JSON.stringify(report, null, 2) + '\n');
console.log(`${report.tests.length} passed, ${report.failures.length} failed`); if (report.failures.length) process.exitCode = 1;
