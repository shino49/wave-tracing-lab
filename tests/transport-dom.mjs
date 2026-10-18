/** DOM/Canvas-spy lifecycle tests and actual CPU/Worker module fixtures.
 * No real browser layout, browser WebGPU or physical GPU validation is claimed. */
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {Worker as NodeWorker} from 'node:worker_threads';
import {prepareCoherentMeshTask, executeCoherentMeshTask} from '../dist/coherent-mesh-runtime.js';
import {installTransportWorker} from '../dist/transport-worker.js';
import {cieXYZ, displaySpectralXYZ, linearToSRGB} from '../dist/plt-spectrum.js';
const {Window} = await import(process.env.WAVE_TEST_DOM || 'happy-dom');
assert(!/物理(?:成像)?相机/.test(fs.readFileSync(new URL('../dist/transport.html', import.meta.url), 'utf8')));
const win = new Window({url: 'https://test.local/transport.html', settings: {disableJavaScriptEvaluation: true, disableCSSFileLoading: true, disableJavaScriptFileLoading: true}});
win.document.write(fs.readFileSync(new URL('../dist/transport.html', import.meta.url), 'utf8'));
const contexts = new WeakMap(), pixels = new WeakMap(), calls = [], draws = [], workers = [], blobs = new Map(), downloads = [], revoked = [], passed = [], failures = [];
win.HTMLCanvasElement.prototype.getContext = function () {
  if (!contexts.has(this)) {
    const canvas = this;
    contexts.set(this, new Proxy({
      createImageData: (width, height) => ({width, height, data: new Uint8ClampedArray(width * height * 4)}),
      createRadialGradient: () => ({addColorStop() {}}), createLinearGradient: () => ({addColorStop() {}}), measureText: text => ({width: String(text).length * 7}),
      clearRect: () => pixels.delete(canvas),
      putImageData: (image, x = 0, y = 0) => {
        let stored = pixels.get(canvas);
        if (!stored || stored.width !== canvas.width || stored.height !== canvas.height) stored = {width: canvas.width, height: canvas.height, data: new Uint8ClampedArray(canvas.width * canvas.height * 4)};
        for (let row = 0; row < image.height; row++) stored.data.set(image.data.subarray(row * image.width * 4, (row + 1) * image.width * 4), ((row + y) * canvas.width + x) * 4);
        pixels.set(canvas, stored); draws.push({canvas, image, x, y});
      },
    }, {get: (target, key) => key in target ? target[key] : (...args) => calls.push({canvas, method: key, args}), set: (target, key, value) => (target[key] = value, true)}));
  }
  return contexts.get(this);
};
win.HTMLCanvasElement.prototype.getBoundingClientRect = () => ({width: 640, height: 400, top: 0, left: 0, right: 640, bottom: 400});
Object.assign(globalThis, {window: win, document: win.document, devicePixelRatio: 1, ResizeObserver: class { observe() {} disconnect() {} }});
Object.defineProperty(globalThis, 'navigator', {value: {}, configurable: true});
let networkCalls = 0, mode = 'manual', actualExecutions = 0, blobIndex = 0;
globalThis.fetch = async () => { networkCalls++; throw new Error('Unexpected network request in local CPU/OBJ fixture'); };
URL.createObjectURL = blob => { const url = `blob:transport-${++blobIndex}`; blobs.set(url, blob); return url; };
URL.revokeObjectURL = url => { revoked.push(url); blobs.delete(url); };
win.HTMLAnchorElement.prototype.click = function () { downloads.push({name: this.download, blob: blobs.get(this.href), url: this.href}); };
class WorkerDouble {
  constructor(url, options) { if (mode === 'factory-error') throw new Error('Worker construction failed'); this.url = String(url); this.options = options; this.mode = mode; this.messages = []; this.terminated = false; workers.push(this); }
  terminate() { this.terminated = true; }
  postMessage(request) {
    this.messages.push(request); if (request.type === 'cancel') return;
    this.request = structuredClone(request); if (this.mode === 'post-error') throw new Error('Worker posting failed');
    if (this.mode !== 'execute') return;
    queueMicrotask(async () => {
      try {
        actualExecutions++;
        const result = await executeCoherentMeshTask(request, {progress: value => { if (!this.terminated) this.emit({type: 'progress', ...value}); }, preview: value => { if (!this.terminated) this.emit({type: 'preview', ...value}); }, reset: value => { if (!this.terminated) this.emit({type: 'reset', ...value}); }});
        this.result = result; if (!this.terminated) this.emit({type: 'result', result});
      } catch (error) { if (!this.terminated) this.emit({type: 'error', error: error.message, name: error.name}); }
    });
  }
  emit(message) { this.onmessage?.({data: {id: this.request.id, ...message}}); }
  error() { this.onerror?.({message: 'Worker/device lost', preventDefault() {}}); }
  messageError() { this.onmessageerror?.({}); }
}
globalThis.Worker = WorkerDouble;
const {boundedTransportReport, transportIntensityRGB, transportSignedRGB} = await import('../dist/transport-app.js');
const $ = id => document.getElementById(id), last = () => workers.at(-1), report = () => JSON.parse($('trReport').textContent);
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const deferred = () => { let resolve; const promise = new Promise(r => resolve = r); return {promise, resolve}; };
const until = async fn => { for (let n = 0; n < 12000; n++) { if (fn()) return; await delay(5); } throw new Error('DOM fixture timed out'); };
const change = (id, value) => { $(id).value = String(value); $(id).dispatchEvent(new win.Event('input', {bubbles: true})); };
const checkbox = (id, value) => { $(id).checked = value; $(id).dispatchEvent(new win.Event('input', {bubbles: true})); };
const fileChange = file => { Object.defineProperty($('trOBJ'), 'files', {value: [file], configurable: true}); $('trOBJ').dispatchEvent(new win.Event('change', {bubbles: true})); };
const fresh = () => { mode = 'manual'; $('trReset').click(); };
const start = () => { $('trRun').click(); return last(); };
function idle() { assert.equal($('trRun').disabled, false); assert.equal($('trRunQuick').disabled, false); assert.equal($('trCancel').hidden, true); assert.equal($('trCancelQuick').hidden, true); assert.equal($('trProgress').hidden, true); assert.equal(coherentMeshLab.getState().busy, false); }
function empty() { for (const name of ['Geometric', 'Coherent', 'Baseline', 'Cross']) { const canvas = $(`tr${name}Canvas`); assert.equal(canvas.width, 512); assert.equal(canvas.height, 512); assert.equal(pixels.has(canvas), false, name); assert.equal($(`tr${name}Empty`).hidden, false); assert.equal($(`tr${name}Partial`).hidden, true); } }
function fixture(request, {attempt = 0, diffraction = false} = {}) {
  const prepared = prepareCoherentMeshTask(request), length = 512 * 512;
  const incoherent = new Float64Array(length).fill(2e-5), geometric = new Float64Array(incoherent), coherent = new Float64Array(length), crossTerm = new Float64Array(length);
  for (let i = 0; i < length; i++) { crossTerm[i] = request.coherence * Math.sin(i * .02) * 1e-5; coherent[i] = incoherent[i] + crossTerm[i]; }
  return withXYZ({width: 512, height: 512, coherent, incoherent, geometric, crossTerm, configuration: prepared.configuration, overview: prepared.overview, backend: {kind: 'cpu', description: 'CPU test fixture'}, timings: {totalMs: 123}, diagnostics: {attempt, diffractionEnabled: diffraction, integratedResponse: {coherent: .1, incoherent: .08}}, complete: true}, request);
}
function withXYZ(frame, request = last()?.request) {
  const spectrum = request?.spectrum ?? request?.source?.spectrum ?? request?.scene?.source?.spectrum;
  const mono = !spectrum || spectrum.mode === 'monochrome';
  frame.spectralComplete = true; frame.spectralSamples = mono ? 1 : spectrum.mode === 'lines' ? spectrum.lines.length : request.spectralSamples;
  frame.spectralEstimator = mono ? 'exact monochrome' : 'DOM synthetic XYZ fixture; not a physics reference';
  for (const key of ['coherent', 'incoherent', 'geometric']) {
    frame[key + 'XYZ'] = new Float64Array(frame[key].length * 3);
    for (let i = 0; i < frame[key].length; i++) {
      const basis = mono ? cieXYZ(request?.wavelengthNm ?? 550) : [.25 + .025 * (i % 3), .3, .2 + .025 * (i % 5)];
      for (let c = 0; c < 3; c++) frame[key + 'XYZ'][3 * i + c] = frame[key][i] * basis[c];
    }
  }
  return frame;
}
function tile(rowStart = 0, rowCount = 8, attempt = 0) { return withXYZ({width: 512, height: 512, rowStart, rowCount, attempt, complete: false, coherent: new Float64Array(512 * rowCount).fill(3e-5), incoherent: new Float64Array(512 * rowCount).fill(2e-5), geometric: new Float64Array(512 * rowCount).fill(2e-5), crossTerm: new Float64Array(512 * rowCount).fill(1e-5)}); }
function deliver() { const worker = start(), result = fixture(worker.request); worker.emit({type: 'result', result}); return {worker, result}; }
async function test(name, fn, reset = true) { if (process.env.WAVE_TEST_FILTER && !new RegExp(process.env.WAVE_TEST_FILTER, 'i').test(name)) return; if (reset) fresh(); try { await fn(); passed.push(name); console.log('PASS', name); } catch (error) { failures.push({name, error: error.stack}); console.error('FAIL', name, error.stack); } }

await test('Initial state is explicit-start, labelled, native512 and shows the actual mesh', () => {
  assert.equal(workers.length, 0); idle(); empty();
  for (const element of document.querySelectorAll('input,select')) assert(element.labels.length, element.id);
  for (const canvas of document.querySelectorAll('canvas')) assert(canvas.getAttribute('aria-label'));
  const ids = [...document.querySelectorAll('[id]')].map(e => e.id); assert.equal(ids.length, new Set(ids).size);
  assert($('trStatus').closest('[aria-live]')); assert.equal(document.querySelector('[aria-current]').getAttribute('href'), 'transport.html');
  assert.equal(coherentMeshLab.getViewScene().mesh.triangles.length, 2);
  assert(calls.some(call => call.canvas === $('trSceneCanvas') && call.method === 'fill'));
  assert.equal(networkCalls, 0); assert.match($('trScope').textContent, /Maxwell/); assert.match($('trScope').textContent, /不是通用光源部分相干模型/);
}, false);
await test('Default engine request and repeated Run click are bounded to one worker', () => {
  const worker = start(); assert.match(worker.url, /transport-worker\.js$/); assert.deepEqual(worker.options, {type: 'module'});
  assert.deepEqual({...worker.request, id: 0}, {id: 0, measurement: 'detector', slab: null, lighting: null, camera: null, preset: 'lloyd', resolution: 512, wavelengthNm: 550, spectrum: {mode: 'broadband', illuminant: 'D65'}, spectralSamples: 64, spectralSeed: 7391, phaseShiftRad: 0, power: 1, coherence: 1, maxReflections: 2, pixelQuadrature: 2, backend: 'auto', diffraction: false, blockReflected: false, customOBJ: null, customMaterial: 'opaque', mixedDiffraction: false, scene: null, source: {position: [0, .08, -50], phaseRad: 0, power: 1, wavelengthMm: .00055, spectrum: {mode: 'broadband', illuminant: 'D65'}}, detector: {center: [0, .5, 50], u: [1, 0, 0], v: [0, 1, 0], width: 6, height: 1}});
  const count = workers.length; $('trRun').click(); assert.equal(workers.length, count); $('trCancel').click(); assert(worker.terminated); idle(); empty();
});
await test('Lloyd color-fringe view changes the real detector and exports its exact pose', async () => {
  assert.equal($('trColorFringeView').checked, true); assert.equal($('trColorFringeControl').hidden, false); assert.equal($('trDetectorY').value, '0.5'); assert.equal($('trDetectorHeight').value, '1');
  const first = start(); assert.deepEqual(first.request.detector.center, [0, .5, 50]); assert.equal(first.request.detector.width, 6); assert.equal(first.request.detector.height, 1); assert.equal(first.request.resolution, 512);
  checkbox('trColorFringeView', false); assert(first.terminated); empty(); assert.equal($('trDetectorY').value, '3.05'); assert.equal($('trDetectorHeight').value, '6');
  const original = start(); assert.deepEqual(original.request.detector.center, [0, 3.05, 50]); assert.equal(original.request.detector.height, 6); $('trCancel').click();
  change('trSourceX', 1.2); checkbox('trColorFringeView', true); assert.equal($('trSourceX').value, '1.2'); assert.equal($('trDetectorHeight').value, '1');
  $('trExportScene').click(); const exported = JSON.parse(await downloads.at(-1).blob.text()); assert.deepEqual(exported.detector.center, [0, .5, 50]); assert.equal(exported.detector.width, 6); assert.equal(exported.detector.height, 1); assert.equal(exported.source.position[0], 1.2); assert.deepEqual(exported.source.spectrum, {mode: 'broadband', illuminant: 'D65'});
  change('trDetectorY', .7); assert.equal($('trColorFringeView').checked, false); assert.equal($('trDetectorY').value, '0.7'); assert.equal($('trDetectorHeight').value, '1');
  $('trReset').click(); assert.equal($('trColorFringeView').checked, true); assert.equal($('trDetectorY').value, '0.5'); assert.equal($('trDetectorHeight').value, '1'); assert.match($('trStatus').textContent, /6 × 1/);
});
await test('Color-fringe detector shortcut does not change other preset or imported geometry', async () => {
  for (const preset of ['two-mirror', 'double-slit', 'reflected-edge']) { change('trPreset', preset); assert.equal($('trColorFringeControl').hidden, true); assert.equal($('trColorFringeView').disabled, true); const worker = start(); assert.equal(worker.request.source, null); assert.equal(worker.request.detector, null); $('trCancel').click(); }
  change('trPreset', 'lloyd'); assert.equal($('trDetectorHeight').value, '1'); assert.equal($('trColorFringeControl').hidden, false);
});
await test('Top-of-results Run and Cancel stay synchronized with settings controls', () => {
  $('trRunQuick').click(); const worker = last(); assert.equal($('trRunQuick').disabled, true); assert.equal($('trRun').disabled, true); assert.equal($('trCancelQuick').hidden, false); assert.equal($('trCancel').hidden, false);
  const count = workers.length; $('trRunQuick').click(); assert.equal(workers.length, count); $('trCancelQuick').click(); assert(worker.terminated); idle(); empty();
});
await test('Real native512 CPU worker fixture finishes all four direct numeric images', async () => {
  change('trSpectrumMode', 'monochrome'); change('trBackendSelect', 'cpu'); mode = 'execute'; const worker = start(); await until(() => !coherentMeshLab.getState().busy);
  assert(worker.result, $('trStatus').textContent); assert(worker.terminated); idle(); assert.match($('trStatus').textContent, /计算完成/);
  for (const name of ['Geometric', 'Coherent', 'Baseline', 'Cross']) { assert.equal(pixels.get($(`tr${name}Canvas`)).width, 512); assert.equal($(`tr${name}Canvas`).dataset.complete, 'true'); }
  assert.equal(report().complete, true); assert.equal(report().backend.kind, 'cpu'); assert.equal(report().diagnostics.productionCPUShadow, false); assert.equal(networkCalls, 0);
});
await test('Physical images use the same fixed CIE transform; signed cross term preserves its sign', () => {
  const {result} = deliver();
  for (const exposure of [0, -2, 4]) {
    change('trExposure', exposure);
    for (const [name, key] of [['Geometric', 'geometric'], ['Coherent', 'coherent'], ['Baseline', 'incoherent'], ['Cross', 'crossTerm']]) {
      const image = pixels.get($(`tr${name}Canvas`));
      for (const p of [0, 79, 153, 8192, 131328, 262143]) {
        const rgb = key === 'crossTerm' ? transportSignedRGB(result.coherentXYZ[3 * p + 1] - result.incoherentXYZ[3 * p + 1], exposure) : displaySpectralXYZ(result[key + 'XYZ'].subarray(3 * p, 3 * p + 3), 16 + exposure).rgb;
        assert.deepEqual([...image.data.slice(p * 4, p * 4 + 4)], [...rgb, 255]);
      }
    }
  }
  assert.deepEqual(transportSignedRGB(0), [0, 0, 0]);
  assert(transportSignedRGB(1e-5)[0] > transportSignedRGB(1e-5)[2]); assert(transportSignedRGB(-1e-5)[2] > transportSignedRGB(-1e-5)[0]);
  assert.deepEqual(transportIntensityRGB(1e-5, 550), displaySpectralXYZ(cieXYZ(550).map(v => v * 1e-5), 16).rgb);
  assert(!calls.some(call => ['trGeometricCanvas', 'trCoherentCanvas', 'trBaselineCanvas', 'trCrossCanvas'].includes(call.canvas.id) && call.method === 'drawImage'), 'No detector upscale or painted fringe proxy');
});
await test('Exposure and observation-camera changes preserve physics, report and worker count', () => {
  deliver(); const count = workers.length, before = report(), scene = JSON.stringify(coherentMeshLab.getViewScene()), old = pixels.get($('trCoherentCanvas')).data.slice();
  change('trExposure', -1); assert.equal(workers.length, count); assert.deepEqual(report(), before); assert.notDeepEqual(pixels.get($('trCoherentCanvas')).data, old);
  $('trSceneCanvas').dispatchEvent(new win.KeyboardEvent('keydown', {key: 'ArrowRight', bubbles: true})); $('trSceneCanvas').dispatchEvent(new win.WheelEvent('wheel', {deltaY: 30, bubbles: true})); $('trResetCamera').click();
  assert.equal(JSON.stringify(coherentMeshLab.getViewScene()), scene); assert.deepEqual(report(), before); assert.equal(workers.length, count);
});
await test('Tile previews update native rows only and never enable final download', () => {
  const worker = start(); worker.emit({type: 'preview', ...tile(8, 8)});
  for (const name of ['Geometric', 'Coherent', 'Baseline', 'Cross']) { const canvas = $(`tr${name}Canvas`), draw = draws.findLast(item => item.canvas === canvas); assert.equal(draw.y, 8); assert.equal(draw.image.height, 8); assert.equal(canvas.dataset.complete, 'false'); assert.equal($(`tr${name}Partial`).hidden, false); assert.match($(`tr${name}Partial`).textContent, /8 \/ 512/); }
  assert.equal($('trDownload').disabled, true); assert.match($('trStatus').textContent, /不是物理零值/);
  worker.emit({type: 'preview', ...tile(16, 8)}); assert.match($('trCoherentPartial').textContent, /16 \/ 512/);
  const count = workers.length; change('trExposure', -1); assert.equal(workers.length, count); assert.equal($('trDownload').disabled, true);
  $('trCancel').click(); idle(); empty();
});
await test('Full-frame preview arrays with row ranges are accepted without scaling', () => {
  const worker = start(), p = tile(24, 8);
  for (const key of ['coherent', 'incoherent', 'geometric', 'crossTerm', 'coherentXYZ', 'incoherentXYZ', 'geometricXYZ']) { const stride = key.endsWith('XYZ') ? 3 : 1, full = new Float64Array(512 * 512 * stride); full.set(p[key], 24 * 512 * stride); p[key] = full; }
  worker.emit({type: 'preview', ...p}); assert.match($('trCoherentPartial').textContent, /8 \/ 512/); const image = pixels.get($('trCoherentCanvas'));
  assert.deepEqual([...image.data.slice(24 * 512 * 4, 24 * 512 * 4 + 4)], [...displaySpectralXYZ(p.coherentXYZ.subarray(24 * 512 * 3, 24 * 512 * 3 + 3), 16).rgb, 255]);
});
await test('Malformed previews reject atomically and remove all incomplete images', () => {
  for (const mutate of [p => p.coherent[0] = NaN, p => p.geometric[0] = -1, p => p.crossTerm = p.crossTerm.slice(1), p => p.width = 256, p => p.rowCount = 0, p => p.rowStart = 510, p => p.complete = true, p => delete p.coherentXYZ, p => p.incoherentXYZ[0] = NaN, p => p.geometricXYZ[0] = -1, p => p.coherentXYZ = p.coherentXYZ.slice(3), p => p.spectralComplete = false, p => p.spectralSamples = 1]) {
    fresh(); const worker = start(), p = tile(); mutate(p); worker.emit({type: 'preview', ...p}); idle(); empty(); assert.match($('trStatus').textContent, /预览被拒绝/); assert.equal(report().complete, false);
  }
});
await test('GPU restart discards partials; old-attempt messages cannot overwrite CPU replacement', () => {
  const worker = start(); worker.emit({type: 'preview', ...tile()}); worker.emit({type: 'reset', attempt: 1, reason: 'device lost'}); empty();
  const count = draws.length; worker.emit({type: 'preview', ...tile(8, 8, 0)}); worker.emit({type: 'result', result: fixture(worker.request)}); assert.equal(draws.length, count); assert.equal(coherentMeshLab.getState().busy, true);
  worker.emit({type: 'progress', reset: true, attempt: 1, fraction: 0}); worker.emit({type: 'preview', ...tile(0, 8, 1)}); assert.equal($('trCoherentPartial').hidden, false);
  worker.emit({type: 'result', result: fixture(worker.request, {attempt: 1})}); idle(); assert.equal(report().diagnostics.attempt, 1); assert.equal($('trCoherentPartial').hidden, true);
});
await test('Final dimension, configuration and field corruption never become completed results', () => {
  for (const mutate of [r => r.width = 256, r => r.coherent[0] = NaN, r => r.incoherent[0] = -1, r => r.complete = false, r => r.configuration = {...r.configuration, wavelengthNm: 630}, r => delete r.backend, r => r.backend = {}, r => r.configuration = {}, r => r.timings.totalMs = NaN, r => delete r.geometricXYZ, r => r.coherentXYZ[0] = Infinity, r => r.incoherentXYZ = new Float32Array(3), r => r.spectralComplete = false, r => r.spectralSamples = 1, r => r.configuration = {...r.configuration, spectrum: {mode: 'broadband', illuminant: 'E'}}, r => r.configuration = {...r.configuration, spectralSamples: 32}]) {
    fresh(); const worker = start(), result = fixture(worker.request); mutate(result); worker.emit({type: 'result', result}); idle(); empty(); assert.match($('trStatus').textContent, /结果被拒绝/); assert.equal(report().complete, false);
  }
});
await test('Every physical input invalidates results and propagates exact next request values', () => {
  for (const [id, value, key, expected] of [['trPreset', 'two-mirror', 'preset', 'two-mirror'], ['trWavelength', 630, 'wavelengthNm', 630], ['trCoherence', .2, 'coherence', .2], ['trPhase', .5, 'phaseShiftRad', Math.PI / 2], ['trPower', .6, 'power', .6], ['trOrder', 1, 'maxReflections', 1], ['trQuadrature', 4, 'pixelQuadrature', 4], ['trBackendSelect', 'cpu', 'backend', 'cpu']]) {
    fresh(); deliver(); change(id, value); empty(); assert.equal($('trDownload').disabled, true); const worker = start(); assert.equal(worker.request[key], expected); $('trCancel').click();
  }
  fresh(); const worker = start(); checkbox('trBlockReflected', true); assert(worker.terminated); empty(); assert.equal(start().request.blockReflected, true);
});
await test('Double-slit chooses admitted first-order diffraction and locks unsupported mirror phase', () => {
  change('trPhase', .5); change('trPreset', 'double-slit'); assert.equal($('trDiffraction').checked, true); assert.equal($('trPhase').value, '0'); assert.equal($('trPhase').disabled, true);
  const worker = start(); assert.equal(worker.request.preset, 'double-slit'); assert.equal(worker.request.diffraction, true); assert.equal(worker.request.phaseShiftRad, 0); assert.equal(coherentMeshLab.getViewScene().detector.width, 3);
  checkbox('trDiffraction', false); assert(worker.terminated); assert.equal($('trPhase').disabled, false); empty(); const geometricOnly = start(); assert.equal(geometricOnly.request.diffraction, false);
  change('trPreset', 'lloyd'); assert.equal($('trDiffraction').checked, false); change('trPhase', .5); checkbox('trDiffraction', true); assert.equal($('trPhase').value, '0'); assert.equal($('trPhase').disabled, true);
});
await test('Tampered UI bounds cannot force unsupported engine values', () => {
  for (const [id, value] of [['trPower', 1001], ['trOrder', 3], ['trBackendSelect', 'all-gpu']]) { fresh(); if (id !== 'trPower') { const option = document.createElement('option'); option.value = String(value); $(id).append(option); } change(id, value); const count = workers.length; start(); assert.equal(workers.length, count); empty(); idle(); assert.match($('trStatus').textContent, /无法开始/); }
});
await test('Cancel, parameter edits and reset suppress every type of late worker response', () => {
  for (const cancel of [() => $('trCancel').click(), () => change('trWavelength', 580), () => $('trReset').click()]) {
    fresh(); const worker = start(), result = fixture(worker.request); cancel(); const text = $('trStatus').textContent, count = draws.length;
    for (const message of [{type: 'progress', fraction: 1, message: 'stale'}, {type: 'preview', ...tile()}, {type: 'reset', attempt: 1}, {type: 'result', result}, {type: 'error', error: 'stale'}]) worker.emit(message);
    worker.error(); worker.messageError(); assert.equal($('trStatus').textContent, text); assert.equal(draws.length, count); assert(worker.terminated); idle(); empty();
  }
});
await test('Pagehide clears interrupted previews and return-to-page can run cleanly', () => {
  const worker = start(); worker.emit({type: 'preview', ...tile()}); win.dispatchEvent(new win.Event('pagehide')); idle(); empty(); assert(worker.terminated); assert.match($('trStatus').textContent, /离开页面时已取消/);
  const count = draws.length; worker.emit({type: 'result', result: fixture(worker.request)}); assert.equal(draws.length, count); const next = start(); next.emit({type: 'result', result: fixture(next.request)}); idle(); assert.equal(report().complete, true);
});
await test('Superseded worker cannot publish into a newer active run; repeated renders work', () => {
  const old = start(); change('trWavelength', 600); const current = start(); old.emit({type: 'result', result: fixture(old.request)}); assert.equal(coherentMeshLab.getState().busy, true); empty();
  current.emit({type: 'result', result: fixture(current.request)}); idle(); assert.equal(report().configuration.wavelengthNm, 600);
  const next = start(); assert.notEqual(next, current); next.emit({type: 'result', result: fixture(next.request)}); idle(); assert.equal(report().complete, true);
});
await test('Worker construction, posting, execution, device and data errors release the UI', () => {
  for (const failure of ['factory-error', 'post-error', 'runtime', 'device', 'deserialize']) { fresh(); if (failure.endsWith('error')) { mode = failure; start(); } else { const worker = start(); if (failure === 'runtime') worker.emit({type: 'error', error: 'Backend failed', diagnostics: {buffer: new Float32Array(100)}}); if (failure === 'device') worker.error(); if (failure === 'deserialize') worker.messageError(); } idle(); empty(); assert.equal(report().complete, false); assert.equal($('trDownload').disabled, false); }
});
const obj = 'mtllib https://invalid.example/never-fetch.mtl\nv 8 0 0\nv 9 0 0\nv 8 1 0\nf 1 2 3\n';
await test('OBJ is local, adds actual geometry and explicitly selects opaque or mirror boundary', async () => {
  fileChange({name: 'addition.obj', size: obj.length, text: async () => obj}); await until(() => !coherentMeshLab.getState().fileReading);
  assert.equal(coherentMeshLab.getState().imported, true, $('trStatus').textContent); assert.equal(coherentMeshLab.getViewScene().mesh.triangles.length, 3); assert.equal($('trMaterial').disabled, false);
  change('trMaterial', 'mirror'); const worker = start(); assert.equal(worker.request.customOBJ, obj); assert.equal(worker.request.customMaterial, 'mirror'); assert.equal(networkCalls, 0);
  $('trClearOBJ').click(); assert(worker.terminated); empty(); assert.equal(coherentMeshLab.getState().imported, false); assert.equal(coherentMeshLab.getViewScene().mesh.triangles.length, 2);
});
await test('Oversized and malformed OBJ never silently render the previous scene', async () => {
  let reads = 0; fileChange({name: 'large.obj', size: 2 * 1024 * 1024 + 1, text: async () => { reads++; return obj; }}); await until(() => !coherentMeshLab.getState().fileReading); assert.equal(reads, 0); assert.match($('trStatus').textContent, /2 MiB/); const count = workers.length; start(); assert.equal(workers.length, count);
  fileChange({name: 'bad.obj', size: 8, text: async () => '<html>'}); await until(() => !coherentMeshLab.getState().fileReading); assert.match($('trStatus').textContent, /未通过检查/); assert.equal(coherentMeshLab.getViewScene(), null); start(); assert.equal(workers.length, count);
});
await test('Late OBJ reads cannot overwrite a newer file, reset, clear, or pagehide', async () => {
  for (const action of ['new-file', 'reset', 'clear', 'pagehide']) {
    fresh(); const slow = deferred(); fileChange({name: 'slow.obj', size: obj.length, text: () => slow.promise}); assert.equal($('trRun').disabled, true);
    if (action === 'new-file') fileChange({name: 'new.obj', size: obj.length, text: async () => obj}); else if (action === 'reset') $('trReset').click(); else if (action === 'clear') $('trClearOBJ').click(); else win.dispatchEvent(new win.Event('pagehide'));
    await delay(5); slow.resolve(obj); await delay(5); assert(!$('trFileName').textContent.includes('slow.obj') || action === 'pagehide');
    assert.equal(coherentMeshLab.getState().imported, action === 'new-file');
  }
});
await test('Diffraction-admitted result is labelled noncoherent path baseline, not geometric optics', () => {
  const worker = start(), result = fixture(worker.request, {diffraction: true}); result.geometric.fill(0); result.geometricXYZ.fill(0); result.diagnostics.edgeWarningCounts = {'endpoint-dominated-diffraction-not-modeled': 7}; result.diagnostics.edgeExclusionCounts = {'occluded-edge-leg': 12}; worker.emit({type: 'result', result});
  assert.match($('trDiffractionSummary').textContent, /端点主导区域未建模：7/); assert.match($('trDiffractionSummary').textContent, /源／接收边腿被遮挡：12/);
  assert.deepEqual([...pixels.get($('trGeometricCanvas')).data.slice(0, 4)], [0, 0, 0, 255]);
  assert.match($('trBaselineTitle').textContent, /独立路径相位基线/); assert.match($('trBaselineFormula').textContent, /衍射/);
  assert.deepEqual([...pixels.get($('trBaselineCanvas')).data.slice(0, 4)], [...displaySpectralXYZ(result.incoherentXYZ.subarray(0, 3), 16).rgb, 255]);
});
await test('Diagnostic download is bounded local JSON with actual backend and no raw arrays or OBJ', async () => {
  deliver(); const downloadsBefore = downloads.length, before = report(); assert.equal(before.backend.kind, 'cpu'); assert.equal(before.display.baseExposureEV, 16); assert.equal(before.diagnostics.integratedResponse.coherent, .1);
  $('trDownload').click(); assert.equal(downloads.length, downloadsBefore + 1); const downloaded = downloads.at(-1); assert.equal(downloaded.name, 'coherent-mesh-run.json'); assert.deepEqual(JSON.parse(await downloaded.blob.text()), before); assert.equal(networkCalls, 0);
  const bounded = boundedTransportReport({customOBJ: obj, coherent: new Float64Array(100), giant: 'a'.repeat(10000), list: Array(100).fill(1), nested: {buffer: new ArrayBuffer(32)}}); assert.equal(bounded.customOBJ, undefined); assert.equal(bounded.coherent, undefined); assert.equal(bounded.giant.length, 2048); assert.equal(bounded.list.length, 32);
  win.dispatchEvent(new win.Event('pagehide')); assert(revoked.includes(downloaded.url));
});
const sceneJSON = () => ({schema: 'coherent-mesh-scene-v1', units: 'mm', name: 'User geometry', materials: [{type: 'opaque'}], meshes: [{vertices: [[30, 0, 0], [31, 0, 0], [30, 1, 0]], faces: [[0, 1, 2]], materialIndex: 0, surface: 'auto'}], source: {position: [2, 3, -20], power: 2.5, phaseRad: .7, wavelengthMm: .0006328}, detector: {center: [2, 3, 20], u: [0, 2, 0], v: [0, 0, 3], width: 3, height: 4}});
const sceneFileChange = file => { Object.defineProperty($('trSceneJSON'), 'files', {value: [file], configurable: true}); $('trSceneJSON').dispatchEvent(new win.Event('change', {bubbles: true})); };
async function importScene(scene = sceneJSON(), name = 'scene.json') { const text = JSON.stringify(scene); sceneFileChange({name, size: text.length, text: async () => text}); await until(() => !coherentMeshLab.getState().fileReading); }
await test('Complete scene JSON replaces all preset geometry and sets actual source and detector', async () => {
  const old = start(); await importScene(); assert(old.terminated); empty();
  assert.equal(coherentMeshLab.getState().customScene, true, $('trStatus').textContent); assert.equal($('trColorFringeControl').hidden, true); assert.equal($('trColorFringeView').checked, false); assert.equal($('trDetectorHeight').value, '4'); assert.equal(coherentMeshLab.getViewScene().mesh.triangles.length, 1);
  assert.deepEqual(coherentMeshLab.getViewScene().sourceOrigin, [2, 3, -20]); assert.deepEqual(coherentMeshLab.getViewScene().detector.u, [0, 1, 0]); assert.equal($('trPower').value, '2.5'); assert.equal($('trWavelength').value, '632.8'); assert.equal($('trSpectrumMode').value, 'monochrome'); assert(!('spectrum' in sceneJSON().source)); assert.equal($('trOBJ').disabled, true); assert.equal($('trPhase').disabled, true); assert.match($('trSceneMode').textContent, /全部几何已替换/);
  const worker = start(); assert(worker.request.scene); assert.equal(worker.request.customOBJ, null); assert.equal(worker.request.scene.source.phaseRad, .7);
  const count = draws.length; old.emit({type: 'result', result: fixture(old.request)}); assert.equal(draws.length, count); assert(coherentMeshLab.getState().busy);
  worker.emit({type: 'result', result: fixture(worker.request)}); idle(); assert.equal(report().request.importedScene, true); assert.equal(report().request.scene, undefined); assert.equal(report().configuration.scene, undefined);
  const repeated = start(); repeated.emit({type: 'result', result: fixture(repeated.request)}); idle(); assert.equal(report().complete, true);
});
await test('Custom source-detector coordinates are real physics inputs and cancel stale workers', () => {
  const expected = [['trSourceX', 1.2, 'source', 'position', 0], ['trSourcePhase', 1.1, 'source', 'phaseRad'], ['trDetectorZ', 60, 'detector', 'center', 2], ['trDetectorWidth', 4.5, 'detector', 'width']];
  for (const [id, value, owner, key, axis] of expected) {
    fresh(); const old = start(); change(id, value); assert(old.terminated); empty(); const worker = start(), got = worker.request[owner][key]; assert.equal(axis === undefined ? got : got[axis], value);
    old.emit({type: 'result', result: fixture(old.request)}); assert(coherentMeshLab.getState().busy); worker.emit({type: 'result', result: fixture(worker.request)}); idle(); assert.equal(report().complete, true);
  }
  fresh(); change('trDetectorUX', 0); assert.match($('trGeometryNote').textContent, /nonzero/); const count = workers.length; start(); assert.equal(workers.length, count);
  fresh(); change('trSourceX', ''); start(); assert.equal(workers.length, count); assert.match($('trStatus').textContent, /finite/);
});
await test('Custom JSON remains editable without preset geometry and exports full local scene', async () => {
  await importScene(); change('trSourceX', 4); change('trDetectorWidth', 5); change('trWavelength', 610); change('trPower', 3);
  const worker = start(); assert.equal(worker.request.scene.source.position[0], 4); assert.equal(worker.request.scene.detector.width, 5); assert.equal(worker.request.scene.source.wavelengthMm, .00061); assert.equal(coherentMeshLab.getViewScene().mesh.triangles.length, 1); $('trCancel').click();
  const before = downloads.length; $('trExportScene').click(); assert.equal(downloads.length, before + 1); const file = downloads.at(-1), exported = JSON.parse(await file.blob.text()); assert.equal(file.name, 'coherent-mesh-scene.json'); assert.equal(exported.source.position[0], 4); assert.equal(exported.source.power, 3); assert.equal(exported.detector.width, 5); assert.equal(exported.meshes.length, 1); assert.equal(exported.materials[0].type, 'opaque'); assert.equal(networkCalls, 0);
  $('trUsePreset').click(); assert.equal(coherentMeshLab.getState().customScene, false); assert.equal(coherentMeshLab.getViewScene().mesh.triangles.length, 2); assert.equal($('trOBJ').disabled, false);
  $('trExportScene').click(); const template = JSON.parse(await downloads.at(-1).blob.text()); assert.equal(template.schema, 'coherent-mesh-scene-v1'); assert.equal(template.meshes[0].faces.length, 2); await importScene(template); assert.equal(coherentMeshLab.getState().customScene, true);
});
await test('Malformed custom scenes never fall back to preset or retain prior images', async () => {
  await importScene(); deliver(); const invalid = sceneJSON(); invalid.detector.v = [0, 1, 1]; await importScene(invalid); empty(); assert.equal(coherentMeshLab.getViewScene(), null); assert.equal(coherentMeshLab.getState().customScene, false); const count = workers.length; start(); assert.equal(workers.length, count); assert.match($('trStatus').textContent, /无法开始/);
  let reads = 0; sceneFileChange({name: 'huge.json', size: 2 * 1024 * 1024 + 1, text: async () => { reads++; return '{}'; }}); await until(() => !coherentMeshLab.getState().fileReading); assert.equal(reads, 0); assert.equal($('trExportScene').disabled, true);
  $('trUsePreset').click(); assert.equal($('trExportScene').disabled, false); assert.equal(coherentMeshLab.getViewScene().mesh.triangles.length, 2);
});
await test('Scene, source and detector mismatch in same-generation results is rejected', async () => {
  for (const field of ['scene', 'source', 'detector']) {
    fresh(); if (field === 'scene') await importScene(); else change('trSourceX', 2);
    const worker = start(), result = fixture(worker.request); result.configuration = structuredClone(result.configuration);
    if (field === 'scene') result.configuration.scene.meshes[0].vertices[0][0] += 1; else if (field === 'source') result.configuration.source.position[0] += 1; else result.configuration.detector.width += 1;
    worker.emit({type: 'result', result}); idle(); empty(); assert.match($('trStatus').textContent, /结果场景不匹配/);
  }
});
await test('Pending JSON reads cannot overwrite new import, pose edit, preset, reset or pagehide', async () => {
  const text = JSON.stringify(sceneJSON());
  for (const action of ['new-file', 'pose-edit', 'preset', 'reset', 'clear', 'pagehide', 'obj']) {
    fresh(); const slow = deferred(); sceneFileChange({name: 'slow.json', size: text.length, text: () => slow.promise}); assert.equal($('trRun').disabled, true);
    if (action === 'new-file') await importScene({...sceneJSON(), name: 'New import'});
    else if (action === 'pose-edit') change('trSourceX', 8);
    else if (action === 'preset') change('trPreset', 'two-mirror');
    else if (action === 'reset') $('trReset').click();
    else if (action === 'clear') $('trUsePreset').click();
    else if (action === 'obj') { fileChange({name: 'addition.obj', size: obj.length, text: async () => obj}); await until(() => !coherentMeshLab.getState().fileReading); }
    else win.dispatchEvent(new win.Event('pagehide'));
    const before = $('trStatus').textContent; slow.resolve(text); await delay(5); assert.equal($('trStatus').textContent, before); assert.equal(coherentMeshLab.getState().customScene, action === 'new-file'); if (action === 'new-file') assert.match($('trSceneMode').textContent, /New import/); if (action === 'pose-edit') assert.equal($('trSourceX').value, '8');
  }
});
await test('Mixed-path controls have explicit prerequisites and reflected-edge preset uses the shared engine', () => {
  assert.equal($('trMixedDiffraction').disabled, true); change('trPreset', 'reflected-edge'); assert.equal($('trDiffraction').checked, true); assert.equal($('trMixedDiffraction').checked, true); assert.equal($('trOrder').value, '2'); assert.equal(coherentMeshLab.getViewScene().detector.width, 2);
  assert.equal($('trPhase').disabled, false); assert.match($('trPhaseLabel').textContent, /外部镜面/); change('trPhase', .25); const worker = start(); assert.equal(worker.request.mixedDiffraction, true); assert.equal(worker.request.phaseShiftRad, Math.PI / 4); assert.equal(worker.request.resolution, 512); assert.equal(coherentMeshLab.getViewScene().mesh.triangles[0].materialIndex, 1); checkbox('trBlockReflected', true); assert(worker.terminated); assert.equal($('trMixedDiffraction').checked, false); assert.equal($('trMixedDiffraction').disabled, true);
  checkbox('trBlockReflected', false); checkbox('trMixedDiffraction', true); change('trOrder', 1); assert.equal($('trMixedDiffraction').checked, false); assert.equal($('trMixedDiffraction').disabled, true); change('trOrder', 2); checkbox('trMixedDiffraction', true); checkbox('trDiffraction', false); assert.equal($('trMixedDiffraction').checked, false); checkbox('trDiffraction', true); assert.equal($('trPhase').value, '0.25');
});
await test('Removing an appended OBJ preserves explicitly edited source-detector coordinates', async () => {
  change('trSourceX', 4); change('trDetectorWidth', 5);
  fileChange({name: 'addition.obj', size: obj.length, text: async () => obj}); await until(() => !coherentMeshLab.getState().fileReading);
  assert.equal(coherentMeshLab.getViewScene().mesh.triangles.length, 3); $('trClearOBJ').click();
  assert.equal($('trSourceX').value, '4'); assert.equal($('trDetectorWidth').value, '5'); assert.equal(coherentMeshLab.getViewScene().mesh.triangles.length, 2);
  const worker = start(); assert.equal(worker.request.source.position[0], 4); assert.equal(worker.request.detector.width, 5); $('trCancel').click();
});
await test('Wavelength conversion roundoff accepts 489 nm and decimal imported wavelengths', async () => {
  change('trWavelength', 489); const first = deliver(); idle(); assert.equal(report().complete, true); assert.equal(first.worker.request.wavelengthNm, 489); assert(Math.abs(first.result.configuration.wavelengthNm - 489) < 1e-10);
  const scene = sceneJSON(); scene.source.wavelengthMm = .000489123456789; await importScene(scene); const second = deliver(); idle(); assert.equal(report().complete, true); assert.equal(second.worker.request.scene.source.wavelengthMm, scene.source.wavelengthMm);
});
await test('Actual custom-scene native512 CPU render returns all direct numeric images', async () => {
  await importScene(); change('trBackendSelect', 'cpu'); change('trQuadrature', 1); mode = 'execute'; const worker = start(); await until(() => !coherentMeshLab.getState().busy); assert(worker.result, $('trStatus').textContent); assert.equal(worker.result.width, 512); assert.equal(worker.result.diagnostics.triangles, 1); assert.equal(worker.result.diagnostics.productionCPUShadow, false); assert.equal(report().complete, true); assert.equal(networkCalls, 0);
});
await test('Actual native512 broadband CPU XYZ reaches the canvases without monochrome tint', async () => {
  change('trBackendSelect', 'cpu'); change('trOrder', 0); change('trQuadrature', 1); change('trSpectralQuality', 16); mode = 'execute';
  const worker = start(); await until(() => !coherentMeshLab.getState().busy); assert(worker.result, $('trStatus').textContent); idle();
  const result = worker.result; assert.equal(result.spectralComplete, true); assert.equal(result.spectralSamples, 16); assert.equal(result.coherentXYZ.length, 512 * 512 * 3); assert.equal(result.configuration.spectrum.illuminant, 'D65');
  const image = pixels.get($('trCoherentCanvas'));
  for (const p of [0, 8192, 131328, 262143]) {
    const xyz = result.coherentXYZ.subarray(3 * p, 3 * p + 3), rgb = displaySpectralXYZ(xyz, 16).rgb;
    assert.deepEqual([...image.data.slice(4 * p, 4 * p + 4)], [...rgb, 255]);
    assert(xyz[2] > .3 * xyz[1], 'D65 blue response must not be the near-zero 550 nm blue response');
    assert.notDeepEqual(rgb, transportIntensityRGB(result.coherent[p], 550));
    assert.equal(result.coherent[p], result.incoherent[p]);
  }
  assert.equal(report().display.spectralComplete, true); assert.equal(report().display.spectralSamples, 16); assert.equal(networkCalls, 0);
});
await test('Visible-source controls produce exact spectra and invalidate stale spectral results', () => {
  for (const illuminant of ['D65', 'E', 'blackbody3000']) {
    fresh(); change('trSpectrumSource', illuminant); const worker = start(); assert.deepEqual(worker.request.spectrum, {mode: 'broadband', illuminant}); assert.equal($('trWavelength').disabled, true); assert.equal($('trSpectrumSource').disabled, false); $('trCancel').click();
  }
  change('trSpectrumMode', 'gaussian'); change('trWavelength', 612); change('trBandwidth', 37); change('trSpectralQuality', 128);
  const worker = start(); assert.deepEqual(worker.request.spectrum, {mode: 'gaussian', centerNm: 612, fwhmNm: 37}); assert.equal(worker.request.spectralSamples, 128); assert.equal($('trWavelength').disabled, false); assert.equal($('trSpectrumSource').disabled, true); assert.equal($('trBandwidth').disabled, false);
  worker.emit({type: 'preview', ...tile()}); change('trBandwidth', 42); assert(worker.terminated); empty();
  change('trSpectrumMode', 'monochrome'); const mono = start(); assert.deepEqual(mono.request.spectrum, {mode: 'monochrome'}); assert.equal($('trSpectralQuality').disabled, true); assert.equal($('trBandwidth').disabled, true); $('trReset').click(); assert.equal($('trSpectrumMode').value, 'broadband'); assert.equal($('trSpectrumSource').value, 'D65');
});
await test('Imported spectral scenes round-trip and mono legacy scenes stay mono', async () => {
  for (const spectrum of [{mode: 'broadband', illuminant: 'blackbody3000'}, {mode: 'gaussian', centerNm: 610, fwhmNm: 26}, {mode: 'lines', lines: [{wavelengthNm: 460, powerFraction: .3}, {wavelengthNm: 630, powerFraction: .7}]}]) {
    fresh(); const input = sceneJSON(); input.source.spectrum = spectrum; await importScene(input); assert.equal(coherentMeshLab.getState().customScene, true, $('trStatus').textContent); assert.equal($('trSpectrumMode').value, spectrum.mode);
    const worker = start(); assert.deepEqual(worker.request.spectrum, spectrum); assert.deepEqual(worker.request.scene.source.spectrum, spectrum); assert.equal(worker.request.scene.source.wavelengthMm, input.source.wavelengthMm); $('trCancel').click();
    $('trExportScene').click(); const exported = JSON.parse(await downloads.at(-1).blob.text()); assert.deepEqual(exported.source.spectrum, spectrum); assert.equal(exported.source.wavelengthMm, input.source.wavelengthMm);
  }
  await importScene(); assert.equal($('trSpectrumMode').value, 'monochrome'); $('trExportScene').click(); const legacy = JSON.parse(await downloads.at(-1).blob.text()); assert(!('spectrum' in legacy.source));
});
await test('Spectral source edits retain mixed-path options and pose geometry', () => {
  change('trPreset', 'reflected-edge'); change('trSourceX', .04); change('trSpectrumSource', 'E'); change('trSpectralQuality', 32); const worker = start(); assert.equal(worker.request.mixedDiffraction, true); assert.equal(worker.request.diffraction, true); assert.equal(worker.request.source.position[0], .04); assert.deepEqual(worker.request.source.spectrum, {mode: 'broadband', illuminant: 'E'}); $('trCancel').click();
});
await test('Spectral invalid values cannot start a worker or preserve stale pixels', () => {
  for (const [id, value] of [['trBandwidth', 0], ['trBandwidth', 471], ['trBandwidth', '']]) { fresh(); change('trSpectrumMode', 'gaussian'); deliver(); change(id, value); empty(); const count = workers.length; start(); assert.equal(workers.length, count); assert.match($('trStatus').textContent, /无法开始/); }
});
await test('Matched point and disk comparison buttons preserve exposure and change only radius', () => {
  change('trExposure', 3); change('trPreset', 'two-mirror'); const old = start();
  $('trPointComparison').click(); assert(old.terminated); idle(); empty(); assert.equal($('trExposure').value, '3');
  assert.match($('trStatus').textContent, /点击“运行引擎”/);
  const point = structuredClone(start().request); $('trCancel').click();
  $('trDiskComparison').click(); idle(); assert.equal($('trExposure').value, '3');
  const disk = structuredClone(start().request); $('trCancel').click();
  for (const request of [point, disk]) {
    assert.equal(request.preset, 'lloyd'); assert.equal(request.resolution, 512); assert.equal(request.pixelQuadrature, 1); assert.equal(request.spectralSamples, 16);
    assert.deepEqual(request.spectrum, {mode: 'broadband', illuminant: 'D65'}); assert.equal(request.source.spatial.samples, 16); assert.deepEqual(request.source.spatial.normal, [0, 0, 1]);
    assert.deepEqual(request.detector.center, [0, .5, 50]); assert.equal(request.detector.width, 6); assert.equal(request.detector.height, 1);
    delete request.id;
  }
  assert.equal(point.source.spatial.radiusMm, 0); assert.equal(disk.source.spatial.radiusMm, .06);
  disk.source.spatial.radiusMm = 0; assert.deepEqual(disk, point);
});
await test('Continuous disk detector intersection surfaces in UI before launch', () => {
  $('trDiskComparison').click(); change('trSourceSamples', 1);
  change('trSourceX', 0); change('trSourceY', .5); change('trSourceZ', 49.99);
  change('trSourceNormalX', 1); change('trSourceNormalZ', 0);
  assert.match($('trGeometryNote').textContent, /Source disk may intersect detector/);
  assert.equal(coherentMeshLab.getViewScene(), null); const count = workers.length; start(); assert.equal(workers.length, count);
  assert.match($('trStatus').textContent, /Source disk may intersect detector/); empty();
});
await test('Disk source controls default to exact point and edits cancel stale work', async () => {
  assert.equal($('trSourceRadius').value, '0'); assert.equal($('trSourceSamples').value, '16');
  assert.deepEqual([...'XYZ'].map(axis => Number($('trSourceNormal' + axis).value)), [0, 0, 1]);
  assert.deepEqual([...$('trSourceSamples').options].map(option => Number(option.value)), [1, 8, 16, 32, 64, 128, 256]);
  const initial = start(); assert.equal(initial.request.source?.spatial, undefined);
  change('trSourceRadius', .015); assert(initial.terminated); idle(); empty();
  change('trSourceNormalY', 1); change('trSourceNormalZ', 0); change('trSourceSamples', 32);
  const worker = start(); assert.deepEqual(worker.request.source.spatial, {type: 'disk', radiusMm: .015, normal: [0, 1, 0], samples: 32, seed: 7391});
  change('trSourceRadius', 0); assert(worker.terminated); empty();
  const zero = start(); assert.equal(zero.request.source.spatial.radiusMm, 0); $('trCancel').click();
  $('trExportScene').click(); const exported = JSON.parse(await downloads.at(-1).blob.text()); assert.equal(exported.source.spatial.radiusMm, 0);
  fresh(); $('trExportScene').click(); assert.equal(JSON.parse(await downloads.at(-1).blob.text()).source.spatial, undefined);
});
await test('Spatial previews remain incomplete until all emitters contribute to each row', () => {
  change('trSourceRadius', .01); change('trSourceSamples', 8); const worker = start();
  worker.emit({type: 'preview', ...tile(0, 512), spatialComplete: false, spatialSamplesIntegrated: 1, spatialSamples: 8});
  assert.equal($('trCoherentCanvas').dataset.complete, 'false'); assert.equal($('trDownload').disabled, true);
  assert.match($('trCoherentPartial').textContent, /0 \/ 512 行/); assert.equal(Number($('trProgress').value), 1 / 8);
  worker.emit({type: 'preview', ...tile(0, 8), spatialComplete: true, spatialSamplesIntegrated: 8, spatialSamples: 8});
  assert.match($('trCoherentPartial').textContent, /8 \/ 512 行/); assert.equal($('trCoherentCanvas').dataset.complete, 'false');
  $('trCancel').click(); assert.equal($('trDownload').disabled, true);
  const incompleteWorker = start(), incompleteResult = fixture(incompleteWorker.request);
  incompleteWorker.emit({type: 'result', result: {...incompleteResult, spatialSamples: 8, spatialComplete: false}});
  empty(); assert.equal(report().complete, false);
  const finalWorker = start(), finalResult = fixture(finalWorker.request);
  finalWorker.emit({type: 'result', result: {...finalResult, spatialSamples: 8, spatialComplete: true}});
  assert.equal($('trCoherentCanvas').dataset.complete, 'true'); assert.equal($('trDownload').disabled, false);
});
await test('Imported disk source retains seed and nonstandard samples through edits and export', async () => {
  const scene = sceneJSON(); scene.source.spatial = {type: 'disk', radiusMm: .02, normal: [1, 0, 0], samples: 23, seed: 4294967295};
  await importScene(scene); assert.equal(coherentMeshLab.getState().customScene, true, $('trStatus').textContent);
  assert.equal($('trSourceRadius').value, '0.02'); assert.equal($('trSourceSamples').value, '23');
  const worker = start(); assert.deepEqual(worker.request.scene.source.spatial, scene.source.spatial); $('trCancel').click();
  change('trSourceRadius', .04); change('trSourceSamples', 64); change('trSourceX', 4);
  $('trExportScene').click(); const exported = JSON.parse(await downloads.at(-1).blob.text());
  assert.deepEqual(exported.source.spatial, {...scene.source.spatial, radiusMm: .04, samples: 64}); assert.equal(exported.source.position[0], 4);
  await importScene(); assert.equal($('trSourceRadius').value, '0'); assert.equal($('trSourceSamples').value, '16'); assert.equal($('trSourceSamples').querySelector('[data-imported]'), null);
});
await test('Invalid disk radii and zero normals prevent worker launch', () => {
  for (const radius of [-1, 1001, '']) { fresh(); change('trSourceRadius', radius); const count = workers.length; start(); assert.equal(workers.length, count); assert.match($('trStatus').textContent, /无法开始/); }
  fresh(); change('trSourceRadius', .01); change('trSourceNormalZ', 0); const count = workers.length; start(); assert.equal(workers.length, count); assert.match($('trStatus').textContent, /无法开始/);
});
await test('Object-imaging preset selects physical camera and economical D65 point illumination', () => {
  change('trSourceRadius', .06); change('trPreset', 'object-imaging');
  assert.equal($('trMeasurementMode').value, 'camera'); assert.equal($('trPhysicalCamera').hidden, false);
  assert.equal($('trExposure').value, '2'); assert.equal($('trExposureValue').textContent, '+2 EV');
  assert.equal($('trColorFringeControl').hidden, true); assert.equal($('trSourceRadius').value, '0');
  const worker = start(); assert.equal(worker.request.measurement, 'camera'); assert.equal(worker.request.resolution, 512);
  assert.deepEqual(worker.request.spectrum, {mode: 'broadband', illuminant: 'D65'}); assert.equal(worker.request.pixelQuadrature, 1); assert.equal(worker.request.spectralSamples, 16); assert.equal(worker.request.diffraction, true);
  assert.deepEqual(worker.request.camera, coherentMeshLab.getViewScene().camera);
  assert.match($('trGeometricTitle').textContent, /几何照明/); assert.match($('trCoherentTitle').textContent, /波动照明/);
  assert.match($('trPhysicalCamera').textContent, /Lambertian-like/); assert.match($('trPhysicalCamera').textContent, /镜头衍射/);
  change('trPreset', 'lloyd'); assert(worker.terminated); empty(); assert.equal($('trMeasurementMode').value, 'detector'); assert.equal($('trPhysicalCamera').hidden, true);
  const legacy = start(); assert.equal(legacy.request.camera, null); assert.equal(legacy.request.measurement, 'detector'); $('trCancel').click();
});
await test('Physical camera edits cancel stale results while display orbit remains display-only', () => {
  change('trPreset', 'object-imaging'); const worker = start(); const result = fixture(worker.request);
  change('trCameraFov', 42); assert(worker.terminated); empty(); worker.emit({type: 'result', result}); empty();
  const next = start(); assert.equal(next.request.camera.fovDeg, 42);
  const camera = structuredClone(next.request.camera); $('trResetCamera').click(); assert.deepEqual(next.request.camera, camera); assert.equal(next.terminated, false);
  next.emit({type: 'preview', ...tile()}); assert.equal($('trCoherentCanvas').dataset.complete, 'false');
  $('trCancel').click(); empty(); idle();
});
await test('Optional imported camera and diffuse reflectance survive edits, detector mode and JSON export', async () => {
  const input = sceneJSON(); input.camera = {position: [2, 3, 30], target: [2, 3, 0], up: [0, 1, 0], fovDeg: 35};
  input.materials = [{type: 'diffuse', reflectance: [{wavelengthNm: 360, value: .2}, {wavelengthNm: 830, value: .8}]}];
  await importScene(input); assert.equal(coherentMeshLab.getState().customScene, true, $('trStatus').textContent);
  assert.equal($('trMeasurementMode').value, 'camera'); assert.equal($('trCameraZ').value, '30');
  change('trCameraX', 4); change('trCameraFov', 50); const worker = start();
  assert.equal(worker.request.camera.position[0], 4); assert.equal(worker.request.scene.camera.fovDeg, 50); $('trCancel').click();
  change('trMeasurementMode', 'detector'); $('trExportScene').click(); const exported = JSON.parse(await downloads.at(-1).blob.text());
  assert.deepEqual(exported.camera, {...input.camera, position: [4, 3, 30], fovDeg: 50, mirrorBounces: 2}); assert.deepEqual(exported.materials, input.materials);
  await importScene(); assert.equal($('trMeasurementMode').value, 'detector'); $('trExportScene').click(); assert.equal(JSON.parse(await downloads.at(-1).blob.text()).camera, undefined);
});
await test('Camera black statuses and mirror transport counts stay explicit', () => {
  change('trPreset', 'object-imaging'); const worker = start(), result = fixture(worker.request);
  for (const key of ['coherent', 'incoherent', 'geometric', 'crossTerm']) result[key][0] = 0;
  for (const key of ['coherentXYZ', 'incoherentXYZ', 'geometricXYZ']) result[key].fill(0, 0, 3);
  Object.assign(result.diagnostics, {cameraMissSamples: 10, cameraHitSamples: 20, cameraUnsupportedSamples: 3, cameraMirrorSamples: 12, cameraMirrorBounces: 18, cameraReflectedHits: 9, cameraMirrorLimitSamples: 2, cameraAbsorbedSamples: 1, cameraOpaqueSamples: 4, cameraBackfaceSamples: 5});
  worker.emit({type: 'result', result}); idle(); assert.equal(report().complete, true);
  for (const name of ['Geometric', 'Coherent', 'Baseline', 'Cross']) assert.deepEqual([...pixels.get($(`tr${name}Canvas`)).data.slice(0, 4)], [0, 0, 0, 255]);
  assert.match($('trWarnings').textContent, /未命中 10/); for (const label of ['经过镜面的采样 12', '镜面交互总次数 18', '镜后漫反射命中 9', '镜面次数上限 2', '零反射率吸收 1', '不透明终止 4', '漫反射背面 5', '不透明终止（旧版计数）3']) assert($('trWarnings').textContent.includes(label), label); assert.match($('trMeasurement').textContent, /投影标量平方律/);
});
await test('Camera mirror budget cancels stale work and is independent of illumination reflections', async () => {
  change('trPreset', 'object-imaging'); assert.equal($('trCameraMirrorBounces').value, '2');
  assert.deepEqual([...$('trCameraMirrorBounces').options].map(option => Number(option.value)), [0, 1, 2]);
  const old = start(); const stale = fixture(old.request); change('trCameraMirrorBounces', 1); assert(old.terminated); idle(); empty();
  old.emit({type: 'result', result: stale}); empty();
  for (const cameraBounces of [0, 1, 2]) for (const illuminationBounces of [0, 1, 2]) {
    change('trCameraMirrorBounces', cameraBounces); change('trOrder', illuminationBounces);
    const worker = start(); assert.equal(worker.request.camera.mirrorBounces, cameraBounces); assert.equal(worker.request.maxReflections, illuminationBounces); $('trCancel').click();
  }
  $('trBlockReflected').checked = true; $('trBlockReflected').dispatchEvent(new win.Event('input', {bubbles: true})); const blocked = start(); assert.equal(blocked.request.camera.mirrorBounces, 2); assert.equal(blocked.request.blockReflected, true); $('trCancel').click();
  for (const mirrorBounces of [0, 1, 2]) {
    const input = sceneJSON(); input.camera = {position: [2, 3, 30], target: [2, 3, 0], up: [0, 1, 0], fovDeg: 35, mirrorBounces};
    await importScene(input); assert.equal($('trCameraMirrorBounces').value, String(mirrorBounces));
    $('trExportScene').click(); assert.deepEqual(JSON.parse(await downloads.at(-1).blob.text()).camera, input.camera);
    change('trCameraMirrorBounces', (mirrorBounces + 1) % 3); const worker = start(); assert.equal(worker.request.scene.camera.mirrorBounces, (mirrorBounces + 1) % 3); $('trCancel').click();
    $('trExportScene').click(); assert.equal(JSON.parse(await downloads.at(-1).blob.text()).camera.mirrorBounces, (mirrorBounces + 1) % 3);
  }
});
await test('Camera quality buttons configure exact native sampling without running or changing exposure', () => {
  change('trPreset', 'object-imaging'); change('trCameraMirrorBounces', 1); change('trOrder', 0);
  for (const [id, q, spectral] of [['trCameraFine', 2, 32], ['trCameraFast', 1, 16]]) {
    const old = start(), count = workers.length; $('' + id).click(); assert(old.terminated); idle(); empty(); assert.equal(workers.length, count);
    assert.equal($('trExposure').value, '2'); assert.match($('trStatus').textContent, /点击“运行引擎”/);
    const worker = start(); assert.equal(worker.request.resolution, 512); assert.equal(worker.request.pixelQuadrature, q); assert.equal(worker.request.spectralSamples, spectral);
    assert.equal(worker.request.camera.mirrorBounces, 1); assert.equal(worker.request.maxReflections, 0); $('trCancel').click();
  }
  change('trExposure', 3); $('trCameraFine').click(); assert.equal($('trExposure').value, '3');
  assert.match($('trPhysicalCamera').textContent, /几何采样量 4 倍、场求值量 8 倍/); assert.match($('trPhysicalCamera').textContent, /不是耗时倍数/);
});
await test('Camera pose and measurement snapshot mismatches reject atomically', () => {
  for (const field of ['camera', 'measurement']) {
    fresh(); change('trPreset', 'object-imaging'); const worker = start(), result = fixture(worker.request); result.configuration = structuredClone(result.configuration);
    if (field === 'camera') result.configuration.camera.fovDeg += 1; else result.configuration.measurement = 'detector';
    worker.emit({type: 'result', result}); idle(); empty(); assert.match($('trStatus').textContent, /结果.*不匹配/);
  }
});
await test('Invalid physical camera prevents worker launch', () => {
  for (const value of [0, 180, '']) { fresh(); change('trPreset', 'object-imaging'); change('trCameraFov', value); const count = workers.length; start(); assert.equal(workers.length, count); assert.match($('trStatus').textContent, /无法开始/); }
  fresh(); change('trPreset', 'object-imaging'); for (const axis of 'XYZ') change('trCamera' + axis, $('trCameraTarget' + axis).value);
  const count = workers.length; start(); assert.equal(workers.length, count); assert.match($('trStatus').textContent, /无法开始/);
});
await test('Lighting starts off, is camera-only and preset only changes lighting', async () => {
  assert.match($('trLightingControls').textContent, /10 分钟运行上限/); assert.match($('trLightingControls').textContent, /软件适配器/);
  assert.equal($('trEnvironmentEnabled').checked, false); assert.equal($('trEnvironmentEnabled').disabled, true);
  const legacy = start(); assert.equal(legacy.request.lighting, null); $('trCancel').click();
  change('trPreset', 'object-imaging'); const old = start(), count = workers.length, exposure = $('trExposure').value;
  $('trLightingPreset').click(); assert(old.terminated); assert.equal(workers.length, count); idle(); empty();
  assert.equal($('trExposure').value, exposure); assert.match($('trStatus').textContent, /点击“运行引擎”/);
  const worker = start(); assert.equal(worker.request.resolution, 512); assert.equal(worker.request.lighting.environment.radiance, 2e-6); assert.equal(worker.request.lighting.environment.lower, .25); assert.equal(worker.request.lighting.environment.samples, 8); assert.equal(worker.request.lighting.indirectSamples, 4); $('trCancel').click();
  $('trExportScene').click(); const exported = JSON.parse(await downloads.at(-1).blob.text()); assert.deepEqual(exported.lighting, worker.request.lighting);
  change('trMeasurementMode', 'detector'); assert.equal($('trIndirectEnabled').disabled, true); const n = workers.length; start(); assert.equal(workers.length, n); assert.match($('trStatus').textContent, /仅支持相机模式/);
});
await test('Lighting edits preserve imported independent spectrum, orientation, seed and precision', async () => {
  const input = sceneJSON(); input.camera = {position: [2, 3, 30], target: [2, 3, 0], up: [0, 1, 0], fovDeg: 35};
  input.lighting = {environment: {radiance: .000001234567891234, spectrum: {mode: 'lines', lines: [{wavelengthNm: 430, powerFraction: .2}, {wavelengthNm: 610, powerFraction: .8}]}, wavelengthNm: 612, up: [1, 0, 0], upper: 3, lower: .1234567891234, samples: 32, seed: 4294967295}, indirectSamples: 2};
  await importScene(input); assert.equal(coherentMeshLab.getState().customScene, true, $('trStatus').textContent);
  change('trSourceX', 4); change('trSpectrumMode', 'broadband'); change('trCameraFov', 44);
  const worker = start(); assert.deepEqual(worker.request.lighting, input.lighting); assert.deepEqual(worker.request.scene.lighting, input.lighting); $('trCancel').click();
  change('trEnvironmentRadiance', 3); change('trIndirectSamples', 8); $('trExportScene').click();
  const exported = JSON.parse(await downloads.at(-1).blob.text()); assert.deepEqual(exported.lighting, {...input.lighting, environment: {...input.lighting.environment, radiance: 3e-6}, indirectSamples: 8});
  checkbox('trEnvironmentEnabled', false); const sourceOnly = start(); assert.deepEqual(sourceOnly.request.lighting, {environment: null, indirectSamples: 8}); $('trCancel').click();
  checkbox('trEnvironmentEnabled', true); const restored = start(); assert.deepEqual(restored.request.lighting.environment, {...input.lighting.environment, radiance: 3e-6}); $('trCancel').click();
  checkbox('trEnvironmentEnabled', false); checkbox('trIndirectEnabled', false); const off = start(); assert.deepEqual(off.request.lighting, {environment: null, indirectSamples: 0}); assert.deepEqual(off.request.scene.lighting, off.request.lighting); $('trCancel').click();
});
await test('Lighting previews and final completion cannot claim unfinished illumination is complete', () => {
  change('trPreset', 'object-imaging'); $('trLightingPreset').click();
  const worker = start(); worker.emit({type: 'preview', ...tile(), lightingComplete: false});
  assert(Number($('trProgress').value) <= .5); assert.match($('trCoherentPartial').textContent, /环境与间接照明尚未完成/); assert.equal($('trDownload').disabled, true); assert.match($('trStatus').textContent, /不是最终结果/);
  const incomplete = fixture(worker.request); worker.emit({type: 'result', result: {...incomplete, lightingComplete: false}}); empty(); idle(); assert.equal(report().complete, false);
  const next = start(), result = fixture(next.request); result.lightingComplete = true;
  Object.assign(result.diagnostics, {lightingEnvironmentRays: 80, lightingEnvironmentEscapes: 60, lightingSecondaryHits: 9, lightingSourceSamples: 18});
  next.emit({type: 'result', result}); idle(); assert.equal(report().complete, true); assert.equal(report().display.lightingComplete, true);
  assert.match($('trWarnings').textContent, /环境射线 80/); assert.match($('trWarnings').textContent, /逃逸至天空 60/); assert.match($('trWarnings').textContent, /次级漫反射命中 9/);
  assert.match($('trMeasurement').textContent, /降低条纹对比度/);
});
await test('Lighting configuration mismatches reject and numeric validation prevents launch', () => {
  change('trPreset', 'object-imaging'); $('trLightingPreset').click(); const worker = start(), result = fixture(worker.request); result.lightingComplete = true; result.configuration = structuredClone(result.configuration); result.configuration.lighting.environment.radiance *= 2;
  worker.emit({type: 'result', result}); empty(); assert.match($('trStatus').textContent, /不匹配/);
  for (const [id, value] of [['trEnvironmentRadiance', -1], ['trEnvironmentRadiance', 1000001], ['trEnvironmentRadiance', ''], ['trEnvironmentUpper', 11], ['trEnvironmentLower', -1]]) {
    fresh(); change('trPreset', 'object-imaging'); $('trLightingPreset').click(); change(id, value); const count = workers.length; start(); assert.equal(workers.length, count); assert.match($('trStatus').textContent, /无法开始/);
  }
});
await test('Worker module suppresses cancelled/superseded results and exposes internal AbortError', async () => {
  const messages = [], calls = [], scope = {postMessage: message => messages.push(message)};
  const remove = installTransportWorker(scope, (request, options) => { const d = deferred(); calls.push({request, options, ...d}); return d.promise; });
  const first = scope.onmessage({data: {id: 1}}), second = scope.onmessage({data: {id: 2}}); assert(calls[0].options.signal.aborted); calls[0].options.preview(tile()); calls[0].resolve({complete: true}); await first; assert.equal(messages.length, 0);
  calls[1].options.progress(.5); calls[1].options.preview(tile()); assert.deepEqual(messages.map(m => m.type), ['progress', 'preview']); assert(messages.every(m => m.id === 2));
  await scope.onmessage({data: {type: 'cancel', id: 2}}); calls[1].resolve({complete: true}); await second; assert.equal(messages.length, 2); remove(); assert.equal(scope.onmessage, null);
  const errors = [], failed = {postMessage: message => errors.push(message)}; installTransportWorker(failed, async () => { throw Object.assign(new Error('Internal cancelled'), {name: 'AbortError'}); }); await failed.onmessage({data: {id: 3}}); assert.equal(errors[0].type, 'error'); assert.equal(errors[0].name, 'AbortError');
});
await test('Actual unchanged worker module round-trips typed arrays through Node worker_threads', async () => {
  const url = new URL('../dist/transport-worker.js', import.meta.url).href;
  const code = `const {parentPort} = require('node:worker_threads'); import(${JSON.stringify(url)}).then(({installTransportWorker})=>{const scope={postMessage:m=>parentPort.postMessage(m)};installTransportWorker(scope);parentPort.on('message',data=>scope.onmessage({data}));parentPort.postMessage({type:'ready'});});`;
  const worker = new NodeWorker(code, {eval: true});
  try {
    await new Promise((resolve, reject) => { worker.once('error', reject); worker.once('message', message => message.type === 'ready' ? resolve() : reject(new Error('No ready message'))); });
    const completed = new Promise((resolve, reject) => { worker.once('error', reject); worker.on('message', message => { if (message.type === 'result') resolve(message.result); if (message.type === 'error') reject(new Error(message.error)); }); });
    worker.postMessage({id: 77, preset: 'double-slit', resolution: 16, backend: 'cpu', pixelQuadrature: 1}); const result = await completed;
    assert(result.coherent instanceof Float64Array); assert.equal(result.coherent.length, 256); assert.equal(result.complete, true); assert.equal(result.backend.kind, 'cpu'); assert.equal(result.diagnostics.diffractionEnabled, true); assert(result.diagnostics.diffractedRoutes > 0);
  } finally { await worker.terminate(); }
});
await test('Glass preset buttons configure exact shared scenes without running and keep native512', async () => {
  for (const [id, file, camera] of [['trGlassCameraPreset', 'coherent-glass-imaging.json', true], ['trGlassInterferencePreset', 'coherent-slab-interference.json', false]]) {
    const count = workers.length; $(id).click(); assert.equal(workers.length, count); empty();
    const worker = start(); assert.equal(workers.length, count + 1, $('trStatus').textContent);
    const expected = JSON.parse(fs.readFileSync(new URL('../dist/examples/' + file, import.meta.url), 'utf8'));
    assert.deepEqual(worker.request.scene, expected); assert.deepEqual(worker.request.slab, expected.slab);
    assert.equal(worker.request.resolution, 512); assert.equal(worker.request.maxReflections, camera ? 0 : 1);
    assert.equal(worker.request.diffraction, false); assert.equal(worker.request.mixedDiffraction, false);
    assert.equal(worker.request.pixelQuadrature, 1); assert.equal(worker.request.spectralSamples, 16); assert.equal($('trExposure').value, camera ? '2' : '-2');
    assert.equal(worker.request.lighting?.indirectSamples ?? 0, 0);
    $('trExportScene').click(); assert.deepEqual(JSON.parse(await downloads.at(-1).blob.text()), expected);
    $('trCancel').click();
    change('trOrder', 2); checkbox('trDiffraction', true); await importScene(expected);
    assert.equal($('trOrder').value, camera ? '0' : '1'); assert.equal($('trDiffraction').checked, false); assert.equal($('trMixedDiffraction').checked, false);
    const imported = start(); assert.deepEqual(imported.request.scene, expected); $('trCancel').click();
  }
});
await test('Glass camera mirror preset preserves lighting and enables one source reflection', () => {
  $('trGlassMirrorCameraPreset').click();
  const expected = JSON.parse(fs.readFileSync(new URL('../dist/examples/coherent-glass-mirror-imaging.json', import.meta.url), 'utf8'));
  assert.equal($('trMeasurementMode').value, 'camera'); assert.equal($('trOrder').value, '1'); assert.equal($('trDiffraction').checked, false);
  const w = start(); assert.deepEqual(w.request.scene, expected); assert.equal(w.request.camera.mirrorBounces, 2); assert(w.request.lighting.environment); assert.equal(w.request.spectrum.mode, 'broadband');
  $('trCancel').click(); assert(w.terminated); empty();
});
await test('Mirror slab preset restores supported color settings and survives cancel/export/import', async () => {
  $('trGlassMirrorPreset').click();
  const expected = JSON.parse(fs.readFileSync(new URL('../dist/examples/coherent-slab-mirror.json', import.meta.url), 'utf8'));
  assert.equal($('trMeasurementMode').value, 'detector'); assert.equal($('trOrder').value, '1'); assert.equal($('trDiffraction').checked, false); assert.equal($('trSlabPropagationWarning').hidden, true);
  const w = start(); assert.deepEqual(w.request.scene, expected); assert.equal(w.request.resolution, 512); assert.equal(w.request.spectrum.mode, 'broadband');
  $('trCancel').click(); assert(w.terminated); empty();
  $('trExportScene').click(); const exported = JSON.parse(await downloads.at(-1).blob.text()); assert.deepEqual(exported, expected);
  change('trOrder', 2); checkbox('trDiffraction', true); assert.equal($('trSlabPropagationWarning').hidden, false); assert.match($('trSlabPropagationWarning').textContent, /未建模/);
  await importScene(exported); assert.equal($('trOrder').value, '1'); assert.equal($('trDiffraction').checked, false);
  const repeat = start(); assert.deepEqual(repeat.request.scene, expected); $('trCancel').click(); empty();
});
await test('Imported slab curve and axes survive edits and disable/re-enable; neutral edit is explicit', async () => {
  const scene = JSON.parse(fs.readFileSync(new URL('../dist/examples/coherent-glass-imaging.json', import.meta.url), 'utf8'));
  scene.slab.u = [0, 1, 0]; scene.slab.v = [-1, 0, 0]; scene.slab.width = 1.6; scene.slab.height = 2.4;
  await importScene(scene); assert.equal($('trSlabAbsorption').value, ''); assert.match($('trStatus').textContent, /显式保存/); assert.equal($('trOrder').value, '0'); assert.equal($('trSlabPropagationWarning').hidden, true);
  change('trOrder', 0); change('trSlabIndex', 1.4); change('trSlabThickness', .7);
  const worker = start(); assert.deepEqual(worker.request.slab.absorption, scene.slab.absorption); assert.deepEqual(worker.request.slab.u, scene.slab.u); assert.deepEqual(worker.request.slab.v, scene.slab.v);
  checkbox('trSlabEnabled', false); assert(worker.terminated); empty(); const disabled = start(); assert.equal(disabled.request.slab, null); assert.equal(disabled.request.scene.slab, undefined);
  checkbox('trSlabEnabled', true); const enabled = start(); assert.deepEqual(enabled.request.slab.absorption, scene.slab.absorption);
  change('trSlabAbsorption', .25); assert(enabled.terminated); const neutral = start(); assert.deepEqual(neutral.request.slab.absorption, [{wavelengthNm: 360, value: .25}, {wavelengthNm: 830, value: .25}]);
  $('trExportScene').click(); const exported = JSON.parse(await downloads.at(-1).blob.text()); assert.deepEqual(exported.slab, neutral.request.slab);
  $('trUsePreset').click(); assert.equal($('trSlabEnabled').checked, false); idle(); empty();
});
await test('Slab snapshots reject mismatches; scalar edits clear completed results and errors remain visible', () => {
  $('trGlassInterferencePreset').click(); const worker = start(), result = fixture(worker.request);
  result.configuration = structuredClone(result.configuration); result.configuration.slab.index = 1.8;
  worker.emit({type: 'result', result}); empty(); assert.match($('trStatus').textContent, /不匹配/);
  for (const [id, value] of [['trSlabIndex', 3.1], ['trSlabThickness', -1], ['trSlabAbsorption', -1]]) {
    $('trGlassInterferencePreset').click(); change(id, value); const count = workers.length; start(); assert.equal(workers.length, count); assert.match($('trStatus').textContent, /无法开始/);
  }
  $('trGlassInterferencePreset').click(); const running = start(); running.emit({type: 'error', error: 'Unsupported reflected route intersects dielectric slab'}); empty(); assert.match($('trStatus').textContent, /Unsupported reflected route/);
  $('trGlassInterferencePreset').click(); const incomplete = start(), incompleteFrame = fixture(incomplete.request);
  Object.assign(incompleteFrame.diagnostics, {modelIncomplete: true, cameraSlabUnsupportedSamples: 2, sourceSlabUnsupportedSamples: 3, lightingSlabUnsupportedRays: 4});
  incomplete.emit({type: 'result', result: incompleteFrame}); assert.match($('trStatus').textContent, /模型不完整/); assert.match($('trWarnings').textContent, /源端不支持采样 3/); assert.match($('trWarnings').textContent, /间接链不支持射线 4/);
  $('trReset').click(); assert.equal($('trSlabEnabled').checked, false);
});
$('trCancel').click(); win.close();
const summary = {passed: !failures.length, tests: passed.length, failures, actualCPUExecutions: actualExecutions, filter: process.env.WAVE_TEST_FILTER ?? null, native512Fixture: actualExecutions > 0, workerModuleRoundtrip: passed.includes('Actual unchanged worker module round-trips typed arrays through Node worker_threads'), DOMValidated: true, browserLayoutValidated: false, browserWorkerValidated: false, GPUValidated: false};
console.log(JSON.stringify(summary, null, 2));
if (failures.length) process.exitCode = 1;
