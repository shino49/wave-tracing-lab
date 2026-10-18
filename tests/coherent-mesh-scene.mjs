import assert from 'node:assert/strict';
import fs from 'node:fs';
import {COHERENT_SCENE_SCHEMA, COHERENT_SCENE_LIMITS, validateCoherentScene, validateCoherentSceneSource, validateCoherentSceneDetector, validateCoherentSourceDetector, parseCoherentScene, serializeCoherentScene, sceneToCoherentData, sceneFromCoherentData} from '../dist/coherent-mesh-scene.js';
import {makeCoherentMeshPreset, detectorPoint} from '../dist/coherent-mesh-core.js';
import {prepareMesh} from '../dist/mesh-geometry.js';
import {prepareCoherentMeshTask, executeCoherentMeshTask} from '../dist/coherent-mesh-runtime.js';
const source = {position: [2, 3, -20], power: 2.5, phaseRad: .7, wavelengthMm: .0006328};
const detector = {center: [2, 3, 20], u: [0, 2, 0], v: [0, 0, 3], width: 3, height: 4};
const raw = () => ({schema: COHERENT_SCENE_SCHEMA, units: 'mm', name: 'User-authored scene', materials: [{type: 'opaque'}], meshes: [{vertices: [[50, 0, 0], [51, 0, 0], [50, 1, 0]], faces: [[0, 1, 2]], materialIndex: 0, surface: 'open-plate'}], source: structuredClone(source), detector: structuredClone(detector)});
let tests = 0;
async function test(name, f) { await f(); tests++; console.log('PASS', name); }
await test('Canonical scenes deeply snapshot and freeze every authored field', () => {
  const input = raw(), scene = validateCoherentScene(input);
  assert(Object.isFrozen(scene)); assert(Object.isFrozen(scene.meshes[0].vertices[0])); assert(Object.isFrozen(scene.source.position)); assert(Object.isFrozen(scene.materials[0]));
  input.meshes[0].vertices[0][0] = 0; input.source.position[0] = 100; input.detector.center[0] = 99;
  assert.equal(scene.meshes[0].vertices[0][0], 50); assert.equal(scene.source.position[0], 2); assert.equal(scene.detector.center[0], 2);
  assert.deepEqual(scene.detector.u, [0, 1, 0]); assert.deepEqual(scene.detector.v, [0, 0, 1]);
  assert.deepEqual(validateCoherentScene(scene), scene);
});
await test('Canonical JSON round-trips and converts to runtime entries with no preset geometry', () => {
  const scene = parseCoherentScene('\uFEFF' + serializeCoherentScene(raw()));
  assert.deepEqual(scene, validateCoherentScene(raw()));
  const data = sceneToCoherentData(scene), prepared = prepareCoherentMeshTask({scene, preset: 'two-mirror', diffraction: false});
  assert.equal(data.entries.length, 1); assert.equal(prepared.scene.triangles.length, 1); assert.equal(prepared.scene.mirrorPlanes.length, 0);
  assert.deepEqual(prepared.source, scene.source); assert.deepEqual(prepared.detector, scene.detector); assert.equal(prepared.configuration.power, 2.5); assert.equal(prepared.configuration.wavelengthNm, 632.8);
  assert.throws(() => prepareCoherentMeshTask({scene, customOBJ: 'v 0 0 0'}), /replaces/);
});
await test('Optional source spectra round-trip without reinterpreting legacy monochrome scenes', () => {
  const legacy = parseCoherentScene(serializeCoherentScene(raw())); assert(!Object.hasOwn(legacy.source, 'spectrum')); assert.equal(legacy.source.wavelengthMm, source.wavelengthMm);
  for (const spectrum of [{mode: 'monochrome'}, {mode: 'broadband', illuminant: 'D65'}, {mode: 'broadband', illuminant: 'E'}, {mode: 'broadband', illuminant: 'blackbody3000'}, {mode: 'gaussian', centerNm: 612, fwhmNm: 22}, {mode: 'lines', lines: [{wavelengthNm: 450, powerFraction: .3}, {wavelengthNm: 630, powerFraction: .7}]}]) {
    const input = raw(); input.source.spectrum = spectrum; const scene = validateCoherentScene(input);
    assert.deepEqual(scene.source.spectrum, spectrum); assert.equal(scene.source.power, 2.5); assert(Object.isFrozen(scene.source.spectrum)); assert.deepEqual(parseCoherentScene(serializeCoherentScene(scene)), scene);
    if (spectrum.mode === 'lines') { assert(Object.isFrozen(scene.source.spectrum.lines)); input.source.spectrum.lines[0].powerFraction = .9; assert.equal(scene.source.spectrum.lines[0].powerFraction, .3); }
  }
});
await test('Invalid spectral source descriptors fail explicitly and never downgrade to monochrome', () => {
  for (const spectrum of [{mode: 'broadband', illuminant: 'unknown'}, {mode: 'broadband', illuminant: 'D65', url: 'https://invalid'}, {mode: 'gaussian', centerNm: 550, fwhmNm: 0}, {mode: 'gaussian', centerNm: 359, fwhmNm: 20}, {mode: 'gaussian', centerNm: 550, fwhmNm: Infinity}, {mode: 'lines', lines: []}, {mode: 'lines', lines: [{wavelengthNm: 550, powerFraction: -1}]}, {mode: 'monochrome', centerNm: 550}]) {
    const input = raw(); input.source.spectrum = spectrum; assert.throws(() => validateCoherentScene(input));
  }
});
await test('D65 color-fringe scene is explicit and leaves the legacy Lloyd preset unchanged', () => {
  const example = parseCoherentScene(fs.readFileSync(new URL('../dist/examples/coherent-lloyd-d65-color.json', import.meta.url), 'utf8'));
  assert.deepEqual(example.source.spectrum, {mode: 'broadband', illuminant: 'D65'}); assert.deepEqual(example.detector.center, [0, .5, 50]); assert.equal(example.detector.width, 6); assert.equal(example.detector.height, 1);
  const preset = makeCoherentMeshPreset('lloyd'); assert.deepEqual(preset.detector.center, [0, 3.05, 50]); assert.equal(preset.detector.width, 6); assert.equal(preset.detector.height, 6); assert(!Object.hasOwn(preset.source, 'spectrum'));
  const legacy = parseCoherentScene(fs.readFileSync(new URL('../dist/examples/coherent-reflected-edge.json', import.meta.url), 'utf8')); assert(!Object.hasOwn(legacy.source, 'spectrum'));
});
await test('Inline OBJ is canonicalized locally and external material references never load', () => {
  const input = raw(); input.meshes = [{obj: 'mtllib https://example.invalid/never.mtl\nv 50 0 0\nv 51 0 0\nv 50 1 0\nf 1 2 3\n', materialIndex: 0}];
  const scene = validateCoherentScene(input); assert(!('obj' in scene.meshes[0])); assert.deepEqual(scene.meshes[0].faces, [[0, 1, 2]]);
  assert.equal(prepareCoherentMeshTask({scene}).scene.triangles.length, 1);
});
await test('Preset exports preserve source, detector and prepared OBJ world geometry', () => {
  const d = makeCoherentMeshPreset('lloyd'); d.source = {...d.source, power: 1, phaseRad: 0, wavelengthMm: .00055};
  d.entries.push({mesh: prepareMesh({vertices: [[30, 0, 0], [31, 0, 0], [30, 1, 0]], faces: [[0, 1, 2]]}, {translation: [7, 0, 0]}), materialIndex: 0});
  const scene = sceneFromCoherentData(d); assert.equal(scene.meshes[1].vertices[0][0], 37); assert.equal(prepareCoherentMeshTask({scene}).scene.triangles.length, 3);
});
await test('Explicit units, keys, plain objects and triangle topology are enforced', () => {
  for (const modify of [s => delete s.units, s => s.units = 'm', s => s.schema = 'arbitrary', s => s.url = 'https://x', s => s.materials[0] = {type: 'diffuse'}, s => s.materials[0].reflectance = .5, s => s.meshes[0].url = 'https://x', s => s.meshes[0].faces = [[0, 1, 2, 0]], s => s.meshes[0].faces = [[0, 0, 1]], s => s.meshes[0].faces[0][0] = 3, s => s.meshes[0].materialIndex = .5, s => s.meshes[0].materialIndex = 1, s => s.meshes[0].surface = 'unrestricted', s => s.meshes[0].obj = 'v 0 0 0', s => s.meshes = []]) {
    const s = raw(); modify(s); assert.throws(() => validateCoherentScene(s));
  }
  assert.throws(() => validateCoherentScene(Object.assign(Object.create({hidden: true}), raw())), /object/);
  assert.throws(() => parseCoherentScene(JSON.stringify(raw()).replace('"units":"mm"', '"units":"mm","__proto__":{}')), /unsupported/);
  assert.throws(() => parseCoherentScene('{malformed'), /valid JSON/);
});
await test('Nonfinite, sparse, empty, oversized and singular source-detector inputs reject', () => {
  for (const modify of [s => s.source.position[0] = Infinity, s => s.source.position = Array(3), s => s.source.power = -1, s => s.source.power = 1001, s => s.source.wavelengthMm = 0, s => s.source.phaseRad = 10, s => s.detector.u = [0, 0, 0], s => s.detector.v = [1, 1, 0], s => s.detector.width = 0, s => s.detector.height = 1001, s => s.detector.center = [10000, 10000, 10000], s => s.source.position = [...s.detector.center], s => s.meshes[0].vertices[0][0] = 10001, s => s.meshes[0].vertices = Array(8001), s => s.meshes = Array(65).fill(s.meshes[0]), s => s.materials = Array(65).fill({type: 'opaque'})]) { const s = raw(); modify(s); assert.throws(() => validateCoherentScene(s)); }
  assert.throws(() => parseCoherentScene(' '.repeat(COHERENT_SCENE_LIMITS.maxBytes + 1)), /2 MiB/);
  const s = raw(); s.meshes = Array.from({length: 64}, (_, j) => ({...s.meshes[0], vertices: Array.from({length: 126}, (_, i) => [i, j + (i === 2 ? 1 : 0), 0])}));
  assert.throws(() => validateCoherentScene(s), /total vertex/);
  assert.throws(() => validateCoherentSourceDetector(validateCoherentSceneSource({...source, position: detector.center}), validateCoherentSceneDetector(detector)), /point source/);
  for (const nm of [360, 830]) assert.equal(validateCoherentSceneSource({...source, wavelengthMm: nm / 1e6}).wavelengthMm, nm / 1e6);
});
await test('Custom source and tilted detector reach real numeric rendering and initial phase cancels', async () => {
  const scene = validateCoherentScene(raw()), input = {scene, resolution: 16, pixelQuadrature: 1, backend: 'cpu', diffraction: false};
  const a = await executeCoherentMeshTask(input), b = await executeCoherentMeshTask({...input, scene: {...scene, source: {...scene.source, phaseRad: -1.2}}});
  assert.equal(a.complete, true); assert.equal(a.diagnostics.triangles, 1); assert.equal(a.diagnostics.reflectedRoutes, 0);
  for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) {
    const p = detectorPoint(scene.detector, x, y, 16, 16), d2 = p.reduce((v, c, i) => v + (c - scene.source.position[i]) ** 2, 0), expected = scene.source.power / (4 * Math.PI * d2), i = y * 16 + x;
    assert(Math.abs(a.coherent[i] - expected) < expected * 1e-12); assert(Math.abs(b.coherent[i] - a.coherent[i]) < expected * 1e-12);
  }
});
await test('Active custom requests keep a private scene snapshot across caller mutation and cancellation', async () => {
  const scene = raw(), controller = new AbortController(); let first = true;
  const result = await executeCoherentMeshTask({scene, resolution: 16, pixelQuadrature: 1, backend: 'cpu'}, {progress: () => { if (first) { first = false; scene.source.power = 0; scene.meshes[0].vertices[0][0] = -1000; } }});
  assert.equal(result.configuration.scene.source.power, 2.5); assert(result.coherent.some(x => x > 0));
  await assert.rejects(executeCoherentMeshTask({scene: raw(), resolution: 16, pixelQuadrature: 1, backend: 'cpu'}, {signal: controller.signal, preview: () => controller.abort()}), e => e.name === 'AbortError');
});
await test('Optional render settings round-trip strictly and retain coherent family constraints', async () => {
  for (const renderSettings of [{maxReflections: 0, diffraction: false, mixedDiffraction: false}, {maxReflections: 1, diffraction: false, mixedDiffraction: false}, {maxReflections: 2, diffraction: true, mixedDiffraction: true}]) {
    const input = {...raw(), renderSettings}, validated = validateCoherentScene(input);
    assert.deepEqual(parseCoherentScene(serializeCoherentScene(input)).renderSettings, renderSettings);
    assert.deepEqual(sceneFromCoherentData(sceneToCoherentData(input)).renderSettings, renderSettings);
    assert(Object.isFrozen(validated.renderSettings));
  }
  for (const renderSettings of [null, {}, {maxReflections: 3, diffraction: false, mixedDiffraction: false}, {maxReflections: .5, diffraction: false, mixedDiffraction: false}, {maxReflections: 0, diffraction: 0, mixedDiffraction: false}, {maxReflections: 2, diffraction: false, mixedDiffraction: true}, {maxReflections: 1, diffraction: true, mixedDiffraction: true}, {maxReflections: 0, diffraction: false, mixedDiffraction: false, typo: true}]) assert.throws(() => validateCoherentScene({...raw(), renderSettings}));
  assert.equal(Object.hasOwn(validateCoherentScene(raw()), 'renderSettings'), false);
});
console.log(JSON.stringify({passed: true, tests, actualCPUExecutions: 4, contract: COHERENT_SCENE_SCHEMA}));
