/** Independent v30 route/scene/runtime regressions. No full-wave certification. */
import assert from 'node:assert/strict';
import {prepareCoherentMeshScene, compileCoherentSource, sumCoherentRoutes} from '../dist/coherent-mesh-core.js';
import {evaluateMeshDiffraction} from '../dist/coherent-mesh-diffraction.js';
import {evaluateMixedDiffraction, prepareMixedDiffraction} from '../dist/coherent-mesh-mixed.js';
import {evaluateMeshTransport, coherentRouteCapacity} from '../dist/coherent-mesh-transport.js';
import {parseCoherentScene, validateCoherentScene, sceneToCoherentData} from '../dist/coherent-mesh-scene.js';
import {executeCoherentMeshTask, prepareCoherentMeshTask} from '../dist/coherent-mesh-runtime.js';
const TAU = 2 * Math.PI;
const soft = {type: 'mirror', reflectance: 1, phaseRad: Math.PI};
const mirror = {type: 'mirror', reflectance: .64, phaseRad: .37};
const source = {position: [-3, 0, -4], wavelengthMm: .05, power: 2.7, phaseRad: .41};
const receiver = [5, .3, 10];
const q = (vertices, materialIndex = 0, faces = [[0, 1, 2], [0, 2, 3]]) => ({mesh: {vertices, faces}, materialIndex, surface: 'open-plate'});
const plate = q([[0, -100, 0], [100, -100, 0], [100, 100, 0], [0, 100, 0]]);
const external = q([[-10, -100, -100], [-10, 100, -100], [-10, 100, 100], [-10, -100, 100]], 1);
const makeScene = (entries = [plate, external], materials = [soft, mirror]) => prepareCoherentMeshScene(entries, materials, {maxReflections: 2});
const d = (a, b) => Math.hypot(...a.map((v, i) => v - b[i]));
const field = routes => routes.reduce((a, r) => [a[0] + r.amplitude * Math.cos(r.phaseRadReduced), a[1] + r.amplitude * Math.sin(r.phaseRadReduced)], [0, 0]);
const rotate = (v, angle, factor = 1) => [factor * (v[0] * Math.cos(angle) - v[1] * Math.sin(angle)), factor * (v[0] * Math.sin(angle) + v[1] * Math.cos(angle))];
const near = (a, b, rel = 1e-10, abs = 1e-13) => assert(Math.abs(a - b) <= abs + rel * Math.max(Math.abs(a), Math.abs(b)), `${a} != ${b}`);
const nearField = (a, b, tol = 2e-11) => assert(Math.hypot(a[0] - b[0], a[1] - b[1]) <= tol, `${a} != ${b}`);
const nearPoint = (a, b, tol = 1e-8) => d(a, b) < tol;
const results = [];
async function check(name, fn) { try { await fn(); results.push({name, pass: true}); console.log('PASS', name); } catch (error) { results.push({name, pass: false, error: error.stack}); console.error('FAIL', name, error.stack); } }
const scene = makeScene(), compiled = compileCoherentSource(scene, source);
const sampleMixed = evaluateMixedDiffraction(scene, compiled, receiver);
const mainRoute = (routes, order) => routes.find(r => r.interactionOrder === order && Math.abs(r.point[0]) < 1e-8 && Math.abs(r.point[2]) < 1e-8);

await check('RD and DR use full image-leg spreading and one physical common carrier', () => {
  const onlyPlate = makeScene([plate], [soft]);
  for (const order of ['RD', 'DR']) {
    const s = order === 'RD' ? [-17, 0, -4] : source.position;
    const r = order === 'DR' ? [-25, .3, 10] : receiver;
    const oracle = evaluateMeshDiffraction(onlyPlate, s, r, {...source, referenceDistance: d(s, r)});
    const actual = sampleMixed.routes.filter(route => route.interactionOrder === order);
    assert(actual.length >= 2);
    for (const route of actual) {
      const reference = oracle.routes.find(x => nearPoint(x.point, route.point)); assert(reference);
      const phaseOffset = source.phaseRad + mirror.phaseRad - TAU / source.wavelengthMm * (d(s, r) - d(source.position, receiver));
      nearField(field([route]), rotate(field([reference]), phaseOffset, Math.sqrt(mirror.reflectance)));
      near(route.ri, reference.ri); near(route.ro, reference.ro); near(route.L, reference.L);
      const physical = route.vertices.slice(1).reduce((sum, p, i) => sum + d(route.vertices[i], p), 0);
      near(route.opticalLengthMm, physical); near(route.opticalExcessMm, physical - d(source.position, receiver));
      near(route.amplitude, reference.amplitude * Math.sqrt(mirror.reflectance));
    }
  }
});

await check('mixed skew routes match independent SciPy UTD-equation reference values', () => {
  const zMirror = q([[-6, -100, -6], [-.1, -100, -6], [-.1, 100, -6], [-6, 100, -6]], 1);
  const local = makeScene([plate, zMirror]), s = {position: [-3, 1.3, -4], wavelengthMm: .05, power: 2.7, phaseRad: 0};
  const routes = evaluateMixedDiffraction(local, compileCoherentSource(local, s), [-2, -.7, 10]).routes;
  // Constants derived independently from SciPy complex-erfc UTD equations.
  // See research/coherent-mesh-mixed-physics-review.md for the derivation.
  for (const [order, point, bounce, ri, ro, L, expected] of [
    ['RD', [0, .3882526681827189, 0], [-2.25, 1.0720631670456797, -6], 8.59251320610425, 10.255939443552053, 4.622771177174412, [.0007533007156798428, .00006881679359350906]],
    ['DR', [0, .9308699344627533, 0], [-.5454545454545454, .4860872250638207, -6], 5.013607184980045, 22.15084054258742, 4.066109176214920, [-.00015792341391640172, -.0004329512058896184]],
  ]) {
    const r = mainRoute(routes, order); assert(r); nearField(field([r]), expected, 3e-13);
    assert(nearPoint(r.point, point)); assert(nearPoint(r.vertices[order === 'RD' ? 1 : 2], bounce));
    near(r.ri, ri); near(r.ro, ro); near(r.L, L);
  }
});

await check('seeded 3-D RD(S,R) equals DR(R,S) without conjugation', () => {
  let seed = 377031, compared = 0;
  const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) | 0; return (seed >>> 0) / 4294967296; };
  for (let trial = 0; trial < 80; trial++) {
    const s = [-8 + 12 * random(), -3 + 6 * random(), -10 + 7 * random()];
    const r = [-8 + 40 * random(), -3 + 6 * random(), 3 + 9 * random()];
    const forward = evaluateMixedDiffraction(scene, compileCoherentSource(scene, {...source, position: s}), r);
    const reverse = evaluateMixedDiffraction(scene, compileCoherentSource(scene, {...source, position: r}), s);
    assert.equal(forward.routes.length, reverse.routes.length);
    for (const route of forward.routes) {
      const opposite = reverse.routes.find(x => x.edgeId === route.edgeId && x.mirrorPlaneId === route.mirrorPlaneId && x.interactionOrder !== route.interactionOrder);
      assert(opposite); nearField(field([route]), field([opposite]));
      route.vertices.forEach((p, i) => assert(nearPoint(p, opposite.vertices.at(-1 - i)))); compared++;
    }
  }
  assert(compared > 150);
});

await check('tessellation seams, split entries and T junctions do not duplicate mixed fields', () => {
  const p = plate.mesh.vertices, m = external.mesh.vertices;
  const variants = [
    [plate, q(m, 1, [[0, 1, 3], [1, 2, 3]])],
    [q(p, 0, [[0, 1, 2]]), q(p, 0, [[0, 2, 3]]), external],
    [q(p, 0, [[0, 1, 2]]), q([...p, [50, 0, 0]], 0, [[0, 4, 3], [4, 2, 3]]), external],
    [plate, q(m, 1, [[0, 1, 2]]), q(m, 1, [[0, 2, 3]])],
  ];
  for (const entries of variants) {
    const candidate = makeScene(entries), actual = evaluateMixedDiffraction(candidate, compileCoherentSource(candidate, source), receiver);
    assert.equal(actual.routes.length, sampleMixed.routes.length); assert.equal(prepareMixedDiffraction(candidate).count, prepareMixedDiffraction(scene).count);
    nearField(field(actual.routes), field(sampleMixed.routes));
    const identities = new Set(actual.routes.map(r => `${r.interactionOrder}/${r.mirrorPlaneId}/${r.edgeId}`)); assert.equal(identities.size, actual.routes.length);
  }
});

function smallBlocker(a, b, f = .5) {
  const p = a.map((v, i) => v + f * (b[i] - v)), delta = b.map((v, i) => Math.abs(v - a[i]));
  const axis = delta.indexOf(Math.max(...delta)), other = [0, 1, 2].filter(i => i !== axis);
  return q([[-1, -1], [1, -1], [1, 1], [-1, 1]].map(sign => p.map((v, i) => v + (i === other[0] ? sign[0] * .03 : i === other[1] ? sign[1] * .03 : 0))), 2);
}
await check('each physical leg can block RD and DR; virtual straight legs cannot', () => {
  for (const order of ['RD', 'DR']) {
    const reference = mainRoute(sampleMixed.routes, order); assert(reference);
    for (let leg = 0; leg < 3; leg++) {
      const blocked = makeScene([plate, external, smallBlocker(reference.vertices[leg], reference.vertices[leg + 1])], [soft, mirror, {type: 'opaque'}]);
      const result = evaluateMixedDiffraction(blocked, compileCoherentSource(blocked, source), receiver);
      assert(!mainRoute(result.routes, order), `${order} physical leg ${leg} remained visible`);
    }
    const a = order === 'RD' ? [-17, 0, -4] : reference.point;
    const b = order === 'RD' ? reference.point : [-25, .3, 10];
    const f = order === 'RD' ? .1 : .8;
    const blockedImage = makeScene([plate, external, smallBlocker(a, b, f)], [soft, mirror, {type: 'opaque'}]);
    const actual = mainRoute(evaluateMixedDiffraction(blockedImage, compileCoherentSource(blockedImage, source), receiver).routes, order);
    assert(actual, `${order} incorrectly tested a virtual segment`); nearField(field([actual]), field([reference]));
  }
});

await check('exact mixed shadow/reflection seams have one GO partner and continuous complex field', () => {
  for (const boundaryPhase of [0, Math.PI]) {
    const testScene = makeScene([plate, external], [{...soft, phaseRad: boundaryPhase}, mirror]);
    for (const reverse of [false, true]) for (const z of [-10, 10]) {
      const original = {...source, phaseRad: .83};
      const values = [];
      for (const delta of [-1e-9, 0, 1e-9]) {
        const target = [42.5 + delta, 0, z];
        const src = reverse ? target : original.position, dst = reverse ? original.position : target;
        const r = evaluateMeshTransport(testScene, compileCoherentSource(testScene, {...original, position: src}), dst, {diffraction: true, mixedDiffraction: true});
        values.push(field(r.routes));
        if (!delta) {
          const seq = z > 0 ? '1' : reverse ? '0,1' : '1,0';
          const partners = r.routes.filter(route => route.kind === 'reflection' && route.sequence.join(',') === seq);
          assert.equal(partners.length, 1); assert.equal(partners[0].boundary, true); assert.equal(partners[0].mixedBoundaryPartner, reverse ? 'DR' : 'RD');
          const geometric = r.routes.filter(route => ['direct', 'reflection'].includes(route.kind)).reduce((sum, route) => sum + route.amplitude ** 2, 0);
          near(r.geometricPower, geometric);
        }
      }
      nearField(values[0], values[2], 2e-9); nearField(values[1], values[0].map((v, i) => .5 * (v + values[2][i])), 1e-10);
    }
  }
});

await check('power, source phase and uniform geometric scale preserve complete mixed fields', () => {
  const scalar = 3;
  const scaledEntries = [plate, external].map(e => ({...e, mesh: {...e.mesh, vertices: e.mesh.vertices.map(p => p.map(x => x * scalar))}}));
  const scaled = makeScene(scaledEntries);
  const reference = field(sampleMixed.routes);
  const actual = evaluateMixedDiffraction(scaled, compileCoherentSource(scaled, {...source, position: source.position.map(x => x * scalar), wavelengthMm: source.wavelengthMm * scalar}), receiver.map(x => x * scalar));
  nearField(field(actual.routes), reference.map(x => x / scalar));
  const shifted = evaluateMixedDiffraction(scene, compileCoherentSource(scene, {...source, power: source.power * 4, phaseRad: source.phaseRad + .7}), receiver);
  nearField(field(shifted.routes), rotate(reference, .7, 2));
});

await check('rigid 3-D rotation and large translation preserve mixed route identity and field', () => {
  const unit = [2 / 3, -2 / 3, 1 / 3], angle = .73, shift = [1000, -2000, 500];
  const transform = p => {
    const projection = p.reduce((sum, v, i) => sum + v * unit[i], 0);
    const cross = [unit[1] * p[2] - unit[2] * p[1], unit[2] * p[0] - unit[0] * p[2], unit[0] * p[1] - unit[1] * p[0]];
    return p.map((v, i) => shift[i] + v * Math.cos(angle) + cross[i] * Math.sin(angle) + unit[i] * projection * (1 - Math.cos(angle)));
  };
  const shiftedScene = makeScene([plate, external].map(e => ({...e, mesh: {...e.mesh, vertices: e.mesh.vertices.map(transform)}})));
  const shifted = evaluateMixedDiffraction(shiftedScene, compileCoherentSource(shiftedScene, {...source, position: transform(source.position)}), transform(receiver));
  assert.equal(shifted.routes.length, sampleMixed.routes.length); nearField(field(shifted.routes), field(sampleMixed.routes), 1e-11);
  for (const r of sampleMixed.routes) {
    const other = shifted.routes.find(x => x.interactionOrder === r.interactionOrder && nearPoint(x.point, transform(r.point)));
    assert(other); near(r.amplitude, other.amplitude); near(r.opticalExcessMm, other.opticalExcessMm);
  }
});

await check('external finite mirror clipping and exact rims never invent mixed paths', () => {
  const original = mainRoute(sampleMixed.routes, 'RD'), point = original.vertices[1];
  const clipped = q([[-10, -1, point[2] + .1], [-10, 1, point[2] + .1], [-10, 1, point[2] + 2], [-10, -1, point[2] + 2]], 1);
  const clippedScene = makeScene([plate, clipped]);
  assert(!mainRoute(evaluateMixedDiffraction(clippedScene, compileCoherentSource(clippedScene, source), receiver).routes, 'RD'));
  const rim = q([[-10, -1, point[2]], [-10, 1, point[2]], [-10, 1, point[2] + 2], [-10, -1, point[2] + 2]], 1);
  const rimScene = makeScene([plate, rim]), r = evaluateMixedDiffraction(rimScene, compileCoherentSource(rimScene, source), receiver);
  assert(!mainRoute(r.routes, 'RD')); assert(r.diagnostics.warnings.includes('mixed-external-mirror-rim-not-modeled'));
});

await check('reflected-endpoint coincidence is harmless for distinct physical endpoints', () => {
  const r = evaluateMeshTransport(scene, compiled, [-17, 0, -4], {diffraction: true, mixedDiffraction: true});
  assert(r.routes.every(route => Number.isFinite(route.amplitude) && Number.isFinite(route.phaseRadReduced)));
});

await check('mixed capacity reserves all families and path controls remove the whole reflected class', () => {
  const capacity = coherentRouteCapacity(scene, {diffraction: true, mixedDiffraction: true});
  assert.equal(capacity.count, scene.routeSequences.length + capacity.diffraction.eligible.length + prepareMixedDiffraction(scene).count);
  const controlled = evaluateMeshTransport(scene, compiled, receiver, {diffraction: true, mixedDiffraction: true, blockReflected: true});
  assert(!controlled.routes.some(r => ['reflection', 'mixed-diffraction'].includes(r.kind)));
  const additional = [-20, -30, -40, -50].map(x => q([[x, -100, -100], [x, 100, -100], [x, 100, 100], [x, -100, 100]], 1));
  const large = makeScene([plate, external, ...additional]);
  assert.throws(() => coherentRouteCapacity(large, {diffraction: true, mixedDiffraction: true}), /65|capacity|families/i);
  const firstOrder = prepareCoherentMeshScene([plate, external], [soft, mirror], {maxReflections: 1});
  assert.throws(() => evaluateMixedDiffraction(firstOrder, compileCoherentSource(firstOrder, source), receiver), /order 2/);
});

const jsonScene = () => ({schema: 'coherent-mesh-scene-v1', units: 'mm', name: 'Independent mixed fixture', materials: [soft, mirror], meshes: [plate, external].map(e => ({vertices: e.mesh.vertices, faces: e.mesh.faces, materialIndex: e.materialIndex, surface: e.surface})), source: {...source, wavelengthMm: .00055}, detector: {center: receiver, u: [1, 0, 0], v: [0, 1, 0], width: 1, height: 1}});
await check('scene import snapshots finite own-property geometry and detector domains', () => {
  const input = structuredClone(jsonScene()), validated = validateCoherentScene(input);
  input.source.position[0] = 123; input.meshes[0].vertices[0][0] = 123;
  assert.equal(validated.source.position[0], -3); assert.equal(validated.meshes[0].vertices[0][0], 0);
  assert(Object.isFrozen(validated) && Object.isFrozen(validated.meshes[0].vertices[0]));
  assert.deepEqual(parseCoherentScene(JSON.stringify(jsonScene())), validated);
  const changes = [
    x => x.source.position[0] = NaN, x => x.source.power = Infinity, x => x.source.position.length = 2,
    x => x.detector.u = [0, 0, 0], x => x.detector.v = [1, 0, 0], x => x.detector.width = 0,
    x => x.detector.center = [10000, 0, 0], x => x.detector.center = [...x.source.position],
    x => x.meshes[0].faces[0][0] = 99, x => x.meshes[0].vertices[0][0] = 10001,
    x => x.meshes[0].faces[0][0] = .5, x => x.meshes[0].materialIndex = 2,
    x => x.materials[0].type = 'diffuse', x => x.units = 'm', x => x.source.position = ['-3', 0, -4],
    x => x.materials.length = 65, x => x.meshes.length = 65,
  ];
  for (const change of changes) { const invalid = structuredClone(jsonScene()); change(invalid); assert.throws(() => validateCoherentScene(invalid)); }
  for (const key of ['__proto__', 'constructor', 'prototype', 'url']) {
    const invalid = JSON.stringify(jsonScene()).replace('"units":"mm"', `"${key}":{},"units":"mm"`);
    assert.throws(() => parseCoherentScene(invalid));
  }
  const inherited = Object.assign(Object.create({source: jsonScene().source}), jsonScene()); assert.throws(() => validateCoherentScene(inherited));
  assert.throws(() => parseCoherentScene(' '.repeat(2 * 1024 * 1024 + 1)));
  const converted = sceneToCoherentData(validated), materialScene = prepareCoherentMeshScene(converted.entries, converted.materials);
  assert.equal(materialScene.triangles.length, 4);
});

await check('custom-scene runtime uses a stable snapshot and honors effective source/detector overrides', async () => {
  const input = structuredClone(jsonScene()), options = {scene: input, resolution: 16, pixelQuadrature: 1, backend: 'cpu', diffraction: true, mixedDiffraction: true};
  const expected = await executeCoherentMeshTask(options);
  const pending = executeCoherentMeshTask(options, {progress(p) { if (p.phase === 'setup') { input.source.position[0] = 456; input.meshes[0].vertices[0][0] = 456; } }});
  const actual = await pending;
  for (const key of ['coherent', 'incoherent', 'geometric', 'crossTerm']) assert.deepEqual(actual[key], expected[key]);
  assert.equal(actual.overview.mesh.triangles.length, 4); assert.equal(actual.configuration.scene.source.position[0], -3);
  assert(actual.diagnostics.reflectionDiffractionRoutes > 0 && actual.diagnostics.diffractionReflectionRoutes > 0);
  const overrides = {source: {...jsonScene().source, position: [-4, 2, -7], phaseRad: .6, power: 3, wavelengthMm: .000489}, detector: {...jsonScene().detector, center: [6, 3, 11], width: 2, height: 3}};
  const prepared = prepareCoherentMeshTask({...options, scene: jsonScene(), ...overrides});
  assert.deepEqual(prepared.source, overrides.source); assert.deepEqual(prepared.detector, overrides.detector);
  near(prepared.configuration.wavelengthNm, 489); assert.equal(prepared.configuration.power, 3);
  assert.throws(() => prepareCoherentMeshTask({...options, scene: jsonScene(), source: {...overrides.source, position: [...jsonScene().detector.center]}}), /contain the point source/);
  assert.throws(() => prepareCoherentMeshTask({...options, scene: jsonScene(), detector: {...overrides.detector, center: [...jsonScene().source.position]}}), /contain the point source/);
  assert.throws(() => prepareCoherentMeshTask({...options, scene: jsonScene(), customOBJ: 'v 0 0 0'}), /replaces|combine/);
});

// Runtime lifecycle probes use the original default fixture too, so they stay
// independent of JSON UI routing and distinguish observer errors from GPU loss.
await check('GPU failure after a committed tile discards it and replays exactly on CPU', async () => {
  const options = {resolution: 16, pixelQuadrature: 1, backend: 'cpu'};
  const expected = await executeCoherentMeshTask(options); let calls = 0, resets = 0, destroyed = 0; const attempts = [];
  const actual = await executeCoherentMeshTask({...options, backend: 'auto'}, {
    gpuFactory: async () => ({backend: {kind: 'webgpu'}, async evaluate({sampleCount}) { calls++; if (calls === 2) throw new Error('injected second tile loss'); return {coherent: new Float32Array(sampleCount).fill(1), incoherent: new Float32Array(sampleCount).fill(2)}; }, destroy() { destroyed++; }}),
    preview: p => attempts.push(p.attempt), reset: () => resets++,
  });
  assert.equal(calls, 2); assert.equal(resets, 1); assert.equal(destroyed, 1); assert.deepEqual(attempts, [0, 1, 1]);
  for (const key of ['coherent', 'incoherent', 'geometric', 'crossTerm']) assert.deepEqual(actual[key], expected[key]);
  assert.equal(actual.diagnostics.attempt, 1); assert.equal(actual.diagnostics.validRoutes, expected.diagnostics.validRoutes);
});

await check('cancellation and caller failure do not become a GPU fallback or complete result', async () => {
  const controller = new AbortController(); let resets = 0, partials = 0;
  await assert.rejects(executeCoherentMeshTask({resolution: 16, pixelQuadrature: 1, backend: 'cpu'}, {signal: controller.signal, preview() { partials++; controller.abort(); }, reset() { resets++; }}), e => e.name === 'AbortError');
  assert.equal(partials, 1); assert.equal(resets, 0);
  let evaluations = 0;
  await assert.rejects(executeCoherentMeshTask({resolution: 16, pixelQuadrature: 1}, {
    gpuFactory: async () => ({backend: {kind: 'webgpu'}, async evaluate({sampleCount}) { evaluations++; return {coherent: new Float32Array(sampleCount), incoherent: new Float32Array(sampleCount)}; }, destroy() {}}),
    preview() { throw new Error('caller rejected preview'); }, reset() { resets++; },
  }), /caller rejected preview/);
  assert.equal(evaluations, 1); assert.equal(resets, 0);
});
console.log(JSON.stringify({scope: 'Independent mixed-family, scene-import and runtime checks; no hardware/browser/full-wave claim', results}, null, 2));
if (results.some(r => !r.pass)) process.exitCode = 1;
