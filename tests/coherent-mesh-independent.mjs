// Independent regressions for physical route identity, phase, and mesh seams.
// Run directly: node tests/coherent-mesh-independent.mjs
import assert from 'node:assert/strict';
import {
  prepareCoherentMeshScene,
  compileCoherentSource,
  evaluateGeometricRoutes,
  sumCoherentRoutes,
  makeCoherentMeshPreset,
  coherentSegmentVisible,
} from '../dist/coherent-mesh-core.js';
import {evaluateMeshDiffraction} from '../dist/coherent-mesh-diffraction.js';
import {makePresetOBJ} from '../dist/mesh-geometry.js';
import {evaluateMeshTransport} from '../dist/coherent-mesh-transport.js';
import {segmentVisible} from '../dist/plt-geometry.js';

const TAU = 2 * Math.PI;
const soft = {type: 'mirror', reflectance: 1, phaseRad: Math.PI};
const vertices = [[-10, 0, -49], [10, 0, -49], [10, 0, 49], [-10, 0, 49]];
const quad = {mesh: {vertices, faces: [[0, 1, 2], [0, 2, 3]]}};
const separatedTriangles = [
  {mesh: {vertices, faces: [[0, 1, 2]]}},
  {mesh: {vertices, faces: [[0, 2, 3]]}},
];
const field = routes => routes.reduce((value, route) => [
  value[0] + route.amplitude * Math.cos(route.phaseRadReduced),
  value[1] + route.amplitude * Math.sin(route.phaseRadReduced),
], [0, 0]);
const near = (a, b, tolerance = 1e-11) => assert(
  Math.abs(a - b) <= tolerance * Math.max(1e-12, Math.abs(a), Math.abs(b)),
  `${a} differs from ${b}`,
);
const nearField = (a, b, absoluteTolerance = 1e-11) => assert(
  Math.hypot(a[0] - b[0], a[1] - b[1]) <= absoluteTolerance,
  `${JSON.stringify(a)} differs from ${JSON.stringify(b)}`,
);

function independentImageField(source, receiver, images, wavelength, power = 1, sourcePhase = 0) {
  // Independent absolute image distances and a receiver-common phase carrier.
  const reference = Math.hypot(...receiver.map((x, i) => x - source[i]));
  return images.reduce((value, image) => {
    const length = Math.hypot(...receiver.map((x, i) => x - image.position[i]));
    const amplitude = Math.sqrt(power / (4 * Math.PI)) * image.amplitude / length;
    const phase = sourcePhase + image.phase - TAU * (length - reference) / wavelength;
    return [value[0] + amplitude * Math.cos(phase), value[1] + amplitude * Math.sin(phase)];
  }, [0, 0]);
}

const results = [];
function check(name, body) {
  try {
    body();
    results.push({name, pass: true});
    console.log(`PASS ${name}`);
  } catch (error) {
    results.push({name, pass: false, error: error.message});
    console.error(`FAIL ${name}: ${error.message}`);
  }
}

check('ordinary direct and lossy-mirror fields use complete-route amplitude and a common phase gauge', () => {
  const material = {type: 'mirror', reflectance: 0.36, phaseRad: 0.7};
  const scene = prepareCoherentMeshScene([quad], [material]);
  const source = {position: [0, 1, -1], wavelengthMm: 0.03, power: 2.7, phaseRad: -1.2};
  const receiver = [1, 3, 2];
  const routes = evaluateGeometricRoutes(scene, compileCoherentSource(scene, source), receiver).routes;
  assert.equal(routes.length, 2);
  const expected = independentImageField(source.position, receiver, [
    {position: source.position, amplitude: 1, phase: 0},
    {position: [0, -1, -1], amplitude: 0.6, phase: material.phaseRad},
  ], source.wavelengthMm, source.power, source.phaseRad);
  nearField(field(routes), expected);
});

check('orthogonal mirror ordering boundary preserves one complete fourth image', () => {
  const preset = makeCoherentMeshPreset('two-mirror');
  const scene = prepareCoherentMeshScene(preset.entries, preset.materials);
  const wavelength = 0.00055;
  const compiled = compileCoherentSource(scene, {...preset.source, wavelengthMm: wavelength});
  const images = [-0.08, 0.08].flatMap(x => [-0.08, 0.08].map(y => ({
    position: [x, y, -50], amplitude: 1,
    phase: (x < 0 ? Math.PI : 0) + (y < 0 ? Math.PI : 0),
  })));
  for (const offset of [0, -1e-10, 1e-10, -1e-8, 1e-8, -1e-5, 1e-5]) {
    const receiver = [2, 2 + offset, 50];
    const routes = evaluateGeometricRoutes(scene, compiled, receiver).routes;
    assert.equal(routes.length, 4, `offset ${offset}: fourth image was lost or duplicated`);
    assert.equal(routes.filter(route => route.sequence.length === 2).length, 1);
    const expected = independentImageField(preset.source.position, receiver, images, wavelength);
    nearField(field(routes), expected, 2e-11);
    near(sumCoherentRoutes(routes).coherent, expected[0] ** 2 + expected[1] ** 2, 2e-8);
  }
});

check('orthogonal limiting paths preserve arbitrary constant complex mirror coefficients', () => {
  const preset = makeCoherentMeshPreset('two-mirror');
  preset.entries[1].materialIndex = 2;
  const materials = [
    {type: 'opaque'},
    {type: 'mirror', reflectance: 0.36, phaseRad: 0.7},
    {type: 'mirror', reflectance: 0.81, phaseRad: -0.3},
  ];
  const scene = prepareCoherentMeshScene(preset.entries, materials);
  const wavelength = 0.00055;
  const compiled = compileCoherentSource(scene, {...preset.source, wavelengthMm: wavelength});
  const images = [-0.08, 0.08].flatMap(x => [-0.08, 0.08].map(y => ({
    position: [x, y, -50],
    amplitude: (x < 0 ? 0.9 : 1) * (y < 0 ? 0.6 : 1),
    phase: (x < 0 ? -0.3 : 0) + (y < 0 ? 0.7 : 0),
  })));
  for (const offset of [-1e-8, 0, 1e-8]) {
    const receiver = [2, 2 + offset, 50];
    const routes = evaluateGeometricRoutes(scene, compiled, receiver).routes;
    assert.equal(routes.length, 4);
    nearField(field(routes), independentImageField(
      preset.source.position, receiver, images, wavelength,
    ), 2e-11);
  }
});

check('the orthogonal corner-limit shortcut excludes the common physical edge endpoint', () => {
  const scene = prepareCoherentMeshScene([
    {mesh: {vertices: [[0, 0, -10], [10, 0, -10], [10, 0, 0], [0, 0, 0]],
      faces: [[0, 1, 2], [0, 2, 3]]}},
    {mesh: {vertices: [[0, 0, -10], [0, 10, -10], [0, 10, 0], [0, 0, 0]],
      faces: [[0, 1, 2], [0, 2, 3]]}},
  ], [soft]);
  const compiled = compileCoherentSource(scene, {position: [1, 1, -1], wavelengthMm: 0.005});
  const endpoint = evaluateGeometricRoutes(scene, compiled, [1, 1, 1]).routes;
  assert(!endpoint.some(route => route.orthogonalCornerImageLimit));
  const interior = evaluateGeometricRoutes(scene, compiled, [1, 1, 0.9998]).routes;
  assert.equal(interior.filter(route => route.orthogonalCornerImageLimit).length, 1);
});

check('coplanar triangle entries preserve the single physical GO mirror field', () => {
  const source = {position: [0, 1, -1], wavelengthMm: 0.005};
  const receiver = [0, 1, 1];
  const values = [[quad], separatedTriangles].map(entries => {
    const scene = prepareCoherentMeshScene(entries, [soft]);
    const routes = evaluateGeometricRoutes(scene, compileCoherentSource(scene, source), receiver).routes;
    assert.equal(routes.length, 2);
    assert.equal(routes[1].boundary, false);
    return field(routes);
  });
  nearField(values[0], values[1]);
});

check('accepted narrow triangles retain an interior specular reflection', () => {
  for (const width of [0.001, 0.0001, 0.00001]) {
    const triangleVertices = [[0, 0, 0], [1000, 0, 1000], [1000 + width, 0, 1000 - width]];
    const centre = [(2000 + width) / 3, 0, (2000 - width) / 3];
    const source = [centre[0] - 0.2, 1, centre[2] - 0.2];
    const receiver = [centre[0] + 0.2, 1, centre[2] + 0.2];
    const scene = prepareCoherentMeshScene([
      {mesh: {vertices: triangleVertices, faces: [[0, 1, 2]]}},
    ], [soft]);
    const routes = evaluateGeometricRoutes(scene, compileCoherentSource(scene, {
      position: source, wavelengthMm: 0.005,
    }), receiver).routes;
    assert.equal(routes.length, 2, `width ${width}: accepted triangle lost its reflection`);
    assert.equal(routes[1].boundary, false);
    const expectedLength = Math.hypot(receiver[0] - source[0], 2, receiver[2] - source[2]);
    near(routes[1].amplitude, 1 / (Math.sqrt(4 * Math.PI) * expectedLength), 1e-10);
  }
});

check('coplanar triangle entries cannot create physical UTD edges or GO seam metadata', () => {
  for (const receiver of [[0, 1, 1], [1, 3, 2], [0, -1, 1]]) {
    const values = [[quad], separatedTriangles].map(entries => {
      const scene = prepareCoherentMeshScene(entries, [soft]);
      return evaluateMeshDiffraction(scene, [0, 1, -1], receiver, {wavelengthMm: 0.005});
    });
    assert.equal(values[1].diagnostics.valid, true);
    assert.equal(values[1].directBoundary, values[0].directBoundary, 'artificial direct boundary');
    assert.equal(values[1].reflectionBoundaries.length, values[0].reflectionBoundaries.length,
      'artificial reflection boundary');
    nearField(field(values[0].routes), field(values[1].routes));
  }
});

check('overlaid alternate tessellation is rejected or does not duplicate the UTD field', () => {
  const second = {mesh: {vertices, faces: [[0, 1, 3], [1, 2, 3]]}};
  const evaluate = entries => evaluateMeshDiffraction(
    prepareCoherentMeshScene(entries, [soft]), [0, 1, -1], [1, 3, 2], {wavelengthMm: 0.005},
  );
  const original = evaluate([quad]);
  let overlay;
  try {
    overlay = evaluate([quad, second]);
  } catch (error) {
    assert.match(error.message, /overlap|duplicate|coincident|topology/i);
    return;
  }
  if (!overlay.diagnostics.valid) return;
  nearField(field(original.routes), field(overlay.routes));
});

check('physically equivalent mirror phases 0 and plus-or-minus 2π admit identical hard diffraction', () => {
  const outcomes = [0, TAU, -TAU].map(phaseRad => evaluateMeshDiffraction(
    prepareCoherentMeshScene([quad], [{type: 'mirror', reflectance: 1, phaseRad}]),
    [0, 1, -1], [1, 3, 2], {wavelengthMm: 0.005},
  ));
  assert(outcomes[0].routes.length > 0);
  for (const outcome of outcomes.slice(1)) {
    assert.equal(outcome.routes.length, outcomes[0].routes.length);
    assert.equal(outcome.diagnostics.materialExcludedEdges, 0);
    nearField(field(outcome.routes), field(outcomes[0].routes));
  }
});

check('ordinary opaque blockers remain occluders for every admitted route family', () => {
  const blocker = {mesh: {
    vertices: [[-20, -20, 0], [20, -20, 0], [20, 20, 0], [-20, 20, 0]],
    faces: [[0, 1, 2], [0, 2, 3]],
  }, materialIndex: 1};
  const scene = prepareCoherentMeshScene([quad, blocker], [soft, {type: 'opaque'}]);
  const source = {position: [0, 1, -1], wavelengthMm: 0.005};
  const receiver = [1, 3, 2];
  assert.equal(evaluateGeometricRoutes(scene, compileCoherentSource(scene, source), receiver).routes.length, 0);
  const diffracted = evaluateMeshDiffraction(scene, source.position, receiver, source);
  assert.equal(diffracted.routes.length, 0);
  assert.equal(diffracted.directBoundary, false);
  assert.equal(diffracted.reflectionBoundaries.length, 0);
  const coplanarSquare = (radius, materialIndex) => ({materialIndex, mesh: {
    vertices: [[-radius, 0, -radius], [radius, 0, -radius],
      [radius, 0, radius], [-radius, 0, radius]],
    faces: [[0, 1, 2], [0, 2, 3]],
  }});
  // A triangle tessellation seam does not make an opaque overlay disappear.
  for (const x of [0, 0.1]) assert.throws(() => {
    const overlap = prepareCoherentMeshScene([coplanarSquare(2, 0), coplanarSquare(1, 1)],
      [soft, {type: 'opaque'}]);
    evaluateGeometricRoutes(overlap, compileCoherentSource(overlap, {
      position: [x, 1, -0.5], wavelengthMm: 0.005,
    }), [x, 1, 0.5]);
  }, /coplanar|overlap|boundary/i);
});

check('conditioned coherent BVH visibility agrees with the ordinary opaque scene oracle', () => {
  const scene = prepareCoherentMeshScene([{mesh: makePresetOBJ('box')}], [{type: 'opaque'}]);
  let seed = 138711;
  const random = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) | 0;
    return (seed >>> 0) / 4294967296;
  };
  let blocked = 0;
  for (let trial = 0; trial < 1000; trial++) {
    const source = Array.from({length: 3}, () => 10 * (random() - 0.5));
    const receiver = Array.from({length: 3}, () => 10 * (random() - 0.5));
    const expected = segmentVisible(scene, source, receiver);
    assert.equal(coherentSegmentVisible(scene, source, receiver), expected);
    if (!expected) blocked++;
  }
  assert(blocked > 50);
});

check('convex solid-wedge fields are reciprocal for seeded 3-D source/receiver pairs', () => {
  let seed = 72913;
  const random = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) | 0;
    return (seed >>> 0) / 4294967296;
  };
  let compared = 0;
  for (const name of ['box', 'tetrahedron', 'triangular-prism']) {
    const scene = prepareCoherentMeshScene([{mesh: makePresetOBJ(name)}], [soft]);
    for (let trial = 0; trial < 100; trial++) {
      const source = Array.from({length: 3}, () => 10 * (random() - 0.5));
      const receiver = Array.from({length: 3}, () => 10 * (random() - 0.5));
      const forward = evaluateMeshDiffraction(scene, source, receiver, {wavelengthMm: 0.01});
      const reverse = evaluateMeshDiffraction(scene, receiver, source, {wavelengthMm: 0.01});
      const byEdge = new Map(reverse.routes.map(route => [route.edgeId, route]));
      assert.equal(forward.routes.length, reverse.routes.length);
      for (const route of forward.routes) {
        assert(byEdge.has(route.edgeId));
        nearField(field([route]), field([byEdge.get(route.edgeId)]), 2e-11);
        compared++;
      }
    }
  }
  assert(compared > 200, 'solid-wedge cases did not exercise enough admitted edges');
});

const halfPlate = {mesh: {
  vertices: [[0, -100, 0], [100, -100, 0], [100, 100, 0], [0, 100, 0]],
  faces: [[0, 1, 2], [0, 2, 3]],
}};

check('assembled transport shares one direct and reflection seam convention', () => {
  for (const phaseRad of [0, Math.PI]) {
    const scene = prepareCoherentMeshScene([halfPlate], [{...soft, phaseRad}]);
    const compiled = compileCoherentSource(scene, {position: [-3, 0, -4], wavelengthMm: 0.05});
    for (const z of [-10, 10]) for (const gap of [1e-8, 1e-10, 1e-12, 1e-13]) {
      const values = [-gap, 0, gap].map(offset => field(evaluateMeshTransport(
        scene, compiled, [7.5 + offset, 0, z], {diffraction: true},
      ).routes));
      nearField(values[0], values[2], 2e-8);
      nearField(values[1], values[0].map((x, i) => (x + values[2][i]) / 2), 2e-9);
    }
  }
});

check('assembled geometric control includes restored half-weight direct routes', () => {
  const scene = prepareCoherentMeshScene([halfPlate], [soft]);
  const compiled = compileCoherentSource(scene, {position: [-3, 0, -4], wavelengthMm: 0.05});
  const result = evaluateMeshTransport(scene, compiled, [7.5, 0, 10], {diffraction: true});
  const expected = result.routes.filter(route => route.kind !== 'diffraction')
    .reduce((total, route) => total + route.amplitude ** 2, 0);
  assert(expected > 0);
  near(result.geometricPower, expected);
});

check('assembled diffraction shares nonzero source phase with geometric fields', () => {
  const scene = prepareCoherentMeshScene([halfPlate], [soft]);
  const phase = 1.13;
  const values = [0, phase].map(phaseRad => field(evaluateMeshTransport(scene,
    compileCoherentSource(scene, {position: [-3, 0, -4], wavelengthMm: 0.05, phaseRad}),
    [5, 0.3, 10], {diffraction: true},
  ).routes));
  nearField(values[1], [
    values[0][0] * Math.cos(phase) - values[0][1] * Math.sin(phase),
    values[0][0] * Math.sin(phase) + values[0][1] * Math.cos(phase),
  ]);
});

console.log(JSON.stringify({scope: 'independent physical route and topology regression', results}, null, 2));
if (results.some(result => !result.pass)) process.exitCode = 1;
