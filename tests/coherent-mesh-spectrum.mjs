import assert from 'node:assert/strict';
import fs from 'node:fs';
import {validateCoherentSpectrum, prepareCoherentSpectrum} from '../dist/coherent-mesh-spectrum.js';
const close = (a, b, tolerance = 1e-12) => assert(Math.abs(a - b) <= tolerance, `${a} != ${b} (tolerance ${tolerance})`);
let checks = 0;
function test(name, fn) { fn(); checks++; console.log('PASS', name); }
const plan = (spectrum, options) => prepareCoherentSpectrum({spectrum, wavelengthMm: .00055, power: 13}, options);

test('Strict spectral descriptors snapshot, normalize, merge duplicates and remain idempotent', () => {
  assert.equal(validateCoherentSpectrum(), null); assert.equal(validateCoherentSpectrum(null), null);
  for (const descriptor of [{mode: 'monochrome'}, {mode: 'broadband', illuminant: 'D65'}, {mode: 'gaussian', centerNm: 360, fwhmNm: 470}, {mode: 'lines', lines: [{wavelengthNm: 600, powerFraction: 2}, {wavelengthNm: 450, powerFraction: 1}, {wavelengthNm: 600, powerFraction: 1}, {wavelengthNm: 550, powerFraction: 0}]}]) {
    const canonical = validateCoherentSpectrum(descriptor); assert(Object.isFrozen(canonical)); assert.deepEqual(validateCoherentSpectrum(canonical), canonical);
  }
  const merged = validateCoherentSpectrum({mode: 'lines', lines: [{wavelengthNm: 600, powerFraction: 2}, {wavelengthNm: 450, powerFraction: 1}, {wavelengthNm: 600, powerFraction: 1}]});
  assert.deepEqual(merged.lines, [{wavelengthNm: 450, powerFraction: .25}, {wavelengthNm: 600, powerFraction: .75}]);
  const huge = validateCoherentSpectrum({mode: 'lines', lines: [{wavelengthNm: 400, powerFraction: Number.MAX_VALUE}, {wavelengthNm: 600, powerFraction: Number.MAX_VALUE}]});
  assert.deepEqual(huge.lines.map(x => x.powerFraction), [.5, .5]);
  const many = validateCoherentSpectrum({mode: 'lines', lines: Array.from({length: 128}, (_, i) => ({wavelengthNm: 360 + 3 * i, powerFraction: i * i + 1}))});
  assert.deepEqual(validateCoherentSpectrum(many), many); close(many.lines.reduce((s, x) => s + x.powerFraction, 0), 1, 2e-15);
});
test('Malformed, sparse, nonfinite, extra-field and unsupported source inputs reject', () => {
  for (const value of [[], '', 1, {}, {mode: 'RGB'}, {mode: 'monochrome', illuminant: 'E'}, {mode: 'broadband', illuminant: 'bad'}, {mode: 'gaussian', centerNm: 350, fwhmNm: 1}, {mode: 'gaussian', centerNm: 550, fwhmNm: 0}, {mode: 'gaussian', centerNm: 550, fwhmNm: Infinity}, {mode: 'lines', lines: []}, {mode: 'lines', lines: Array(3)}, {mode: 'lines', lines: Array.from({length: 129}, () => ({wavelengthNm: 500, powerFraction: 1}))}, {mode: 'lines', lines: [{wavelengthNm: 500, powerFraction: 0}]}, {mode: 'lines', lines: [{wavelengthNm: 500, powerFraction: -1}]}, {mode: 'lines', lines: [{wavelengthNm: 500, powerFraction: NaN}]}]) assert.throws(() => validateCoherentSpectrum(value));
  assert.throws(() => validateCoherentSpectrum(Object.create({mode: 'monochrome'})));
  assert.throws(() => plan(null, {seed: -1})); assert.throws(() => plan(null, {spectralSamples: 63}));
  assert.throws(() => plan({mode: 'gaussian', centerNm: 550, fwhmNm: Number.MIN_VALUE}));
});
test('Legacy mono and one line are exact with unchanged source wavelength and unit power', () => {
  const source = {wavelengthMm: .0006328, power: 17}, p = prepareCoherentSpectrum(source), explicit = prepareCoherentSpectrum({...source, spectrum: {mode: 'monochrome'}});
  assert(p.exact); assert(!p.stochastic); assert.equal(p.samples, 1); assert.deepEqual(p.sampleNodes(0), explicit.sampleNodes(900));
  assert.equal(p.sampleNodes(0)[0].wavelengthMm, source.wavelengthMm); assert.equal(p.sampleNodes(0)[0].weight, 1);
  const line = plan({mode: 'lines', lines: [{wavelengthNm: 632.8, powerFraction: 999}]}).sampleNodes(0)[0];
  close(line.wavelengthMm, source.wavelengthMm); assert.deepEqual(line.xyz, p.sampleNodes(0)[0].xyz);
});
test('Exact multiple lines preserve weighted intensities and never combine different-frequency fields', () => {
  const p = plan({mode: 'lines', lines: [{wavelengthNm: 450, powerFraction: 1}, {wavelengthNm: 650, powerFraction: 3}]});
  const n = p.sampleNodes(0); assert.deepEqual(n, p.sampleNodes(0xffffffff)); assert(p.exact); assert.equal(n.reduce((s, n) => s + n.weight, 0), 1);
  const powers = [4, 9], measured = n.reduce((s, x, i) => s + x.weight * powers[i], 0); close(measured, 7.75);
  assert.notEqual(measured, (Math.sqrt(.25 * powers[0]) + Math.sqrt(.75 * powers[1])) ** 2);
});

const sources = [{mode: 'broadband', illuminant: 'E'}, {mode: 'broadband', illuminant: 'D65'}, {mode: 'broadband', illuminant: 'blackbody3000'}, {mode: 'gaussian', centerNm: 550, fwhmNm: 10}, {mode: 'gaussian', centerNm: 360, fwhmNm: .001}, {mode: 'gaussian', centerNm: 830, fwhmNm: 100}];
test('Source CDF inversion is monotonic and normalized, including narrow truncated Gaussian', () => {
  for (const descriptor of sources) {
    const p = plan(descriptor); close(p.cdf(360), 0); close(p.cdf(830), 1); let previous = 0;
    for (let i = 0; i <= 1024; i++) { const u = i / 1024, wavelength = p.quantile(u); assert(wavelength >= previous); previous = wavelength; close(p.cdf(wavelength), u, descriptor.fwhmNm === .001 ? 1e-9 : 2e-12); }
  }
  const p = plan({mode: 'gaussian', centerNm: 550, fwhmNm: .001}), sigma = .001 / Math.sqrt(8 * Math.log(2));
  close(p.quantile(.5), 550, 2e-12); close(p.cdf(550 + sigma), .8413447460685429, 4e-7); close(p.cdf(550 + 2 * sigma), .9772498680518208, 2e-7);
});
test('Every detector sample independently covers all CDF strata, with deterministic scheduling', () => {
  for (const descriptor of sources) for (const spectralSamples of [16, 32, 64, 128]) {
    const p = plan(descriptor, {spectralSamples, seed: 0xffffffff});
    for (const sampleIndex of [0, 17, 0xffffffff]) {
      const nodes = p.sampleNodes(sampleIndex); assert.deepEqual(nodes, p.sampleNodes(sampleIndex)); assert.equal(nodes.length, spectralSamples); close(nodes.reduce((s, x) => s + x.weight, 0), 1, 0);
      nodes.forEach((node, i) => { const u = p.cdf(node.wavelengthNm); assert(u >= i / spectralSamples - 2e-9 && u <= (i + 1) / spectralSamples + 2e-9); assert.equal(node.weight, 1 / spectralSamples); for (let c = 0; c < 3; c++) close(node.xyzWeight[c], node.weight * node.xyz[c], 0); });
      if (sampleIndex) assert.notDeepEqual(nodes.map(n => n.wavelengthNm), p.sampleNodes(0).map(n => n.wavelengthNm));
    }
    assert.throws(() => p.sampleNodes(-1)); assert.throws(() => p.sampleNodes(.5));
  }
});
test('Allocation-free GPU packing matches source nodes, raw CIE and split-precision cycles', () => {
  for (const descriptor of [null, {mode: 'lines', lines: [{wavelengthNm: 450, powerFraction: 1}, {wavelengthNm: 650, powerFraction: 3}]}, ...sources]) {
    const p = plan(descriptor), target = new Float32Array(p.samples * 8 + 3).fill(-1), nodes = p.sampleNodes(77);
    assert.equal(p.writePackedNodes(77, target, 2), p.samples); assert.equal(target[0], -1); assert.equal(target[1], -1); assert.equal(target.at(-1), -1);
    nodes.forEach((n, i) => { const at = 2 + i * 8, cycles = 1e6 / n.wavelengthNm; assert.equal(target[at], Math.fround(cycles)); assert.equal(target[at + 1], Math.fround(cycles - Math.fround(cycles))); close(target[at] + target[at + 1], cycles, 4e-12); assert.equal(target[at + 2], Math.fround(2 * Math.PI * cycles)); assert.equal(target[at + 3], Math.fround(n.weight)); for (let c = 0; c < 3; c++) assert.equal(target[at + 4 + c], Math.fround(n.xyz[c])); assert.equal(target[at + 7], 0); });
    assert.throws(() => p.writePackedNodes(0, new Float32Array(1))); assert.throws(() => p.writePackedNodes(0, target, -1)); assert.throws(() => p.writePackedNodes(0, new Float64Array(target.length)));
  }
});

const csv = name => fs.readFileSync(new URL(`../research/spectral-data/${name}`, import.meta.url), 'utf8').trim().split(/\r?\n/).map(line => line.split(',').map(Number));
const cmf = csv('CIE_xyz_1931_2deg.csv'), d65 = csv('CIE_std_illum_D65.csv');
function lookup(rows, nm) { const at = Math.min(rows.length - 2, Math.floor(nm - rows[0][0])), f = nm - rows[at][0]; return rows[at].slice(1).map((v, c) => v * (1 - f) + rows[at + 1][c + 1] * f); }
const planck = nm => { const l = nm * 1e-9; return 1 / (l ** 5 * Math.expm1(6.62607015e-34 * 299792458 / (l * 1.380649e-23 * 3000))); };
function denseXYZ(illuminant) {
  const xyz = [0, 0, 0]; let mass = 0; const step = .05;
  for (let i = 0; i < 9400; i++) { const nm = 360 + (i + .5) * step, s = illuminant === 'E' ? 1 : illuminant === 'D65' ? lookup(d65, nm)[0] : planck(nm), c = lookup(cmf, nm); mass += s; for (let k = 0; k < 3; k++) xyz[k] += s * c[k]; }
  return xyz.map(v => v / mass);
}
test('Raw CIE nodes match official CSV and no hidden luminance/source-power normalization exists', () => {
  const p = plan({mode: 'broadband', illuminant: 'D65'});
  for (const n of p.sampleNodes(51)) { const expected = lookup(cmf, n.wavelengthNm); n.xyz.forEach((v, c) => close(v, expected[c], 4e-16)); }
  const a = prepareCoherentSpectrum({power: 1, spectrum: sources[0]}), b = prepareCoherentSpectrum({power: 100, spectrum: sources[0]}); assert.deepEqual(a.sampleNodes(9), b.sampleNodes(9));
  const equal = denseXYZ('E'); close(equal[1], 106.8569149167669 / 470, 2e-9); assert(equal[1] < .23);
});
test('Reproducible stratified ensembles converge to independent official-data/Planck XYZ integrals', () => {
  for (const illuminant of ['E', 'D65', 'blackbody3000']) {
    const p = plan({mode: 'broadband', illuminant}), result = [0, 0, 0], expected = denseXYZ(illuminant), count = 4096;
    for (let pixel = 0; pixel < count; pixel++) for (const n of p.sampleNodes(pixel)) for (let c = 0; c < 3; c++) result[c] += n.xyzWeight[c] / count;
    result.forEach((v, c) => close(v, expected[c], 3e-4));
    console.log('  XYZ', illuminant, result, 'dense', expected);
  }
});
test('Long optical delays become disclosed sampling noise with falling ensemble RMS, not grid fringes', () => {
  const count = 2048, differenceNm = 25e6, rms = [], means = [];
  for (const spectralSamples of [16, 128]) {
    const p = plan({mode: 'broadband', illuminant: 'E'}, {spectralSamples, seed: 81173}); let mean = 0, squared = 0;
    for (let pixel = 0; pixel < count; pixel++) { let crossTerm = 0; for (const n of p.sampleNodes(pixel)) crossTerm += 2 * n.weight * Math.cos(2 * Math.PI * differenceNm / n.wavelengthNm); mean += crossTerm / count; squared += crossTerm ** 2 / count; }
    means.push(mean); rms.push(Math.sqrt(squared)); assert(Math.abs(mean) < .025);
  }
  assert(rms[1] < .5 * rms[0]); console.log('  long-OPD cross-term mean', means, 'RMS16/128', rms);
});
console.log(`Coherent spectral source: ${checks} checks passed`);
