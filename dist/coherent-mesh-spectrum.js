/** Spectral measure for the deterministic, complete-route coherent engine.
 * A node is ONE wavelength: sum its complex routes, square, then apply weight.
 * Never sum complex fields across these nodes. Source power is already in the
 * route amplitudes; the weights here sum to one and do not contain that power.
 * Continuous sources use independently jittered, equal-probability strata.
 * Their result has spectral sampling noise, not deterministic convergence.
 */
import {cieXYZ, sourceSpectrum, hashSpectralIndex} from './plt-spectrum.js';
import {CIE_START_NM, CIE_XYZ_1NM} from './plt-cie.js';

export const COHERENT_SPECTRAL_SAMPLES = Object.freeze([16, 32, 64, 128]);
export const COHERENT_SPECTRAL_SOURCES = Object.freeze(['D65', 'E', 'blackbody3000']);
export const COHERENT_SPECTRAL_SCOPE = '360–830 nm; source power integrated over the visible band; CIE 1931 2° XYZ, K=1; complete-field intensity sampled within each wavelength; no cross-wavelength amplitude sum';
const MIN = 360, MAX = 830, UINT32 = 0xffffffff;
const freeze = value => { if (value && typeof value === 'object') { for (const v of Object.values(value)) freeze(v); Object.freeze(value); } return value; };
function object(value, keys, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value) || ![Object.prototype, null].includes(Object.getPrototypeOf(value))) throw new TypeError(`${label} must be a plain object`);
  for (const k of Object.keys(value)) if (!keys.includes(k)) throw new Error(`${label}: unsupported field ${k}`);
}
function finite(value, min, max, label) { if (!Number.isFinite(value) || value < min || value > max) throw new RangeError(`Invalid ${label}`); return value; }
function index(value, label) { if (!Number.isInteger(value) || value < 0 || value > UINT32) throw new RangeError(`Invalid spectral ${label}`); return value; }

/** Snapshot and canonicalize the physical source, independent of run quality.
 * Duplicate spectral lines are the SAME frequency and become one power bin.
 * Input line weights are relative powers; canonical powerFraction sums to one.
 */
export function validateCoherentSpectrum(value) {
  if (value === null || value === undefined) return null;
  object(value, ['mode', 'illuminant', 'centerNm', 'fwhmNm', 'lines'], 'spectrum');
  if (value.mode === 'monochrome') { object(value, ['mode'], 'spectrum'); return freeze({mode: 'monochrome'}); }
  if (value.mode === 'broadband') {
    object(value, ['mode', 'illuminant'], 'spectrum');
    if (!COHERENT_SPECTRAL_SOURCES.includes(value.illuminant)) throw new RangeError('Invalid spectral illuminant');
    return freeze({mode: value.mode, illuminant: value.illuminant});
  }
  if (value.mode === 'gaussian') {
    object(value, ['mode', 'centerNm', 'fwhmNm'], 'spectrum');
    finite(value.centerNm, MIN, MAX, 'Gaussian centerNm'); finite(value.fwhmNm, 0, 470, 'Gaussian fwhmNm');
    if (!(value.fwhmNm > 0)) throw new RangeError('Gaussian fwhmNm must be positive; use monochrome for a spectral line');
    const sigma = value.fwhmNm / Math.sqrt(8 * Math.log(2));
    if (!(sigma > 0) || value.centerNm + sigma === value.centerNm || value.centerNm - sigma === value.centerNm) throw new RangeError('Gaussian bandwidth is below floating-point wavelength resolution');
    return freeze({mode: value.mode, centerNm: value.centerNm, fwhmNm: value.fwhmNm});
  }
  if (value.mode === 'lines') {
    object(value, ['mode', 'lines'], 'spectrum');
    if (!Array.isArray(value.lines) || !value.lines.length || value.lines.length > 128) throw new RangeError('Spectrum requires 1–128 lines');
    const input = Array.from(value.lines, line => { object(line, ['wavelengthNm', 'powerFraction'], 'spectral line'); return {wavelengthNm: finite(line.wavelengthNm, MIN, MAX, 'line wavelengthNm'), powerFraction: finite(line.powerFraction, 0, Number.MAX_VALUE, 'line powerFraction')}; });
    const maximum = Math.max(...input.map(line => line.powerFraction));
    if (!(maximum > 0)) throw new RangeError('Spectral lines require positive total power');
    // Scaling first avoids overflow for legitimate finite relative weights.
    const combined = new Map();
    for (const line of input) if (line.powerFraction > 0) combined.set(line.wavelengthNm, (combined.get(line.wavelengthNm) ?? 0) + line.powerFraction / maximum);
    const total = [...combined.values()].reduce((s, v) => s + v, 0);
    const lines = [...combined].sort((a, b) => a[0] - b[0]).map(([wavelengthNm, weight]) => ({wavelengthNm, powerFraction: weight / total})).filter(line => line.powerFraction > 0);
    // Preserve canonical input byte-for-byte when its already-normalized powers
    // are distinct. This makes repeated scene validation idempotent.
    const originalTotal = input.reduce((s, line) => s + line.powerFraction, 0);
    if (input.length === lines.length && Math.abs(originalTotal - 1) <= 8 * Number.EPSILON * input.length) for (const line of lines) line.powerFraction = input.find(v => v.wavelengthNm === line.wavelengthNm).powerFraction;
    return freeze({mode: value.mode, lines});
  }
  throw new RangeError('Invalid coherent spectral mode');
}

/** Piecewise-linear SPD integrated exactly in its own coordinate. Gaussian
 * coordinates are standardized sigma units, resolving widths far below 1 nm.
 */
function distribution(descriptor) {
  let coordinate = x => x, inverseCoordinate = x => x, density, lo = MIN, hi = MAX, divisions;
  let approximation;
  if (descriptor.mode === 'gaussian') {
    const sigma = descriptor.fwhmNm / Math.sqrt(8 * Math.log(2)), center = descriptor.centerNm;
    if (!(sigma > 0) || center + sigma === center || center - sigma === center) throw new RangeError('Gaussian bandwidth is below floating-point wavelength resolution');
    lo = Math.max(-12, (MIN - center) / sigma); hi = Math.min(12, (MAX - center) / sigma);
    coordinate = x => center + sigma * x; inverseCoordinate = nm => (nm - center) / sigma;
    density = x => Math.exp(-.5 * x * x); divisions = Math.max(1, Math.ceil((hi - lo) * 256));
    approximation = {kind: 'piecewise-linear Gaussian in sigma coordinates',maxStepSigma: 1 / 256, omittedGaussianTailMassUpperBound: 4e-33, visibleBandTruncated: true};
  } else {
    density = nm => sourceSpectrum(nm, descriptor.illuminant);
    divisions = descriptor.illuminant === 'E' ? 1 : descriptor.illuminant === 'D65' ? 470 : 3760;
    approximation = {kind: descriptor.illuminant === 'blackbody3000' ? 'piecewise-linear 3000 K Planck spectrum' : 'exact piecewise-linear published source', maxStepNm: (hi - lo) / divisions, visibleBandTruncated: true};
  }
  const x = new Float64Array(divisions + 1), y = new Float64Array(divisions + 1), cumulative = new Float64Array(divisions + 1);
  for (let i = 0; i <= divisions; i++) { x[i] = lo + (hi - lo) * i / divisions; y[i] = density(x[i]); if (i) cumulative[i] = cumulative[i - 1] + (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2; }
  const area = cumulative[divisions];
  if (!(area > 0 && Number.isFinite(area))) throw new Error('Invalid integrated source spectrum');
  function intervalFor(value, values) { let a = 0, b = divisions; while (b - a > 1) { const mid = (a + b) >>> 1; if (values[mid] <= value) a = mid; else b = mid; } return a; }
  function quantile(u) {
    finite(u, 0, 1, 'source quantile');
    if (u === 0) return Math.max(MIN, coordinate(lo)); if (u === 1) return Math.min(MAX, coordinate(hi));
    const target = u * area, i = intervalFor(target, cumulative), mass = cumulative[i + 1] - cumulative[i];
    if (!(mass > 0)) throw new Error('Unresolved source CDF interval');
    const f = Math.max(0, Math.min(1, (target - cumulative[i]) / mass)), a = y[i], b = y[i + 1];
    // Stable quadratic inverse for an affine density. No subtraction of roots.
    const root = Math.sqrt(Math.max(0, a * a * (1 - f) + b * b * f));
    const t = f === 0 ? 0 : f * (a + b) / (a + root);
    return Math.max(MIN, Math.min(MAX, coordinate(x[i] + t * (x[i + 1] - x[i]))));
  }
  function cdf(nm) {
    finite(nm, MIN, MAX, 'source CDF wavelength'); const v = inverseCoordinate(nm);
    if (v <= lo) return 0; if (v >= hi) return 1;
    const i = intervalFor(v, x), d = v - x[i], h = x[i + 1] - x[i];
    return Math.max(0, Math.min(1, (cumulative[i] + d * (y[i] + .5 * (y[i + 1] - y[i]) * d / h)) / area));
  }
  return {quantile, cdf, approximation: freeze({...approximation, knots: divisions + 1}), supportNm: Object.freeze([Math.max(MIN, coordinate(lo)), Math.min(MAX, coordinate(hi))])};
}

function node(wavelengthNm, weight) { const xyz = cieXYZ(wavelengthNm); return {wavelengthNm, wavelengthMm: wavelengthNm / 1e6, weight, xyz, xyzWeight: xyz.map(v => v * weight)}; }

/** Canonical call: prepareCoherentSpectrum(source, {spectralSamples, seed}).
 * The descriptor/options overload is retained for standalone module callers.
 * sampleIndex is the global detector spatial-subsample index, never tile index.
 */
export function prepareCoherentSpectrum(source = {}, options = {}) {
  const descriptorInput = source?.mode ? source : source?.spectrum;
  const descriptor = validateCoherentSpectrum(descriptorInput);
  const wavelengthNm = finite(options.wavelengthNm ?? (source?.wavelengthMm === undefined ? 550 : source.wavelengthMm * 1e6), MIN, MAX, 'monochrome wavelengthNm');
  const samples = options.spectralSamples ?? options.samples ?? 64, seed = index(options.seed ?? 7391, 'seed');
  if (!COHERENT_SPECTRAL_SAMPLES.includes(samples)) throw new RangeError('spectralSamples must be 16, 32, 64 or 128');
  const mode = descriptor?.mode ?? 'monochrome', exact = mode === 'monochrome' || mode === 'lines';
  const fixedNode = mode === 'monochrome' ? node(wavelengthNm, 1) : null;
  if (fixedNode && options.wavelengthNm === undefined && source?.wavelengthMm !== undefined) fixedNode.wavelengthMm = source.wavelengthMm;
  const fixed = exact ? freeze(mode === 'lines' ? descriptor.lines.map(line => node(line.wavelengthNm, line.powerFraction)) : [fixedNode]) : null;
  const d = exact ? null : distribution(descriptor);
  const drawWavelength = (sampleIndex, stratum) => {
    const jitter = (hashSpectralIndex(seed, sampleIndex, stratum, 0x53504543) + .5) / 4294967296;
    return d.quantile((stratum + jitter) / samples);
  };
  const sampleNodes = sampleIndex => {
    index(sampleIndex, 'detector sample index');
    if (fixed) return fixed;
    return Array.from({length: samples}, (_, stratum) => {
      // Every stratum gets its own pseudorandom jitter. There is no single grid
      // offset whose harmonics could retain a long-OPD spectral resonance.
      return node(drawWavelength(sampleIndex, stratum), 1 / samples);
    });
  };
  function pack(target, offset, nm, weight) {
    const cycles = 1e6 / nm, high = Math.fround(cycles), p = nm - CIE_START_NM, i = Math.min(Math.floor(p), CIE_XYZ_1NM.length - 2), fraction = p - i;
    const a = CIE_XYZ_1NM[i], b = CIE_XYZ_1NM[i + 1];
    target[offset] = high; target[offset + 1] = Math.fround(cycles - high); target[offset + 2] = 2 * Math.PI * cycles; target[offset + 3] = weight;
    target[offset + 4] = a[0] * (1 - fraction) + b[0] * fraction;
    target[offset + 5] = a[1] * (1 - fraction) + b[1] * fraction;
    target[offset + 6] = a[2] * (1 - fraction) + b[2] * fraction;
    target[offset + 7] = 0;
  }
  const fixedPacked = fixed ? new Float32Array(fixed.length * 8) : null;
  if (fixed) for (let i = 0; i < fixed.length; i++) pack(fixedPacked, i * 8, fixed[i].wavelengthNm, fixed[i].weight);
  /** Offset is a Float32 ELEMENT index. No full-image storage or node objects.
   * [cyclesHi,cyclesLo,k,weight,xbar,ybar,zbar,0]; cycles in mm^-1, XYZ raw.
   */
  const writePackedNodes = (sampleIndex, target, offset = 0) => {
    index(sampleIndex, 'detector sample index');
    const count = fixed?.length ?? samples;
    if (!(target instanceof Float32Array) || !Number.isSafeInteger(offset) || offset < 0 || offset + count * 8 > target.length) throw new RangeError('Invalid packed spectral Float32 target or offset');
    if (fixedPacked) target.set(fixedPacked, offset);
    else for (let i = 0; i < count; i++) pack(target, offset + i * 8, drawWavelength(sampleIndex, i), 1 / samples);
    return count;
  };
  return Object.freeze({descriptor, mode, samples: fixed?.length ?? samples, spectralSamples: fixed?.length ?? samples, seed, exact, stochastic: !exact,
    scope: COHERENT_SPECTRAL_SCOPE, supportNm: fixed ? Object.freeze([Math.min(...fixed.map(n => n.wavelengthNm)), Math.max(...fixed.map(n => n.wavelengthNm))]) : d.supportNm,
    approximation: d?.approximation ?? Object.freeze({kind: 'exact discrete spectral measure'}), sampleNodes, writePackedNodes,
    ...(d ? {quantile: d.quantile, cdf: d.cdf} : {})});
}
