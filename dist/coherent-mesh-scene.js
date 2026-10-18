import {validateCoherentSlab} from './coherent-mesh-slab.js';
import {validateCoherentLighting} from './coherent-mesh-environment.js';
/** Versioned, local-only scene exchange. All distances are mm, phases radians.
 * Geometry is replaced, never appended to a preset. No URLs, scripts, textures,
 * transforms or third-party material loaders are part of this contract.
 */
import {validateCoherentCamera,validateDiffuseReflectance} from './coherent-mesh-camera.js';
import {prepareMesh, parseOBJ} from './mesh-geometry.js';
import {validateCoherentSpatialSource} from './coherent-mesh-spatial.js';
import {validateCoherentSpectrum} from './coherent-mesh-spectrum.js';
export {validateCoherentSpectrum as validateCoherentSceneSpectrum} from './coherent-mesh-spectrum.js';
export const COHERENT_SCENE_SCHEMA = 'coherent-mesh-scene-v1';
export const COHERENT_SCENE_LIMITS = Object.freeze({maxBytes: 2 * 1024 * 1024, maxMeshes: 64, maxMaterials: 64, maxVertices: 8000, maxTriangles: 12000, maxCoordinateMm: 10000});
const own = (object, key) => Object.prototype.hasOwnProperty.call(object, key);
function record(value, keys, name) {
  if (!value || typeof value !== 'object' || Array.isArray(value) || ![Object.prototype, null].includes(Object.getPrototypeOf(value))) throw new TypeError(`${name} must be an object`);
  for (const key of Object.keys(value)) if (!keys.includes(key)) throw new Error(`${name}: unsupported field ${key}`);
  return value;
}
function number(value, name, min, max) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max) throw new RangeError(`${name} must be finite within ${min}…${max}`);
  return value;
}
function vector(value, name, limit = COHERENT_SCENE_LIMITS.maxCoordinateMm) {
  if (!Array.isArray(value) || value.length !== 3) throw new TypeError(`${name} must have three coordinates`);
  return [0, 1, 2].map(i => number(value[i], `${name}[${i}]`, -limit, limit));
}
const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);
function unit(value, name) {
  const v = vector(value, name), length = Math.hypot(...v);
  if (!(length > 1e-12)) throw new RangeError(`${name} must be nonzero`);
  return Math.abs(length - 1) <= 8 * Number.EPSILON ? v : v.map(x => x / length);
}
function freeze(value) {
  if (value && typeof value === 'object') { for (const child of Object.values(value)) freeze(child); Object.freeze(value); }
  return value;
}
export function validateCoherentSceneSource(value) {
  record(value, ['position', 'power', 'phaseRad', 'wavelengthMm', 'spectrum', 'spatial'], 'source');
  const spatial=validateCoherentSpatialSource(value.spatial);
  if(spatial&&value.position?.some(x=>Math.abs(x)+spatial.radiusMm>10000))throw new RangeError('Disk extent exceeds coordinate bounds');
  return freeze({...(spatial?{spatial}:{}),position: vector(value.position, 'source.position'), power: number(value.power, 'source.power', 0, 1000), phaseRad: number(value.phaseRad, 'source.phaseRad', -2 * Math.PI, 2 * Math.PI), wavelengthMm: number(value.wavelengthMm, 'source.wavelengthMm', .00036, .00083), ...(own(value, 'spectrum') ? {spectrum: validateCoherentSpectrum(value.spectrum)} : {})});
}
export function validateCoherentSceneDetector(value) {
  record(value, ['center', 'u', 'v', 'width', 'height'], 'detector');
  const center = vector(value.center, 'detector.center'), u = unit(value.u, 'detector.u'), v = unit(value.v, 'detector.v');
  if (Math.abs(dot(u, v)) > 1e-10) throw new Error('Detector axes must be orthogonal');
  const width = number(value.width, 'detector.width', .0001, 1000), height = number(value.height, 'detector.height', .0001, 1000);
  for (const su of [-1, 1]) for (const sv of [-1, 1]) vector(center.map((x, i) => x + su * width / 2 * u[i] + sv * height / 2 * v[i]), 'detector corner');
  return freeze({center, u, v, width, height});
}
/** Validate the final effective pose after optional request overrides. */
export function validateCoherentSourceDetector(source, detector) {
  const delta = source.position.map((x, i) => x - detector.center[i]);
  const normal = [detector.u[1] * detector.v[2] - detector.u[2] * detector.v[1], detector.u[2] * detector.v[0] - detector.u[0] * detector.v[2], detector.u[0] * detector.v[1] - detector.u[1] * detector.v[0]];
  if(source.spatial?.radiusMm>0){const r=source.spatial.radiusMm, sn=source.spatial.normal, extent=r*Math.sqrt(Math.max(0,1-dot(sn,normal)**2));if(Math.abs(dot(delta,normal))<=extent+1e-6&&Math.abs(dot(delta,detector.u))<=detector.width/2+r&&Math.abs(dot(delta,detector.v))<=detector.height/2+r)throw new Error('Source disk may intersect detector; separate their extents');}
  if (Math.abs(dot(delta, normal)) < 1e-6 && Math.abs(dot(delta, detector.u)) <= detector.width / 2 + 1e-6 && Math.abs(dot(delta, detector.v)) <= detector.height / 2 + 1e-6) throw new Error('Detector must not contain the point source');
  return true;
}
function material(value, index) {
  record(value, ['type', 'reflectance', 'phaseRad'], `materials[${index}]`);
  if(value.type==='diffuse'){if(own(value,'phaseRad'))throw new Error('Diffuse coating has no coherent material phase');return {type:'diffuse',reflectance:validateDiffuseReflectance(value.reflectance)};}
  if (value.type === 'opaque') {
    if (own(value, 'reflectance') || own(value, 'phaseRad')) throw new Error('Opaque material accepts only type; no scattering coefficient is modeled');
    return {type: 'opaque'};
  }
  if (value.type !== 'mirror') throw new Error('Only opaque and ideal planar mirror materials are supported');
  return {type: 'mirror', reflectance: number(value.reflectance ?? 1, 'mirror.reflectance', 0, 1), phaseRad: number(value.phaseRad ?? Math.PI, 'mirror.phaseRad', -2 * Math.PI, 2 * Math.PI)};
}
function jsonBytes(value) {
  let text;
  try { text = JSON.stringify(value); } catch { throw new TypeError('Scene must be finite, acyclic JSON data'); }
  if (!text || text.length > COHERENT_SCENE_LIMITS.maxBytes || new TextEncoder().encode(text).length > COHERENT_SCENE_LIMITS.maxBytes) throw new RangeError('Scene JSON capacity is 2 MiB');
}
export function validateCoherentSceneRenderSettings(value) {
  record(value, ['maxReflections', 'diffraction', 'mixedDiffraction'], 'renderSettings');
  const maxReflections = number(value.maxReflections, 'renderSettings.maxReflections', 0, 2);
  if (!Number.isInteger(maxReflections)) throw new Error('renderSettings.maxReflections must be 0, 1 or 2');
  if (typeof value.diffraction !== 'boolean' || typeof value.mixedDiffraction !== 'boolean') throw new TypeError('renderSettings diffraction flags must be booleans');
  if (value.mixedDiffraction && (!value.diffraction || maxReflections !== 2)) throw new Error('Mixed diffraction requires diffraction and reflection order 2');
  return freeze({maxReflections, diffraction: value.diffraction, mixedDiffraction: value.mixedDiffraction});
}
export function validateCoherentScene(value) {
  record(value, ['schema', 'units', 'name', 'materials', 'meshes', 'source', 'detector', 'camera', 'lighting', 'slab', 'renderSettings'], 'scene');
  if (value.schema !== COHERENT_SCENE_SCHEMA) throw new Error(`Scene schema must be ${COHERENT_SCENE_SCHEMA}`);
  if (value.units !== 'mm') throw new Error('Scene units must explicitly be mm');
  if (value.name !== undefined && (typeof value.name !== 'string' || value.name.length > 120 || /[\0-\x1f]/.test(value.name))) throw new Error('Scene name must be at most 120 printable characters');
  if (!Array.isArray(value.materials) || !value.materials.length || value.materials.length > COHERENT_SCENE_LIMITS.maxMaterials) throw new Error('Scene requires 1–64 materials');
  if (!Array.isArray(value.meshes) || !value.meshes.length || value.meshes.length > COHERENT_SCENE_LIMITS.maxMeshes) throw new Error('Scene requires 1–64 meshes');
  jsonBytes(value);
  const materials = Array.from(value.materials, material), source = validateCoherentSceneSource(value.source), detector = validateCoherentSceneDetector(value.detector);
  validateCoherentSourceDetector(source, detector);
  let vertexCount = 0, triangleCount = 0;
  const meshes = Array.from(value.meshes, (entry, index) => {
    record(entry, ['vertices', 'faces', 'obj', 'materialIndex', 'surface'], `meshes[${index}]`);
    const materialIndex = number(entry.materialIndex, 'mesh.materialIndex', 0, materials.length - 1);
    if (!Number.isSafeInteger(materialIndex)) throw new Error('Material index must be an integer');
    const surface = entry.surface ?? 'auto';
    if (!['auto', 'open-plate', 'closed-solid'].includes(surface)) throw new Error('Unsupported mesh surface declaration');
    let input;
    if (own(entry, 'obj')) {
      if (own(entry, 'vertices') || own(entry, 'faces') || typeof entry.obj !== 'string') throw new Error('A mesh requires either inline OBJ or vertices and faces');
      input = parseOBJ(entry.obj);
    } else {
      if (!Array.isArray(entry.vertices) || !entry.vertices.length || entry.vertices.length > COHERENT_SCENE_LIMITS.maxVertices || !Array.isArray(entry.faces) || !entry.faces.length || entry.faces.length > COHERENT_SCENE_LIMITS.maxTriangles) throw new Error('Mesh vertex/triangle capacity exceeded or empty mesh');
      const vertices = Array.from(entry.vertices, v => vector(v, 'mesh vertex'));
      const faces = Array.from(entry.faces, face => {
        if (!Array.isArray(face) || face.length !== 3) throw new Error('Indexed mesh faces must contain exactly three indices');
        return Array.from(face, i => { if (!Number.isSafeInteger(i) || i < 0 || i >= vertices.length) throw new Error('Mesh vertex index out of range'); return i; });
      });
      input = {vertices, faces};
    }
    vertexCount += input.vertices.length; triangleCount += input.faces.length;
    if (vertexCount > COHERENT_SCENE_LIMITS.maxVertices || triangleCount > COHERENT_SCENE_LIMITS.maxTriangles) throw new Error('Scene total vertex/triangle capacity exceeded');
    const mesh = prepareMesh(input, {rejectNonManifold: false});
    return {vertices: mesh.authoredVertices.map(v => vector(v, 'mesh vertex')), faces: mesh.authoredFaces.map(f => [...f]), materialIndex, surface};
  });
  const scene = {...(own(value,'renderSettings')?{renderSettings:validateCoherentSceneRenderSettings(value.renderSettings)}:{}),...(value.slab!=null?{slab:validateCoherentSlab(value.slab)}:{}),schema: COHERENT_SCENE_SCHEMA, units: 'mm', name: value.name ?? 'Custom scene', materials, meshes, source, detector,...(own(value,'camera')?{camera:validateCoherentCamera(value.camera)}:{}),...(own(value,'lighting')?{lighting:validateCoherentLighting(value.lighting)}:{})};
  jsonBytes(scene);
  return freeze(scene);
}
export function parseCoherentScene(text) {
  if (typeof text !== 'string' || text.length > COHERENT_SCENE_LIMITS.maxBytes || new TextEncoder().encode(text).length > COHERENT_SCENE_LIMITS.maxBytes) throw new Error('Scene JSON capacity is 2 MiB');
  let value; try { value = JSON.parse(text.replace(/^\uFEFF/, '')); } catch { throw new Error('Scene file is not valid JSON'); }
  return validateCoherentScene(value);
}
export function serializeCoherentScene(value) { return JSON.stringify(validateCoherentScene(value)) + '\n'; }
export function sceneToCoherentData(value) {
  const scene = validateCoherentScene(value);
  return {...(own(scene,'renderSettings')?{renderSettings:scene.renderSettings}:{}),...(scene.slab?{slab:scene.slab}:{}),name: scene.name, materials: scene.materials, source: scene.source, detector: scene.detector, ...(scene.camera?{camera:scene.camera}:{}),...(own(scene,'lighting')?{lighting:scene.lighting}:{}),entries: scene.meshes.map(({vertices, faces, materialIndex, surface}) => ({mesh: {vertices, faces}, materialIndex, surface}))};
}
export function sceneFromCoherentData(data) {
  return validateCoherentScene({... (own(data,'renderSettings')?{renderSettings:data.renderSettings}:{}),... (data.slab?{slab:data.slab}:{}),schema: COHERENT_SCENE_SCHEMA, units: 'mm', name: data.name ?? 'Custom scene', materials: data.materials, source: data.source,...(data.camera?{camera:data.camera}:{}),...(own(data,'lighting')?{lighting:data.lighting}:{}), detector: {center: data.detector.center, u: data.detector.u, v: data.detector.v, width: data.detector.width, height: data.detector.height}, meshes: data.entries.map(entry => {
    const mesh = prepareMesh(entry.mesh?.authoredVertices ? {vertices: entry.mesh.vertices, faces: entry.mesh.authoredFaces} : entry.mesh, {rejectNonManifold: false});
    return {vertices: mesh.authoredVertices, faces: mesh.authoredFaces, materialIndex: entry.materialIndex ?? 0, surface: entry.surface ?? 'auto'};
  })});
}
