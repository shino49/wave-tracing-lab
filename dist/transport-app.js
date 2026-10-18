import {prepareCoherentMeshTask} from './coherent-mesh-runtime.js';
import {makeCoherentMeshPreset} from './coherent-mesh-core.js';
import {COHERENT_SCENE_LIMITS, parseCoherentScene, serializeCoherentScene, sceneFromCoherentData, validateCoherentSceneSource, validateCoherentSceneDetector, validateCoherentSceneSpectrum} from './coherent-mesh-scene.js';
import {validateCoherentLighting} from './coherent-mesh-environment.js';
import {validateCoherentSlab} from './coherent-mesh-slab.js';
import {makeCoherentCameraPreset, validateCoherentCamera} from './coherent-mesh-camera.js';
import {setupMeshView} from './mesh-view.js';
import {cieXYZ, displaySpectralXYZ, linearToSRGB} from './plt-spectrum.js';

const $ = id => document.getElementById(id);
export const TRANSPORT_RESOLUTION = 512;
export const TRANSPORT_BASE_EXPOSURE_EV = 16;
const MAX_OBJ_BYTES = 2 * 1024 * 1024;
const physicalInputs = ['trMeasurementMode', 'trPreset', 'trColorFringeView', 'trMaterial', 'trSpectrumMode', 'trSpectrumSource', 'trBandwidth', 'trSpectralQuality', 'trWavelength', 'trCoherence', 'trPhase', 'trBlockReflected', 'trDiffraction', 'trMixedDiffraction', 'trPower', 'trOrder', 'trQuadrature', 'trBackendSelect'];
const slabInputs = ['trSlabEnabled', 'trSlabIndex', 'trSlabThickness', 'trSlabAbsorption', 'trSlabInternalReflections'];
const lightingInputs = ['trEnvironmentEnabled', 'trEnvironmentRadiance', 'trEnvironmentUpper', 'trEnvironmentLower', 'trEnvironmentSamples', 'trIndirectEnabled', 'trIndirectSamples'];
const spatialInputs = ['trSourceRadius', 'trSourceNormalX', 'trSourceNormalY', 'trSourceNormalZ', 'trSourceSamples'];
const poseInputs = ['trSourceX', 'trSourceY', 'trSourceZ', 'trSourcePhase', 'trDetectorX', 'trDetectorY', 'trDetectorZ', 'trDetectorUX', 'trDetectorUY', 'trDetectorUZ', 'trDetectorVX', 'trDetectorVY', 'trDetectorVZ', 'trDetectorWidth', 'trDetectorHeight'];
const cameraInputs = ['trCameraX', 'trCameraY', 'trCameraZ', 'trCameraTargetX', 'trCameraTargetY', 'trCameraTargetZ', 'trCameraUpX', 'trCameraUpY', 'trCameraUpZ', 'trCameraFov', 'trCameraMirrorBounces'];
const channels = ['coherent', 'incoherent', 'geometric', 'crossTerm'];
const xyzChannels = ['coherentXYZ', 'incoherentXYZ', 'geometricXYZ'];
const figures = ['Geometric', 'Coherent', 'Baseline', 'Cross'];
const finite = value => typeof value === 'number' && Number.isFinite(value);
const floatArray = value => value instanceof Float32Array || value instanceof Float64Array;
const ensure = (condition, message) => { if (!condition) throw new Error(message); };
const boundedText = (value, fallback = '') => typeof value === 'string' ? value.slice(0, 2048) : fallback;
const view = setupMeshView($('trSceneCanvas'));
const downloads = new Set();
let generation = 0, worker = null, busy = false, timer = null, started = 0;
let activeRequest = null, lastResult = null, partial = null, lastExport = null, activeAttempt = 0;
let customScene = null, poseEdited = false, importedLineSpectrum = null;
let lightingState = null, savedEnvironment = null;
let slabState = null;
let spatialPresent = false, spatialSeed = 7391, cameraPresent = false;
let customOBJ = null, fileName = '', fileReading = false, fileError = '', fileGeneration = 0;
const status = value => { $('trStatus').textContent = value; };
function syncRunningControls() {
  for (const id of ['trRun', 'trRunQuick']) { $(id).disabled = busy || fileReading; $(id).setAttribute('aria-busy', String(busy)); }
  for (const id of ['trCancel', 'trCancelQuick']) $(id).hidden = !busy;
}

function stop() {
  const old = worker, id = activeRequest?.id;
  worker = null; generation++;
  if (old) { try { old.postMessage({type: 'cancel', id}); } catch {} old.terminate(); }
  if (timer !== null) { clearInterval(timer); timer = null; }
  busy = false;
  syncRunningControls(); $('trProgress').hidden = true;
}
function clearImages() {
  for (const name of figures) {
    const canvas = $(`tr${name}Canvas`);
    canvas.width = TRANSPORT_RESOLUTION; canvas.height = TRANSPORT_RESOLUTION;
    canvas.dataset.complete = 'false'; canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    $(`tr${name}Empty`).hidden = false; $(`tr${name}Partial`).hidden = true;
    $(`tr${name}Resolution`).textContent = name === 'Cross' ? '蓝：相消 · 黑：零 · 橙：相长' : '512 × 512 原生输出 · 尚未计算';
  }
  view.setTexture(null);
}
function clearResult() {
  activeRequest = null; lastResult = null; partial = null; lastExport = null; activeAttempt = 0;
  clearImages();
  $('trDownload').disabled = true; $('trReport').textContent = '尚无结果';
  $('trBackend').textContent = '尚未计算'; $('trNative').textContent = '—'; $('trElapsed').textContent = '—';
  $('trPathSummary').textContent = '有限阶 · 尚未运行'; $('trWarnings').textContent = ''; $('trDiffractionSummary').textContent = ''; $('trCriticalWarnings').textContent = '';
  $('trMeasurement').textContent = '完整路径的相位在复振幅上累计；同一像素内再按采样积分强度。';
  baselineLabels(false);
}
function spectralLabels() {
  const mode = $('trSpectrumMode').value, broadband = mode === 'broadband', gaussian = mode === 'gaussian', mono = mode === 'monochrome';
  $('trSpectrumSourceField').hidden = !broadband; $('trSpectrumSource').disabled = !broadband;
  $('trBandwidthField').hidden = !gaussian; $('trBandwidth').disabled = !gaussian;
  $('trSpectralQualityField').hidden = mono || mode === 'lines'; $('trSpectralQuality').disabled = mono || mode === 'lines';
  $('trWavelengthField').hidden = broadband || mode === 'lines'; $('trWavelength').disabled = broadband || mode === 'lines'; $('trWavelengthLabel').textContent = gaussian ? '中心真空波长' : '真空波长';
  $('trImportedLinesOption').hidden = !importedLineSpectrum;
  const source = broadband ? {D65: 'D65 日光', E: '等能白光', blackbody3000: '3000 K 暖光'}[$('trSpectrumSource').value] : gaussian ? '可见范围截断的高斯谱' : mode === 'lines' ? `导入的 ${importedLineSpectrum?.lines?.length ?? 0} 条谱线` : '单一波长';
  $('trSpectrumNote').textContent = `${source}。${mono || mode === 'lines' ? '逐条谱线计算，无连续光谱采样。' : `每个像素积分位置使用 ${$('trSpectralQuality').value} 个固定种子的随机分层光谱样本；增加数量通常可降低彩色采样噪声，仍需检查收敛。`}同一波长叠加路径复振幅，不同波长只加光强。源强度为 360–830 nm 内的全光谱积分，改变光谱时相同源强度不保证相同视觉亮度。`;
  $('trDisplayNote').textContent = `${lightingState?.environment ? '另加独立天空光谱的 XYZ；' : ''}${mono ? '单色线谱' : '逐波长相干强度的光谱积分'} → CIE 1931 XYZ → sRGB。直接显示求解器的 XYZ，四图共用固定增益，不按每幅峰值归一化。交叉项显示有符号 Y 差值。`;
}
function labels() {
  spectralLabels();
  lightingLabels();
  slabLabels();
  const cameraMode = $('trMeasurementMode').value === 'camera';
  $('trPhysicalCamera').hidden = !cameraMode;
  baselineLabels($('trDiffraction').checked);
  $('trGeometricTitle').textContent = cameraMode ? '物体表面 · 几何照明' : '几何光路强度和';
  $('trCoherentTitle').textContent = cameraMode ? '物体表面 · 波动照明' : '相干传输';
  $('trGeometricFormula').textContent = cameraMode ? '几何照明 × 被动反射率 / π · 终端闭合' : '不计 UTD 衍射 · 不合成路径间相位';
  $('trCoherentFormula').textContent = cameraMode ? '相干照明 × 被动反射率 / π · 终端闭合' : '(1 − γ) Σ |Eₚ|² + γ |Σ Eₚ|²';
  for (const name of ['Geometric', 'Coherent']) $(`tr${name}Canvas`).setAttribute('aria-label', cameraMode ? `${name === 'Geometric' ? '几何' : '波动'}照明下的被动漫反射物体，原生 512 乘 512，固定共用曝光` : `${name === 'Geometric' ? '几何光路强度和' : '完整路径相干传输强度'}，原生 512 乘 512，固定共用曝光`);
  $('trColorFringeControl').hidden = customScene !== null || $('trPreset').value !== 'lloyd' || cameraMode;
  $('trColorFringeView').disabled = customScene !== null || $('trPreset').value !== 'lloyd' || cameraMode;
  $('trWavelengthValue').textContent = `${$('trWavelength').value} nm`;
  $('trCoherenceValue').textContent = Number($('trCoherence').value).toFixed(2);
  $('trPhaseValue').textContent = `${Number($('trPhase').value).toFixed(2)} π`;
  $('trExposureValue').textContent = `${Number($('trExposure').value) >= 0 ? '+' : ''}${$('trExposure').value} EV`;
  $('trMaterial').disabled = customOBJ === null || customScene !== null;
  $('trOBJ').disabled = customScene !== null;
  const externalMirrorPhase = $('trPreset').value === 'reflected-edge' && customScene === null;
  $('trPhase').disabled = $('trDiffraction').checked && !externalMirrorPhase || customScene !== null;
  $('trPhaseLabel').textContent = externalMirrorPhase ? '外部镜面的附加相位' : '每次反射的附加相位';
  $('trPhaseNote').textContent = customScene ? '自定义场景的镜面相位由 JSON 材料记录指定；此预设相位控件不修改材料。' : externalMirrorPhase ? '只改变外部反射镜（及另行添加镜面）的附加相位；主绕射刀口保持理想边界。' : '叠加在理想镜固有反射相位上。启用衍射时复位到 0 并锁定，以满足边缘边界条件。';
  $('trMixedDiffraction').disabled = !$('trDiffraction').checked || Number($('trOrder').value) < 2 || $('trBlockReflected').checked;
  $('trSceneMode').textContent = customScene ? `自定义场景：${customScene.name}。全部几何已替换预设。` : fileReading ? '正在本地读取场景…' : fileError ? '导入未通过检查；修正或返回预设后才能运行。' : $('trPreset').value === 'lloyd' && $('trColorFringeView').checked ? 'Lloyd 彩色条纹观察区域：真实探测器 6 × 1 mm，中心 [0, 0.5, 50] mm。' : poseEdited ? '验证预设几何；使用自定义光源与探测器。' : '当前使用验证预设。';
  $('trUsePreset').hidden = customScene === null && !fileError && !fileReading;
  $('trExportScene').disabled = fileReading || !!fileError;
  $('trDiffractionNote').textContent = $('trDiffraction').checked ? `已启用有限源一阶 UTD。${$('trMixedDiffraction').checked ? '另计入准入的 R–D / D–R 路径。' : 'R–D / D–R 混合路径关闭。'}角点与连续两次衍射未建模；不适用路径会被排除。${customScene ? 'JSON 材料相位保持原值，准入由引擎检查。' : externalMirrorPhase ? '外部镜面相位可调，主刀口保持理想边界。' : '预设附加镜面相位锁定为 0。'}` : '衍射已关闭。双缝预设自动开启；未开启时不对任何几何边缘添加衍射。';
  $('trClearOBJ').hidden = customScene !== null || /\.json$/i.test(fileName) || customOBJ === null && !fileError && !fileReading;
  $('trFileName').textContent = fileReading ? '正在本地读取…' : customScene ? '完整 JSON 场景模式；OBJ 添加入口已停用' : fileName || '使用内置几何';
  updateWorkbenchLabels();
}

// Presentation-only summaries. Rendering requests and numerical settings remain unchanged.
function requestGlassSceneLabel() { return Number($('trOrder').value) > 0 ? '玻璃物体 + 镜面照明' : '玻璃后的彩色物体'; }
function updateWorkbenchLabels() {
  const descriptions = {
    'lloyd': '直达光与镜面反射光相遇，观察宽带光的彩色干涉条纹。',
    'object-imaging': '用几何针孔相机观察三维彩色物体，比较几何照明与波动照明。',
    'two-mirror': '让多个完整反射路径叠加，观察路径相位如何影响光强。',
    'double-slit': '观察双缝网格的边缘衍射，以及完整路径之间的干涉。',
    'reflected-edge': '观察反射与边缘绕射；在传播设置中检查混合路径的准入。'
  };
  const camera = $('trMeasurementMode').value === 'camera';
  const preset = $('trPreset').selectedOptions[0]?.textContent || '验证预设';
  $('trPresetDescription').textContent = customScene ? '已载入独立场景。重新选择基础场景会替换当前配置。' : descriptions[$('trPreset').value] || '';
  const incompatibleLighting = !camera && ($('trEnvironmentEnabled').checked || $('trIndirectEnabled').checked);
  $('trConfigurationWarning').hidden = !incompatibleLighting;
  $('trConfigurationWarning').textContent = incompatibleLighting ? '当前仍启用了环境光或间接光，但这些功能仅支持相机模式。请切回相机模式。若要使用探测器，请先在相机模式的“环境与间接光”中关闭这两项。' : '';
  $('trCurrentScene').textContent = fileReading ? '正在读取…' : fileError ? '导入未通过检查' : customScene ? ({'Colored solids through a finite glass slab': requestGlassSceneLabel(), 'Lloyd interference with direct-arm glass slab': '单色玻璃干涉', 'Broadband one-mirror glass transmission': '宽带镜面 + 玻璃'}[customScene.name] ?? customScene.name) : customOBJ ? `${preset} + ${fileName}` : preset;
  const spectrum = $('trSpectrumMode').value;
  const light = spectrum === 'monochrome' ? `${$('trWavelength').value} nm 单色` : spectrum === 'gaussian' ? `${$('trWavelength').value} nm 高斯光谱` : spectrum === 'lines' ? '导入的离散光谱' : $('trSpectrumSource').selectedOptions[0]?.textContent;
  $('trCurrentSetup').textContent = `${camera ? '三维物体成像' : '探测器平面'} · ${light} · 原生 512 × 512${$('trSlabEnabled').checked ? ' · 玻璃已启用' : ''}`;
  $('trModeHelp').textContent = camera ? '相机模式：在物体表面计算照明。位置、视场角与镜面次数在下方“相机与探测器”中调整。' : '探测器模式：观察接收平面上的局部光强。观察区域与坐标在下方“相机与探测器”中调整。';
  for (const [id, q, n] of [['trCameraFast', 1, 16], ['trCameraFine', 2, 32]]) {
    const selected = Number($('trQuadrature').value) === q && (spectrum === 'monochrome' || spectrum === 'lines' || Number($('trSpectralQuality').value) === n);
    $(id).setAttribute('aria-pressed', String(selected));
    $(id).title = `${q} × ${q} 像素积分 / ${n} 连续光谱样本；仅设置参数`;
  }
}

function defaultSlab() { return {center: [0, .09, 0], u: [1, 0, 0], v: [0, 1, 0], width: 1, height: .03, thickness: .2, index: 1.5, absorption: [{wavelengthNm: 360, value: 0}, {wavelengthNm: 830, value: 0}], internalReflections: 1}; }
function setSlab(slab) {
  slabState = structuredClone(slab ?? defaultSlab());
  $('trSlabEnabled').checked = !!slab;
  $('trSlabIndex').value = String(slabState.index);
  $('trSlabThickness').value = String(slabState.thickness);
  $('trSlabInternalReflections').value = String(slabState.internalReflections);
  $('trSlabAbsorption').value = slabState.absorption.every(p => p.value === slabState.absorption[0].value) ? String(slabState.absorption[0].value) : '';
}
function slabLabels() {
  const restricted = $('trSlabEnabled').checked && (Number($('trOrder').value) > 1 || $('trDiffraction').checked);
  $('trSlabPropagationWarning').hidden = !restricted;
  $('trSlabPropagationWarning').textContent = restricted ? '注意：当前启用了源侧反射或衍射。现支持一次镜面反射与一次双界面透射的两种顺序。穿板的两次镜面反射、绕射及重复穿板仍未建模；请设源反射 0 或 1 并关闭衍射，或保证这些路径绕过玻璃。玻璃物体成像示例需将光源最大反射次数设为 0、关闭衍射；玻璃干涉示例使用 1 次反射、关闭衍射。导入的显式运行选项会恢复；手动覆盖后请核对。' : '';
  for (const id of slabInputs.slice(1)) $(id).disabled = !$('trSlabEnabled').checked;
  const spectral = slabState && !slabState.absorption.every(p => p.value === slabState.absorption[0].value);
  $('trSlabCurveNote').textContent = spectral ? `保留 ${slabState.absorption.length} 点光谱吸收曲线。中性吸收留空表示保留；修改此项才替换为常数 α。` : '中性吸收 α 为功率衰减系数（每 mm），不是透明度。导入的板位置、轴与尺寸保持不变。';
}
function currentSlab() { return $('trSlabEnabled').checked ? validateCoherentSlab(slabState) : null; }
export function makeTransportSlabScene(kind) {
  const camera = kind === 'camera' || kind === 'camera-mirror', data = camera ? makeCoherentCameraPreset() : makeCoherentMeshPreset('lloyd');
  data.name = camera ? 'Colored solids through a finite glass slab' : 'Lloyd interference with direct-arm glass slab';
  data.slab = camera ? {...defaultSlab(), center: [0, .9, 48], width: 3.2, height: 1.6, thickness: .8, absorption: [[360,.6],[450,.4],[550,.08],[650,.02],[830,.01]].map(([wavelengthNm,value]) => ({wavelengthNm,value}))} : defaultSlab();
  data.source = {power: 1, phaseRad: 0, wavelengthMm: .00055, ...data.source, spectrum: camera ? {mode: 'broadband', illuminant: 'D65'} : {mode: 'monochrome'}};
  if (camera) { data.camera.mirrorBounces = 2; data.lighting = {environment: defaultEnvironment(), indirectSamples: 0}; }
  else { data.source.position = [0, 2, -50]; data.slab.center = [0, 1.05, 0]; data.slab.height = .04; data.detector = {...data.detector, center: [0, .10, 50], width: .02, height: .06}; }
  if (kind === 'mirror') { data.name = 'Broadband one-mirror glass transmission'; data.source.position = [0, .2, -50]; data.source.spectrum = {mode: 'broadband', illuminant: 'D65'}; data.slab.center = [0, .125, -25]; data.slab.width = .1; data.slab.height = .04; data.slab.thickness = .0002; data.detector = {...data.detector, center: [0, .1, 50], width: .02, height: .12}; }
  return {...sceneFromCoherentData(data), renderSettings: {maxReflections: kind === 'camera' ? 0 : 1, diffraction: false, mixedDiffraction: false}};
}
function defaultEnvironment() { return {radiance: 2e-6, spectrum: {mode: 'broadband', illuminant: 'D65'}, up: [0, 1, 0], upper: 1, lower: .2, samples: 8, seed: 7391}; }
function setLighting(lighting) {
  lightingState = lighting ? structuredClone(lighting) : null;
  savedEnvironment = structuredClone(lighting?.environment ?? defaultEnvironment());
  $('trEnvironmentEnabled').checked = !!lighting?.environment;
  $('trEnvironmentRadiance').value = String(savedEnvironment.radiance * 1e6);
  $('trEnvironmentUpper').value = String(savedEnvironment.upper);
  $('trEnvironmentLower').value = String(savedEnvironment.lower);
  $('trEnvironmentSamples').value = String(savedEnvironment.samples);
  $('trIndirectEnabled').checked = (lighting?.indirectSamples ?? 0) > 0;
  $('trIndirectSamples').value = String(lighting?.indirectSamples || 4);
}
function lightingLabels() {
  const camera = $('trMeasurementMode').value === 'camera';
  for (const id of lightingInputs) $(id).disabled = !camera || (id.startsWith('trEnvironment') && id !== 'trEnvironmentEnabled' && !$('trEnvironmentEnabled').checked) || (id === 'trIndirectSamples' && !$('trIndirectEnabled').checked);
  $('trLightingPreset').disabled = !camera;
  $('trLightingModeNote').textContent = camera ? '仅相机模式。环境光与一次漫反射间接光独立于点源波动求解；修改后需重新运行。' : '环境与间接光仅支持相机模式；请先切换测量模式。启用的照明配置不会静默丢弃。';
}
function currentLighting() {
  if (!lightingState) return null;
  const environment = lightingState.environment;
  if (environment) {
    ensure(finite(environment.radiance) && environment.radiance >= 0 && environment.radiance <= 1, '环境辐亮度必须在 0–1 内（显示值 0–1000000）。');
    for (const key of ['upper', 'lower']) ensure(finite(environment[key]) && environment[key] >= 0 && environment[key] <= 10, '环境上下半球倍率必须在 0–10 内。');
    ensure([4, 8, 16, 32].includes(environment.samples), '环境采样数必须是 4、8、16 或 32。');
  }
  ensure([0, 1, 2, 4, 8].includes(lightingState.indirectSamples), '间接采样数必须是 0、1、2、4 或 8。');
  return validateCoherentLighting(lightingState);
}
function setDetectorPose(detector) {
  for (const [prefix, vector] of [['trDetector', detector.center], ['trDetectorU', detector.u], ['trDetectorV', detector.v]]) for (const [i, axis] of [...'XYZ'].entries()) $(prefix + axis).value = String(vector[i]);
  $('trDetectorWidth').value = String(detector.width); $('trDetectorHeight').value = String(detector.height);
}
function setCamera(camera) {
  cameraPresent = !!camera;
  const value = camera ?? {position: [0, 0, 100], target: [0, 0, 0], up: [0, 1, 0], fovDeg: 35};
  for (const [prefix, vector] of [['trCamera', value.position], ['trCameraTarget', value.target], ['trCameraUp', value.up]]) for (const [i, axis] of [...'XYZ'].entries()) $(prefix + axis).value = String(vector[i]);
  $('trCameraFov').value = String(value.fovDeg);
  $('trCameraMirrorBounces').value = String(value.mirrorBounces ?? 2);
}
function currentCamera() {
  return validateCoherentCamera({position: [...'XYZ'].map(axis => inputNumber('trCamera' + axis)), target: [...'XYZ'].map(axis => inputNumber('trCameraTarget' + axis)), up: [...'XYZ'].map(axis => inputNumber('trCameraUp' + axis)), fovDeg: inputNumber('trCameraFov'), mirrorBounces: inputNumber('trCameraMirrorBounces')});
}
function setSourceSpatial(spatial) {
  spatialPresent = spatial !== undefined; spatialSeed = spatial?.seed ?? 7391;
  $('trSourceRadius').value = String(spatial?.radiusMm ?? 0);
  for (const [i, axis] of [...'XYZ'].entries()) $('trSourceNormal' + axis).value = String((spatial?.normal ?? [0, 0, 1])[i]);
  const samples = String(spatial?.samples ?? 16), select = $('trSourceSamples');
  select.querySelector('[data-imported]')?.remove();
  if (![...select.options].some(option => option.value === samples)) {
    const option = document.createElement('option'); option.value = samples; option.textContent = `${samples} 点 · 导入值`; option.dataset.imported = 'true'; select.append(option);
  }
  select.value = samples;
}
function currentSpatial() {
  if (!spatialPresent) return {};
  return {spatial: {type: 'disk', radiusMm: inputNumber('trSourceRadius'), normal: [...'XYZ'].map(axis => inputNumber('trSourceNormal' + axis)), samples: inputNumber('trSourceSamples'), seed: spatialSeed}};
}
function setPose(source, detector) {
  setSourceSpatial(source.spatial);
  for (const [i, axis] of [...'XYZ'].entries()) $('trSource' + axis).value = String(source.position[i]);
  $('trSourcePhase').value = String(source.phaseRad ?? 0); setDetectorPose(detector);
}
function presetDetector() {
  const detector = makeCoherentMeshPreset($('trPreset').value).detector;
  return $('trPreset').value === 'lloyd' && $('trColorFringeView').checked ? {...detector, center: [0, .5, 50], width: 6, height: 1} : detector;
}
function presetPose() { setLighting(null); setSlab(null); const data = makeCoherentMeshPreset($('trPreset').value); setPose(data.source, presetDetector()); setCamera(data.camera); $('trMeasurementMode').value = data.camera ? 'camera' : 'detector'; poseEdited = $('trPreset').value === 'lloyd' && $('trColorFringeView').checked; }
const inputNumber = id => $(id).value.trim() === '' ? NaN : Number($(id).value);
function currentSpectrum() {
  const mode = $('trSpectrumMode').value;
  const descriptor = mode === 'monochrome' ? {mode} : mode === 'broadband' ? {mode, illuminant: $('trSpectrumSource').value} : mode === 'gaussian' ? {mode, centerNm: inputNumber('trWavelength'), fwhmNm: inputNumber('trBandwidth')} : mode === 'lines' ? importedLineSpectrum : null;
  ensure(descriptor, '光源光谱模式无效。'); return validateCoherentSceneSpectrum(descriptor);
}
function sourceWavelengthNm() { return customScene && $('trSpectrumMode').value !== 'monochrome' ? customScene.source.wavelengthMm * 1e6 : inputNumber('trWavelength'); }
function setSpectrum(source) {
  const spectrum = source.spectrum;
  importedLineSpectrum = spectrum?.mode === 'lines' ? spectrum : null;
  $('trSpectrumMode').value = spectrum?.mode ?? 'monochrome';
  $('trWavelength').value = String(spectrum?.mode === 'gaussian' ? spectrum.centerNm : source.wavelengthMm * 1e6);
  if (spectrum?.mode === 'broadband') $('trSpectrumSource').value = spectrum.illuminant;
  if (spectrum?.mode === 'gaussian') $('trBandwidth').value = String(spectrum.fwhmNm);
}
function currentSource() {
  const spectrum = currentSpectrum();
  return validateCoherentSceneSource({...currentSpatial(), position: [...'XYZ'].map(axis => inputNumber('trSource' + axis)), phaseRad: inputNumber('trSourcePhase'), power: inputNumber('trPower'), wavelengthMm: sourceWavelengthNm() / 1e6, ...(spectrum.mode === 'monochrome' && !customScene?.source?.spectrum ? {} : {spectrum})});
}
function currentDetector() { return validateCoherentSceneDetector({center: [...'XYZ'].map(axis => inputNumber('trDetector' + axis)), u: [...'XYZ'].map(axis => inputNumber('trDetectorU' + axis)), v: [...'XYZ'].map(axis => inputNumber('trDetectorV' + axis)), width: inputNumber('trDetectorWidth'), height: inputNumber('trDetectorHeight')}); }
function defaults() {
  for (const [id, value] of Object.entries({trPreset: 'lloyd', trMaterial: 'opaque', trSpectrumMode: 'broadband', trSpectrumSource: 'D65', trBandwidth: 20, trSpectralQuality: 64, trWavelength: 550, trCoherence: 1, trPhase: 0, trPower: 1, trOrder: 2, trQuadrature: 2, trBackendSelect: 'auto', trExposure: 0})) $(id).value = String(value);
  $('trColorFringeView').checked = true; $('trBlockReflected').checked = false; $('trDiffraction').checked = false; $('trMixedDiffraction').checked = false;
  customScene = null; importedLineSpectrum = null; $('trSceneJSON').value = ''; presetPose();
  $('trOBJ').value = ''; customOBJ = null; fileName = ''; fileError = ''; fileReading = false; fileGeneration++;
  labels();
}
function requestFor(id) {
  const measurement = $('trMeasurementMode').value;
  const request = {id, measurement, slab: currentSlab(), lighting: currentLighting(), camera: measurement === 'camera' ? currentCamera() : null, preset: $('trPreset').value, resolution: TRANSPORT_RESOLUTION,
    wavelengthNm: sourceWavelengthNm(), spectrum: currentSpectrum(), spectralSamples: Number($('trSpectralQuality').value), spectralSeed: 7391, phaseShiftRad: Number($('trPhase').value) * Math.PI,
    power: inputNumber('trPower'), coherence: Number($('trCoherence').value),
    maxReflections: Number($('trOrder').value), pixelQuadrature: Number($('trQuadrature').value),
    backend: $('trBackendSelect').value, mixedDiffraction: $('trMixedDiffraction').checked, diffraction: $('trDiffraction').checked, blockReflected: $('trBlockReflected').checked, customOBJ, customMaterial: $('trMaterial').value, scene: null, source: null, detector: null};
  if (customScene) { request.scene = {...customScene, slab: request.slab, ...(request.lighting || 'lighting' in customScene ? {lighting: request.lighting} : {}), source: currentSource(), detector: currentDetector(), ...(cameraPresent || measurement === 'camera' ? {camera: currentCamera()} : {})}; if (!request.slab) delete request.scene.slab; if (request.slab || customScene.renderSettings) request.scene.renderSettings = {maxReflections: request.maxReflections, diffraction: request.diffraction, mixedDiffraction: request.mixedDiffraction}; request.phaseShiftRad = 0; request.customOBJ = null; }
  else if (poseEdited) { request.source = currentSource(); request.detector = currentDetector(); }
  ensure(!fileReading, '场景文件仍在读取，请稍候。'); ensure(!fileError, fileError);
  ensure(['lloyd', 'two-mirror', 'double-slit', 'reflected-edge', 'object-imaging'].includes(request.preset), '未知验证场景。');
  ensure(measurement === 'camera' || !request.lighting?.environment && !request.lighting?.indirectSamples, '环境与间接光仅支持相机模式；请切换相机模式后关闭照明或运行。');
  ensure(['detector', 'camera'].includes(measurement), '未知测量模式。');
  ensure(finite(request.wavelengthNm) && request.wavelengthNm >= 360 && request.wavelengthNm <= 830, '波长必须在 360–830 nm 内。');
  ensure(finite(request.phaseShiftRad) && Math.abs(request.phaseShiftRad) <= Math.PI, '附加反射相位必须在 −π 至 π 内。');
  ensure(finite(request.power) && request.power >= 0 && request.power <= 1000, '源功率必须在 0–1000 内。');
  ensure(finite(request.coherence) && request.coherence >= 0 && request.coherence <= 1, '相干度必须在 0–1 内。');
  ensure([0, 1, 2].includes(request.maxReflections), '最大反射次数必须在 0–2 内。');
  ensure([1, 2, 4].includes(request.pixelQuadrature), '像素积分必须为 1 × 1、2 × 2 或 4 × 4。');
  ensure([16, 32, 64, 128].includes(request.spectralSamples), '光谱采样数必须是 16、32、64 或 128。');
  ensure(['auto', 'cpu'].includes(request.backend), '未知计算后端。');
  ensure(['opaque', 'mirror'].includes(request.customMaterial), '未知导入网格边界。');
  ensure(!request.mixedDiffraction || request.diffraction && request.maxReflections === 2 && !request.blockReflected, 'R–D / D–R 需要启用衍射、反射阶数 2 且不屏蔽反射。');
  return request;
}
function showOverview(overview) {
  if (!overview) return;
  ensure(overview.mesh && Array.isArray(overview.mesh.triangles), '场景缺少真实三角网格。');
  const vector = v => Array.isArray(v) && v.length === 3 && v.every(finite);
  ensure(vector(overview.sourceOrigin) && vector(overview.detector?.center) && vector(overview.detector?.u) && vector(overview.detector?.v), '场景坐标无效。');
  ensure(finite(overview.detector.width) && overview.detector.width > 0 && finite(overview.detector.height) && overview.detector.height > 0, '探测器尺寸无效。');
  for (const triangle of overview.mesh.triangles) ensure(['a', 'b', 'c'].every(key => vector(triangle[key])), '三角网格坐标无效。');
  view.setScene(overview);
  $('trMeshCount').textContent = `${overview.mesh.triangles.length.toLocaleString()} 个真实三角形`;
  $('trGeometryNote').textContent = `显示引擎实际几何；探测器 ${overview.detector.width} × ${overview.detector.height} mm。坐标单位 mm，观察相机只改变投影。${overview.camera ? `几何针孔相机视场 ${overview.camera.fovDeg}°；拖动此预览仅改变观察视角，不改变几何针孔相机。` : ''}${overview.sourceDisk ? '圆环表示真实圆盘光源范围；法向量决定圆盘朝向，不是光束方向。' : ''}`;
}
function updateGeometry() {
  view.setScene(null); $('trMeshCount').textContent = '检查几何';
  try {
    const prepared = prepareCoherentMeshTask(requestFor(Math.max(1, generation)));
    showOverview(prepared.overview);
  } catch (error) {
    $('trMeshCount').textContent = '几何未通过检查';
    $('trGeometryNote').textContent = boundedText(error.message, '当前参数无法准备真实几何。');
  }
}
function invalidate() {
  if (fileReading) { fileGeneration++; fileReading = false; fileError = '参数已改变，未完成的文件读取已取消；请重新选择文件或返回预设。'; fileName = ''; customScene = null; customOBJ = null; $('trSceneJSON').value = ''; $('trOBJ').value = ''; }
  stop(); clearResult(); labels(); updateGeometry();
  status('参数已改变，旧结果已清除。重新运行后将使用当前条件。');
}
function baselineLabels(diffraction) {
  const cameraMode = $('trMeasurementMode').value === 'camera';
  $('trBaselineTitle').textContent = cameraMode ? '物体表面 · 独立路径相位基线' : '独立路径相位基线';
  $('trBaselineFormula').textContent = diffraction ? 'Σ |Eₚ|² · 含准入衍射，去掉全部路径交叉项' : 'Σ |Eₚ|² · 去掉全部完整路径交叉项';
  if (cameraMode) $('trBaselineFormula').textContent += ' · 被动反射率 / π 终端闭合';
  $('trBaselineCanvas').setAttribute('aria-label', `${diffraction ? '包含准入衍射路径的独立路径相位基线' : '完整路径独立相位随机化基线'}，原生 512 乘 512，固定共用曝光`);
}
function includesDiffraction(frame, request) {
  return frame?.diagnostics?.diffraction?.enabled === true || frame?.diagnostics?.diffractionEnabled === true || frame?.configuration?.diffractionEnabled === true || request?.diffraction === true;
}
function sameSnapshot(a, b) {
  if (a === b) return true;
  if (finite(a) && finite(b)) return Math.abs(a - b) <= 16 * Number.EPSILON * Math.max(Math.abs(a), Math.abs(b), 1e-12);
  if (!a || !b || typeof a !== 'object' || typeof b !== 'object' || Array.isArray(a) !== Array.isArray(b)) return false;
  const keys = Object.keys(a); return keys.length === Object.keys(b).length && keys.every(key => Object.prototype.hasOwnProperty.call(b, key) && sameSnapshot(a[key], b[key]));
}
export function validateTransportFrame(frame, request, {partial: isPartial = false} = {}) {
  ensure(frame && request && frame.complete === !isPartial, '完成状态与消息类型不符。');
  ensure(frame.width === TRANSPORT_RESOLUTION && frame.height === TRANSPORT_RESOLUTION && request.resolution === TRANSPORT_RESOLUTION, '图像必须是原生 512 × 512。');
  const rowStart = isPartial ? frame.rowStart ?? 0 : 0, rowCount = isPartial ? frame.rowCount ?? frame.height : frame.height;
  ensure(Number.isSafeInteger(rowStart) && Number.isSafeInteger(rowCount) && rowStart >= 0 && rowCount > 0 && rowStart + rowCount <= frame.height, '分块行范围无效。');
  const fullLength = frame.width * frame.height, tileLength = frame.width * rowCount;
  let layout = null;
  for (const key of channels) {
    const array = frame[key];
    ensure(floatArray(array) && (array.length === fullLength || isPartial && array.length === tileLength), `${key} 数组形状无效。`);
    if (layout === null) layout = array.length; else ensure(layout === array.length, '各图数组的分块形状不一致。');
    for (let i = 0; i < array.length; i++) ensure(finite(array[i]) && (key === 'crossTerm' || array[i] >= 0), `${key} 含非有限值或负强度。`);
  }
  if (!isPartial && (request.lighting?.environment || request.lighting?.indirectSamples)) ensure(frame.lightingComplete === true, '环境与间接照明尚未积分完成。');
  ensure(frame.spectralComplete === true, '光谱未积分完成，不能显示该图像或分块。');
  const spatial = request.scene?.source?.spatial ?? request.source?.spatial;
  const spatialSamples = spatial?.radiusMm > 0 ? spatial.samples : 1;
  if (spatial?.radiusMm > 0) {
    ensure(frame.spatialSamples === spatialSamples, '结果空间采样数与请求不符。');
    ensure(isPartial || frame.spatialComplete === true, '空间光源尚未积分完成。');
    if (isPartial) ensure(Number.isSafeInteger(frame.spatialSamplesIntegrated) && frame.spatialSamplesIntegrated >= 1 && frame.spatialSamplesIntegrated <= spatialSamples && frame.spatialComplete === (frame.spatialSamplesIntegrated === spatialSamples), '空间光源分块积分记录无效。');
  }
  const spectrum = request.spectrum ?? request.source?.spectrum ?? request.scene?.source?.spectrum;
  const expectedSamples = !spectrum || spectrum.mode === 'monochrome' ? 1 : spectrum.mode === 'lines' ? spectrum.lines.length : request.spectralSamples;
  ensure(Number.isSafeInteger(frame.spectralSamples) && frame.spectralSamples === expectedSamples, '结果光谱采样数与请求不符。');
  for (const key of xyzChannels) {
    const array = frame[key];
    ensure(floatArray(array) && array.length === layout * 3, `${key} 数组形状无效，必须与强度行块一致。`);
    for (const value of array) ensure(finite(value) && value >= 0, `${key} 含非有限值或负 XYZ。`);
  }
  if (!isPartial) {
    ensure(frame.configuration && typeof frame.configuration === 'object', '结果缺少运行参数。');
    ensure(typeof frame.backend === 'string' && frame.backend.trim().length > 0 || frame.backend && ['cpu', 'hybrid'].includes(frame.backend.kind), '结果缺少实际后端。');
    ensure(finite(frame.timings?.totalMs) && frame.timings.totalMs >= 0, '结果缺少有效的实际用时。');
  }
  if (frame.configuration) for (const key of ['measurement', 'preset', 'resolution', 'wavelengthNm', 'spectralSamples', 'spectralSeed', 'phaseShiftRad', 'power', 'coherence', 'maxReflections', 'pixelQuadrature', 'blockReflected', 'customMaterial', 'diffraction', 'mixedDiffraction']) {
    if (!isPartial || frame.configuration[key] !== undefined) ensure(key === 'wavelengthNm' ? sameSnapshot(frame.configuration[key], request[key]) : frame.configuration[key] === request[key], `结果参数不匹配：${key}。`);
  }
  if (frame.configuration) for (const key of ['scene', 'source', 'detector', 'spectrum', 'camera', 'lighting', 'slab']) {
    if (!isPartial || frame.configuration[key] !== undefined) ensure(sameSnapshot(frame.configuration[key] ?? null, request[key] ?? null), `结果场景不匹配：${key}。`);
  }
  return {frame, rowStart, rowCount, fullFrame: layout === fullLength};
}
export function transportIntensityRGB(intensity, wavelengthNm, exposureEV = 0) {
  return displaySpectralXYZ(cieXYZ(wavelengthNm).map(value => value * intensity), TRANSPORT_BASE_EXPOSURE_EV + exposureEV).rgb;
}
export function transportSignedRGB(value, exposureEV = 0) {
  const gain = 2 ** (TRANSPORT_BASE_EXPOSURE_EV + exposureEV);
  const v = Math.round(255 * linearToSRGB(-Math.expm1(-gain * Math.abs(value))));
  return value < 0 ? [Math.round(v * .28), Math.round(v * .68), v] : [v, Math.round(v * .57), Math.round(v * .2)];
}
function paintFrame(frame, request, incomplete, rowStart = 0, rowCount = frame.height) {
  const exposure = Number($('trExposure').value);
  const diffraction = includesDiffraction(frame, request); baselineLabels(diffraction);
  for (const [name, key] of [['Geometric', 'geometric'], ['Coherent', 'coherent'], ['Baseline', 'incoherent'], ['Cross', 'crossTerm']]) {
    const canvas = $(`tr${name}Canvas`), ctx = canvas.getContext('2d'), image = ctx.createImageData(frame.width, rowCount);
    for (let pixel = 0; pixel < frame.width * rowCount; pixel++) {
      const p = rowStart * frame.width + pixel;
      const rgb = key === 'crossTerm' ? transportSignedRGB(frame.coherentXYZ[3 * p + 1] - frame.incoherentXYZ[3 * p + 1], exposure) : displaySpectralXYZ(frame[key + 'XYZ'].subarray(3 * p, 3 * p + 3), TRANSPORT_BASE_EXPOSURE_EV + exposure).rgb;
      image.data.set([...rgb, 255], pixel * 4);
    }
    ctx.putImageData(image, 0, rowStart); canvas.dataset.complete = String(!incomplete);
    $(`tr${name}Empty`).hidden = true; $(`tr${name}Partial`).hidden = !incomplete;
    if (incomplete) $(`tr${name}Partial`).querySelector('span').textContent = frame.lightingComplete === false ? '点源预览 · 环境与间接照明尚未完成' : frame.spatialSamples > 1 ? `${frame.rowsComplete} / 512 行完成全部发光点与光谱 · 空间积分尚未完成` : `${frame.rowsComplete} / 512 行完成全部光谱 · 未计算区域为黑色`;
    $(`tr${name}Resolution`).textContent = `512 × 512 原生计算 · ${incomplete ? '分块预览，尚未完成' : '已完成'}${name === 'Cross' ? ' · 蓝负 / 橙正' : ''}`;
  }
  // The textured detector is display-only; it never feeds the field solver.
  view.setTexture(request.measurement === 'camera' ? null : $('trCoherentCanvas'));
}
function acceptPreview(message, request) {
  const {rowStart, rowCount, fullFrame} = validateTransportFrame(message, request, {partial: true});
  if (!partial) partial = {width: TRANSPORT_RESOLUTION, height: TRANSPORT_RESOLUTION, complete: false, rows: new Uint16Array(TRANSPORT_RESOLUTION), rowsComplete: 0, spatialSamples: message.spatialSamples ?? 1, ...Object.fromEntries(channels.map(key => [key, new Float64Array(TRANSPORT_RESOLUTION ** 2)])), ...Object.fromEntries(xyzChannels.map(key => [key, new Float64Array(3 * TRANSPORT_RESOLUTION ** 2)]))};
  const start = rowStart * TRANSPORT_RESOLUTION, end = (rowStart + rowCount) * TRANSPORT_RESOLUTION;
  for (const key of channels) partial[key].set(fullFrame ? message[key].subarray(start, end) : message[key], start);
  for (const key of xyzChannels) partial[key].set(fullFrame ? message[key].subarray(3 * start, 3 * end) : message[key], 3 * start);
  for (let row = rowStart; row < rowStart + rowCount; row++) partial.rows[row] = message.spatialSamplesIntegrated ?? 1;
  partial.rowsComplete = partial.rows.reduce((count, samples) => count + (samples === partial.spatialSamples ? 1 : 0), 0);
  partial.lightingComplete = message.lightingComplete;
  partial.diagnostics = message.diagnostics; partial.configuration = message.configuration;
  paintFrame(partial, request, true, rowStart, rowCount);
  $('trNative').textContent = '512 × 512 · 未完成';
  $('trProgress').value = Math.min(.99, Math.max(Number($('trProgress').value), partial.rows.reduce((sum, samples) => sum + samples, 0) / (TRANSPORT_RESOLUTION * partial.spatialSamples) * (message.lightingComplete === false ? .5 : 1)));
  status(partial.spatialSamples > 1 ? `空间光源积分预览：${partial.rowsComplete} / 512 行完成全部 ${partial.spatialSamples} 个发光点与光谱，尚未完成。四幅图共用曝光。` : `已计算 ${partial.rowsComplete} / 512 行的全部光谱。四幅图共用曝光；黑色未计算区域不是物理零值。`);
  if (message.lightingComplete === false) status('点源照明预览；环境与一次漫反射间接照明尚未完成，当前亮度不是最终结果。');
  if (message.overview) showOverview(message.overview);
}
/** Bounded local evidence only. Never include imported source text or images. */
export function boundedTransportReport(value, depth = 0) {
  if (depth > 6 || ArrayBuffer.isView(value) || value instanceof ArrayBuffer) return undefined;
  if (typeof value === 'string') return value.slice(0, 2048);
  if (typeof value === 'number') return finite(value) ? value : String(value);
  if (value === null || typeof value === 'boolean') return value;
  if (Array.isArray(value)) return value.slice(0, 32).map(item => boundedTransportReport(item, depth + 1)).filter(item => item !== undefined);
  if (value && typeof value === 'object') {
    const out = {};
    for (const [key, item] of Object.entries(value).slice(0, 50)) {
      if (/^(customOBJ|scene|meshes|faces|obj|image|pixels|vertices|triangles|mesh|overview|sourceText|objText)$/i.test(key)) continue;
      const cleaned = boundedTransportReport(item, depth + 1);
      if (cleaned !== undefined) out[key.slice(0, 128)] = cleaned;
    }
    return out;
  }
}
function backendLabel(backend) {
  if (backend?.kind === 'cpu') return 'CPU 几何 + CPU 复振幅叠加';
  if (backend?.kind === 'hybrid') return 'CPU 几何 + WebGPU 复振幅叠加';
  const raw = typeof backend === 'string' ? backend : backend?.label ?? backend?.description ?? backend?.actual ?? backend?.used ?? backend?.name;
  const label = boundedText(raw, '详见实际后端记录');
  return /gpu/i.test(label) && !/cpu/i.test(label) ? `CPU 几何 + ${label} 相干叠加` : label;
}
function exportRecord(record) {
  lastExport = boundedTransportReport(record);
  let text = JSON.stringify(lastExport, null, 2);
  if (text.length > 64000) { lastExport = {...lastExport, diagnostics: {note: '诊断超过导出上限，仅保留参数与阶段耗时。'}}; text = JSON.stringify(lastExport, null, 2); }
  $('trReport').textContent = text; $('trDownload').disabled = false;
}
function diffractionSummary(diagnostics, enabled, mixed) {
  if (!enabled) return '本次未启用衍射；几何边缘不添加波动修正。';
  const labels = {'no-interior-finite-edge-saddle': '有限边无内部驻相点', 'nonideal-or-nonreflecting-edge-material': '非理想／吸收边界未建模', 'edge-material-diffraction-not-modeled': '边缘材料衍射未建模', 'endpoint-dominated-diffraction-not-modeled': '端点主导区域未建模', 'endpoint-stationary-point-not-modeled': '端点驻相点未建模', 'occluded-edge-leg': '源／接收边腿被遮挡', 'edge-grazing-not-modeled': '沿边掠射未建模', 'face-grazing-not-modeled': '沿面掠射未建模', 'inside-solid-angular-sector': '位于实体角域', 'incident-inside-or-tangent-to-wedge': '入射位于楔内或切向', 'unsupported-wedge-topology': '不支持的楔拓扑'};
  const entries = [], seen = new Set();
  for (const container of [diagnostics, diagnostics?.diffraction]) {
    if (!container || typeof container !== 'object') continue;
    for (const key of ['edgeWarningCounts', 'edgeExclusionCounts', 'warningCounts', 'excludedCounts', 'exclusionCounts', 'edgeExclusions', 'diffractionWarnings', 'diffractionWarningCounts', 'diffractionExclusions', 'diffractionExclusionCounts']) {
      const counts = container[key]; if (!counts || typeof counts !== 'object' || Array.isArray(counts)) continue;
      for (const [reason, count] of Object.entries(counts)) if (finite(count) && count > 0 && !seen.has(`${reason}:${count}`)) { seen.add(`${reason}:${count}`); entries.push(`${labels[reason] ?? reason}：${count.toLocaleString()}`); }
    }
  }
  return `一阶 UTD 已开启；${mixed ? '包含准入的 R–D / D–R；' : '未计入 R–D / D–R；'}角点与连续两次衍射未计入。${entries.length ? '诊断事件计数（接收采样累计，并非独立边缘数）：' + entries.slice(0, 12).join('；') + '。' : '准入与排除明细见下方诊断。'}`;
}
// Keep numerical records in diagnostics, and actionable model limitations beside the result.
function showResultCautions(result, request, warnings) {
  const d = result.diagnostics ?? {}, messages = [];
  const raw = Array.isArray(warnings) ? warnings : [warnings];
  for (const value of raw.slice(0, 12)) { const text = boundedText(value); if (text) messages.push(text); }
  if (d.modelIncomplete || d.cameraSlabUnsupportedSamples > 0 || d.sourceSlabUnsupportedSamples > 0 || d.lightingSlabUnsupportedRays > 0) messages.push(`模型不完整：存在未建模的玻璃路径（相机 ${d.cameraSlabUnsupportedSamples ?? 0}、光源 ${d.sourceSlabUnsupportedSamples ?? 0}、环境／间接光 ${d.lightingSlabUnsupportedRays ?? 0}）。相关暗区不能解释为物理黑色或吸收。`);
  if (d.cameraMirrorLimitSamples > 0) messages.push(`有 ${d.cameraMirrorLimitSamples} 个相机采样因镜面次数上限终止；这些黑色区域不代表物理吸收。`);
  if (d.cameraSlabTruncatedPowerSum > 0) messages.push('玻璃板内反射采用有限阶截断，仍有未追踪的尾项；权重记录见下方诊断。');
  if (includesDiffraction(result, request)) {
    const counts = [...Object.values(d.edgeWarningCounts ?? {}), ...Object.values(d.edgeExclusionCounts ?? {})];
    if (d.samplesWithModelWarnings > 0 || counts.some(v => finite(v) && v > 0)) messages.push('部分衍射采样触及模型边界或被排除；结果不包含被排除的路径。原因与计数见下方诊断。');
  }
  if (d.fallbackReason) messages.push(`已回退到 CPU 完成计算：${boundedText(d.fallbackReason)}`);
  $('trCriticalWarnings').textContent = messages.join('\n');
}

function resultEvidence(result, request) {
  const totalMs = finite(result.timings?.totalMs) ? result.timings.totalMs : finite(result.totalMs) ? result.totalMs : performance.now() - started;
  $('trBackend').textContent = backendLabel(result.backend);
  $('trNative').textContent = '512 × 512'; $('trElapsed').textContent = `${(totalMs / 1000).toFixed(2)} s`;
  $('trPathSummary').textContent = `${request.blockReflected ? '光源照明反射路径已屏蔽' : `光源照明反射阶数 ≤ ${request.maxReflections}`} · ${request.diffraction ? request.mixedDiffraction ? '一次 UTD + R–D / D–R' : '一次 UTD' : '无衍射'} · γ = ${request.coherence.toFixed(2)}${request.measurement === 'camera' ? ` · 每段几何链镜面次数 ≤ ${request.camera.mirrorBounces ?? 2}` : ''}`;
  const spectrum = request.spectrum ?? request.source?.spectrum ?? request.scene?.source?.spectrum;
  const spectralDescription = !spectrum || spectrum.mode === 'monochrome' ? `单色 λ = ${request.wavelengthNm} nm` : spectrum.mode === 'broadband' ? `${spectrum.illuminant} 宽带，360–830 nm，${result.spectralSamples} 个随机分层光谱样本 / 空间积分位置` : spectrum.mode === 'gaussian' ? `高斯谱中心 ${spectrum.centerNm} nm，FWHM ${spectrum.fwhmNm} nm，${result.spectralSamples} 个随机分层光谱样本 / 空间积分位置` : `${result.spectralSamples} 条离散谱线`;
  const spatial = request.scene?.source?.spatial ?? request.source?.spatial;
  const spatialDescription = spatial?.radiusMm > 0 ? `均匀圆盘半径 ${spatial.radiusMm} mm，${spatial.samples} 个独立发光点按固定总功率加权求和` : '点源（半径 0）';
  $('trMeasurement').textContent = `${request.measurement === 'camera' ? `物体成像：针孔射线经至多 ${request.camera.mirrorBounces ?? 2} 次双面理想镜反射到终端漫反射表面；沿途镜面反射率乘积传递终端辐亮度。投影标量平方律照明近似与一次 Lambertian-like 终端闭合；不含相机侧相干混合／镜头衍射及多次漫反射全局光照。${request.lighting ? '已启用几何／非相干照明扩展：按方向可见性积分天空，最多一次漫反射间接反弹；镜面预算分别用于主相机链、次级表面链及天空链，不是全路径总上限；不是常数环境补亮，真实环境光会降低条纹对比度。' : '环境与间接照明关闭。'}` : ''}${spatialDescription}；${spectralDescription}；像素积分 ${request.pixelQuadrature} × ${request.pixelQuadrature}。源强度 P 为全光谱积分；单个发光点自由空间参考归一化为其功率 /(4πL²)，长度单位 mm。理想局部平方律标量响应；探测器积分不是标定瓦数或精确法向通量。固定基准 +16 EV，再加共用曝光，无逐图归一化。`;
  const warnings = result.diagnostics?.warnings ?? result.warnings ?? [];
  const notes = Array.isArray(warnings) ? warnings.slice(0, 12).map(value => boundedText(value)).filter(Boolean) : [boundedText(warnings)];
  if (request.measurement === 'camera') {
    const diagnostics = result.diagnostics ?? {};
    if (request.lighting?.environment || request.lighting?.indirectSamples) notes.push(`照明扩展：环境射线 ${diagnostics.lightingEnvironmentRays ?? '未提供'}；逃逸至天空 ${diagnostics.lightingEnvironmentEscapes ?? '未提供'}；次级漫反射命中 ${diagnostics.lightingSecondaryHits ?? '未提供'}；间接源求值 ${diagnostics.lightingSourceSamples ?? '未提供'}。`);
    notes.push(`相机采样：漫反射命中 ${diagnostics.cameraHitSamples ?? '未提供'}；经过镜面的采样 ${diagnostics.cameraMirrorSamples ?? '未提供'}；镜面交互总次数 ${diagnostics.cameraMirrorBounces ?? '未提供'}；镜后漫反射命中 ${diagnostics.cameraReflectedHits ?? '未提供'}；镜面次数上限 ${diagnostics.cameraMirrorLimitSamples ?? '未提供'}；零反射率吸收 ${diagnostics.cameraAbsorbedSamples ?? '未提供'}；不透明终止 ${diagnostics.cameraOpaqueSamples ?? '未提供'}；未命中 ${diagnostics.cameraMissSamples ?? '未提供'}；漫反射背面 ${diagnostics.cameraBackfaceSamples ?? '未提供'}；不透明终止（旧版计数）${diagnostics.cameraUnsupportedSamples ?? '未提供'}。次数上限、不透明终止与漫反射背面显示黑色，但不代表物理吸收；零反射率镜面吸收也为黑色。未命中时，启用环境则显示逃逸方向的天空，否则为黑色。`);
  }
  if (request.slab) { const d = result.diagnostics ?? {}; notes.push(`有限玻璃板：相机板交互采样 ${d.cameraSlabSamples ?? '未提供'}；相机不支持侧面退出 ${d.cameraSlabUnsupportedSamples ?? '未提供'}；源端不支持采样 ${d.sourceSlabUnsupportedSamples ?? '未提供'}；天空／间接链不支持射线 ${d.lightingSlabUnsupportedRays ?? '未提供'}。${d.modelIncomplete ? '模型不完整：存在未建模的玻璃路径；相关暗区不是物理黑色或吸收。' : '不支持路径的暗区不能解释为物理黑色或吸收。'}截断路径权重和 ${d.cameraSlabTruncatedPowerSum ?? '未提供'}（不是归一化功率）；源透射路径 ${d.transmittedRoutes ?? '未提供'}，其中镜面＋透射 ${d.reflectedTransmittedRoutes ?? '未提供'}。源路径支持首次透射与一次镜面反射的两种顺序，不含相干 Fabry–Pérot；相机使用有限阶 Fresnel 分支采样。`); }
  if (result.diagnostics?.fallbackReason) notes.push(`本次最终结果使用 CPU：${boundedText(result.diagnostics.fallbackReason)}`);
  $('trWarnings').textContent = notes.filter(Boolean).join('\n');
  showResultCautions(result, request, warnings);
  $('trDiffractionSummary').textContent = diffractionSummary(result.diagnostics, includesDiffraction(result, request), request.mixedDiffraction);
  exportRecord({schema: 'coherent-mesh-local-run-v1', generatedAt: new Date().toISOString(), complete: true,
    request: {...request, id: undefined, importedScene: customScene !== null, importedOBJ: customOBJ !== null, importedFile: fileName || undefined, importedBytes: customOBJ ? new TextEncoder().encode(customOBJ).length : 0},
    configuration: result.configuration, width: result.width, height: result.height, backend: result.backend, timings: result.timings,
    totalMs, clientElapsedThroughFinalCanvasWriteMs: Math.max(0, performance.now() - started), diagnostics: result.diagnostics, scope: result.scope, meshTriangles: result.overview?.mesh?.triangles?.length,
    display: {baseExposureEV: TRANSPORT_BASE_EXPOSURE_EV, sharedExposureAtCompletionEV: Number($('trExposure').value), mapping: 'solver spectral XYZ → fixed sRGB exponential compression; no image-peak normalization; signed coherentXYZ.Y − incoherentXYZ.Y uses blue-negative/orange-positive diagnostic palette', spectralComplete: result.spectralComplete, spectralSamples: result.spectralSamples, spectralEstimator: result.spectralEstimator, spatialComplete: result.spatialComplete, spatialSamples: result.spatialSamples, lightingComplete: result.lightingComplete}});
}
function fail(message, details = {}) {
  const request = activeRequest, elapsedMs = Math.max(0, performance.now() - started);
  stop(); clearResult(); status(message);
  $('trBackend').textContent = '未完成'; $('trElapsed').textContent = `${(elapsedMs / 1000).toFixed(2)} s · 未完成`;
  $('trWarnings').textContent = '本次未发布完成图像。可调整参数后重新运行；旧图与未完成分块均已清除。';
  exportRecord({schema: 'coherent-mesh-local-run-v1', generatedAt: new Date().toISOString(), complete: false, request, elapsedMs, error: message, errorName: details.name, backend: details.backend, timings: details.timings, diagnostics: details.diagnostics});
}
function run() {
  if (busy || fileReading) return;
  stop(); clearResult(); started = performance.now();
  let request;
  try { request = requestFor(generation); showOverview(prepareCoherentMeshTask(request).overview); } catch (error) { status(`无法开始：${error.message}`); return; }
  const token = generation;
  activeRequest = request; busy = true;
  syncRunningControls();
  $('trProgress').hidden = false; $('trProgress').value = 0;
  $('trBackend').textContent = 'CPU 几何 · 正在选择叠加后端';
  status('正在逐波长计算完整路径传输；原生 512 × 512，每个行块完成全部光谱后显示。');
  timer = setInterval(() => { if (busy && generation === token) $('trElapsed').textContent = `${((performance.now() - started) / 1000).toFixed(1)} s · 计算中`; }, 250);
  try {
    const instance = new Worker(new URL('./transport-worker.js', import.meta.url), {type: 'module'});
    worker = instance;
    const current = () => worker === instance && generation === token && busy;
    instance.onmessage = ({data: message}) => {
      if (!current() || !message || message.id !== token) return;
      const attempt = message.type === 'result' ? message.result?.diagnostics?.attempt ?? 0 : message.attempt ?? activeAttempt;
      if (Number.isSafeInteger(attempt) && attempt < activeAttempt) return;
      if (message.type === 'reset' || message.type === 'progress' && message.reset === true) {
        if (attempt === activeAttempt) return;
        if (attempt !== activeAttempt + 1 || attempt > 1) { fail('计算重试记录被拒绝。'); return; }
        activeAttempt = attempt; partial = null; clearImages(); $('trProgress').value = 0;
        if (message.backend) $('trBackend').textContent = backendLabel(message.backend);
        status('GPU 叠加未完成，已清除未完成分块；正在 CPU 上从头计算。');
      } else if (attempt !== activeAttempt) { fail('计算批次不匹配。');
      } else if (message.type === 'progress') {
        if (finite(message.fraction)) $('trProgress').value = Math.max(Number($('trProgress').value), Math.min(.99, Math.max(0, message.fraction)));
        status(boundedText(message.message, '正在计算完整路径传输。'));
        if (message.backend) $('trBackend').textContent = backendLabel(message.backend);
      } else if (message.type === 'preview') {
        try { acceptPreview(message, request); } catch (error) { fail(`分块预览被拒绝：${error.message}`); }
      } else if (message.type === 'result') {
        try {
          const result = validateTransportFrame(message.result, request).frame;
          if (result.overview) showOverview(result.overview);
          lastResult = result; partial = null; paintFrame(result, request, false); resultEvidence(result, request);
          stop();
          status(result.diagnostics?.modelIncomplete ? '计算结束，但模型不完整：存在未建模的玻璃路径。相关暗区不是物理吸收；请查看下方路径诊断。四图为原生 512 × 512。' : '计算完成：四幅图均为原生 512 × 512。改变共用曝光可观察暗部；改变传播参数需重新运行。');
        } catch (error) { fail(`结果被拒绝：${error.message}`); }
      } else if (message.type === 'error') fail(`计算未完成：${boundedText(message.error, '未知计算错误')}`, message);
    };
    instance.onerror = event => { if (current()) { event.preventDefault?.(); fail(`计算线程失败：${boundedText(event.message, '无法执行计算')}`); } };
    instance.onmessageerror = () => { if (current()) fail('计算线程数据无法读取。请重新运行。'); };
    instance.postMessage(request);
  } catch (error) { fail(`无法启动计算：${error.message}`); }
}
for (const id of physicalInputs) $(id).addEventListener('input', () => {
  if (id === 'trMeasurementMode' && $('trMeasurementMode').value === 'camera') cameraPresent = true;
  if (id === 'trColorFringeView' && customScene === null && $('trPreset').value === 'lloyd') { setDetectorPose(presetDetector()); poseEdited = true; }
  if (id === 'trPreset') { fileGeneration++; fileReading = false; fileError = ''; fileName = customOBJ ? fileName : ''; customScene = null; $('trSceneJSON').value = ''; presetPose(); $('trDiffraction').checked = ['double-slit', 'reflected-edge', 'object-imaging'].includes($('trPreset').value); $('trMixedDiffraction').checked = $('trPreset').value === 'reflected-edge'; if ($('trMixedDiffraction').checked) { $('trOrder').value = '2'; $('trBlockReflected').checked = false; } }
  if (id === 'trPreset' && $('trPreset').value === 'object-imaging') { $('trExposure').value = '2'; $('trSpectrumMode').value = 'broadband'; $('trSpectrumSource').value = 'D65'; $('trQuadrature').value = '1'; $('trSpectralQuality').value = '16'; setSourceSpatial(undefined); }
  if (!$('trDiffraction').checked || Number($('trOrder').value) < 2 || $('trBlockReflected').checked) $('trMixedDiffraction').checked = false;
  if (id === 'trPreset' || id === 'trDiffraction' && $('trDiffraction').checked && $('trPreset').value !== 'reflected-edge') $('trPhase').value = '0';
  invalidate();
});
for (const id of slabInputs) $(id).addEventListener('input', () => {
  if (id === 'trSlabIndex') slabState.index = inputNumber(id);
  if (id === 'trSlabThickness') slabState.thickness = inputNumber(id);
  if (id === 'trSlabInternalReflections') slabState.internalReflections = inputNumber(id);
  if (id === 'trSlabAbsorption') slabState.absorption = [{wavelengthNm: 360, value: inputNumber(id)}, {wavelengthNm: 830, value: inputNumber(id)}];
  invalidate();
});
for (const [id, kind] of [['trGlassCameraPreset', 'camera'], ['trGlassInterferencePreset', 'detector'], ['trGlassMirrorPreset', 'mirror'], ['trGlassMirrorCameraPreset', 'camera-mirror']]) $(id).addEventListener('click', () => {
  defaults();
  const scene = makeTransportSlabScene(kind), cameraMode = kind === 'camera' || kind === 'camera-mirror';
  customScene = scene; $('trColorFringeView').checked = false;
  setPose(scene.source, scene.detector); setCamera(scene.camera); setLighting(scene.lighting); setSlab(scene.slab); setSpectrum(scene.source);
  $('trMeasurementMode').value = cameraMode ? 'camera' : 'detector'; $('trPower').value = String(scene.source.power);
  $('trOrder').value = kind === 'camera' ? '0' : '1'; $('trDiffraction').checked = false; $('trMixedDiffraction').checked = false;
  $('trQuadrature').value = '1'; $('trSpectralQuality').value = '16'; $('trExposure').value = cameraMode ? '2' : '-2';
  invalidate();
  status(kind === 'camera-mirror' ? '已设置玻璃后彩色物体＋一次镜面光源照明：保留天空、玻璃和相机镜面链；源端支持镜面与透射组合，未启用绕射。原生 512 × 512。天空侧面退出仍可能标记模型不完整。点击运行。' : kind === 'mirror' ? '已设置 D65 宽带镜面＋玻璃透射实验：折射后反射的驻相路径与直达参考臂叠加，原生 512 × 512。不是透明度混色；也未启用绕射或两次反射穿板。点击运行。' : kind === 'camera' ? '已设置有限厚玻璃后的真实彩色物体：D65、天空 8 样本、间接光关闭，原生 512 × 512，共用总曝光 +18 EV。点击“运行引擎”计算。' : '已设置 550 nm 单色相位实验：玻璃直达臂与绕过玻璃的镜面参考臂，原生 512 × 512，共用总曝光 +14 EV。绿色来自单一波长；宽带彩色成像请选玻璃后彩色物体。厚度改变约 0.0011 mm 对应一个近法向相位周期。点击“运行引擎”计算。');
});
for (const id of lightingInputs) $(id).addEventListener('input', () => {
  if (id === 'trEnvironmentRadiance') savedEnvironment.radiance = inputNumber(id) / 1e6;
  if (id === 'trEnvironmentUpper') savedEnvironment.upper = inputNumber(id);
  if (id === 'trEnvironmentLower') savedEnvironment.lower = inputNumber(id);
  if (id === 'trEnvironmentSamples') savedEnvironment.samples = inputNumber(id);
  lightingState = {environment: $('trEnvironmentEnabled').checked ? structuredClone(savedEnvironment) : null, indirectSamples: $('trIndirectEnabled').checked ? inputNumber('trIndirectSamples') : 0};
  invalidate();
});
$('trLightingPreset').addEventListener('click', () => {
  setLighting({environment: {...defaultEnvironment(), lower: .25}, indirectSamples: 4}); invalidate();
  status('已为当前物体场景设置方向天空：辐亮度 2 × 10⁻⁶、上／下半球 1 / 0.25、8 个天空样本与 4 个一次漫反射样本。原生 512 × 512、光源与曝光保持不变。点击“运行引擎”开始计算。');
});
for (const id of cameraInputs) $(id).addEventListener('input', () => { cameraPresent = true; invalidate(); });
for (const [id, quadrature, spectralSamples] of [['trCameraFast', 1, 16], ['trCameraFine', 2, 32]]) $(id).addEventListener('click', () => {
  $('trQuadrature').value = String(quadrature); $('trSpectralQuality').value = String(spectralSamples);
  invalidate();
  status(`已设置${quadrature === 1 ? '快速' : '精细'}采样：原生 512 × 512、${quadrature} × ${quadrature} 像素积分、${spectralSamples} 连续光谱样本。共用曝光保持不变。点击“运行引擎”开始计算。`);
});
for (const id of spatialInputs) $(id).addEventListener('input', () => { spatialPresent = true; poseEdited = true; invalidate(); });
for (const id of poseInputs) $(id).addEventListener('input', () => { if (id.startsWith('trDetector')) $('trColorFringeView').checked = false; poseEdited = true; invalidate(); });
$('trForm').addEventListener('submit', event => event.preventDefault());
$('trRun').addEventListener('click', run);
$('trRunQuick').addEventListener('click', run);
$('trCancelQuick').addEventListener('click', () => $('trCancel').click());
$('trCancel').addEventListener('click', () => { stop(); clearResult(); status('已取消，未完成图像已清除。可以调整参数后重新运行。'); });
$('trReset').addEventListener('click', () => { defaults(); stop(); clearResult(); updateGeometry(); view.resetCamera(); status('已恢复 D65 宽带与 Lloyd 彩色条纹观察区域（真实探测器 6 × 1 mm）。点击“运行引擎”开始计算。'); });
$('trResetCamera').addEventListener('click', () => view.resetCamera());
for (const [id, radius] of [['trPointComparison', 0], ['trDiskComparison', .06]]) $(id).addEventListener('click', () => {
  const exposure = $('trExposure').value;
  defaults(); $('trExposure').value = exposure;
  $('trQuadrature').value = '1'; $('trSpectralQuality').value = '16';
  setSourceSpatial({type: 'disk', radiusMm: radius, normal: [0, 0, 1], samples: 16, seed: 7391}); poseEdited = true;
  invalidate();
  status(`已设置${radius === 0 ? '点源' : '有限圆盘'}对照：Lloyd / D65，半径 ${radius} mm；512 × 512、1 × 1 像素积分、16 光谱样本、16 发光点。共用曝光保持不变。点击“运行引擎”开始计算。`);
});
$('trExposure').addEventListener('input', () => {
  const exposure = Number($('trExposure').value);
  if (!finite(exposure) || exposure < -8 || exposure > 4) return;
  labels(); if (lastResult) paintFrame(lastResult, activeRequest, false); else if (partial) paintFrame(partial, activeRequest, true);
});
$('trOBJ').addEventListener('change', async () => {
  const file = $('trOBJ').files?.[0]; if (!file) return;
  const token = ++fileGeneration;
  fileReading = true; fileError = ''; customOBJ = null; customScene = null; fileName = file.name;
  stop(); clearResult(); labels(); view.setScene(null); $('trMeshCount').textContent = '读取 OBJ'; status('正在本地读取 OBJ；不请求材质或纹理。');
  try {
    ensure(file.size <= MAX_OBJ_BYTES, 'OBJ 超过 2 MiB 上限。');
    ensure(/\.obj$/i.test(file.name), '请选择 .obj 文件。');
    const text = await file.text(); if (token !== fileGeneration) return;
    ensure(new TextEncoder().encode(text).length <= MAX_OBJ_BYTES && text.trim().length > 0, 'OBJ 为空或超过 2 MiB 上限。');
    customOBJ = text; fileReading = false;
    const prepared = prepareCoherentMeshTask(requestFor(Math.max(1, generation))); showOverview(prepared.overview);
    status('OBJ 已在本地读取并检查。选择边界后运行引擎。');
  } catch (error) {
    if (token !== fileGeneration) return;
    customOBJ = null; fileReading = false; fileError = `OBJ 未通过检查：${boundedText(error.message)}`;
    view.setScene(null); $('trMeshCount').textContent = 'OBJ 未通过检查'; $('trGeometryNote').textContent = fileError; status(fileError);
  } finally { if (token === fileGeneration) { fileReading = false; syncRunningControls(); labels(); } }
});
$('trClearOBJ').addEventListener('click', () => { fileGeneration++; customScene = null; customOBJ = null; fileName = ''; fileError = ''; fileReading = false; $('trOBJ').value = ''; $('trSceneJSON').value = ''; invalidate(); });
function downloadScene() {
  ensure(!fileReading && !fileError, fileError || '请等待读取完成。');
  const request = requestFor(Math.max(1, generation));
  let scene = request.scene;
  if (!scene) { const data = makeCoherentMeshPreset(request.preset, request); scene = sceneFromCoherentData({...data, slab: request.slab, ...(request.lighting ? {lighting: request.lighting} : {}), source: currentSource(), detector: currentDetector(), ...(cameraPresent || request.measurement === 'camera' ? {camera: currentCamera()} : {})}); }
  if (request.slab || scene.renderSettings) scene = {...scene, renderSettings: {maxReflections: request.maxReflections, diffraction: request.diffraction, mixedDiffraction: request.mixedDiffraction}};
  const text = serializeCoherentScene(scene), blob = new Blob([text], {type: 'application/json'}), url = URL.createObjectURL(blob);
  downloads.add(url); const link = document.createElement('a'); link.href = url; link.download = 'coherent-mesh-scene.json'; document.body.append(link); link.click(); link.remove();
  setTimeout(() => { URL.revokeObjectURL(url); downloads.delete(url); }, 1000);
}
$('trExportScene').addEventListener('click', () => { try { downloadScene(); status('已在本地导出完整场景 JSON，可修改后重新导入。诊断下载不包含网格原文。'); } catch (error) { status(`场景导出失败：${boundedText(error.message)}`); } });
$('trUsePreset').addEventListener('click', () => { fileGeneration++; fileReading = false; fileError = ''; fileName = ''; customScene = null; customOBJ = null; $('trSceneJSON').value = ''; $('trOBJ').value = ''; presetPose(); invalidate(); });
$('trSceneJSON').addEventListener('change', async () => {
  const file = $('trSceneJSON').files?.[0]; if (!file) return;
  const token = ++fileGeneration;
  fileReading = true; fileError = ''; customOBJ = null; customScene = null; fileName = file.name; $('trOBJ').value = '';
  stop(); clearResult(); labels(); view.setScene(null); $('trMeshCount').textContent = '读取场景 JSON'; status('正在本地读取完整场景 JSON；不会读取外部文件。');
  try {
    ensure(file.size <= COHERENT_SCENE_LIMITS.maxBytes, '场景 JSON 超过 2 MiB 上限。'); ensure(/\.json$/i.test(file.name), '请选择 .json 文件。');
    const text = await file.text(); if (token !== fileGeneration) return;
    const scene = parseCoherentScene(text);
    customScene = scene; fileReading = false; poseEdited = false; $('trColorFringeView').checked = false; setPose(scene.source, scene.detector); setCamera(scene.camera); setLighting(scene.lighting); setSlab(scene.slab); $('trMeasurementMode').value = scene.camera ? 'camera' : 'detector';
    $('trPower').value = String(scene.source.power); setSpectrum(scene.source); $('trPhase').value = '0';
    if (scene.renderSettings) { $('trOrder').value = String(scene.renderSettings.maxReflections); $('trDiffraction').checked = scene.renderSettings.diffraction; $('trMixedDiffraction').checked = scene.renderSettings.mixedDiffraction; }
    const prepared = prepareCoherentMeshTask(requestFor(Math.max(1, generation))); showOverview(prepared.overview);
    status(scene.slab ? (scene.renderSettings ? '玻璃场景已导入，并恢复 JSON 中显式保存的源反射／衍射运行选项。点击运行开始计算。' : '玻璃场景已导入；文件未保存运行选项，保留当前设置。玻璃成像示例需要源反射 0、衍射关闭；玻璃干涉示例需要源反射 1、衍射关闭。请核对。') : '完整场景已在本地检查并替换预设几何。可编辑光源、探测器与可选几何针孔相机后运行。');
  } catch (error) {
    if (token !== fileGeneration) return;
    customScene = null; fileReading = false; fileError = `场景未通过检查：${boundedText(error.message)}`;
    view.setScene(null); $('trMeshCount').textContent = '场景未通过检查'; $('trGeometryNote').textContent = fileError; status(fileError);
  } finally { if (token === fileGeneration) { fileReading = false; syncRunningControls(); labels(); } }
});
$('trDownload').addEventListener('click', () => {
  if (!lastExport) return;
  try {
    const blob = new Blob([JSON.stringify(lastExport, null, 2) + '\n'], {type: 'application/json'}), url = URL.createObjectURL(blob);
    downloads.add(url); const link = document.createElement('a'); link.href = url; link.download = `coherent-mesh-${lastExport.complete ? 'run' : 'failure'}.json`; document.body.append(link); link.click(); link.remove();
    setTimeout(() => { URL.revokeObjectURL(url); downloads.delete(url); }, 1000);
  } catch (error) { status(`诊断下载失败：${error.message}`); }
});
window.addEventListener('pagehide', () => {
  const interrupted = busy || fileReading, interruptedRead = fileReading;
  fileGeneration++; fileReading = false; stop();
  if (interruptedRead) { customScene = null; customOBJ = null; fileName = ''; fileError = ''; $('trOBJ').value = ''; $('trSceneJSON').value = ''; presetPose(); }
  if (interrupted) { clearResult(); labels(); updateGeometry(); status('离开页面时已取消未完成计算或读取。返回后可重新运行。'); }
  for (const url of downloads) URL.revokeObjectURL(url); downloads.clear();
});
// These disclosures are interface state only; toggling them never restarts a calculation.
document.querySelector('.tr-method-link').addEventListener('click', () => { $('trScope').open = true; });
$('trControlsTitle').setAttribute('tabindex', '-1');
for (const id of ['trRun', 'trRunQuick']) $(id).addEventListener('click', () => {
  if (window.matchMedia?.('(max-width: 760px)').matches) $('trStatus').scrollIntoView?.({block: 'center', behavior: 'auto'});
});
window.addEventListener('resize', () => view.draw());
globalThis.coherentMeshLab = {run, cancel: () => $('trCancel').click(), getState: () => ({busy, generation, complete: !!lastResult, backend: lastResult?.backend, imported: customOBJ !== null, customScene: customScene !== null, poseEdited, fileReading}), getResult: () => lastResult, getViewScene: () => view.getScene()};
defaults(); clearResult(); updateGeometry();
