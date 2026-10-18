/** Camera canvases contain computed XYZ only. The separate apparatus view is
 * a projection of the scene geometry, never a substitute for a camera result. */
import {displaySpectralXYZ, linearToSRGB} from './plt-spectrum.js';
import {createPinholeScene, pinholeBasisLinearRGB, pinholeSceneOverview, PINHOLE_BASIS_NAMES} from './pinhole-scene.js';

const $ = id => document.getElementById(id);
const FINAL_RESOLUTION = 512;
const BASE_EXPOSURE_EV = 16;
const transportIds = ['phDiameter', 'phMode', 'phSpectrum', 'phWavelength', 'phCompute', 'phLayers'];
const floatArray = value => value instanceof Float32Array || value instanceof Float64Array;
const finite = value => typeof value === 'number' && Number.isFinite(value);
const ensure = (value, message) => { if (!value) throw new Error(message); };
const boundedText = (value, fallback = '') => typeof value === 'string' ? value.slice(0, 2048) : fallback;
let worker = null, generation = 0, activeRequest = null, lastResult = null, lastPartial = null, lastExport = null;
let activeAttempt = 0;
let busy = false, startTime = 0, timer = null, currentOverview = null;
const downloads = new Set();
const scene = createPinholeScene();
const basisColors = pinholeBasisLinearRGB();
const initialOverview = pinholeSceneOverview(scene, {diameterMm: .2, resolution: FINAL_RESOLUTION});
const status = value => { $('phStatus').textContent = value; };

function stop() {
  const old = worker, oldId = activeRequest?.id;
  worker = null;
  generation++;
  if (old) { try { old.postMessage({type: 'cancel', id: oldId}); } catch {} old.terminate(); }
  if (timer !== null) { clearInterval(timer); timer = null; }
  busy = false;
  $('phRun').disabled = false;
  $('phRun').setAttribute('aria-busy', 'false');
  $('phCancel').hidden = true;
  $('phProgress').hidden = true;
}
function clearCameraImages() {
  for (const name of ['Geometric', 'Wave']) {
    const canvas = $(`ph${name}Canvas`);
    canvas.width = FINAL_RESOLUTION; canvas.height = FINAL_RESOLUTION;
    canvas.dataset.complete = 'false';
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    $(`ph${name}Empty`).hidden = false;
    $(`ph${name}PartialWrap`).hidden = true;
    $(`ph${name}Resolution`).textContent = '512 × 512 原生输出 · 尚未计算';
  }
  const diff = $('phDifferenceCanvas'); diff.getContext('2d').clearRect(0, 0, diff.width, diff.height);
  $('phDifferenceNote').textContent = '完成后可查看。差值为 CIE Y 的绝对差，使用与相机相同的曝光，无单独归一化。';
}
function clearResult() {
  lastResult = null; lastPartial = null; lastExport = null; activeRequest = null; currentOverview = null; activeAttempt = 0;
  clearCameraImages();
  for (const id of ['phNative', 'phSpectralCount', 'phElapsed', 'phExecution']) $(id).textContent = '—';
  $('phBackend').textContent = '尚未计算'; $('phWarnings').textContent = '';
  $('phMeasurement').textContent = '同一孔径、同一曝光。结果完成后显示实际后端、分阶段耗时与数值记录。';
  $('phReport').textContent = '尚无结果'; $('phDownload').disabled = true;
}
function labels() {
  const d = Number($('phDiameter').value), wavelength = Number($('phWavelength').value);
  $('phDiameterValue').textContent = `${d.toFixed(3)} mm`;
  $('phWavelengthValue').textContent = `${wavelength} nm`;
  $('phExposureValue').textContent = `${Number($('phExposure').value) > 0 ? '+' : ''}${$('phExposure').value} EV`;
  $('phOrbitValue').textContent = `${$('phOrbit').value}°`;
  const narrow = $('phSpectrum').value === 'narrowband';
  $('phWavelength').disabled = !narrow;
  $('phSpectrumNote').textContent = narrow ? '窄带仍按 CIE 颜色匹配函数显示；极端可见光颜色可能超出 sRGB 色域。' : '宽带模式对多波长的强度积分；彩色来自光谱，不是后期着色。';
  const relativeArea = (d / .2) ** 2;
  $('phTradeoff').textContent = `孔面积为 0.20 mm 默认孔的 ${relativeArea.toFixed(2)} 倍。小孔更暗，衍射也更宽；共用曝光不自动补亮。`;
}
function invalidate() {
  stop(); clearResult(); labels(); drawRig();
  status('参数已改变。重新计算后，两幅图都会使用新的条件。');
}
function defaults() {
  const values = {phDiameter: .2, phMode: 'scene', phSpectrum: 'broadband', phWavelength: 550, phCompute: 'auto', phLayers: 8, phExposure: 0, phOrbit: 0};
  for (const [id, value] of Object.entries(values)) $(id).value = String(value);
  labels();
}
function makeRequest(id) {
  const diameterMm = Number($('phDiameter').value), wavelengthNm = Number($('phWavelength').value), depthLayers = Number($('phLayers').value);
  const mode = $('phMode').value, spectralMode = $('phSpectrum').value, backend = $('phCompute').value;
  ensure(finite(diameterMm) && diameterMm >= .04 && diameterMm <= .5, '孔径必须在 0.04–0.50 mm 内');
  ensure(finite(wavelengthNm) && wavelengthNm >= 420 && wavelengthNm <= 680, '窄带波长必须在 420–680 nm 内');
  ensure(['scene', 'point'].includes(mode) && ['broadband', 'narrowband'].includes(spectralMode), '未知对象或光谱模式');
  ensure(['auto', 'cpu'].includes(backend) && [4, 8, 16].includes(depthLayers), '未知后端或深度层数');
  return {id, mode, diameterMm, wavelengthNm, spectralMode, resolution: FINAL_RESOLUTION, depthLayers, backend};
}
export function validatePinholeFrame(frame, request, {partial = false, previousProgressive = null, attempt = 0} = {}) {
  ensure(frame && typeof frame === 'object' && request, '没有匹配的相机请求');
  ensure(frame.complete === !partial, '完成状态与消息类型不符');
  const n = frame.width;
  ensure(Number.isInteger(n) && n === frame.height, '相机图像必须为正方形');
  ensure(n === request.resolution && n === FINAL_RESOLUTION, '相机结果不是原生 512 × 512');
  if (frame.stageResolution !== undefined) ensure(frame.stageResolution === n, '原生图像阶段尺寸不符');
  if (frame.requestedResolution !== undefined) ensure(frame.requestedResolution === request.resolution, '目标尺寸不符');
  const contribution = frame.progressive;
  if (partial || contribution || previousProgressive || request.mode === 'scene') {
    ensure(contribution && contribution.kind === 'contribution-accumulation', '缺少逐项贡献累积记录');
    ensure(Number.isSafeInteger(contribution.attempt) && [0, 1].includes(contribution.attempt) && contribution.attempt === attempt, '计算重试序号不符');
    const {completedContributions: completed, totalContributions: total, sourcePowerFraction, representedSourceFraction} = contribution;
    ensure(Number.isSafeInteger(total) && total > 0 && Number.isSafeInteger(completed) && completed > 0 && completed <= total, '贡献计数无效');
    for (const value of [sourcePowerFraction, representedSourceFraction]) ensure(finite(value) && value >= 0 && value <= 1, '源贡献比例无效');
    if (previousProgressive) {
      ensure(contribution.attempt === previousProgressive.attempt && total === previousProgressive.totalContributions, '贡献批次或总数改变');
      ensure(completed >= previousProgressive.completedContributions && sourcePowerFraction + 1e-12 >= previousProgressive.sourcePowerFraction && representedSourceFraction + 1e-12 >= previousProgressive.representedSourceFraction, '贡献累积进度发生倒退');
    }
    if (!partial) ensure(completed === total && sourcePowerFraction === 1 && representedSourceFraction === 1, '贡献尚未全部完成');
  } else ensure(attempt === 0, '最终结果缺少当前重试记录');
  for (const key of ['waveXYZ', 'geometricXYZ']) {
    const array = frame[key];
    ensure(floatArray(array) && array.length === n * n * 3, 'XYZ 数组形状无效');
    for (let i = 0; i < array.length; i++) ensure(finite(array[i]) && array[i] >= 0, 'XYZ 含非有限值或负值');
  }
  if (frame.configuration) {
    for (const key of ['mode', 'diameterMm', 'spectralMode', 'wavelengthNm', 'depthLayers']) {
      if (frame.configuration[key] !== undefined) ensure(frame.configuration[key] === request[key], `结果参数不匹配：${key}`);
    }
    if (frame.configuration.sensorWidthMm !== undefined) ensure(frame.configuration.sensorWidthMm === 8, '探测器物理尺寸改变');
    if (frame.configuration.detectorDistanceMm !== undefined) ensure(frame.configuration.detectorDistanceMm === 50, '像距改变');
  }
  if (!partial) ensure(finite(frame.totalMs) && frame.totalMs >= 0, '最终用时无效');
  return frame;
}
function paintXYZ(canvas, xyz, resolution) {
  ensure(canvas && xyz, '缺少计算数组');
  if (canvas.width !== resolution || canvas.height !== resolution) { canvas.width = resolution; canvas.height = resolution; }
  const ctx = canvas.getContext('2d'), image = ctx.createImageData(resolution, resolution);
  const exposure = BASE_EXPOSURE_EV + Number($('phExposure').value);
  for (let p = 0, q = 0; p < xyz.length; p += 3, q += 4) {
    const rgb = displaySpectralXYZ([xyz[p], xyz[p + 1], xyz[p + 2]], exposure).rgb;
    image.data[q] = rgb[0]; image.data[q + 1] = rgb[1]; image.data[q + 2] = rgb[2]; image.data[q + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
}
const contributionLabel = frame => {
  const p = frame.progressive;
  return `深度 / 材质贡献 ${p.completedContributions}/${p.totalContributions} · 源像素 / 材质覆盖 ${(100 * p.representedSourceFraction).toFixed(1)}% · 累计源 Y 光量 ${(100 * p.sourcePowerFraction).toFixed(1)}%`;
};
function paintFrame(frame, partial) {
  for (const [name, key] of [['Geometric', 'geometricXYZ'], ['Wave', 'waveXYZ']]) {
    const canvas = $(`ph${name}Canvas`);
    paintXYZ(canvas, frame[key], frame.width);
    canvas.dataset.complete = String(!partial);
    $(`ph${name}Empty`).hidden = true;
    $(`ph${name}PartialWrap`).hidden = !partial;
    if (partial) {
      $(`ph${name}PartialWrap`).querySelector('.ph-partial-label').textContent = contributionLabel(frame);
      $(`ph${name}Resolution`).textContent = '512 × 512 原生累积 · 未完成，仍缺少源光贡献';
    } else $(`ph${name}Resolution`).textContent = '512 × 512 原生计算 · 已完成';
  }
  if (!partial) paintDifference(frame);
}
function paintDifference(frame) {
  const canvas = $('phDifferenceCanvas'), ctx = canvas.getContext('2d'), image = ctx.createImageData(frame.width, frame.height);
  const gain = 2 ** (BASE_EXPOSURE_EV + Number($('phExposure').value));
  for (let p = 0; p < frame.width * frame.height; p++) {
    const value = Math.round(255 * linearToSRGB(-Math.expm1(-gain * Math.abs(frame.waveXYZ[p * 3 + 1] - frame.geometricXYZ[p * 3 + 1]))));
    image.data.set([value, value, value, 255], p * 4);
  }
  ctx.putImageData(image, 0, 0);
  $('phDifferenceNote').textContent = '原生 512 × 512 的 CIE Y 绝对差。固定基准 +16 EV 与当前曝光共用，无独立拉伸；这里是亮度差，不是颜色图。';
}
/** Keep diagnostic exports bounded and exclude images, buffers and large arrays. */
export function boundedPinholeReport(value, depth = 0) {
  if (depth > 6) return '[depth limit]';
  if (typeof value === 'string') return value.slice(0, 1024);
  if (value === null || typeof value === 'boolean') return value;
  if (typeof value === 'number') return Number.isFinite(value) ? value : String(value);
  if (ArrayBuffer.isView(value) || value instanceof ArrayBuffer) return undefined;
  if (Array.isArray(value)) return value.slice(0, 24).map(v => boundedPinholeReport(v, depth + 1)).filter(v => v !== undefined);
  if (value && typeof value === 'object') {
    const out = {};
    for (const [key, item] of Object.entries(value).slice(0, 40)) {
      if (/^(waveXYZ|geometricXYZ|image|pixels|meshes|vertices|triangles|sourceMaps|previewRGBA|sourceLayers|layers)$/i.test(key)) continue;
      const clean = boundedPinholeReport(item, depth + 1);
      if (clean !== undefined) out[key.slice(0, 128)] = clean;
    }
    return out;
  }
  return undefined;
}
function backendLabel(value) {
  if (typeof value === 'string') return value.slice(0, 160);
  if (value && typeof value === 'object') return boundedText(value.label ?? value.used ?? value.actual ?? value.name, '运行记录中详列');
  return '运行记录中详列';
}
function setEvidence(result, request) {
  const b = backendLabel(result.backend);
  $('phBackend').textContent = b;
  $('phNative').textContent = `${result.width} × ${result.height}`;
  const spectral = result.diagnostics?.spectral?.samples ?? result.diagnostics?.spectralSamples ?? result.diagnostics?.wavelengthCount ?? result.configuration?.spectralSamples;
  $('phSpectralCount').textContent = finite(spectral) ? `${spectral} 个波长` : request.spectralMode === 'narrowband' ? `${request.wavelengthNm} nm 窄带` : '宽带光谱积分';
  $('phElapsed').textContent = `${(result.totalMs / 1000).toFixed(2)} s`;
  $('phExecution').textContent = b;
  $('phMeasurement').textContent = `原生 ${result.width}² · 8 × 8 mm 探测器 · 50 mm 像距 · D = ${request.diameterMm.toFixed(3)} mm。固定基准增益 65,536（+16 EV），再叠加共用曝光。原始 XYZ 未经逐图归一化。`;
  const warnings = result.diagnostics?.warnings ?? result.warnings ?? [];
  $('phWarnings').textContent = Array.isArray(warnings) ? warnings.slice(0, 8).map(v => boundedText(v)).filter(Boolean).join('\n') : boundedText(warnings);
  lastExport = boundedPinholeReport({schema: 'pinhole-local-run-v1', generatedAt: new Date().toISOString(), request: {...request, id: undefined}, width: result.width, height: result.height, complete: result.complete, backend: result.backend, configuration: result.configuration, progressive: result.progressive, timings: result.timings, stages: result.stages, diagnostics: result.diagnostics, totalMs: result.totalMs, scope: result.scope, display: {baseExposureEV: BASE_EXPOSURE_EV, sharedExposureAtCompletionEV: Number($('phExposure').value), mapping: 'CIE XYZ to linear sRGB, negative display gamut clipping, fixed exponential compression, standard sRGB transfer; no per-image normalization'}});
  let report = JSON.stringify(lastExport, null, 2);
  if (report.length > 64000) { lastExport = {...lastExport, diagnostics: {note: 'Diagnostics exceeded export limit; see bounded timings and configuration.'}}; report = JSON.stringify(lastExport, null, 2); }
  $('phReport').textContent = report;
  $('phDownload').disabled = false;
}
function fail(message, details = {}) {
  const request = activeRequest;
  const elapsedMs = startTime ? Math.max(0, performance.now() - startTime) : 0;
  stop(); clearResult(); status(message);
  $('phWarnings').textContent = '本次结果未发布。可调整参数或改用 CPU 重新计算；不会将旧图标成新结果。';
  $('phBackend').textContent = '未完成';
  $('phElapsed').textContent = `${(elapsedMs / 1000).toFixed(2)} s · 未完成`;
  lastExport = boundedPinholeReport({schema: 'pinhole-local-run-v1', generatedAt: new Date().toISOString(), complete: false, request, error: message, errorName: details.name, elapsedMs, diagnostics: details.diagnostics, timings: details.timings, backend: details.backend});
  $('phReport').textContent = JSON.stringify(lastExport, null, 2);
  $('phDownload').disabled = false;
}
function run() {
  if (busy) return;
  stop(); clearResult();
  let request;
  try { request = makeRequest(generation); } catch (error) { status(error.message); return; }
  const token = generation;
  activeRequest = request; busy = true; startTime = performance.now();
  $('phRun').disabled = true; $('phRun').setAttribute('aria-busy', 'true'); $('phCancel').hidden = false;
  $('phProgress').hidden = false; $('phProgress').value = 0;
  $('phBackend').textContent = '正在选择计算后端';
  status(request.mode === 'point' ? '正在计算一次原生 512 × 512 点源图像。' : '正在计算一次原生 512 × 512 图像；逐项累积深度 / 材质贡献。');
  timer = setInterval(() => { if (busy && token === generation) $('phElapsed').textContent = `${((performance.now() - startTime) / 1000).toFixed(1)} s · 计算中`; }, 250);
  let instance;
  try {
    instance = new Worker(new URL('./pinhole-worker.js', import.meta.url), {type: 'module'});
    worker = instance;
    const current = () => worker === instance && token === generation && busy;
    instance.onmessage = ({data: message}) => {
      if (!current() || !message || message.id !== token) return;
      const messageAttempt = message.type === 'result' ? message.result?.progressive?.attempt ?? 0 : message.type === 'preview' ? message.progressive?.attempt : message.attempt ?? 0;
      // A reset establishes the sole active accumulation. Late GPU messages must
      // never repaint its CPU replacement or move the replacement's progress.
      if (['progress', 'preview', 'result', 'reset'].includes(message.type) && Number.isSafeInteger(messageAttempt) && messageAttempt < activeAttempt) return;
      if (message.type === 'reset') {
        if (messageAttempt === activeAttempt) return;
        if (messageAttempt !== activeAttempt + 1 || messageAttempt !== 1) { fail('计算重试记录被拒绝：重试序号无效'); return; }
        activeAttempt = messageAttempt; lastPartial = null;
        clearCameraImages();
        $('phProgress').value = 0;
        $('phNative').textContent = '512 × 512 · CPU 从零累积';
        $('phBackend').textContent = 'GPU 未完成 · CPU 重试';
        $('phMeasurement').textContent = 'GPU 的未完成累积图已清除。CPU 将从零累积同一原生 512 × 512 图像。';
        status('GPU 计算未完成，已清除部分图像；正在 CPU 上从零重新累积。');
      } else if (message.type === 'progress') {
        if (messageAttempt !== activeAttempt) { fail('计算进度被拒绝：重试序号无效'); return; }
        if (finite(message.fraction)) $('phProgress').value = Math.max(Number($('phProgress').value), Math.max(0, Math.min(.99, message.fraction)));
        status(boundedText(message.message, '正在累积原生 512 × 512 图像的源贡献。'));
      } else if (message.type === 'preview') {
        try {
          validatePinholeFrame(message, request, {partial: true, previousProgressive: lastPartial?.progressive, attempt: activeAttempt});
          lastPartial = message; paintFrame(message, true);
          $('phNative').textContent = '512 × 512 · 未完成累积';
          $('phMeasurement').textContent = `${contributionLabel(message)}。曝光与最终图相同；尚缺源光贡献，亮度和颜色仍会改变。源覆盖表示每组源像素 / 材质至少计入一个深度权重，不代表全部光量。这是逐项贡献累积，不表示噪声或采样收敛。`;
          if (message.backend) $('phBackend').textContent = backendLabel(message.backend);
        } catch (error) { fail(`渐进帧被拒绝：${error.message}`); }
      } else if (message.type === 'result') {
        try {
          const result = validatePinholeFrame(message.result, request, {previousProgressive: lastPartial?.progressive, attempt: activeAttempt});
          lastResult = result; lastPartial = null;
          paintFrame(result, false); setEvidence(result, request);
          if (result.sceneOverview) { currentOverview = result.sceneOverview; drawRig(); }
          stop();
          status(`计算完成：两幅图均为原生 512 × 512，用时 ${(result.totalMs / 1000).toFixed(2)} 秒。可改变共用曝光观察暗部，或换孔径重新计算。`);
        } catch (error) { fail(`结果被拒绝：${error.message}`); }
      } else if (message.type === 'error') fail(`计算未完成：${boundedText(message.error, '未知计算错误')}`, message);
    };
    instance.onerror = event => { if (current()) { event.preventDefault?.(); fail(`计算线程失败：${boundedText(event.message, '无法执行计算')}`); } };
    instance.onmessageerror = () => { if (current()) fail('计算线程数据无法读取。请重新计算。'); };
    instance.postMessage(request);
  } catch (error) { fail(`无法启动计算：${error.message}`); }
}

/* The scene module supplies actual geometry. Populated below after its import. */
function drawRig() {
  const canvas = $('phRigCanvas'), ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#101915'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  drawSceneOverview(ctx, canvas.width, canvas.height, currentOverview);
  drawAperture();
}
function drawAperture() {
  const canvas = $('phApertureCanvas'), ctx = canvas.getContext('2d'), w = canvas.width;
  ctx.clearRect(0, 0, w, w); ctx.fillStyle = '#25382b'; ctx.fillRect(0, 0, w, w);
  ctx.strokeStyle = '#38563f'; ctx.lineWidth = 1;
  for (let i = 0; i <= 10; i++) { ctx.beginPath(); ctx.moveTo(i * w / 10, 0); ctx.lineTo(i * w / 10, w); ctx.moveTo(0, i * w / 10); ctx.lineTo(w, i * w / 10); ctx.stroke(); }
  const radius = Number($('phDiameter').value) * w / 2;
  ctx.fillStyle = '#030a05'; ctx.beginPath(); ctx.arc(w / 2, w / 2, radius, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = '#d3edb6'; ctx.lineWidth = 1.4; ctx.stroke();
  ctx.strokeStyle = '#e2ebca'; ctx.beginPath(); ctx.moveTo(w / 2 - radius, w / 2); ctx.lineTo(w / 2 + radius, w / 2); ctx.stroke();
  ctx.fillStyle = '#cde0bd'; ctx.font = '11px sans-serif'; ctx.textAlign = 'center'; ctx.fillText(`${Number($('phDiameter').value).toFixed(3)} mm`, w / 2, w - 11);
}
function drawSceneOverview(ctx, width, height, overview) {
  const geometry = overview?.objects ? overview : initialOverview;
  const pointMode = $('phMode').value === 'point';
  const orbit = Number($('phOrbit').value) * Math.PI / 180;
  const objects = geometry.objects ?? scene.meshes;
  const vecSub = (a, b) => a.map((v, i) => v - b[i]);
  const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
  const detailCoordinates = p => {
    const x = p[0], y = p[1] + 1, z = p[2] + 250, yaw = .30 + orbit, pitch = .22;
    const xx = x * Math.cos(yaw) - z * Math.sin(yaw), zz = x * Math.sin(yaw) + z * Math.cos(yaw);
    return [xx, y * Math.cos(pitch) - zz * Math.sin(pitch), zz];
  };
  const detailPoints = (objects.length ? objects : scene.meshes).flatMap(o => o.vertices ?? []).map(detailCoordinates);
  const detailBounds = [0,1].map(k => [Math.min(...detailPoints.map(p => p[k])), Math.max(...detailPoints.map(p => p[k]))]);
  const detailScale = Math.min(310 / (detailBounds[0][1]-detailBounds[0][0]), 225 / (detailBounds[1][1]-detailBounds[1][0]));
  const projectDetail = p => { const q = detailCoordinates(p); return [200+(q[0]-(detailBounds[0][0]+detailBounds[0][1])/2)*detailScale,258-(q[1]-(detailBounds[1][0]+detailBounds[1][1])/2)*detailScale,q[2]]; };
  const projectRig = p => {
    const yaw = .65 + orbit * .35, scale = 1.65;
    return [885 + (p[2] * Math.cos(.16) + p[0] * Math.sin(yaw)) * scale, 339 - p[1] * scale - p[0] * Math.cos(yaw) * scale * .6 + p[2] * Math.sin(.16) * scale];
  };
  const line = (a, b, stroke = '#5a7859', dash = []) => {
    ctx.strokeStyle = stroke; ctx.lineWidth = 1.2; ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke(); ctx.setLineDash([]);
  };
  const text = (words, x, y, color = '#b4c6ac', size = 12, align = 'left') => {
    ctx.fillStyle = color; ctx.font = `${size}px sans-serif`; ctx.textAlign = align; ctx.fillText(words, x, y);
  };
  const polygon = (points, fill, stroke = null) => {
    ctx.beginPath(); ctx.moveTo(points[0][0], points[0][1]); for (const p of points.slice(1)) ctx.lineTo(p[0], p[1]); ctx.closePath();
    ctx.fillStyle = fill; ctx.fill(); if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = .6; ctx.stroke(); }
  };
  const paintMeshes = (projection, detail) => {
    const faces = [];
    for (const object of objects) {
      if (!object?.vertices || !object?.faces) continue;
      const color = basisColors[PINHOLE_BASIS_NAMES.indexOf(object.basis)] ?? basisColors[0];
      for (const indices of object.faces) {
        const vertices = indices.map(i => object.vertices[i]); if (vertices.some(v => !v || v.length !== 3)) continue;
        const projected = vertices.map(projection);
        const normal = cross(vecSub(vertices[1], vertices[0]), vecSub(vertices[2], vertices[0])), length = Math.hypot(...normal) || 1;
        const lighting = .30 + .70 * Math.max(0, (normal[0]*-.3 + normal[1]*.65 + normal[2]*.85) / (length * 1.11));
        const rgb = color.map(v => Math.round(255 * linearToSRGB(-Math.expm1(-Math.max(0, v) * object.albedo * lighting * 1.8))));
        faces.push({points: projected, depth: vertices.reduce((a, v) => a + v[2], 0) / 3, projectedDepth: projected.reduce((a, p) => a + (p[2] ?? 0), 0), color: `rgb(${rgb.join(',')})`});
      }
    }
    faces.sort((a, b) => detail ? a.projectedDepth - b.projectedDepth : a.depth - b.depth);
    for (const face of faces) polygon(face.points, face.color);
  };
  // Enlarged source view and dimensioned whole-rig view use independent, labelled scales.
  ctx.fillStyle = '#142019'; ctx.fillRect(19, 65, 362, 351);
  ctx.strokeStyle = '#304231'; ctx.lineWidth = 1; ctx.strokeRect(19.5, 65.5, 362, 351);
  text(pointMode ? '轴上点源 · 同一装置' : '场景细节 · 同一三角网格', 36, 92, '#c7ddba', 14);
  text('局部放大 · 材料以 D65 颜色示意', 36, 116, '#82997e', 11);
  if (pointMode) {
    const p = [200, 250];
    ctx.fillStyle = '#dcecbf'; ctx.beginPath(); ctx.arc(p[0], p[1], 5, 0, 2*Math.PI); ctx.fill();
    line([p[0]-40,p[1]], [p[0]+40,p[1]], '#476341', [3,4]); line([p[0],p[1]-40], [p[0],p[1]+40], '#476341', [3,4]);
    text('点源符号 · 物距 250 mm', 200, 369, '#b5c6a8', 12, 'center');
  } else {
    paintMeshes(projectDetail, true);
    text('红色 F · 蓝球 · 绿块 · 前景遮挡片', 200, 389, '#9bb38e', 11, 'center');
  }
  text('完整光路 · mm', 420, 105, '#c7ddba', 14);
  text('物体在圆孔前，倒像在圆孔后', 420, 129, '#82997e', 11);
  line(projectRig([0,0,-290]), projectRig([0,0,62]), '#3d593d', [5,7]);
  const pupil = projectRig([0,0,0]), detector = projectRig([0,0,50]);
  const sourcePoint = pointMode ? (geometry.point?.positionMm ?? [0, 0, -250]) : (objects[0]?.vertices?.[1] ?? [-8, 6, -250]);
  const q = [50*sourcePoint[0]/sourcePoint[2],50*sourcePoint[1]/sourcePoint[2],50];
  line(projectRig(sourcePoint), projectRig(q), '#acc986', [4,5]);
  if (!pointMode) paintMeshes(projectRig, false);
  else { const p = projectRig(sourcePoint); ctx.fillStyle = '#e3efc7'; ctx.beginPath(); ctx.arc(p[0],p[1],3,0,2*Math.PI); ctx.fill(); }
  // The drawn plate is a mounting illustration; the optical pupil is the small circle at the origin.
  const plate = [[-13,-13,0],[13,-13,0],[13,13,0],[-13,13,0]].map(projectRig);
  polygon(plate, '#2e4739cc', '#769568');
  const actualPupilRadius = Number($('phDiameter').value) / 2 * 1.65;
  ctx.fillStyle = '#07120b'; ctx.beginPath(); ctx.arc(pupil[0],pupil[1],actualPupilRadius,0,2*Math.PI); ctx.fill();
  const corners = geometry.detector?.cornersMm ?? initialOverview.detector.cornersMm;
  polygon(corners.map(projectRig), '#7d917299', '#d0e8b3');
  const sourceLabel = projectRig([0,0,-250]);
  text(pointMode ? '点源' : '三维场景', sourceLabel[0], sourceLabel[1]-35, '#c3d6b8', 12, 'center');
  text('圆孔平面', pupil[0]-9, pupil[1]-41, '#c9dfa9', 12, 'right');
  text('8 × 8 mm', detector[0], detector[1]+37, '#c5d9ab', 12, 'center');
  text('探测器', detector[0], detector[1]+54, '#859d80', 11, 'center');
  const a = projectRig([0,-34,-250]), b = projectRig([0,-34,0]), c = projectRig([0,-34,50]);
  line(a,b,'#617951'); line(b,c,'#617951');
  for (const p of [a,b,c]) line([p[0],p[1]-4],[p[0],p[1]+4],'#839c69');
  text(pointMode ? '250 mm' : '物距 200–300 mm', (a[0]+b[0])/2, (a[1]+b[1])/2+20, '#9eb389', 12, 'center');
  text('50 mm', (b[0]+c[0])/2, (b[1]+c[1])/2+20, '#b4c99b', 12, 'center');
  text('孔径在此尺度下非常小，见右上角独立放大图。', 420, 465, '#7c9576', 11);
  $('phSceneDescription').textContent = pointMode ? '轴上点源位于圆孔前 250 mm。环纹来自同一圆孔的 Fresnel 计算；选择窄带更容易观察，宽带会把不同波长的强度相加。点源与彩色场景的亮度单位不同，请勿直接比较它们的绝对亮度。' : '图中的红色 F、蓝球、绿色块、遮挡片和底座，都是相机采样的实际三角网格。装置与局部场景分别按标注尺度投影；装置视角只用于观察，小孔探测器上会形成倒像。';
}

for (const id of transportIds) $(id).addEventListener('input', () => { if (id === 'phMode' && $('phMode').value === 'point') { $('phSpectrum').value = 'narrowband'; $('phDiameter').value = '0.08'; $('phWavelength').value = '550'; } invalidate(); });
for (const button of document.querySelectorAll('[data-diameter]')) button.addEventListener('click', () => { $('phDiameter').value = button.dataset.diameter; invalidate(); });
$('phExposure').addEventListener('input', () => { labels(); if (lastResult) paintFrame(lastResult, false); else if (lastPartial) paintFrame(lastPartial, true); });
$('phOrbit').addEventListener('input', () => { labels(); drawRig(); });
$('phRun').addEventListener('click', run);
$('phCancel').addEventListener('click', () => { stop(); clearResult(); status('已取消。未完成的累积图已清除；可随时重新计算。'); });
$('phReset').addEventListener('click', () => { stop(); clearResult(); defaults(); drawRig(); status('已恢复默认：0.20 mm 圆孔、宽带彩色场景、原生 512²。点击开始计算。'); });
$('phDownload').addEventListener('click', () => {
  if (!lastExport) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(lastExport, null, 2) + '\n'], {type: 'application/json'}));
  downloads.add(url); const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'pinhole-camera-local-run.json'; anchor.click();
  setTimeout(() => { if (downloads.delete(url)) URL.revokeObjectURL(url); }, 1000);
});
window.addEventListener('pagehide', () => { stop(); clearResult(); status('计算已停止。重新计算可生成当前参数的原生 512² 图像。'); for (const url of downloads) URL.revokeObjectURL(url); downloads.clear(); });
defaults(); clearResult(); drawRig();
