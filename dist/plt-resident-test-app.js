/** Opt-in test UI only. The stable Wave/spectral application is not imported.
 * Worker data is validated before a canvas or downloadable report is exposed. */
const $ = id => document.getElementById(id);
const presets = Object.freeze({
  'structured-mirror-panels': {title:'镜面与发光面板 · 结构场景', description:'真实相机光线经过有限镜面、自发光面板与黑色挡板，经一次 / 两次镜面反射与前景遮挡形成明暗结构；不含漫反射照明。', expected:null},
  'two-mirrors-emission': {title:'双镜面 → 发光面', description:'两次镜面反射后到达发光面，解析相对响应为 3。', expected:3},
  'four-scale-shifted-zxy': {title:'4 倍尺度 · 平移 · ZXY', description:'冻结的四倍尺度、平移与轴置换场景，解析相对响应为 3。', expected:3},
  'half-scale-three-mirrors-negative-permutation': {title:'半尺度 · 三镜面 · 负坐标置换', description:'冻结的半尺度三镜面场景，解析相对响应为 0.75。', expected:.75}
});
const defaults = {residentPreset:'structured-mirror-panels',residentDimension:'32',residentLanes:'128'};
const resultIds = ['residentAdmission','residentExecuted','residentAdapter','residentReplay','residentRetained','residentSetupTime','residentRenderTime','residentShadowTime','residentAdmissionTime','residentTotalTime'];
const comparisonViews = {cpu:['residentCanvas','residentEmpty'],gpuAssisted:['residentGpuCanvas','residentGpuEmpty'],absoluteDifference:['residentDifferenceCanvas','residentDifferenceEmpty'],replayMask:['residentReplayCanvas','residentReplayEmpty']};
const differenceScale = 1e-6;
const baselineWarning = '执行证据只适用于本次冻结场景和参数，不能推广为物理 GPU、完整 Wave 或光谱功能已验证。';
let generation = 0, worker = null, lastExport = null;
const downloads = new Set(), retiring = new Map();
const status = text => { $('residentStatus').textContent = text; };
const smallText = (value, fallback = '') => typeof value === 'string' ? value.slice(0,2048) : fallback;
const isObject = value => value !== null && typeof value === 'object' && !Array.isArray(value) && !ArrayBuffer.isView(value);
const finiteTime = value => typeof value === 'number' && Number.isFinite(value) && value >= 0;
const requireThat = (condition,message) => { if (!condition) throw new Error(message); };
function stop(grace = false) {
  const previous = generation++;
  if (!grace) for (const [old,timer] of retiring) { clearTimeout(timer); old.terminate(); retiring.delete(old); }
  if (worker) {
    const old = worker; worker = null;
    try { old.postMessage({type:'cancel',id:previous}); } catch { /* Termination remains the final cancellation boundary. */ }
    if (grace) retiring.set(old,setTimeout(() => { old.terminate(); retiring.delete(old); },250));
    else old.terminate();
  }
  $('residentRun').disabled = false;
  $('residentCancel').hidden = true;
  $('residentProgress').hidden = true;
  $('residentProgress').removeAttribute('value');
}
function clearResult() {
  lastExport = null;
  $('residentDownload').disabled = true;
  $('residentReport').textContent = '尚无结果';
  $('residentBackend').textContent = '尚未运行';
  $('residentEmpty').hidden = false;
  $('residentEmpty').textContent = '点击开始后，显示完整核验的结果。';
  $('residentMeasurement').textContent = '当前没有运行结果。';
  $('residentImageCaption').textContent = 'CPU / GPU 辅助使用相同固定灰度映射；仅显示已核验结果。';
  $('residentDifferenceMax').textContent = '最大绝对差值：—';
  $('residentReplayHistogram').textContent = '显式重放与数值恢复去重；完整 CPU shadow 不计入此图。';
  $('residentMirrorHistogram').textContent = '路径按镜面反射次数的统计将在核验后显示。';
  $('residentWarnings').textContent = baselineWarning;
  for (const id of resultIds) $(id).textContent = '—';
  for (const [canvasId,emptyId] of Object.values(comparisonViews)) { const canvas=$(canvasId); canvas.getContext('2d')?.clearRect(0,0,canvas.width,canvas.height); $(emptyId).hidden=false; }
  $('residentGpuEmpty').textContent = '需要真实 WebGPU 执行与完整准入。';
  $('residentDifferenceEmpty').textContent = '尚无已准入 GPU 辅助结果，无法计算差值。';
  $('residentReplayEmpty').textContent = '尚无已准入 GPU 辅助结果，无法显示重放覆盖。';
}
function fixtureText() {
  const preset = presets[$('residentPreset').value];
  $('residentFixtureTitle').textContent = preset?.title ?? '不支持的场景';
  $('residentFixtureDescription').textContent = preset?.description ?? '请选择一个冻结场景。';
}
function invalidate() {
  stop(true); clearResult(); fixtureText();
  status('参数已改变。点击开始，运行新的独立测试。');
}
function request() {
  const preset = $('residentPreset').value, size = Number($('residentDimension').value), laneChunkSize = Number($('residentLanes').value);
  requireThat(Object.hasOwn(presets,preset) && [8,16,32].includes(size) && [32,128].includes(laneChunkSize),'请选择支持的冻结场景、分辨率和 lane 数');
  return {preset,size,laneChunkSize};
}
/** Reject unsafe/non-JSON data rather than silently stringify NaN, cyclic data,
 * huge exception objects or raw buffers. Include every bounded metadata field. */
function boundedJSON(value) {
  let nodes = 0, characters = 0;
  const seen = new Set();
  function copy(v,depth) {
    requireThat(++nodes <= 100000 && depth <= 24,'诊断记录过大或嵌套过深');
    if (v === null || typeof v === 'boolean') return v;
    if (typeof v === 'string') { characters += v.length; requireThat(v.length <= 65536 && characters <= 2000000,'诊断文字超过本地导出上限'); return v; }
    if (typeof v === 'number') { requireThat(Number.isFinite(v),'诊断记录包含非有限数值'); return v; }
    requireThat(v && typeof v === 'object' && !seen.has(v),'诊断记录不是有效 JSON');
    requireThat(!(v instanceof ArrayBuffer) && !ArrayBuffer.isView(v),'诊断记录不应包含图像或缓冲区二进制');
    seen.add(v);
    let output;
    if (Array.isArray(v)) { requireThat(v.length <= 16384,'诊断数组超过本地导出上限'); output = v.map(item => copy(item,depth+1)); }
    else {
      requireThat(Object.getPrototypeOf(v) === Object.prototype || Object.getPrototypeOf(v) === null,'诊断记录包含不支持的对象');
      output = {};
      for (const [key,item] of Object.entries(v)) {
        requireThat(key.length <= 256 && !['__proto__','prototype','constructor'].includes(key),'诊断字段名称无效');
        if (item !== undefined) output[key] = copy(item,depth+1);
      }
    }
    seen.delete(v); return output;
  }
  const text = JSON.stringify(copy(value,0),null,2);
  requireThat(text.length <= 4000000,'诊断记录超过本地导出上限');
  return text;
}
function storeReport(report) {
  lastExport = boundedJSON(report);
  $('residentReport').textContent = lastExport;
  $('residentDownload').disabled = false;
}
function failed(message,config,diagnostic = null,name = 'Error') {
  stop(); clearResult();
  $('residentBackend').textContent = '未生成图像';
  $('residentEmpty').textContent = '本次测试未生成可显示的已核验图像。';
  status(message);
  const error = {name:smallText(name,'Error'),message:smallText(message,'测试失败'),diagnostic};
  try { storeReport({schemaVersion:'resident-test-ui-error-v1',status:'error',requested:config??null,noImageDisplayed:true,error,limitations:[baselineWarning,'错误记录不是 GPU 执行或准入证明。']}); }
  catch { storeReport({schemaVersion:'resident-test-ui-error-v1',status:'error',requested:config??null,noImageDisplayed:true,error:{name:error.name,message:error.message,diagnosticOmitted:'诊断无法安全序列化或超出有界导出限制'},limitations:[baselineWarning]}); }
}
function adapterLabel(adapter) {
  if (!isObject(adapter)) return '未提供 / 不可用';
  const labels = ['vendor','architecture','device','description'].map(key => typeof adapter[key] === 'string' ? adapter[key].trim() : '').filter(Boolean);
  const label = labels.length ? labels.join(' · ') : '适配器信息未公开';
  return label + (adapter.isFallbackAdapter === true ? ' · 浏览器报告为软件回退适配器' : ' · 物理 GPU 未独立确认');
}
function milliseconds(value) { return value === null ? '未单独测量' : value < 1000 ? `${value.toFixed(1)} ms` : `${(value/1000).toFixed(2)} s`; }

// Versioned display contract: complete shadow and admission precede every pixel.
function validateReport(report,config) {
  requireThat(isObject(report),'缺少运行记录');
  requireThat(report.schemaVersion === 2,'运行记录版本不匹配');
  requireThat(isObject(report.requested) && report.requested.preset === config.preset && report.requested.size === config.size && report.requested.laneChunkSize === config.laneChunkSize,'返回参数与当前测试不符');
  requireThat(['gpu-validated','cpu-only-fallback'].includes(report.status),'未知执行状态');
  const fixed = {width:config.size,height:config.size,samplesPerPixel:2,maxDepth:6,seed:7391,wave:false,spectral:false,spectralMode:'monochrome',wavelengthMm:.025,boundary:'unpolarized',nee:false,mode:'ray-only-monochrome',guard:'axis-mirror-camera',axisSubmission:'round',recoveryMode:'selective',sampleStart:0};
  requireThat(Object.entries(fixed).every(([key,value])=>report.requested[key]===value),'返回结果改变了固定采样或守卫配置');
  requireThat(isObject(report.requested.tile) && Object.entries({x:0,y:0,width:config.size,height:config.size}).every(([key,value])=>report.requested.tile[key]===value),'返回结果不是完整冻结 tile');
  const pixels = config.size**2, paths = pixels*2, expected = presets[config.preset].expected;
  const shadow = report.cpuFullShadow;
  requireThat(isObject(shadow) && shadow.required === true && shadow.complete === true && shadow.exactCanonicalRng === true && shadow.paths === paths && shadow.pixels === pixels && shadow.analyticPower === expected,'缺少完整独立 CPU shadow 证明');
  const oracle = expected === null ? 'dyadic-mirror-panels-v1' : 'constant-dyadic-mirror-v1';
  requireThat(shadow.analyticOracle === oracle,'CPU 参考的独立解析模型不符');
  requireThat(report.physicalGPUValidated === false,'本测试不能证明使用物理 GPU');
  const {image,gpu,validation,timings,recovery} = report;
  requireThat(isObject(image) && image.validated === true && image.width === config.size && image.height === config.size,'图像未核验或尺寸与当前测试不符');
  requireThat((Array.isArray(image.values) || ArrayBuffer.isView(image.values)) && image.values.length === config.size*config.size,'图像数组缺失或长度不符');
  const values = Array.from(image.values);
  requireThat(values.every(v => Number.isFinite(v) && v >= 0),'图像包含无效数值');
  requireThat(isObject(validation) && validation.passed === true && validation.completeImage === true && validation.exactRng === true,'结果缺少完整图像 / RNG 核验');
  requireThat(validation.expectedPower === expected && validation.validatedPixels === pixels && validation.validatedPaths === paths && validation.imageSource === image.source,'图像核验范围或来源不一致');
  requireThat(validation.analyticOracle === oracle && validation.analyticOraclePassed === true,'独立逐路径 / 像素解析核验未通过');
  if (expected !== null) requireThat(values.every(v => v === expected),'图像不满足冻结场景的解析响应');
  requireThat(isObject(gpu) && ['apiAvailable','adapterAvailable','deviceCreated','executionAttempted','executed'].every(key=>typeof gpu[key] === 'boolean'),'WebGPU 执行证据不完整');
  requireThat(['computeSubmissions','completedComputeSubmissions'].every(key=>Number.isSafeInteger(gpu[key])&&gpu[key]>=0) && gpu.completedComputeSubmissions<=gpu.computeSubmissions && typeof gpu.injectedRunner==='boolean','WebGPU 提交记录无效');
  requireThat(!gpu.executed || (gpu.apiAvailable && gpu.adapterAvailable && gpu.deviceCreated && gpu.executionAttempted && gpu.completedComputeSubmissions>0 && !gpu.injectedRunner),'WebGPU 执行证据不一致');
  if (report.status === 'gpu-validated') requireThat(gpu.executed && gpu.candidateCompleted === true && gpu.candidateAdmitted === true && gpu.fullRenderCompleted === true && validation.admissionPassed === true && image.source === 'gpu-with-selective-recovery','GPU 图像缺少完整执行和严格准入证明');
  else requireThat(image.source === 'cpu-full-shadow','CPU 回退只能显示独立 CPU shadow 图像');
  requireThat(isObject(timings) && ['coldSetupMs','cpuFullShadowMs','endToEndMs'].every(key=>finiteTime(timings[key])) && ['compilationMs','rendererIncludingReadbackReplayMs','admissionMs'].every(key=>timings[key]===null||finiteTime(timings[key])),'计时记录缺失或无效');
  if (report.status === 'gpu-validated') requireThat(finiteTime(timings.rendererIncludingReadbackReplayMs) && finiteTime(timings.admissionMs),'GPU 完成结果缺少渲染或准入计时');
  requireThat(isObject(recovery) && ['explicitReplayPaths','auditNumericalRecoveryPaths','retainedPaths','totalRecoveredPaths'].every(key=>recovery[key]===null || (Number.isSafeInteger(recovery[key])&&recovery[key]>=0&&recovery[key]<=paths)),'缺少有效路径重放记录');
  if (report.status === 'gpu-validated') {
    requireThat(['explicitReplayPaths','auditNumericalRecoveryPaths','retainedPaths','totalRecoveredPaths'].every(key=>Number.isSafeInteger(recovery[key])) && recovery.totalRecoveredPaths+recovery.retainedPaths===paths,'路径重放与保留计数不完整');
    requireThat(recovery.totalRecoveredPaths>=Math.max(recovery.explicitReplayPaths,recovery.auditNumericalRecoveryPaths) && recovery.totalRecoveredPaths<=recovery.explicitReplayPaths+recovery.auditNumericalRecoveryPaths,'去重 CPU 重放路径数不一致');
    requireThat(Number.isFinite(recovery.recoveryFraction) && Math.abs(recovery.recoveryFraction-recovery.totalRecoveredPaths/paths)<1e-12,'CPU 重放比例不一致');
  }
  let replayOwners = null;
  if (recovery.totalRecoveredPaths !== null) {
    const readKeys = (key,count) => {
      const array=recovery[key];requireThat(Array.isArray(array)&&array.length===count&&array.length<=paths,'CPU 重放路径键缺失或数量不符');
      requireThat(array.every(value=>typeof value==='string'&&/^0:(0|[1-9]\d*):[01]$/.test(value)&&Number(value.split(':')[1])<pixels),'CPU 重放路径键超出像素 / 样本范围');
      requireThat(new Set(array).size===array.length,'CPU 重放路径键重复');return array;
    };
    replayOwners = new Set([...readKeys('explicitReplayKeys',recovery.explicitReplayPaths),...readKeys('auditNumericalRecoveryKeys',recovery.auditNumericalRecoveryPaths)]);
    requireThat(replayOwners.size===recovery.totalRecoveredPaths,'去重重放路径键与总数不一致');
  } else requireThat(recovery.explicitReplayKeys===null && recovery.auditNumericalRecoveryKeys===null,'未准入的候选不应包含重放路径键');
  const comparison = report.comparison, diagnostics = report.diagnostics;
  requireThat(isObject(comparison) && Object.keys(comparison).length===4 && Object.keys(comparisonViews).every(key=>Object.hasOwn(comparison,key)) && isObject(diagnostics),'缺少四视图比较与解析诊断');
  const views = {};
  function readView(key) {
    const view = comparison[key];
    requireThat(isObject(view) && view.width===config.size && view.height===config.size,'比较图尺寸与当前测试不符');
    requireThat((Array.isArray(view.values)||ArrayBuffer.isView(view.values)) && view.values.length===pixels,'比较图数组缺失或长度不符');
    const array = Array.from(view.values);
    requireThat(array.every(v=>Number.isFinite(v)&&v>=0),'比较图包含无效数值');
    views[key] = array; return array;
  }
  const cpu = readView('cpu');
  const allowedPowers = expected === null ? [0,.75,1.5,3,6] : [expected];
  const allowedPixels = new Set(allowedPowers.flatMap(a=>allowedPowers.map(b=>(a+b)/2)));
  requireThat(cpu.every(value=>allowedPixels.has(value)),'CPU 图像不满足独立模型的像素响应范围');
  function histogram(record,allowed,total,label) {
    requireThat(isObject(record) && Object.keys(record).length>0 && Object.entries(record).every(([key,count])=>allowed.includes(key)&&Number.isSafeInteger(count)&&count>=0),'无效的'+label+'分布');
    requireThat(Object.values(record).reduce((a,b)=>a+b,0)===total,label+'分布覆盖不完整');
  }
  const powers = diagnostics.pathPowerHistogram, mirrors = diagnostics.mirrorCountHistogram;
  histogram(powers,allowedPowers.map(String),paths,'路径响应');
  histogram(mirrors,expected===null?['0','1','2']:[config.preset.startsWith('half-')?'3':'2'],paths,'镜面次数');
  requireThat(Object.entries(powers).reduce((n,[power,count])=>n+Number(power)*count,0)===cpu.reduce((n,value)=>n+2*value,0),'CPU 图像与逐路径响应统计不一致');
  if (expected===null && config.size===32) {
    requireThat(Object.entries({'0':181,'0.75':511,'1.5':424,'3':423,'6':509}).every(([key,count])=>powers[key]===count),'结构场景 32² 路径响应分布与冻结模型不符');
    requireThat(Object.entries({'0':181,'1':933,'2':934}).every(([key,count])=>mirrors[key]===count),'结构场景 32² 镜面路径分布与冻结模型不符');
  }
  const admitted = report.status==='gpu-validated';
  if (admitted) {
    const assisted=readView('gpuAssisted'),difference=readView('absoluteDifference'),mask=readView('replayMask');
    requireThat(assisted.every((value,i)=>value===values[i]),'已选择图像与 GPU 辅助比较图不一致');
    requireThat(assisted.every((value,i)=>value===cpu[i]),'GPU 辅助图像不满足冻结模型的逐像素精确准入');
    requireThat(difference.every((value,i)=>value===Math.abs(assisted[i]-cpu[i])),'绝对差值图与两幅相机图不一致');
    requireThat(diagnostics.maxAbsoluteDifference===Math.max(...difference),'最大绝对差值与图像不一致');
    requireThat(mask.every(value=>Number.isInteger(value)&&value<=2) && mask.reduce((a,b)=>a+b,0)===recovery.totalRecoveredPaths,'CPU 重放覆盖与去重路径计数不一致');
    const ownedMask=Array(pixels).fill(0);for(const key of replayOwners)ownedMask[Number(key.split(':')[1])]++;
    requireThat(mask.every((value,pixel)=>value===ownedMask[pixel]),'CPU 重放覆盖与逐像素路径归属不一致');
    const maskHistogram = {'0':0,'1':0,'2':0};for (const value of mask) maskHistogram[value]++;
    requireThat(isObject(diagnostics.replayMaskHistogram) && Object.keys(diagnostics.replayMaskHistogram).length===3 && Object.entries(maskHistogram).every(([key,count])=>diagnostics.replayMaskHistogram[key]===count),'CPU 重放覆盖分布不一致');
    const byMirror = diagnostics.byMirrorCount;
    requireThat(isObject(byMirror) && Object.keys(byMirror).length===Object.keys(mirrors).length,'缺少按镜面次数分类的重放统计');
    const sums={paths:0,retainedPaths:0,explicitReplayPaths:0,auditNumericalRecoveryPaths:0,recoveredPaths:0};
    for (const [key,count] of Object.entries(mirrors)) {
      const group=byMirror[key];requireThat(isObject(group) && Object.keys(sums).every(field=>Number.isSafeInteger(group[field])&&group[field]>=0&&group[field]<=count),'按镜面次数分类的路径计数无效');
      requireThat(group.paths===count && group.retainedPaths+group.recoveredPaths===count && group.recoveredPaths>=Math.max(group.explicitReplayPaths,group.auditNumericalRecoveryPaths) && group.recoveredPaths<=group.explicitReplayPaths+group.auditNumericalRecoveryPaths,'按镜面次数分类的重放计数不一致');
      for(const field of Object.keys(sums)) sums[field]+=group[field];
    }
    requireThat(sums.paths===paths && sums.retainedPaths===recovery.retainedPaths && sums.explicitReplayPaths===recovery.explicitReplayPaths && sums.auditNumericalRecoveryPaths===recovery.auditNumericalRecoveryPaths && sums.recoveredPaths===recovery.totalRecoveredPaths,'分类统计与总重放计数不一致');
  } else {
    requireThat(cpu.every((value,i)=>value===values[i]),'CPU 回退图像与完整 CPU 比较图不一致');
    requireThat(['gpuAssisted','absoluteDifference','replayMask'].every(key=>comparison[key]===null),'CPU 回退不能伪造 GPU 图像、差值或重放覆盖');
    requireThat(diagnostics.maxAbsoluteDifference===null && diagnostics.replayMaskHistogram===null && diagnostics.byMirrorCount===null,'CPU 回退不能提供伪造的 GPU 比较诊断');
  }
  const omitValues = view => { if (view===null) return null; const {values:omitted,...metadata}=view;return {...metadata,valuesOmitted:true}; };
  const exported = {...report,image:omitValues(image),comparison:Object.fromEntries(Object.entries(comparison).map(([key,view])=>[key,omitValues(view)]))};
  const json = boundedJSON(exported);
  return {report,values,views,json};
}
function show({report,values,views,json}) {
  const {image,gpu,timings,recovery,diagnostics} = report;
  const admitted = report.status === 'gpu-validated';
  // Allocate every available view before painting any, so a missing context
  // cannot leave a partial comparison that looks like a validated result.
  const frames=[];
  for (const [key,array] of Object.entries(views)) {
    const [canvasId,emptyId]=comparisonViews[key],canvas=$(canvasId),context=canvas.getContext('2d');
    requireThat(context,'当前浏览器无法创建比较图画布');
    const pixels=context.createImageData(image.width,image.height);
    for(let i=0;i<array.length;i++) {
      let rgb;
      if(key==='replayMask') rgb=[[16,27,25],[227,186,117],[186,139,212]][array[i]];
      else { const gray=key==='absoluteDifference'?Math.round(255*Math.min(1,array[i]/differenceScale)):Math.round(255*Math.pow(-Math.expm1(-array[i]*.25),1/2.2));rgb=[gray,gray,gray]; }
      pixels.data.set([...rgb,255],4*i);
    }
    frames.push({canvas,context,pixels,emptyId});
  }
  for(const {canvas,context,pixels,emptyId} of frames) { canvas.width=image.width;canvas.height=image.height;context.putImageData(pixels,0,0);$(emptyId).hidden=true; }
  if(!admitted) for(const key of ['gpuAssisted','absoluteDifference','replayMask']) $(comparisonViews[key][1]).textContent='不可用：本次显示 CPU 回退，没有已准入 GPU 辅助结果。';
  $('residentBackend').textContent = admitted ? 'GPU 辅助 + CPU 重放' : 'CPU 回退图像';
  $('residentAdmission').textContent = admitted ? 'GPU 辅助结果通过完整 CPU 核验' : '完整 CPU 参考图像';
  $('residentExecuted').textContent = gpu.executed ? (admitted ? '已完成执行 · 已准入' : '曾完成计算提交 · 未采用 GPU 图像') : '无已完成 WebGPU 计算证据';
  $('residentAdapter').textContent = gpu.adapterAvailable ? adapterLabel(gpu.adapter) : '未取得 WebGPU 适配器';
  $('residentReplay').textContent = recovery.totalRecoveredPaths === null ? '无候选准入统计 · 图像全部来自 CPU' : `去重 ${recovery.totalRecoveredPaths} / ${values.length*2} (${(100*recovery.totalRecoveredPaths/(values.length*2)).toFixed(1)}%)；显式 ${recovery.explicitReplayPaths}，数值 ${recovery.auditNumericalRecoveryPaths}${admitted ? '' : ' · 图像全部来自 CPU'}`;
  $('residentRetained').textContent = admitted ? `${(100*recovery.retainedPaths/(values.length*2)).toFixed(1)}%` : '0% · 采用完整 CPU 图像';
  for (const [id,key] of Object.entries({residentSetupTime:'coldSetupMs',residentRenderTime:'rendererIncludingReadbackReplayMs',residentShadowTime:'cpuFullShadowMs',residentAdmissionTime:'admissionMs',residentTotalTime:'endToEndMs'})) $(id).textContent = milliseconds(timings[key]);
  const structured=report.requested.preset==='structured-mirror-panels';
  $('residentMeasurement').textContent = `${image.width} × ${image.height} · 2 spp · ${structured?'明暗结构来自实际有限镜面、发光面板和黑色挡板；逐路径与逐像素通过独立解析模型核验。':`解析响应 ${presets[report.requested.preset].expected}。均匀灰图符合这个冻结验证场景的预期。`}`;
  $('residentImageCaption').textContent = 'CPU / GPU 辅助使用相同固定映射 (1 − exp(−0.25 × 响应))^(1/2.2)；GPU 辅助包含选择性 CPU 重放。';
  $('residentDifferenceMax').textContent = admitted ? `最大绝对差值：${diagnostics.maxAbsoluteDifference===0?'0':diagnostics.maxAbsoluteDifference.toExponential(3)}` : '最大绝对差值：不可用';
  $('residentReplayHistogram').textContent = admitted ? [0,1,2].map(n=>`${n}/2：${diagnostics.replayMaskHistogram[n]} 像素`).join(' · ') : 'CPU 重放覆盖：不可用';
  $('residentMirrorHistogram').textContent = '路径镜面反射次数：'+Object.entries(diagnostics.mirrorCountHistogram).map(([count,paths])=>`${count} 次 ${paths} 条`).join(' · ');
  const failure = report.failure?.message;
  $('residentWarnings').textContent = [baselineWarning,typeof failure === 'string' ? `回退原因：${smallText(failure)}` : '', '候选渲染为墙钟耗时，包含读回 / 重放；编译包含在设置中。各项不能作为可直接相加的 GPU 核心计时。'].filter(Boolean).join('\n');
  lastExport = json; $('residentReport').textContent = json; $('residentDownload').disabled = false;
  status(`${admitted ? '测试完成：GPU 辅助结果已通过完整 CPU 核验。' : '测试完成：只显示已核验的 CPU 参考；其他比较图不可用。'}总耗时 ${milliseconds(timings.endToEndMs)}。`);
}
$('residentRun').addEventListener('click',() => {
  // Even a programmatically repeated click replaces the old generation first.
  stop(); clearResult();
  let config;
  try {
    config = request();
    const token = generation;
    $('residentRun').disabled = true; $('residentCancel').hidden = false;
    $('residentProgress').hidden = false; $('residentProgress').removeAttribute('value');
    status('正在准备完整 CPU shadow 与 WebGPU 管线；首次编译可能需要数十秒…');
    const active = new Worker(new URL('./plt-resident-test-worker.js',import.meta.url),{type:'module'});
    worker = active;
    const current = () => worker === active && generation === token;
    active.onmessage = ({data}) => {
      if (!current() || data?.id !== token) return;
      if (data.type === 'progress') {
        const progress = data.progress;
        if (!isObject(progress)) return;
        const stages = {'cpu-shadow':'正在构建完整 CPU shadow','setup':'正在设置新的 WebGPU 驻留管线','adapter':'正在请求 WebGPU 适配器','device':'正在创建设备','shader-compilation':'正在编译 WGSL 模块','pipeline-compilation':'正在编译 WebGPU 管线','renderer':'正在渲染完整图像并执行必要的 CPU 重放','admission':'正在严格核验完整路径、图像与 RNG','complete':'核验完成，正在整理本地记录'};
        const message = stages[progress.stage] ?? smallText(progress.message,'正在计算…'), entry = smallText(progress.entryPoint), elapsed = finiteTime(progress.elapsedMs) ? ` · ${(progress.elapsedMs/1000).toFixed(1)} 秒` : '';
        status(`${message}${entry ? ` · ${entry}` : ''}${elapsed}`);
      } else if (data.type === 'result') {
        try { const valid = validateReport(data.result,config); stop(); show(valid); }
        catch (error) { failed(`结果被拒绝：${error.message}`,config); }
      } else if (data.type === 'error') {
        failed(`测试未完成：${smallText(data.error,'Worker 返回未知错误')}`,config,data.diagnostic??null,data.name);
      } else { failed('结果被拒绝：Worker 返回未知消息类型。',config); }
    };
    active.onerror = event => { if (current()) failed(`计算 Worker 出错：${smallText(event.message,'无法初始化')}`,config); };
    active.onmessageerror = () => { if (current()) failed('计算 Worker 的结果无法读取，未显示图像。',config); };
    active.postMessage({id:token,request:config});
  } catch (error) { failed(`无法启动测试：${error.message}`,config); }
});
$('residentCancel').addEventListener('click',() => { stop(true); clearResult(); status('已取消。没有保留未完成的图像或旧记录。'); });
for (const id of Object.keys(defaults)) for (const event of ['input','change']) $(id).addEventListener(event,invalidate);
$('residentReset').addEventListener('click',() => { for (const [id,value] of Object.entries(defaults)) $(id).value = value; invalidate(); status('已恢复默认：结构场景、32 × 32、研究选项 128 lanes。'); });
$('residentDownload').addEventListener('click',() => {
  if (!lastExport) return;
  let url;
  try {
    url = URL.createObjectURL(new Blob([lastExport],{type:'application/json'})); downloads.add(url);
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'resident-ray-local-test.json'; anchor.click();
    setTimeout(() => { URL.revokeObjectURL(url); downloads.delete(url); },1000);
  } catch (error) { if (url) { URL.revokeObjectURL(url); downloads.delete(url); } status(`本地下载失败：${smallText(error.message,'请重试')}`); }
});
window.addEventListener('pagehide',() => { stop(); clearResult(); for (const url of downloads) URL.revokeObjectURL(url); downloads.clear(); });
for (const [id,value] of Object.entries(defaults)) $(id).value=value;
fixtureText();
