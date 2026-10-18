/** The demonstration renders only computed camera arrays. Normal execution does
 * not imply parity validation, device type, or a GPU-resident path tracer. */
import {PLT_DEMO_DEFAULTS} from './plt-demo-scene.js';
const $=id=>document.getElementById(id);
const transportIds=['demoWavelength','demoYaw','demoResolution','demoSamples','demoCompute','demoDebug'];
const pixelTypes=v=>v instanceof Float64Array||v instanceof Float32Array;
const countTypes=v=>v instanceof Uint32Array||v instanceof Uint16Array;
const requireThat=(condition,message)=>{if(!condition)throw new Error(message);};
const text=(v,fallback='')=>typeof v==='string'?v.slice(0,2048):fallback;
const finite=v=>typeof v==='number'&&Number.isFinite(v);
const integer=v=>Number.isSafeInteger(v)&&v>=0;
const warningBaseline='UTD 是高频渐近模型；有限端点、角点和高阶项未完整建模。低样本下的明暗差异可能含明显蒙特卡洛噪声。';
let worker=null,generation=0,activeRequest=null,lastResult=null,lastPreview=null,lastExport=null,busy=false;
const downloadURLs=new Set();
const status=value=>{$('demoStatus').textContent=value;};
function stop(){
 generation++;
 if(worker){const old=worker;worker=null;try{old.postMessage({type:'cancel',id:generation-1});}catch{}old.terminate();}
 busy=false;$('demoRun').disabled=false;$('demoRun').setAttribute('aria-busy','false');$('demoCancel').hidden=true;$('demoProgress').hidden=true;
}
function clearResult(){
 activeRequest=null;lastResult=null;lastPreview=null;lastExport=null;
 for(const mode of ['Wave','Ray']){$(`demo${mode}Samples`).textContent='尚未采样';$(`demo${mode}Empty`).hidden=false;}
 for(const mode of ['Wave','Ray','Difference']){const canvas=$(`demo${mode}Canvas`);canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);}
 $('demoDifferenceEmpty').hidden=false;$('demoBackend').textContent='尚未计算';
 for(const id of ['demoGeometry','demoEvents','demoTime','demoValidation'])$(id).textContent='—';
 $('demoMeasurement').textContent='图像显示相机的相对响应；当前尚无计算结果。';$('demoWarnings').textContent=warningBaseline;
 $('demoReport').textContent='尚无结果';$('demoDownload').disabled=true;
}
function controlLabels(){
 $('demoWavelengthValue').textContent=`${(Number($('demoWavelength').value)*1000).toFixed(0)} µm`;
 $('demoYawValue').textContent=`${$('demoYaw').value}°`;
 $('demoExposureValue').textContent=`${$('demoExposure').value} EV`;
}
function invalidate(){stop();clearResult();controlLabels();updateReferenceControls();status('参数已改变。重新计算后，两幅图都会使用新的条件。');}
function setDefaults(){
 const d=PLT_DEMO_DEFAULTS;
 $('demoWavelength').min=String(d.limits.wavelengthMinMm);$('demoWavelength').max=String(d.limits.wavelengthMaxMm);
 const values={demoWavelength:d.options.wavelengthMm,demoYaw:d.camera?.yaw??0,demoResolution:d.options.width,demoSamples:d.options.samplesPerPixel,demoCompute:'auto',demoExposure:0,demoDifferenceMode:'absolute-utd',demoReference:'geometric'};
 for(const [id,value]of Object.entries(values))$(id).value=String(value);
 $('demoDebug').checked=false;updateReferenceControls();
 $('demoSceneTitle').textContent=d.title??'三维光学台';$('demoSceneDescription').textContent=d.description??'实际三角网格与原生局部 UTD 传输。';$('demoScaleNote').textContent=d.scaleNote??'场景单位 mm；单色红外波长，灰度相对相机响应。';
 $('demoRunNote').textContent=`低样本有噪声；分辨率会改变像素光束足迹。单次计算上限 ${(d.progressive?.maxRuntimeMs??120000)/1000} 秒，可随时取消。`;
 controlLabels();
}
function request(id){
 const width=Number($('demoResolution').value),samplesPerPixel=Number($('demoSamples').value),wavelengthMm=Number($('demoWavelength').value),yaw=Number($('demoYaw').value),backend=$('demoCompute').value;
 requireThat([32,48,64,96,128].includes(width)&&[4,16,32,64,128,256].includes(samplesPerPixel),'分辨率或样本数超出本页支持范围');
 requireThat(finite(wavelengthMm)&&wavelengthMm>=PLT_DEMO_DEFAULTS.limits.wavelengthMinMm&&wavelengthMm<=PLT_DEMO_DEFAULTS.limits.wavelengthMaxMm&&finite(yaw)&&yaw>=-35&&yaw<=35,'波长或相机角度无效');
 requireThat(['auto','cpu'].includes(backend),'未知计算后端');
 return {id,backend,compare:true,debugValidation:$('demoDebug').checked,utdAblation:!$('demoDebug').checked,scene:{...PLT_DEMO_DEFAULTS.scene},camera:{...PLT_DEMO_DEFAULTS.camera,yaw},options:{...PLT_DEMO_DEFAULTS.options,width,height:Math.round(width*.75),samplesPerPixel,wavelengthMm,spectralMode:'monochrome'},progressive:{tileSize:32,batchSamples:1,maxRuntimeMs:120000,previewIntervalMs:100,...PLT_DEMO_DEFAULTS.progressive}};
}
function validateModes(frame,partial){
 const o=activeRequest?.options;requireThat(o,'没有对应的活动请求');
 requireThat(Array.isArray(frame.results)&&frame.results.length===2&&frame.results.filter(r=>r?.wave===true).length===1&&frame.results.filter(r=>r?.wave===false).length===1,'缺少波动或几何对照结果');
 if(partial)requireThat(frame.partial===true&&frame.width===o.width&&frame.height===o.height&&frame.targetSamplesPerPixel===o.samplesPerPixel,'渐进帧尺寸或目标样本不符');
 for(const r of frame.results){
  requireThat(r.width===o.width&&r.height===o.height&&pixelTypes(r.image)&&r.image.length===o.width*o.height&&countTypes(r.sampleCounts)&&r.sampleCounts.length===r.image.length,'相机数组或样本计数形状无效');
  requireThat(r.spectralMode!=='visible'&&!r.xyz,'单色结果包含不匹配的光谱数据');
  requireThat(r.partial===partial,'结果完成状态与帧类型不符');
  if(r.wave&&activeRequest.utdAblation)requireThat(pixelTypes(r.utdOffImage)&&r.utdOffImage.length===r.image.length&&r.utdOffImage.every(v=>finite(v)&&v>=0),'同路径 UTD 关闭对照数组缺失或无效');
  let min=Infinity,max=0;
  const before=lastPreview?.results.find(p=>p.wave===r.wave);
  for(let i=0;i<r.image.length;i++){
   const v=r.image[i],n=r.sampleCounts[i];requireThat(finite(v)&&v>=0&&integer(n)&&n<=o.samplesPerPixel&&(n>0||(v===0&&(!r.utdOffImage||r.utdOffImage[i]===0))),'图像含非有限、负值或未采样的非零数值');
   if(before)requireThat(n>=before.sampleCounts[i],'渐进样本计数发生倒退');
   min=Math.min(min,n);max=Math.max(max,n);
  }
  requireThat(r.completedSamplesPerPixel===min&&r.minSamplesPerPixel===min&&r.maxSamplesPerPixel===max,'渐进采样计数与报告不一致');
  if(!partial)requireThat(min===o.samplesPerPixel&&max===o.samplesPerPixel,'未完成的图像不能标为最终结果');
  requireThat(typeof r.backend==='string'&&r.backend.length<=2048,'缺少实际后端说明');
 }
 const completed=Math.min(...frame.results.map(r=>r.completedSamplesPerPixel));
 if(partial)requireThat(frame.completedSamplesPerPixel===completed,'完整采样轮数不一致');
 else{
  requireThat(frame.options&&['width','height','samplesPerPixel','maxDepth','wavelengthMm','seed'].every(k=>frame.options[k]===o[k]),'结果参数与当前请求不符');
  requireThat((frame.options.spectralMode??'monochrome')==='monochrome'&&frame.scene?.preset===activeRequest.scene.preset,'场景或光谱模式与当前请求不符');
  requireThat(frame.progressive?.completedSamplesPerPixel===o.samplesPerPixel&&frame.partial!==true,'未完成的采样轮数不能发布');
  requireThat(frame.executionMode===(activeRequest.debugValidation?'debug-strict':'demo'),'核验模式与当前请求不符');
  requireThat(finite(frame.totalMs)&&frame.totalMs>=0&&integer(frame.scene?.triangles)&&frame.scene.triangles<=4096&&integer(frame.scene?.wedges)&&frame.scene.wedges<=12288,'场景统计或耗时无效');
  requireThat(Array.isArray(frame.warnings)&&frame.warnings.length<=128&&frame.warnings.every(v=>typeof v==='string'&&v.length<=8192),'运行警告超出安全范围');
  for(const r of frame.results){
   requireThat(integer(r.events)&&integer(r.stats?.maxDiffractingPerPath)&&integer(r.stats?.maxSurfacePerPath)&&finite(r.measurement?.mean)&&r.measurement.mean>=0,'路径统计或测量无效');
   if(!activeRequest.debugValidation){requireThat(r.validation?.parityChecked===false&&r.validation?.mode==='finite'&&r.validation?.passed===true,'普通模式必须通过基本安全检查，且不能声明完整参考核验');const e=r.execution;requireThat(e&&['cpuBatches','gpuBatches','cpuOperatorEvaluations','cpuReferenceImages'].every(k=>integer(e[k])),'缺少实际 CPU / GPU 执行计数');}
  }
 }
}
function gray(v,gain){return Math.round(255*Math.min(1,Math.log1p(4*gain*v)/Math.log1p(32))**(1/2.2));}
function paintImage(result,canvasId){
 const canvas=$(canvasId),gain=2**Number($('demoExposure').value);canvas.width=result.width;canvas.height=result.height;
 const ctx=canvas.getContext('2d'),pixels=ctx.createImageData(result.width,result.height);
 for(let i=0;i<result.image.length;i++){const v=gray(result.image[i],gain);pixels.data.set([v,v,v,255],4*i);}ctx.putImageData(pixels,0,0);
}
function paintDifference(){
 if(!lastResult)return;
 const wave=lastResult.results.find(r=>r.wave),ray=lastResult.results.find(r=>!r.wave),canvas=$('demoDifferenceCanvas'),gain=2**Number($('demoExposure').value),mode=$('demoDifferenceMode').value,signed=mode.startsWith('signed'),utd=mode.endsWith('utd'),other=utd?wave.utdOffImage:ray.image;
 if(!other){canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);$('demoDifferenceEmpty').hidden=false;return;}
 canvas.width=wave.width;canvas.height=wave.height;const ctx=canvas.getContext('2d'),pixels=ctx.createImageData(wave.width,wave.height);
 for(let i=0;i<wave.image.length;i++){const delta=wave.image[i]-other[i],v=gray(Math.abs(delta),gain);pixels.data.set(signed?(delta>=0?[Math.round(v*.4),v,Math.round(v*.9),255]:[v,Math.round(v*.6),Math.round(v*.3),255]):[v,v,v,255],i*4);}
 ctx.putImageData(pixels,0,0);$('demoDifferenceEmpty').hidden=true;
 $('demoDifferenceNote').textContent=(signed?'蓝绿：Wave 更亮；橙：对照更亮。诊断色不代表波长。':'绝对差值显示为灰度。')+'采用同一固定 HDR 映射，不做独立最大值归一化。'+(utd?'同一批波动路径中，只把局部事件的系数替换为直接项振幅的平方；保留采样策略与光束。差值用于检查局部 UTD 影响，不是精确场解或置信区间。':'Wave 与独立几何路径还存在光束足迹和采样策略差异。差值包含随机采样噪声，不能全部归因于衍射。');
}
function paint(){const frame=lastResult??lastPreview;if(!frame)return;const wave=frame.results.find(r=>r.wave),ray=frame.results.find(r=>!r.wave),utd=$('demoReference').value==='utd-off';paintImage(wave,'demoWaveCanvas');$('demoWaveEmpty').hidden=true;if(!utd||wave.utdOffImage){paintImage(utd?{...wave,image:wave.utdOffImage}:ray,'demoRayCanvas');$('demoRayEmpty').hidden=true;const r=utd?wave:ray;$('demoRaySamples').textContent=`${r.minSamplesPerPixel===r.maxSamplesPerPixel?r.minSamplesPerPixel:`${r.minSamplesPerPixel}–${r.maxSamplesPerPixel}`} / ${activeRequest.options.samplesPerPixel} spp · ${lastResult?'已完成':'计算中'}`;}paintDifference();}
function updateReferenceControls(){const debug=$('demoDebug').checked;for(const o of $('demoReference').options)if(o.value==='utd-off')o.disabled=debug;for(const o of $('demoDifferenceMode').options)if(o.value.endsWith('utd'))o.disabled=debug;if(debug){$('demoReference').value='geometric';if($('demoDifferenceMode').value.endsWith('utd'))$('demoDifferenceMode').value='absolute-ray';}const utd=$('demoReference').value==='utd-off';$('demoReferenceTitle').textContent=utd?'Same Paths · UTD Off':'Geometric Tracing';$('demoReferenceCaption').textContent=utd?'保留 Wave 路径与光束 · 仅关闭局部 UTD':'同一场景 · 几何采样';$('demoReferenceNote').textContent=utd?'同一批波动路径，仅将每个局部事件的系数替换为直接项振幅的平方。差别更直接反映局部 UTD 算子的作用；仍保留随机采样与模型近似。':debug?'调试模式进行完整 CPU / GPU 核验；本次不额外计算同路径 UTD 关闭对照。':'Wave 与几何图还存在光束足迹和采样策略差异；选择「同一路径关闭 UTD」可更直接检查局部衍射算子的影响。';}
function backendText(results){const labels=[...new Set(results.map(r=>text(r.backend)))];return labels.length===1?labels[0]:`Wave：${labels[0]}；Geometric：${text(results.find(r=>!r.wave).backend)}`;}
function preview(frame){
 validateModes(frame,true);lastPreview=frame;
 for(const r of frame.results)$(r.wave?'demoWaveSamples':'demoRaySamples').textContent=`${r.minSamplesPerPixel===r.maxSamplesPerPixel?r.minSamplesPerPixel:`${r.minSamplesPerPixel}–${r.maxSamplesPerPixel}`} / ${activeRequest.options.samplesPerPixel} spp · 计算中`;
 paint();$('demoBackend').textContent=backendText(frame.results);$('demoMeasurement').textContent=`两幅图均已完成 ${frame.completedSamplesPerPixel} 轮全画幅采样，目标 ${activeRequest.options.samplesPerPixel} 轮。尚未采样的位置为黑色；这是未完成的渐进图像。`;
 $('demoValidation').textContent=activeRequest.debugValidation?'调试核验中':'安全检查 · 无完整 parity';$('demoReport').textContent='渐进计算中，尚未生成最终运行记录。';
}
/** Bound all exported metadata. Image/event buffers never become JSON arrays. */
function boundedJSON(value){
 let nodes=0,characters=0;const seen=new Set();
 function copy(v,depth){
  requireThat(++nodes<=50000&&depth<=20,'运行记录过大或嵌套过深');
  if(v===null||typeof v==='boolean')return v;
  if(typeof v==='number'){requireThat(finite(v),'运行记录包含非有限数值');return v;}
  if(typeof v==='string'){characters+=v.length;requireThat(v.length<=65536&&characters<=1000000,'运行文字超过导出上限');return v;}
  requireThat(v&&typeof v==='object'&&!seen.has(v),'运行记录不是有效 JSON');
  if(ArrayBuffer.isView(v))return {type:v.constructor.name,length:v.length??v.byteLength,omitted:true};
  requireThat(!(v instanceof ArrayBuffer),'运行记录包含原始缓冲区');seen.add(v);let out;
  if(Array.isArray(v)){requireThat(v.length<=4096,'运行记录数组超过导出上限');out=v.map(item=>copy(item,depth+1));}
  else{requireThat(Object.getPrototypeOf(v)===Object.prototype||Object.getPrototypeOf(v)===null,'运行记录包含不支持的对象');out={};const entries=Object.entries(v);requireThat(entries.length<=256,'运行记录字段过多');for(const [key,item]of entries){requireThat(key.length<=256&&!['__proto__','prototype','constructor'].includes(key),'运行记录字段无效');if(item!==undefined)out[key]=copy(item,depth+1);}}
  seen.delete(v);return out;
 }
 const result=JSON.stringify(copy(value,0),null,2);requireThat(result.length<=2000000,'运行记录超过导出上限');return result;
}
function finish(result){
 validateModes(result,false);
 const report={schema:'native-wave-demo-ui-v1',requested:{...activeRequest},display:{sharedExposureAtCompletionEV:Number($('demoExposure').value),mapping:'log1p(4 * 2^EV * I) / log1p(32), gamma 1/2.2; display-only clamp at 1; fixed relative white point 8 at 0 EV; no per-image normalization'},...result,results:result.results.map(({image,xyz,utdOffImage,sampleCounts,...metadata})=>metadata)};
 delete report.requested.id;
 // Serialize before painting, so corrupt evidence cannot leave apparent success.
 const exported=boundedJSON(report),wave=result.results.find(r=>r.wave);
 lastResult=result;lastPreview=null;lastExport=exported;paint();
 for(const r of result.results)$(r.wave?'demoWaveSamples':'demoRaySamples').textContent=`${result.options.samplesPerPixel} spp · 已完成`;
 $('demoBackend').textContent=backendText(result.results);$('demoGeometry').textContent=`${result.scene.triangles} / ${result.scene.wedges}`;$('demoEvents').textContent=wave.events.toLocaleString();$('demoTime').textContent=`${(result.totalMs/1000).toFixed(1)} s`;
 $('demoValidation').textContent=result.executionMode==='demo'?'安全检查 · 无完整 parity':result.results.some(r=>r.validation?.passed===false)?'核验失败 · CPU 回退':result.results.every(r=>r.validation?.passed===true)?'完整 parity 通过':'CPU 参考 · 无 GPU parity';
 $('demoMeasurement').textContent=`Wave 相对相机响应均值 ${wave.measurement.mean.toPrecision(4)}；每像素 ${result.options.samplesPerPixel} 样本。当前路径最多经过 ${wave.stats.maxDiffractingPerPath} 个 UTD 区域、${wave.stats.maxSurfacePerPath} 次表面交互。`;
 const fallback=result.results.map(r=>r.execution?`${r.wave?'Wave':'Geometric'}：${r.execution.gpuBatches} 个 GPU 分块，${r.execution.cpuBatches} 个 CPU 分块。`:null).filter(Boolean);
 const localWarnings=Object.entries(wave.stats.approximationWarnings??{}).map(([key,n])=>`${key}：${n} 个局部算子`);
 $('demoWarnings').textContent=[...result.warnings,...fallback,...localWarnings,warningBaseline,'不推断适配器是物理 GPU；CPU 负责三维遍历与采样，WebGPU 可负责局部 UTD 与相机累积。'].join('\n');
 $('demoReport').textContent=lastExport;$('demoDownload').disabled=false;
 status(`计算完成 · ${(result.totalMs/1000).toFixed(1)} 秒${finite(result.progressive.firstFullPassMs)?` · 首轮完整预览 ${(result.progressive.firstFullPassMs/1000).toFixed(1)} 秒`:''} · 同一曝光比较。`);
}
function failed(prefix,error){stop();clearResult();status(`${prefix}：${text(error?.message??error,'未知错误')}`);}
$('demoRun').addEventListener('click',()=>{
 if(busy)return;stop();clearResult();const id=generation;
 try{
  activeRequest=request(id);busy=true;$('demoRun').disabled=true;$('demoRun').setAttribute('aria-busy','true');$('demoCancel').hidden=false;$('demoProgress').hidden=false;$('demoProgress').value=0;
  status('构建实际三维场景与相机路径，准备第一轮采样…');worker=new Worker(new URL('./plt-demo-worker.js',import.meta.url),{type:'module'});
  worker.onmessage=({data})=>{
   if(id!==generation||data?.id!==id||!busy)return;
   try{
    if(data.type==='progress'){requireThat(finite(data.fraction)&&data.fraction>=0&&data.fraction<=1&&typeof data.text==='string'&&data.text.length<=8192,'无效的进度消息');status(data.text);$('demoProgress').value=data.fraction;}
    else if(data.type==='preview')preview(data.preview??data);
    else if(data.type==='result'){finish(data.result);stop();}
    else if(data.type==='error')failed('未生成图像',data.error);
    else throw new Error('未知 Worker 消息类型');
   }catch(error){failed(data.type==='preview'?'渐进帧被拒绝':'结果被拒绝',error);}
  };
  worker.onerror=event=>{if(id===generation&&busy)failed('计算 Worker 出错',event.message||'无法初始化');};
  worker.onmessageerror=()=>{if(id===generation&&busy)failed('结果消息无法读取','没有保留不完整图像');};
  worker.postMessage(activeRequest);
 }catch(error){failed('无法启动',error);}
});
$('demoCancel').addEventListener('click',()=>{stop();clearResult();status('已取消。本次未完成的图像和运行记录已清除。');});
for(const id of transportIds)$(id).addEventListener('input',invalidate);
$('demoExposure').addEventListener('input',()=>{controlLabels();paint();});
$('demoDifferenceMode').addEventListener('input',paintDifference);
$('demoReference').addEventListener('input',()=>{updateReferenceControls();paint();});
$('demoReset').addEventListener('click',()=>{stop();clearResult();setDefaults();status('已恢复推荐场景与采样参数。点击开始计算。');});
$('demoDownload').addEventListener('click',()=>{
 if(!lastResult||!lastExport)return;
 try{const url=URL.createObjectURL(new Blob([lastExport],{type:'application/json'}));downloadURLs.add(url);const a=document.createElement('a');a.href=url;a.download='native-wave-demo-local-run.json';a.click();setTimeout(()=>{URL.revokeObjectURL(url);downloadURLs.delete(url);},1000);}catch(error){status(`无法导出运行记录：${text(error.message)}`);}
});
window.addEventListener('pagehide',()=>{stop();clearResult();for(const url of downloadURLs)URL.revokeObjectURL(url);downloadURLs.clear();});
setDefaults();clearResult();
